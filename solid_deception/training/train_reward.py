import os
import random
import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from accelerate import PartialState
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig  # type: ignore
from torch.distributed.fsdp import FullStateDictConfig  # type: ignore
from torch.distributed.fsdp import StateDictType  # type: ignore
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map
from trl.commands.cli_utils import RewardScriptArguments  # type: ignore

from solid_deception.utils import utils  # type: ignore
from solid_deception.utils.training import UpdateConfigCallback  # type: ignore
from solid_deception.utils.training import handle_caching_from_config  # type: ignore

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
tqdm.pandas()
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")


@dataclass
class MyRewardScriptArguments(RewardScriptArguments):
    debug_training: bool = False
    null_answer_path: str = "/workspace/data/null_answers.txt"
    logical_batch_size: Optional[int] = None


@dataclass
class MyRewardConfig(RewardConfig):
    eval_steps: float | None = 200  # Needs to be float for typechecker..
    max_length: int | None = 896
    warmup_ratio: float = 0.1
    weight_decay: float = 1e-2
    do_train: bool = True
    do_eval: bool = True
    eval_strategy: str = "steps"
    num_train_epochs: float = 4
    learning_rate: float = 1e-5
    logging_steps: float = 5
    ddp_find_unused_parameters: bool | None = False
    lr_scheduler_type: str = "cosine"
    center_rewards_coefficient: float = 0.001  # type: ignore
    report_to: str | None = "wandb"  # type: ignore
    bf16: bool = True
    experiment_set_name: Optional[str] = None
    experiment_type: str = "RM_BT"
    no_cache: bool = False


if __name__ == "__main__":
    parser = HfArgumentParser(
        (MyRewardScriptArguments, MyRewardConfig, ModelConfig)
    )  # type: ignore
    args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    fsdp_is_activated = bool(os.environ.get("ACCELERATE_USE_FSDP", False))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    seed = training_args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    training_args.adam_beta1 = 0.95
    training_args.adam_beta2 = 0.98
    training_args.save_steps = 10_000_000_000
    ################
    # Model & Tokenizer
    ################

    if args.logical_batch_size is not None:  # type: ignore
        assert training_args.gradient_accumulation_steps == 1
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        gathered_microbatch_size = training_args.per_device_train_batch_size * world_size
        mult, rem = divmod(args.logical_batch_size, gathered_microbatch_size)
        if rem != 0:
            print(
                "WARNING: cannot achieve logical batch size of "
                f"{args.logical_batch_size} with microbatch size of {gathered_microbatch_size} "
                f"{world_size} devices with PDTBS of {training_args.per_device_train_batch_size} "
                f"setting to {mult * gathered_microbatch_size} instead"
            )
            args.logical_batch_size = mult * gathered_microbatch_size
        training_args.gradient_accumulation_steps = mult
        print(f"Set gradient accumulation steps to {training_args.gradient_accumulation_steps}")

    cached_adapter_path: Optional[str]
    cached_config_path: Optional[str]
    config_path = Path(training_args.output_dir).parent / Path("configs")
    config_path_ = str(config_path / Path(training_args.experiment_type))  # type: ignore
    adapter_path = training_args.output_dir + "_adapter"

    if not training_args.no_cache:  # type: ignore
        loaded_from_cache, (cached_adapter_path, cached_config_path) = handle_caching_from_config(
            [args, training_args, model_config],
            [adapter_path, config_path_],
            "RM",
            local_rank == 0,
            exclude_strs=["SFT/"],
        )
        if loaded_from_cache:
            sys.exit()
    else:
        cached_adapter_path = None
        cached_config_path = None

    torch_dtype = torch.bfloat16  # Override since the --bf16 argument wasn't being picked up
    model_config.lora_task_type = "SEQ_CLS"

    if model_config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    else:
        quantization_config = None
    model_kwargs = dict(
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        num_labels=1,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)

    if model_config.use_peft:
        peft_config = LoraConfig(
            lora_alpha=model_config.lora_r * 2,
            lora_dropout=0,
            r=model_config.lora_r,
            bias="none",
            target_modules="all-linear",
            task_type="SEQ_CLS",
            modules_to_save=["score"],
        )
        model.add_adapter(peft_config)
    else:
        peft_config = None

    # We have to be careful choosing the pad token here.
    # The default in trl implementations is to use the 'eos_token'
    # but the chat format for llama has the eos_token '<|eot_id|>' after
    # every turn in the conversation. A lot of the methods in trl and
    # transformers library assume that the pad_token only appears in padding
    # positions. Here it's important because we pick the first padding token to
    # use as the CLS token.
    # So we use the reserved '<|end_of_text|>' token which shouldn't appear
    # in the training data at all

    tokenizer.pad_token_id = tokenizer("<|end_of_text|>")["input_ids"][1]
    tokenizer.eot_id = 128009
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than "
            "`SEQ_CLS` for PEFT. This will lead to silent bugs "
            "Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT."
        )
    #############################
    # Load and preprocess dataset
    #############################
    ds = load_from_disk(args.dataset_name)  # type: ignore

    def preprocess_function(examples):
        new_examples: Dict[str, List[List[int]]] = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)
            if tokenized_chosen["input_ids"][-1] != tokenizer.eot_id:
                tokenized_chosen["input_ids"] += [tokenizer.eot_id]
                tokenized_chosen["attention_mask"] += [tokenized_chosen["attention_mask"][-1]]
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])

            if tokenized_rejected["input_ids"][-1] != tokenizer.eot_id:
                tokenized_rejected["input_ids"] += [tokenizer.eot_id]
                tokenized_rejected["attention_mask"] += [tokenized_rejected["attention_mask"][-1]]
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])

        return new_examples

    with PartialState().local_main_process_first():
        # Wrap inputs with chat template.
        # This assumes the chosen/rejected columns are in the OpenAI messages format.
        # chosen_fn = conversations_formatting_function(tokenizer, "chosen")
        # rejected_fn = conversations_formatting_function(tokenizer, "rejected")
        if args.debug_training:
            ds = DatasetDict(
                {
                    "train": ds["train"].select(range(100)),  # type: ignore
                    "test": ds["test"].select(range(100)),  # type: ignore
                }
            )
        else:
            null_answers = open(args.null_answer_path, "r").readlines()  # type: ignore
            null_answers = [a[:-1] if a[-1] == "\n" else a for a in null_answers]
            for example, null_answer in zip(ds["train"], null_answers):
                new_example = deepcopy(example)
                new_example["rejected"] = null_answer  # type: ignore
                ds["train"] = ds["train"].add_item(example)  # type: ignore

            if not args.debug_training:  # type: ignore
                # If we just add all the null examples to the start of the dataset,
                # this won't train very much on them since the LR is low
                for i in range(1_000, len(ds["train"]), len(ds["train"]) // 5):
                    for j, (example, null_answer) in enumerate(
                        zip(ds["train"][i // 2 :]["rejected"], null_answers)  # type: ignore
                    ):
                        new_example = deepcopy(ds["train"][j])
                        new_example["rejected"] = null_answer
                        ds["train"] = ds["train"].add_item(new_example)  # type: ignore

            for example, null_answer in zip(ds["test"], null_answers):
                new_example = deepcopy(example)
                new_example["rejected"] = null_answer  # type: ignore
                ds["test"] = ds["test"].add_item(new_example)  # type: ignore

        ds = ds.map(
            lambda x: {
                "chosen": x["prompt"] + x["chosen"],
                "rejected": x["prompt"] + x["rejected"],
            },
            num_proc=training_args.dataset_num_proc,
        )
        # Tokenize inputs
        ds = ds.map(
            preprocess_function,
            batched=True,
            num_proc=training_args.dataset_num_proc,
        )
        # if not args.debug_training:
        #     # Add examples with non-terminating sequences
        #     subset_size = min(int(len(ds["train"]) * 0.05), 400)  # 5% or max 100 examples
        #     subset = ds["train"].select(  # type: ignore
        #         list(np.random.permutation(len(ds["train"])))[:subset_size]
        #     )

        #     new_examples: Dict[str, List[Any]] = {
        #         "input_ids_chosen": [],
        #         "attention_mask_chosen": [],
        #         "input_ids_rejected": [],
        #         "attention_mask_rejected": [],
        #     }

        #     for example in subset:
        #         new_example = deepcopy(example)
        #         # Copy rejected sequence to chosen (with EOT)
        #         new_example["input_ids_chosen"] = example["input_ids_rejected"]  # type: ignore
        #         new_example["attention_mask_chosen"] = example[  # type: ignore
        #             "attention_mask_rejected"
        #         ]

        #         # Copy chosen sequence to rejected but remove EOT
        #         rejected_seq = example["input_ids_chosen"]  # type: ignore
        #         if rejected_seq[-1] == tokenizer.eot_id:
        #             rejected_seq = rejected_seq[:-1]
        #         new_example["input_ids_rejected"] = rejected_seq  # type: ignore
        #         new_example["attention_mask_rejected"] = example[  # type: ignore
        #             "attention_mask_chosen"
        #         ][: len(rejected_seq)]
        #         ds["train"] = ds["train"].add_item(new_example)  # type: ignore

        #     test_subset_size = min(int(len(ds["test"]) * 0.05), 400)  # 5% or max 100 examples
        #     test_subset = ds["test"].select(  # type: ignore
        #         list(np.random.permutation(len(ds["test"])))[:test_subset_size]
        #     )

        #     new_examples = {
        #         "input_ids_chosen": [],
        #         "attention_mask_chosen": [],
        #         "input_ids_rejected": [],
        #         "attention_mask_rejected": [],
        #     }

        #     for example in test_subset:
        #         new_example = deepcopy(example)
        #         # Copy rejected sequence to chosen (with EOT)
        #         new_example["input_ids_chosen"] = example["input_ids_rejected"]  # type: ignore
        #         new_example["attention_mask_chosen"] = example[  # type: ignore
        #             "attention_mask_rejected"
        #         ]

        #         # Copy chosen sequence to rejected but remove EOT
        #         rejected_seq = example["input_ids_chosen"]  # type: ignore
        #         if rejected_seq[-1] == tokenizer.eot_id:
        #             rejected_seq = rejected_seq[:-1]
        #         new_example["input_ids_rejected"] = rejected_seq  # type: ignore
        #         new_example["attention_mask_rejected"] = example[  # type: ignore
        #             "attention_mask_chosen"
        #         ][: len(rejected_seq)]
        #         ds["test"] = ds["test"].add_item(new_example)  # type: ignore

        #     max_test_size = min(int(len(ds["test"])), 2_000)
        #     ds["test"] = ds["test"].select(range(max_test_size))  # type: ignore

        #     # breakpoint()
        #     # # Add new examples to dataset
        #     # for k, v in new_examples.items():
        #     #     ds["train"] = ds["train"].add_item(example)  # type: ignore

        # Filter out examples that are too long
        ds = ds.filter(
            lambda x: len(x["input_ids_chosen"]) <= training_args.max_length
            and len(x["input_ids_rejected"]) <= training_args.max_length,
            num_proc=training_args.dataset_num_proc,
        )

        # shuffle training data
        ds["train"] = ds["train"].shuffle(seed=seed)  # type: ignore

    ##########
    # Training
    ##########

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,  # type: ignore
        args=training_args,
        train_dataset=ds["train"],  # type: ignore
        eval_dataset=ds["test"],  # type: ignore
        callbacks=[
            UpdateConfigCallback(
                args,
                training_args,
                model_config,
                local_rank=local_rank,
                config_cache_path=cached_config_path,
            )
        ],
    )

    ##########
    # Debug printouts
    ##########
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        lengths = []
        trainloader = iter(trainer.get_train_dataloader())
        for i, example_batch in enumerate(trainloader):
            lengths += [example_batch["input_ids_chosen"].shape[-1]]
            lengths += [example_batch["input_ids_rejected"].shape[-1]]
            if i == 0:
                print(example_batch)
                print(tokenizer.batch_decode(example_batch["input_ids_chosen"]))
                print(tokenizer.batch_decode(example_batch["input_ids_rejected"]))
                print(
                    tokenizer.batch_decode(
                        example_batch["input_ids_chosen"] * example_batch["attention_mask_chosen"]
                    )
                )
                print(
                    tokenizer.batch_decode(
                        example_batch["input_ids_rejected"]
                        * example_batch["attention_mask_rejected"]
                    )
                )
        print(sorted(lengths)[::-1][:100])
        del trainloader

    ##########
    # Num params printout
    ##########

    print("-" * 80)
    print("total trainable params: ", "{:,}".format(utils.count_trainable_parameters(model)))

    total_params = utils.count_quantized_model_parameters(model)
    print("total param count: ", "{:,}".format(total_params))

    trainable_params_bytes = utils.count_trainable_parameters_bits(model) / 8
    total_params_bytes = utils.count_parameters_bits(model) / 8
    activation_bytes = (
        4
        * training_args.per_device_train_batch_size
        * 2048
        * model.config.hidden_size  # type: ignore
        * model.config.num_hidden_layers  # type: ignore
    )

    print("Trainable param size: ~", utils.pretty_print(trainable_params_bytes))
    print("Total param size: ~", utils.pretty_print(total_params_bytes))
    print(
        "Total Memory: ",
        utils.pretty_print(8 * trainable_params_bytes + total_params_bytes + activation_bytes),
    )
    print("-" * 80)

    if model_config.use_peft:
        # handle PEFT+FSDP case
        if fsdp_is_activated:
            from peft.utils.other import fsdp_auto_wrap_policy

            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)  # type: ignore

    trainer.train()

    def save_with_lock(data, path):
        # There is a strange bug which this works as a workaround for
        # On azure we get crashes if saving on the non-main process.
        # since we have to call save_pretrained with all processes to gather
        # the shards, we have a custom save function which drops on the non-main process
        if local_rank != 0:
            return
        print(f"saving on local rank {local_rank}")
        return torch.save(data, path, _use_new_zipfile_serialization=True)

    ############################
    # Save model and push to Hub
    ############################
    # Save and merge adapters on rank 0
    # If we are doing FSDP, we want to save on all processes to collect shards.
    # Otherwise, save on one process to avoid collisions
    if bool(os.environ.get("ACCELERATE_USE_FSDP", False)):
        if model_config.use_peft:
            os.system(f"rm -rf {adapter_path}")
            trainer.accelerator.wait_for_everyone()
            os.makedirs(f"{adapter_path}", exist_ok=True)
            print(f"attempting to save adapters to {adapter_path}")

            trainer.accelerator.wait_for_everyone()
            # Attempting to save on cpu with main process...
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = model.state_dict()
                if local_rank == 0:
                    cpu_state_dict = {
                        k.replace(".default", ""): v for k, v in cpu_state_dict.items()
                    }
                    cpu_state_dict = {k: v for k, v in cpu_state_dict.items() if "lora" in k}
                    torch.save(cpu_state_dict, os.path.join(adapter_path, "adapter_model.bin"))
                    active_adapter = "default"
                    current_peft_config = trainer.model.peft_config[active_adapter]  # type: ignore
                    current_peft_config.save_pretrained(adapter_path)
        else:
            raise NotImplementedError

    else:  # DDP case
        if local_rank == 0:
            # trainer.save_model(adapter_path)
            trainer.model.save_pretrained(  # type: ignore
                adapter_path,
                is_main_process=local_rank == 0,
                safe_serialization=False,
                save_function=save_with_lock,
            )  # This will work to save the adapter
            print("SAVED!")
            print(f"SAVED! to {adapter_path}")
            if cached_adapter_path is not None:
                trainer.model.save_pretrained(  # type: ignore
                    cached_adapter_path,
                    is_main_process=local_rank == 0,
                    safe_serialization=False,
                    save_function=save_with_lock,
                )

            score_to_save = trainer.accelerator.unwrap_model(
                model
            ).score.modules_to_save.default.cpu()
            save_with_lock(score_to_save, f"{adapter_path}" + "/score_head.pt")
            if cached_adapter_path is not None:
                save_with_lock(score_to_save, f"{cached_adapter_path}" + "/score_head.pt")
    trainer.accelerator.wait_for_everyone()
