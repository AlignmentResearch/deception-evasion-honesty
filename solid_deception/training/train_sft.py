import logging
import multiprocessing
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import torch
import transformers
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig  # type: ignore
from torch.distributed.fsdp import FullStateDictConfig  # type: ignore
from torch.distributed.fsdp import StateDictType  # type: ignore
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

# from peft import peft_model_casting_to_bf16, prepare_model_for_kbit_training
from trl import (
    DataCollatorForCompletionOnlyLM,
    ModelConfig,
    SFTConfig,
    SFTScriptArguments,
    SFTTrainer,
    get_kbit_device_map,
    get_peft_config,
)
from trl.commands.cli_utils import TrlParser

from solid_deception.utils.training import (  # type: ignore
    UpdateConfigCallback,
    handle_caching_from_config,
)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger(__name__)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")


@dataclass
class MySFTScriptArguments(SFTScriptArguments):
    debug_training: bool = False
    experiment_set_name: Optional[str] = None
    logical_batch_size: Optional[int] = None
    experiment_type: str = "SFT"
    no_cache: bool = False


if __name__ == "__main__":
    parser = TrlParser((MySFTScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()  # type: ignore

    seed = training_args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)

    fsdp_is_activated = bool(os.environ.get("ACCELERATE_USE_FSDP", False))
    deepspeed_is_activated = bool(os.environ.get("ACCELERATE_USE_DEEPSPEED", False))
    fsdp_is_activated = bool(os.environ.get("ACCELERATE_USE_FSDP", False))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.system(f"rm -rf {training_args.output_dir}")

    # Calculate the batch size
    if args.logical_batch_size is not None:  # type: ignore
        assert training_args.gradient_accumulation_steps == 1
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        gathered_mb_size = training_args.per_device_train_batch_size * world_size
        mult, rem = divmod(args.logical_batch_size, gathered_mb_size)  # type: ignore
        logical_batch_size = args.logical_batch_size  # type: ignore
        if rem != 0:
            print(
                "WARNING: cannot achieve logical batch size of "
                f"{logical_batch_size} with microbatch size of {gathered_mb_size} "  # type: ignore
                f"{world_size} devices with PDTBS of {training_args.per_device_train_batch_size} "
                f"setting to {mult * gathered_mb_size} instead"
            )
            args.logical_batch_size = mult * gathered_mb_size  # type: ignore
        training_args.gradient_accumulation_steps = mult
        print(f"Set gradient accumulation steps to {training_args.gradient_accumulation_steps}")

    adapter_path = training_args.output_dir + "_adapter"

    if fsdp_is_activated:
        # We will always use fsdp-level gradient checkpointing
        # and this crashes if we use trainer-level checkpointing too
        training_args.gradient_checkpointing = False

    training_args = SFTConfig(
        output_dir=training_args.output_dir,
        report_to="wandb",
        gradient_checkpointing=training_args.gradient_checkpointing,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        do_train=True,
        do_eval=True,
        logging_steps=5,
        eval_strategy="epoch",
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if training_args.gradient_checkpointing else None
        ),
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ddp_find_unused_parameters=False,
        learning_rate=training_args.learning_rate,
        bf16=True,
        weight_decay=1e-2,
        warmup_ratio=0.2,
        num_train_epochs=training_args.num_train_epochs,
        neftune_noise_alpha=5,
        max_seq_length=896,
        group_by_length=True,
        eval_accumulation_steps=16,
        bf16_full_eval=True,
        run_name=training_args.run_name,
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.95,
        skip_memory_metrics=True,
        save_strategy="no",
    )
    config_path = Path(training_args.output_dir).parent / Path("configs")
    config_path_ = str(config_path / Path(args.experiment_type))  # type: ignore

    cached_adapter_path: Optional[str]
    cached_config_path: Optional[str]

    if not args.no_cache:  # type: ignore
        loaded_from_cache, (cached_adapter_path, cached_config_path) = handle_caching_from_config(
            [args, training_args, model_config],
            [adapter_path, config_path_],
            "SFT",
            local_rank == 0,
            exclude_strs=["RM/"],  # This doesn't depend on the reward model, so exclude
        )
        if loaded_from_cache:
            exit()
    else:
        cached_adapter_path = None
        cached_config_path = None

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    print(f"torch dtype is {torch_dtype}")
    torch_dtype = torch.bfloat16  # Override since the --bf16 argument wasn't being picked up

    quantization_config = None
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=False,  # Since we are not generating, no need for the cache here
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # if model_config.lora_r is not None:
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    if model_config.use_peft:
        peft_config = get_peft_config(model_config)
        peft_config = LoraConfig(
            lora_alpha=model_config.lora_r * 2,
            lora_dropout=0,
            r=model_config.lora_r,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            use_rslora=False,
            # init_lora_weights="pissa",
        )
        model.add_adapter(peft_config)
    else:
        peft_config = None
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer("<|end_of_text|>")["input_ids"][1]

    ################
    # Dataset
    ################

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["prompt"])):
            text = example["prompt"][i] + example["chosen"][i]
            # We are not including the eot id in the responses, so need to add this in
            text += "<|eot_id|>"
            output_texts.append(text)
        return output_texts

    ds = load_from_disk(args.dataset_name, keep_in_memory=False)  # type: ignore
    if args.debug_training:  # type: ignore
        if len(ds["train"]) > 2048:
            ds = DatasetDict(
                {
                    "train": ds["train"].select(range(2048)),  # type: ignore
                    "test": ds["test"].select(range(2048)),  # type: ignore
                }
            )

    def process(row):
        return row

    ds = ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    # ds = ds["train"].train_test_split(test_size=32, shuffle=True)
    train_dataset = ds["train"]
    test_dataset = ds["test"]

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    log_level = "DEBUG"
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)  # type: ignore
    transformers.utils.logging.set_verbosity(log_level)  # type: ignore
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Total number of parameters: {n_params}")

    response_template = "assistant<|end_header_id|>"
    # If we encode separately, this becomes [128000, 198, 78191, 128007]
    # but in the context of the response, it is [128006, 198, 78191, 128007]
    response_template_id = tokenizer(response_template)["input_ids"][1:]
    collator = DataCollatorForCompletionOnlyLM(response_template_id, tokenizer=tokenizer)

    ################
    # Training
    ################

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=test_dataset,  # type: ignore
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
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

    if local_rank == 0:
        lengths = []
        trainloader = iter(trainer.get_train_dataloader())
        for example_batch in trainloader:
            lengths += [example_batch["input_ids"].shape[-1]]

        print(sorted(lengths)[::-1][:100])
        print(example_batch["input_ids"])  # type: ignore
        print(tokenizer.decode(example_batch["input_ids"][0]))  # type: ignore
        del trainloader
        # breakpoint()

    if model_config.use_peft:
        # handle PEFT+FSDP case
        if fsdp_is_activated:
            from peft.utils.other import fsdp_auto_wrap_policy

            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)  # type: ignore

    trainer.accelerator.wait_for_everyone()
    trainer.train()  # type: ignore

    def save_with_lock(data, path):
        # There is a strange bug which this works as a workaround for
        # On azure we get crashes if saving on the non-main process.
        # since we have to call save_pretrained with all processes to gather
        # the shards, we have a custom save function which drops on the non-main process
        if local_rank != 0:
            return
        print(f"saving on local rank {local_rank}")
        return torch.save(data, path, _use_new_zipfile_serialization=False)

    tmp_save_dir = training_args.output_dir + "_adapter"
    if fsdp_is_activated:
        if model_config.use_peft:
            os.system(f"rm -rf {tmp_save_dir}")
            trainer.accelerator.wait_for_everyone()
            os.makedirs(f"{tmp_save_dir}", exist_ok=True)
            print(f"attempting to save adapters to {tmp_save_dir}")

            trainer.accelerator.wait_for_everyone()
            # Attempting to save on cpu with main process...
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = model.state_dict()
                print("Fetched state dict on all devices")
                if local_rank == 0:
                    cpu_state_dict = {
                        k.replace(".default", ""): v for k, v in cpu_state_dict.items()
                    }
                    cpu_state_dict = {k: v for k, v in cpu_state_dict.items() if "lora" in k}
                    print("Loaded CPU state dict to memory on device 0")
                    torch.save(cpu_state_dict, os.path.join(tmp_save_dir, "adapter_model.bin"))
                    print(cpu_state_dict.keys())
                    active_adapter = "default"
                    current_peft_config = trainer.model.peft_config[active_adapter]  # type: ignore
                    current_peft_config.save_pretrained(tmp_save_dir)
                # trainer._save(tmp_save_dir, state_dict=cpu_state_dict)  # noqa

            print(f"saved on process {local_rank}")
            trainer.accelerator.wait_for_everyone()
    else:
        tmp_save_dir = training_args.output_dir + "_adapter"
        trainer.model.save_pretrained(  # type: ignore
            tmp_save_dir,
            is_main_process=local_rank == 0,
            safe_serialization=False,
            save_function=save_with_lock,
        )  # This will work to save the adapter
        trainer.model.save_pretrained(  # type: ignore
            cached_adapter_path,  # type: ignore
            is_main_process=local_rank == 0,
            safe_serialization=False,
            save_function=save_with_lock,
        )
    exit()
