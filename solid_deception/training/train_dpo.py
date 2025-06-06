import multiprocessing
import os
import random
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from accelerate import init_empty_weights
from datasets import DatasetDict, load_from_disk
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model  # type: ignore
from peft.utils.other import fsdp_auto_wrap_policy
from torch.distributed.fsdp import StateDictType  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer, ModelConfig, get_kbit_device_map, get_peft_config
from trl.commands.cli_utils import DPOScriptArguments, TrlParser

from solid_deception.training import collators  # type: ignore

# from solid_deception.training.rloo_trainer import fsdp_auto_wrap_policy
from solid_deception.utils.training import UpdateConfigCallback  # type: ignore
from solid_deception.utils.training import handle_caching_from_config

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")

torch.set_float32_matmul_precision("high")


class NoSaveAtEndCallback(TrainingArguments):
    """Callback to prevent saving checkpoints at the end of training."""

    def on_train_end(self, args, state, control, **kwargs):
        # Override the default behavior to avoid saving
        control.should_save = False
        return control


@dataclass
class MyDPOScriptArguments(DPOScriptArguments):
    debug_training: bool = False
    logical_batch_size: Optional[int] = None
    experiment_set_name: Optional[str] = None
    experiment_type: str = "DPO"
    kl_beta: float = 0.2
    no_cache: bool = False
    rpo_alpha: float = 0.2
    null_answer_path: str = "/workspace/data/null_answers.txt"


if __name__ == "__main__":
    parser = TrlParser((MyDPOScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()  # type: ignore
    fsdp_is_activated = bool(os.environ.get("ACCELERATE_USE_FSDP", False))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    seed = training_args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)

    if args.logical_batch_size is not None:  # type: ignore
        assert training_args.gradient_accumulation_steps == 1
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        gathered_microbatch_size = training_args.per_device_train_batch_size * world_size
        logical_batch_size = args.logical_batch_size  # type: ignore
        mult, rem = divmod(logical_batch_size, gathered_microbatch_size)
        if rem != 0:
            print(
                "WARNING: cannot achieve logical batch size of "
                f"{logical_batch_size} with microbatch size of {gathered_microbatch_size} "
                f"{world_size} devices with PDTBS of {training_args.per_device_train_batch_size} "
                f"setting to {mult * gathered_microbatch_size} instead"
            )
            args.logical_batch_size = mult * gathered_microbatch_size  # type: ignore
        training_args.gradient_accumulation_steps = mult
        print(f"Set gradient accumulation steps to {training_args.gradient_accumulation_steps}")

    adapter_save_path = training_args.output_dir + "_adapter"
    config_path = Path(training_args.output_dir).parent / Path("configs")
    config_path_ = str(config_path / Path(args.experiment_type))  # type: ignore

    cached_adapter_path: Optional[str]
    cached_config_path: Optional[str]

    if not args.no_cache:  # type: ignore
        loaded_from_cache, (cached_adapter_path, cached_config_path) = handle_caching_from_config(
            [args, training_args, model_config],
            [adapter_save_path, config_path_],
            "DPO",
            local_rank == 0,
        )
        if loaded_from_cache:
            exit()
    else:
        cached_adapter_path = None
        cached_config_path = None

    ################
    # Model & Tokenizer
    ################

    if fsdp_is_activated:
        loading_handler = init_empty_weights
    else:
        loading_handler = nullcontext
    loading_handler = nullcontext

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    print(f"torch dtype is {torch_dtype}")
    torch_dtype = torch.bfloat16  # Override since the --bf16 argument wasn't being picked up

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
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    load_sft_from_adapter = os.path.exists(
        os.path.join(model_config.model_name_or_path, "adapter_config.json")
    )

    with loading_handler():
        if load_sft_from_adapter:
            print("Loading from adapter")
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_config.model_name_or_path,
                adapter_name="ref",
                **model_kwargs,  # type: ignore
            )
            print("Loaded model!")
            model = model.merge_and_unload()
            print("Merged adapter weights")
            if not model_config.use_peft:
                for n, p in model.named_parameters():
                    p.requires_grad = True
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_name_or_path, **model_kwargs
            )
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
            )

            model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)  # type: ignore

            model_ref = None
        else:
            model_ref = AutoModelForCausalLM.from_pretrained(
                model_config.model_name_or_path, **model_kwargs
            )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        print("Assuming you are using Llama to set the pad token!")
        tokenizer.pad_token_id = tokenizer("<|end_of_text|>")["input_ids"][1]
        tokenizer.eot_id = 128009
    else:
        tokenizer.pad_token = tokenizer.eos_token
        raise NotImplementedError

    ################
    # Dataset
    ################

    ds = load_from_disk(args.dataset_name, keep_in_memory=True)  # type: ignore
    if args.debug_training:  # type: ignore
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

    def process(row):
        return row

    ds = ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    train_dataset = ds["train"]
    test_dataset = ds["test"]

    collator = collators.PromptAugmentedPreferenceCollator(  # type: ignore
        pad_token_id=tokenizer.pad_token_id
    )

    ################
    # Training
    ################
    training_args = DPOConfig(
        seed=training_args.seed,
        beta=args.kl_beta,  # type: ignore
        output_dir=training_args.output_dir,
        report_to="wandb",
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        gradient_checkpointing=training_args.gradient_checkpointing,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        do_train=True,
        do_eval=True,
        logging_steps=5,
        eval_strategy=training_args.eval_strategy,
        eval_steps=training_args.eval_steps,
        max_prompt_length=1024,
        max_target_length=1024,
        max_length=1024,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if training_args.gradient_checkpointing else None
        ),
        label_smoothing=training_args.label_smoothing_factor,
        ddp_find_unused_parameters=False,
        generate_during_eval=not fsdp_is_activated,
        learning_rate=training_args.learning_rate,
        bf16=True,
        weight_decay=1e-4,
        warmup_ratio=0.2,
        num_train_epochs=2.0,
        run_name=training_args.run_name,
        lr_scheduler_type="cosine",
        rpo_alpha=args.rpo_alpha,  # type: ignore
        remove_unused_columns=False,
        adam_beta1=0.95,
        adam_beta2=0.98,
        save_steps=500_000_000_000,  # Don't checkpoint during training
        save_strategy="no",
    )

    print("Starting training!")
    trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=test_dataset,  # type: ignore
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[
            UpdateConfigCallback(
                args,
                training_args,
                model_config,
                local_rank=local_rank,
                config_cache_path=cached_config_path,
            ),
        ],
    )

    if fsdp_is_activated:
        fsdp_plugin = trainer.accelerator.state.fsdp_plugin  # type: ignore

    if local_rank == 0:
        time.sleep(int(local_rank) * 1)
        print(f"LOCAL_RANK: {local_rank}")
        print("-" * 80)

        examples = next(iter(trainer.get_train_dataloader()))
        print('prompt:, {examples["prompt"][0]}')
        print('chosen: {examples["chosen"][0]}')
        print('rejected: {examples["rejected"][0]}')
        print(tokenizer.batch_decode(examples["prompt_input_ids"]))

    examples = next(iter(trainer.get_train_dataloader()))
    print(examples)
    print(tokenizer.batch_decode(examples["prompt_input_ids"]))

    def save_with_lock(data, path):
        # There is a strange bug which this works as a workaround for
        # On azure we get crashes if saving on the non-main process.
        # since we have to call save_pretrained with all processes to gather
        # the shards, we have a custom save function which drops on the non-main process
        if local_rank != 0:
            return
        print(f"saving on local rank {local_rank}")
        return torch.save(data, path, _use_new_zipfile_serialization=False)

    trainer.accelerator.wait_for_everyone()
    time.sleep(10)
    trainer.train()
    print("Done training!")

    # If we are doing FSDP, we want to save on all processes to collect shards.
    # Otherwise, save on one process to avoid collisions
    if fsdp_is_activated:
        print("Saving model across different processes")
        assert trainer.accelerator.state.fsdp_plugin is not None
        # We have to do this strange approach to set the plugin state dict type
        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.state_dict_type = StateDictType.FULL_STATE_DICT
        fsdp_plugin.set_state_dict_type()
        print("changed state dict type!")
        trainer.accelerator.wait_for_everyone()
        trainer.save_model(adapter_save_path)
    else:  # DDP case
        if local_rank == 0:
            trainer.model.save_pretrained(  # type: ignore
                adapter_save_path,
                is_main_process=local_rank == 0,
                safe_serialization=False,
                save_function=save_with_lock,
            )  # This will work to save the adapter
            if cached_adapter_path:
                trainer.model.save_pretrained(  # type: ignore
                    cached_adapter_path,
                    is_main_process=local_rank == 0,
                    safe_serialization=False,
                    save_function=save_with_lock,
                )  # This will work to save the adapter
            print("")
