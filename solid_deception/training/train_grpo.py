import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from accelerate import init_empty_weights
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig  # type: ignore
from peft import AutoPeftModelForCausalLM, get_peft_model  # type: ignore
from peft.auto import AutoPeftModelForSequenceClassification
from torch.distributed.fsdp import FullStateDictConfig  # type: ignore
from torch.distributed.fsdp import StateDictType  # type: ignore
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
from torch.profiler import ProfilerActivity, profile
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig

from solid_deception.training.grpo_trainer import (  # type: ignore
    MyGRPOConfig,
    MyGRPOTrainer,
)
from solid_deception.training.reward_functions import (  # type: ignore
    GPT4LRRewardFunction,
)
from solid_deception.utils.training import (
    UpdateConfigCallback,
    handle_caching_from_config,
)
from solid_deception.utils.utils import print_memory_summary  # type: ignore

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.serialization.add_safe_globals([torch.nn.modules.linear.Linear])  # type: ignore
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])  # type: ignore
torch.serialization.add_safe_globals(
    [np.ndarray, np.dtype, np.core.multiarray._reconstruct, np.dtypes.UInt32DType]  # type: ignore
)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")


# torch.use_deterministic_algorithms(True)


@dataclass
class MyModelConfig(ModelConfig):
    use_triple_peft: bool = False
    do_categorical_labels: bool = False
    experiment_type: str = "GRPO"
    n_bins: Optional[int] = None


if __name__ == "__main__":
    parser = HfArgumentParser((MyGRPOConfig, MyModelConfig))  # type: ignore
    training_args, model_config = parser.parse_args_into_dataclasses()

    seed = training_args.seed

    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)

    if training_args.profile:
        torch.cuda.memory._record_memory_history(
            max_entries=10_000_000, enabled="all"
        )  # type: ignore

    # remove output_dir if exists
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("local rank is 0")
        # shutil.rmtree(training_args.output_dir, ignore_errors=True)
    fsdp_is_activated = bool(os.environ.get("ACCELERATE_USE_FSDP", False))
    deepspeed_is_activated = bool(os.environ.get("ACCELERATE_USE_DEEPSPEED", False))

    if training_args.logical_batch_size is not None:
        assert training_args.gradient_accumulation_steps == 1
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        gathered_microbatch_size = training_args.per_device_train_batch_size * world_size
        mult, rem = divmod(training_args.logical_batch_size, gathered_microbatch_size)
        logical_batch_size = training_args.logical_batch_size  # type: ignore
        if rem != 0:
            print(
                "WARNING: cannot achieve logical batch size of "
                f"{logical_batch_size} with microbatch size of {gathered_microbatch_size} "
                f"{world_size} devices with PDTBS of {training_args.per_device_train_batch_size} "
                f"setting to {mult * gathered_microbatch_size} instead"
            )
            training_args.logical_batch_size = mult * gathered_microbatch_size
        training_args.gradient_accumulation_steps = mult
        print(f"Set gradient accumulation steps to {training_args.gradient_accumulation_steps}")

    training_args.warmup_ratio = 0.1
    training_args.weight_decay = 1e-3
    training_args.do_train = True
    training_args.do_eval = True
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 100
    training_args.num_train_epochs = 1.0
    training_args.logging_steps = 25
    training_args.ddp_find_unused_parameters = False
    training_args.lr_scheduler_type = "cosine"
    training_args.report_to = ("wandb",)
    training_args.bf16 = True
    training_args.temperature = 1.0
    training_args.missing_eos_penalty = 1.0
    training_args.num_mini_batches = 1
    training_args.save_steps = 30  # Checkpointing is quick, so do it frequently
    training_args.stop_token = "eos"
    training_args.save_total_limit = 2  # Don't clutter with too many checkpoints
    training_args.num_ppo_epochs = 2  # Following RLOO paper, appendix
    training_args.adam_beta1 = 0.95
    training_args.adam_beta2 = 0.98
    # training_args.max_grad_norm = 10_000
    training_args.per_device_eval_batch_size = training_args.per_device_train_batch_size
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    if model_config.attn_implementation is None:  # default
        training_args.bf16 = True
        torch_dtype = torch.bfloat16
    else:
        training_args.bf16 = False
        torch_dtype = torch.float32

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

    adapter_path = training_args.output_dir + "_adapter"
    config_path = Path(training_args.output_dir).parent / Path("configs")
    config_path_ = str(config_path / Path(model_config.experiment_type))  # type: ignore

    cached_adapter_path: Optional[str]
    cached_config_path: Optional[str]

    if not training_args.no_cache:
        loaded_from_cache, (cached_adapter_path, cached_config_path) = handle_caching_from_config(
            [training_args, model_config],
            [adapter_path, config_path_],
            "GRPO",
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path,
            padding_side="left",
            trust_remote_code=model_config.trust_remote_code,
        )
    except OSError as e:
        if "llama" in model_config.model_name_or_path:
            print(
                "Tokenizer not found for model, falling back"
                " to Llama tokenizer since llama was in the model name"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-3B-Instruct",
                padding_side="left",
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            raise e

    # Load models

    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation=(
            "flash_attention_2" if model_config.attn_implementation is None else None
        ),
        torch_dtype=torch_dtype,
        # torch_dtype="auto",
        use_cache=False if training_args.gradient_checkpointing else True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    quantized_extra_kwargs = dict(
        # device_map=get_kbit_device_map() if quantization_config is not None else {"": local_rank},
        quantization_config=quantization_config,
    )

    will_load_from_checkpoint = (
        os.path.exists(training_args.output_dir)
        and get_last_checkpoint(training_args.output_dir)
        and not training_args.no_resume_checkpoint
    )
    print(f"Will load from checkpoint: {will_load_from_checkpoint}")
    load_ref_from_adapter = os.path.exists(
        os.path.join(training_args.sft_model_path, "adapter_config.json")
    )

    if load_ref_from_adapter:
        if model_config.use_triple_peft or model_config.use_peft:
            # We are in the 'full peft' situation with
            # reference, reward and policy as adapters
            ref_policy = None
        else:
            ref_policy = AutoPeftModelForCausalLM.from_pretrained(
                training_args.sft_model_path,
                **model_kwargs,  # type: ignore
                **quantized_extra_kwargs,  # type: ignore
            )
            ref_policy = ref_policy.merge_and_unload()
    else:
        ref_policy = AutoPeftModelForCausalLM.from_pretrained(
            training_args.sft_model_path,
            **model_kwargs,  # type: ignore
            **quantized_extra_kwargs,  # type: ignore
        )

    load_rm_from_adapter = os.path.exists(
        os.path.join(training_args.reward_model_path, "adapter_config.json")
    )

    score_head = None
    if load_rm_from_adapter:
        if model_config.use_triple_peft:
            # We are in the 'full peft' situation with
            # reference, reward and policy as adapters
            reward_model = None
            score_head = torch.load(
                training_args.reward_model_path + "/score_head.pt"  # , map_location="cpu"
            )
            score_head = score_head.to("cpu")
            # The following is needed for the FSDP wrap policy to work properly
            score_head._tied_weights_keys = []
            print("score head!!")
            print(score_head)
        else:
            reward_model = AutoPeftModelForSequenceClassification.from_pretrained(
                training_args.reward_model_path,
                **model_kwargs,  # type: ignore
                **quantized_extra_kwargs,  # type: ignore
            )
            reward_model = reward_model.merge_and_unload()
    else:
        reward_model = AutoPeftModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path,
            **model_kwargs,  # type: ignore
            **quantized_extra_kwargs,  # type: ignore
        )

    if load_ref_from_adapter:
        if fsdp_is_activated:
            loading_handler = init_empty_weights
        else:
            loading_handler = nullcontext
        with loading_handler():
            print("Attempting to load with empty weights")
            policy = AutoPeftModelForCausalLM.from_pretrained(
                training_args.sft_model_path,
                adapter_name="ref",
                **model_kwargs,  # type: ignore
                **quantized_extra_kwargs,  # type: ignore
            )
        print("Loaded model!")
        if model_config.use_triple_peft:
            policy = None
        else:
            policy = policy.merge_and_unload()
    else:
        policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, **model_kwargs, **quantized_extra_kwargs
        )

    if local_rank == 0:
        print(reward_model)

    if model_config.use_peft or model_config.use_triple_peft:
        peft_config = LoraConfig(
            lora_alpha=model_config.lora_r * 2,
            lora_dropout=0,
            r=model_config.lora_r,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            use_rslora=False,
        )

        if model_config.use_peft:
            policy = get_peft_model(
                policy, peft_config, autocast_adapter_dtype=False  # type: ignore
            )  # type: ignore
        else:
            print("attempting to load policy")
            policy = AutoModelForCausalLM.from_pretrained(
                model_config.model_name_or_path, **model_kwargs, **quantized_extra_kwargs
            )
            print("loaded policy!")
            if will_load_from_checkpoint:
                pass  # Will load it when loading checkpoint
            else:
                policy.load_adapter(
                    training_args.sft_model_path, adapter_name="policy", low_cpu_mem_usage=True
                )
            policy.load_adapter(
                training_args.sft_model_path, adapter_name="ref", low_cpu_mem_usage=True
            )
            policy.load_adapter(
                training_args.reward_model_path, adapter_name="rm", low_cpu_mem_usage=True
            )
            if not will_load_from_checkpoint:
                policy.set_adapter("policy")

    if local_rank == 0:
        print_memory_summary(
            policy, ref_policy, training_args.per_device_train_batch_size, training_args.max_length
        )

    do_gt_reward = False
    if do_gt_reward:
        rewards = {
            "deceptive": {True: -1.0, False: 2.0},
            "truthful": {True: 1.0, False: 1.0},
        }
        model_name = "gpt-4o-mini"
        gpt4_lr_rm = GPT4LRRewardFunction(
            lr_path="./detectors/bow_lr",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            system_prompt=open("solid_deception/training/gpt4_reward_prompt.txt", "r").read(),
            tokenizer=tokenizer,
            response_to_rewards=rewards,
            model=model_name,
        )

    if model_config.use_triple_peft:
        # policy.config.pad_token_id = tokenizer("[PAD]")["input_ids"][1]  # type: ignore
        policy.config.pad_token_id = 128001  # type: ignore
        tokenizer.pad_token_id = 128001
    else:
        print(f"Reward model pad token: {reward_model.config.pad_token_id}")  # type: ignore
        print(reward_model.config.pad_token_id)  # type: ignore
        reward_model.config.pad_token_id = tokenizer("[PAD]")["input_ids"][1]  # type: ignore
    # reward_model.config.pad_token_id = tokenizer("[PAD]")
    # print(f"changing reward model pad token to {tokenizer('[PAD]')['input_ids'][1]}")

    ds = load_from_disk(training_args.dataset_name, keep_in_memory=True)  # type: ignore

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element["prompt"],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names["train"],
            num_proc=training_args.dataset_num_proc,
            keep_in_memory=True,
        )

    def shorter_than_max_length(example):
        return len(example["input_ids"]) <= training_args.max_length

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    # with PartialState().local_main_process_first():

    if training_args.debug_training:  # type: ignore
        ds = DatasetDict(
            {
                "train": ds["train"].select(  # type: ignore
                    range(training_args.logical_batch_size // 2)
                ),
                "test": ds["test"].select(  # type: ignore
                    range(training_args.logical_batch_size // 2)
                ),
            }
        )
        training_args.total_episodes = 64
    else:
        # Can't figure out how to stop the eval taking ages,
        # so we use this workaround
        test_len = min(len(ds["test"]), 2048)
        ds = DatasetDict(
            {
                "train": ds["train"],
                "test": ds["test"].select(range(test_len)),  # type: ignore
            }
        )
    ds = prepare_dataset(ds, tokenizer)

    ds_length = len(ds["train"]) + len(ds["test"])
    max_length = max(sorted([len(x["input_ids"]) for x in ds["train"]]))  # type: ignore
    ds = ds.filter(shorter_than_max_length, num_proc=training_args.dataset_num_proc)
    print(
        f"Filtered out {ds_length - len(ds['train'])-len(ds['test'])} examples "
        f"with length greater than {training_args.max_length} "
        f"(originally {ds_length} examples with max length ({max_length}))"
    )
    ################
    # Training
    ################
    REWARD_TO_CATEGORY = {training_args.null_example_reward: 0, 1.0: 1, 2.0: 2, -1.0: 3}
    CATEGORY_TO_REWARD = {v: k for k, v in REWARD_TO_CATEGORY.items()}
    print(f"categorical rewards: {REWARD_TO_CATEGORY}")

    trainer = MyGRPOTrainer(
        train_dataset=ds["train"],  # type: ignore
        config=training_args,
        processing_class=tokenizer,  # type: ignore
        policy=policy,  # type: ignore
        ref_policy=ref_policy,
        reward_function=reward_model,  # type: ignore
        eval_dataset=ds["test"],  # type: ignore
        score_head=score_head,
        rm_adapter_name="rm",
        ref_policy_adapter_name="ref",
        policy_adapter_name="policy",
        categorical_labels=CATEGORY_TO_REWARD if model_config.do_categorical_labels else None,
        callbacks=[
            UpdateConfigCallback(
                training_args,
                model_config,
                local_rank=local_rank,
                config_cache_path=cached_config_path,
            )
        ],
    )

    print(f"FSDP is activated: {fsdp_is_activated}")
    print(f"use peft: {model_config.use_peft}")
    if training_args.profile:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            # Add these:
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile"),
        ) as prof:
            print("Profiling training loop")
            trainer.train()
        # else:
        #     if local_rank == 0:
        #         trainer._load_from_checkpoint(
        #             "/home/dev/persistent/code/solid_deception/outputs/rd-394-8b-seed-0--cat-rm--grpo-8hvsz/policy/checkpoint-4",
        #             model=trainer.accelerator.unwrap_model(trainer.model),
        #         )
        #     else:
        #         import time

        #         time.sleep(600)

    else:
        trainer.train()

    print("Done training!")
    trainer.accelerator.wait_for_everyone()
    os.system(f"rm -rf {adapter_path}")
    trainer.accelerator.wait_for_everyone()
    # trainer.save_model(adapter_path)
    # If we are doing FSDP, we want to save on all processes to collect shards.
    # Otherwise, save on one process to avoid collisions
    if fsdp_is_activated:
        if model_config.use_peft or model_config.use_triple_peft:
            os.system(f"rm -rf {adapter_path}")
            trainer.accelerator.wait_for_everyone()
            os.makedirs(f"{adapter_path}", exist_ok=True)
            print(f"attempting to save adapters to {adapter_path}")

            trainer.accelerator.wait_for_everyone()
            # Attempting to save on cpu with main process...
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = trainer.model.state_dict()
                if local_rank == 0:
                    cpu_state_dict = {k: v for k, v in cpu_state_dict.items() if "lora" in k}
                    cpu_state_dict = {
                        k.replace(".default", ""): v for k, v in cpu_state_dict.items()
                    }
                    torch.save(cpu_state_dict, os.path.join(adapter_path, "adapter_model.bin"))
                    active_adapter = "policy"
                    current_peft_config = trainer.model.peft_config[active_adapter]
                    current_peft_config.save_pretrained(adapter_path)
                # trainer._save(adapter_path, state_dict=cpu_state_dict)  # noqa

            print(f"saved on process {local_rank}")
            trainer.accelerator.wait_for_everyone()
        else:
            print("Saving model across different processes")
            assert trainer.accelerator.state.fsdp_plugin is not None
            # We have to do this strange approach to set the plugin state dict type
            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.state_dict_type = StateDictType.FULL_STATE_DICT
            fsdp_plugin.set_state_dict_type()
            print("changed state dict type!")
            trainer.accelerator.wait_for_everyone()
            trainer.save_model(adapter_path)
    else:  # DDP case
        if local_rank == 0:
            # This is all quite badly documented but I believe this should
            # save the policy adapter, which can be loaded in again
            # via
            # model = AutoModel... ; model.load_adapter($adapter_path)
            model = trainer.accelerator.unwrap_model(trainer.model)
            model.set_adapter("policy")
            model.save_pretrained(adapter_path)  # type: ignore
            if cached_adapter_path:
                model.save_pretrained(cached_adapter_path)  # type: ignore

    trainer.accelerator.wait_for_everyone()
