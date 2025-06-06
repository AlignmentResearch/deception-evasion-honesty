import multiprocessing
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig  # type: ignore
from sklearn.metrics import confusion_matrix
from torch.distributed.fsdp import FullStateDictConfig  # type: ignore
from torch.distributed.fsdp import FullyShardedDataParallel  # type: ignore
from torch.distributed.fsdp import StateDictType  # type: ignore
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from trl import ModelConfig, SFTScriptArguments, get_kbit_device_map, get_peft_config
from trl.commands.cli_utils import TrlParser

from solid_deception.training.models import (  # type: ignore
    MyLlamaForSequenceClassification,
)
from solid_deception.utils.training import UpdateConfigCallback  # type: ignore
from solid_deception.utils.training import handle_caching_from_config  # type: ignore

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")


@dataclass
class MySFTScriptArguments(SFTScriptArguments):
    debug_training: bool = False
    only_score: bool = False
    do_categorical_labels: bool = False
    do_mlp_head: bool = False
    only_lies: bool = False
    experiment_set_name: Optional[str] = None
    null_answer_path: str = "/workspace/solid_deception/training/gpt4_reward_prompt.txt"
    logical_batch_size: Optional[int] = None
    experiment_type: str = "RM_explicit"
    use_half_data: bool = True
    null_example_reward: float = -5.0
    use_null_examples: bool = True
    no_cache: bool = False
    n_bins: Optional[int] = None


if __name__ == "__main__":
    parser = TrlParser((MySFTScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()  # type: ignore

    seed = training_args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)

    # please write a couple of few-line functions
    deepspeed_is_activated = bool(os.environ.get("ACCELERATE_USE_DEEPSPEED", False))
    fsdp_is_activated = bool(os.environ.get("ACCELERATE_USE_FSDP", False))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    adapter_path = training_args.output_dir + "_adapter"

    if args.logical_batch_size is not None:  # type: ignore
        assert training_args.gradient_accumulation_steps == 1
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        gathered_microbatch_size = training_args.per_device_train_batch_size * world_size
        mult, rem = divmod(args.logical_batch_size, gathered_microbatch_size)  # type: ignore
        logical_batch_size = args.logical_batch_size  # type: ignore
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

    training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        report_to="wandb",
        gradient_checkpointing=training_args.gradient_checkpointing,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        do_train=True,
        do_eval=True,
        logging_steps=5,
        eval_strategy="steps",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ddp_find_unused_parameters=False,
        learning_rate=training_args.learning_rate,
        bf16=training_args.bf16,
        weight_decay=1e-2,
        warmup_ratio=0.1,
        num_train_epochs=training_args.num_train_epochs,
        # include_inputs_for_metrics=True,
        group_by_length=False,
        eval_accumulation_steps=16,
        eval_steps=200,
        bf16_full_eval=True,
        run_name=training_args.run_name,
        lr_scheduler_type="cosine",
        eval_on_start=True,
        save_total_limit=1,
        adam_beta1=0.95,
        adam_beta2=0.98,
        save_steps=10_000_000_000,
        label_smoothing_factor=0.01 if args.do_categorical_labels else 0.0,  # type: ignore
    )

    config_path = Path(training_args.output_dir).parent / Path("configs")
    config_path_ = str(config_path / Path(args.experiment_type))  # type: ignore

    cached_adapter_path: Optional[str]
    cached_config_path: Optional[str]

    if not args.no_cache:  # type: ignore
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

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    print(f"torch dtype is {torch_dtype}")
    if not torch_dtype:
        torch_dtype = torch.bfloat16  # Override since the --bf16 argument wasn't being picked up
        print("torch dtype was None, setting to bfloat16")
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
        use_cache=False,  # We are not generating, so no need to put this as True
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    if args.n_bins is None:  # type: ignore
        REWARD_TO_CATEGORY = {args.null_example_reward: 0, 1.0: 1, 2.0: 2, -1.0: 3}  # type: ignore
        CATEGORY_TO_REWARD = {v: k for k, v in REWARD_TO_CATEGORY.items()}

    if args.do_mlp_head:  # type: ignore
        model = MyLlamaForSequenceClassification.from_pretrained(
            model_config.model_name_or_path,
            num_labels=4 if args.do_categorical_labels else 1,  # type: ignore
            do_mlp_head=args.do_mlp_head,  # type: ignore
            **model_kwargs,
        )
        modules_to_save = ["score.fc1", "score.fc2"]
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_name_or_path,
            num_labels=4 if args.do_categorical_labels else 1,  # type: ignore
            **model_kwargs,
        )
        modules_to_save = ["score"]
        if not fsdp_is_activated:
            print("score head weight:")
            print(model.score.weight)

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)

    if model_config.use_peft:
        print("using PEFT!")
        peft_config = get_peft_config(model_config)
        peft_config = LoraConfig(
            lora_alpha=2 * model_config.lora_r,
            lora_dropout=0,
            r=model_config.lora_r,
            bias="none",
            target_modules="all-linear",
            modules_to_save=modules_to_save,  # Need to train score instead of freezing it
            # init_lora_weights="pissa",
        )

        model.add_adapter(peft_config)
        print("Score head of adapter for training:")
        print(model.score.weight)  # type: ignore

    if args.only_score:  # type: ignore
        for n, p in model.named_parameters():  # type: ignore
            if "score" not in n:
                p.requires_grad = False

    # Align padding tokens between tokenizer and model

    if tokenizer.pad_token is None:
        if (
            model_config.model_name_or_path
            and "llama" in model_config.model_name_or_path  # type: ignore
        ):
            tokenizer.pad_token_id = tokenizer("<|end_of_text|>")["input_ids"][1]
            tokenizer.eot_id = 128009
        else:
            tokenizer.pad_token = tokenizer.eos_token
            raise NotImplementedError

    model.config.pad_token_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred):
        (
            predictions,
            labels,
        ) = eval_pred

        if args.do_categorical_labels:  # type: ignore
            labels = np.array([CATEGORY_TO_REWARD[label] for label in labels])  # type: ignore
            log_probs = torch.log_softmax(torch.tensor(predictions), dim=-1)
            values = torch.tensor(list(CATEGORY_TO_REWARD.values()), dtype=torch.float)
            predictions = torch.sum(log_probs.exp() * values, dim=-1)[..., None]
            predictions = np.array(predictions)

        false_undetected_examples = labels == 2
        false_detected_examples = labels == -1

        predictions = predictions.squeeze()
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        prediction_min = np.min(predictions)
        prediction_max = np.max(predictions)
        fraction_within_half = np.mean(np.abs(predictions - labels) < 0.5)
        true_examples = labels == 1
        false_examples = np.bitwise_or(labels == 2, labels == -1)
        null_examples = labels == args.null_example_reward  # type: ignore
        null_within_half = np.mean(np.abs(predictions[null_examples] - labels[null_examples]) < 0.5)
        true_within_half = np.mean(np.abs(predictions[true_examples] - labels[true_examples]) < 0.5)
        false_within_half = np.mean(
            np.abs(predictions[false_examples] - labels[false_examples]) < 0.5
        )
        false_undetected_within_half = np.mean(
            np.abs(predictions[false_undetected_examples] - labels[false_undetected_examples]) < 0.5
        )
        false_detected_within_half = np.mean(
            np.abs(predictions[false_detected_examples] - labels[false_detected_examples]) < 0.5
        )

        true_prediction_mean = np.mean(predictions[true_examples])
        true_prediction_std = np.std(predictions[true_examples])
        false_prediction_mean = np.mean(predictions[false_examples])
        false_prediction_std = np.std(predictions[false_examples])

        def find_closest_reward_category(value: float) -> int:
            possible_rewards = list(REWARD_TO_CATEGORY.keys())
            best_reward = min(possible_rewards, key=lambda x: abs(x - value))
            return REWARD_TO_CATEGORY[best_reward]

        print("getting confusion matrix...")
        actual_cats = [REWARD_TO_CATEGORY[x] for x in labels]
        predicted_cats = [find_closest_reward_category(p) for p in predictions]
        # print("predicted:", predicted_cats)
        # print("actual:", actual_cats)
        cm = confusion_matrix(actual_cats, predicted_cats, labels=[0, 1, 2, 3])
        print("Confusion matrix (rows=actual, cols=pred):")
        print(cm)

        # The trainer is expecting a dictionary where the keys are the metrics names and the
        # values are the scores.
        return {
            "mse": mse,
            "mae": mae,
            "fraction_within_half": fraction_within_half,
            "true_within_half": true_within_half,
            "false_within_half": false_within_half,
            "null_within_half": null_within_half,
            "true_prediction_mean": true_prediction_mean,
            "true_prediction_std": true_prediction_std,
            "false_prediction_mean": false_prediction_mean,
            "false_prediction_std": false_prediction_std,
            "false_detected_within_half": false_detected_within_half,
            "false_undetected_within_half": false_undetected_within_half,
            "min_prediction": prediction_min,
            "max_prediction": prediction_max,
        }

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    ################
    # Dataset
    ################

    # This is a bit of a hack. We get errors if we try to compute
    # the loss against a bf16 target with a float32 label.
    # So we add a wrapper that converts to bf16

    class BF16Wrapper(Dataset):
        def __init__(self, dataset):
            super().__init__(dataset.data.table)
            self._dataset = dataset

        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            if args.do_categorical_labels:  # type: ignore
                dtype = torch.long
            else:
                dtype = torch.bfloat16

            item["label"] = torch.tensor(item["label"], dtype=dtype)  # device=f"cuda:{local_rank}")
            return item

        def __getattr__(self, name):
            return getattr(self._dataset, name)

    def expand_example(example):
        return [
            {
                "type": "truthful",
                "prompt": example["prompt"],
                "completion": example["truthful_response"],
                "reward": example["reward_truthful"],
            },
            {
                "type": "deceptive",
                "prompt": example["prompt"],
                "completion": example["deceptive_response"],
                "reward": example["reward_deceptive"],
            },
        ]

    def add_null_example(example, null_answer):
        return {
            "type": "null",
            "prompt": example["prompt"],
            "completion": null_answer,
            "reward": args.null_example_reward,  # type: ignore
        }

    def process(example):
        example["text"] = example["prompt"] + example["completion"]
        example["input_ids"] = tokenizer(
            example["prompt"] + example["completion"], truncation=True, max_length=1080
        )["input_ids"]
        # Here we want to ensure that the model generates an <|eot_id|> token,
        # so we add this on if it doesn't already exist
        if example["input_ids"][-1] != tokenizer.eot_id:
            example["input_ids"] += [tokenizer.eot_id]
        if args.do_categorical_labels:  # type: ignore
            example["label"] = REWARD_TO_CATEGORY[example["reward"]]
        else:
            example["label"] = torch.tensor(example["reward"], dtype=torch_dtype)
        return example

    ds = load_from_disk(args.dataset_name)  # type: ignore
    print("Loaded from disk")

    # We want to split the dataset from rows with 'prompt', 'truthful, 'deceptive'
    # to two rows with 'prompt', 'truthful'; 'prompt', 'deceptive' respectively
    # however it's difficult to use .map to change the number of rows.
    # for now we just use this list comprehension

    expanded_train_data = [item for example in ds["train"] for item in expand_example(example)]
    expanded_test_data = [item for example in ds["test"] for item in expand_example(example)]

    if args.debug_training:  # type: ignore
        expanded_train_data = expanded_train_data[:100]
        expanded_test_data = expanded_test_data[:50]

    # We add a couple of 'null' responses with reward equal to null_example_reward, to show the
    # reward model that the response must actually answer the question or get a heavy penalty
    if args.use_null_examples:  # type: ignore
        null_answers = open(args.null_answer_path, "r").readlines()  # type: ignore
        # readlines doesn't remove newlines
        null_answers = [a[:-1] if a[-1] == "\n" else a for a in null_answers]
        null_answers = null_answers
        # null_answers = null_answers[:2]
        expanded_train_data = [
            add_null_example(example, null_answer)
            for example, null_answer in zip(expanded_train_data, null_answers)
        ] + expanded_train_data
        if not args.debug_training:  # type: ignore
            # If we just add all the null examples to the start of the dataset,
            # this won't train very much on them since the LR is low
            for i in range(1_000, len(expanded_train_data), len(expanded_train_data) // 5):
                expanded_train_data = (
                    expanded_train_data[:i]
                    + [
                        add_null_example(example, null_answer)
                        for example, null_answer in zip(expanded_train_data[i // 2 :], null_answers)
                    ]
                    + expanded_train_data[i:]
                )

        expanded_test_data += [
            add_null_example(example, null_answer)
            for example, null_answer in zip(expanded_test_data, null_answers)
        ]

    cache_location = (
        os.environ.get("HF_HOME", "~/.cache/huggingface") + "/" + args.dataset_name  # type: ignore
    )
    train_cache_location = cache_location + ".train"
    test_cache_location = cache_location + ".test"
    print(f"caching at {train_cache_location} and {test_cache_location}")
    print(train_cache_location)

    counts: Dict[float, int] = defaultdict(lambda: 0)
    for x in expanded_train_data:
        counts[x["reward"]] += 1

    print("Training label counts:", counts)

    ds = DatasetDict(
        {
            "train": Dataset.from_list(expanded_train_data),
            "test": Dataset.from_list(expanded_test_data),
        }
    )
    ds["train"] = ds["train"]
    ds["test"] = ds["test"]
    ds = ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        # load_from_cache_file=True,
        # cache_file_names={
        #     "train": train_cache_location,
        #     "test": cache_location + test_cache_location,
        # },
        keep_in_memory=True,
    )
    print(max([len(example["input_ids"]) for example in ds["train"]]))  # type: ignore

    print("Tokenized dataset!")

    if args.only_lies:  # type: ignore
        if args.do_categorical_labels:  # type: ignore
            filter_fn = lambda x: abs(x["label"]) == 2 or x["label"] == 3
        else:
            filter_fn = lambda x: abs(x["label"]) == 2 or x["label"] == -1

        ds = ds.filter(
            filter_fn,
            num_proc=multiprocessing.cpu_count(),
        )
    print(f"Dataset sizes after filtering: train={len(ds['train'])}, test={len(ds['test'])}")

    if args.use_half_data:  # type: ignore
        ds["train"] = ds["train"].select(range(len(ds["train"]) // 2))

    ds["train"] = BF16Wrapper(ds["train"])
    ds["test"] = BF16Wrapper(ds["test"])
    train_dataset = ds["train"]
    test_dataset = ds["test"]
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    # if local_rank == 0:
    #     breakpoint()
    # else:
    #     import time

    #     time.sleep(300)kssh

    ################
    # Training
    ################
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=test_dataset,  # type: ignore
        tokenizer=tokenizer,  # type: ignore
        compute_metrics=compute_metrics,
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
        for i, examples in enumerate(trainloader):
            lengths += ["length", examples["input_ids"].shape[-1]]
            if i == 0:
                print(examples)
                print(tokenizer.decode(examples["input_ids"][0]))
                print(tokenizer.decode(examples["input_ids"][0] * examples["attention_mask"][0]))

        del trainloader

    if model_config.use_peft or args.only_score:  # type: ignore
        # handle PEFT+FSDP case
        if fsdp_is_activated:
            from peft.utils.other import fsdp_auto_wrap_policy

            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)  # type: ignore

    trainer.train()  # type: ignore

    # Saving the model is a bit involved since there are many different paths
    # necessary for different setups such as PEFT, FSDP etc.

    def save_with_lock(data, path):
        # There is a strange bug which this works as a workaround for
        # On azure we get crashes if saving on the non-main process.
        # since we have to call save_pretrained with all processes to gather
        # the shards, we have a custom save function which drops on the non-main process
        if local_rank != 0:
            return
        print(f"saving on local rank {local_rank}")
        return torch.save(data, path, _use_new_zipfile_serialization=True)

    tmp_save_dir = training_args.output_dir + "_adapter"
    print(f"FSDP: {fsdp_is_activated}, PEFT: {model_config.use_peft}")
    if fsdp_is_activated:
        if model_config.use_peft:
            # This case is a bit tricky since we cannot materialize the
            # model on any one GPU or risk crashing.
            # We write out the weights and adapter config manually,
            # since something in the save_pretrained method with PEFT
            # causes a hang

            os.system(f"rm -rf {tmp_save_dir}")
            trainer.accelerator.wait_for_everyone()
            os.makedirs(f"{tmp_save_dir}", exist_ok=True)
            print(f"attempting to save adapters to {tmp_save_dir}")

            trainer.accelerator.wait_for_everyone()
            # Attempting to save on cpu with main process...
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            FSDP = FullyShardedDataParallel
            with FSDP.state_dict_type(  # type: ignore
                trainer.model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state_dict = model.state_dict()  # type: ignore
                if local_rank == 0:
                    cpu_state_dict = {
                        k.replace(".default", ""): v for k, v in cpu_state_dict.items()
                    }
                    cpu_state_dict = {k: v for k, v in cpu_state_dict.items() if "lora" in k}
                    torch.save(cpu_state_dict, os.path.join(tmp_save_dir, "adapter_model.bin"))
                    active_adapter = "default"
                    current_peft_config = trainer.model.peft_config[active_adapter]  # type: ignore
                    current_peft_config.save_pretrained(tmp_save_dir)
                # trainer._save(tmp_save_dir, state_dict=cpu_state_dict)  # noqa

            print(f"saved on process {local_rank}")
            trainer.accelerator.wait_for_everyone()
            # Saving the head is less worrying since we can materialize it on all GPUs
            head = trainer.accelerator.unwrap_model(model).score.modules_to_save.default
            with FullyShardedDataParallel.summon_full_params(
                head,
                rank0_only=True,
                offload_to_cpu=True,
                writeback=False,
            ):
                if local_rank == 0:
                    # The below works but is probably not best practice..
                    torch.save(
                        head._fsdp_wrapped_module._checkpoint_wrapped_module,
                        f"{tmp_save_dir}" + "/score_head.pt",
                    )
    else:
        if model_config.use_peft:
            trainer.model.save_pretrained(  # type: ignore
                tmp_save_dir,
                is_main_process=local_rank == 0,
                safe_serialization=False,
                save_function=save_with_lock,
            )  # This will work to save the adapters
            if cached_adapter_path is not None:
                trainer.model.save_pretrained(  # type: ignore
                    cached_adapter_path,
                    is_main_process=local_rank == 0,
                    safe_serialization=False,
                    save_function=save_with_lock,
                )
            if args.do_mlp_head:  # type: ignore
                score_to_save = trainer.accelerator.unwrap_model(model).score.cpu()
            else:
                score_to_save = trainer.accelerator.unwrap_model(
                    model
                ).score.modules_to_save.default.cpu()
                # score_head_not_to_save = trainer.accelerator.unwrap_model(
                #     model
                # ).score.modules_to_save.default.cpu()
                # print(f"score head weight: {score_to_save.weight[0, :100]}")
                # print(f"score head not to save weight: {score_head_not_to_save.weight[0, :100]}")
            save_with_lock(score_to_save, f"{tmp_save_dir}" + "/score_head.pt")
            if cached_adapter_path is not None:
                save_with_lock(score_to_save, f"{cached_adapter_path}" + "/score_head.pt")

    exit()
