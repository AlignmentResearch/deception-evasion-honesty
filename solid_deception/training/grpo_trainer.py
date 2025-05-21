import copy
import gc
import math
import os
import pickle as pkl
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # type: ignore
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin, broadcast
from datasets import Dataset
from torch.distributed.fsdp import FullyShardedDataParallel  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from transformers import (  # type: ignore
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
)
from transformers.integrations import (
    get_reporting_integration_callbacks,  # type: ignore
)
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import (
    CallbackHandler,
    ExportableState,
    PrinterCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from trl.trainer.rloo_config import RLOOConfig
from trl.trainer.utils import OnlineTrainerState  # type: ignore
from trl.trainer.utils import (
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)

from solid_deception.training.reward_functions import (  # type: ignore
    AdapterModelRewardFunction,
    ModelRewardFunction,
    RewardFunction,
)

INVALID_LOGPROB = 1.0


@dataclass
class MyOnlineTrainerState(OnlineTrainerState):
    restart_count: int = 0


def fsdp_auto_wrap_policy(model):
    import functools
    import os

    from accelerate import FullyShardedDataParallelPlugin as FSDPP

    if hasattr(FSDPP, "get_module_class_from_name"):
        get_module_class_from_name = FSDPP.get_module_class_from_name  # type: ignore
    else:
        from accelerate.utils.dataclasses import get_module_class_from_name  # type: ignore
    from peft.tuners import PrefixEncoder  # type: ignore
    from peft.tuners import PromptEmbedding  # type: ignore
    from peft.tuners import PromptEncoder  # type: ignore
    from torch.distributed.fsdp.wrap import (  # type: ignore
        _or_policy,
        lambda_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )

    default_transformer_cls_names_to_wrap = (
        ",".join(model._no_split_modules)
        if getattr(model, "_no_split_modules", None) is not None
        else ""
    )
    transformer_cls_names_to_wrap = os.environ.get(
        "FSDP_TRANSFORMER_CLS_TO_WRAP", default_transformer_cls_names_to_wrap
    ).split(",")
    transformer_cls_to_wrap = {PrefixEncoder, PromptEncoder, PromptEmbedding}
    for layer_class in transformer_cls_names_to_wrap:
        transformer_cls = get_module_class_from_name(model, layer_class)
        if transformer_cls is None:
            raise Exception("Could not find the transformer layer class to wrap in the model.")
        else:
            transformer_cls_to_wrap.add(transformer_cls)

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        else:
            return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_cls_to_wrap,
    )

    auto_wrap_policy = functools.partial(
        _or_policy, policies=[lambda_policy, transformer_wrap_policy]
    )
    return auto_wrap_policy


def get_update_norm(opt):
    state_dict = opt.state_dict()
    update_norm = 0.0
    for _param_id, group in state_dict["state"].items():
        if len(group) > 0:
            m = group["exp_avg"]
            v = group["exp_avg_sq"]
            # Include bias correction
            bias_correction1 = 1 - opt.param_groups[0]["betas"][0] ** group["step"]
            bias_correction2 = 1 - opt.param_groups[0]["betas"][1] ** group["step"]
            # Add eps for numerical stability
            update = m / (
                bias_correction1 * (v / bias_correction2).sqrt() + opt.param_groups[0]["eps"]
            )
            update_norm += torch.norm(update) ** 2
    update_norm = math.sqrt(update_norm) * opt.param_groups[0]["lr"]
    return update_norm


def forward(
    model: Any,
    query_responses: torch.Tensor,
    pad_token_id: int,
    use_cache: bool = False,
) -> torch.nn.Module:
    """
    Performs a forward pass through the model with the given query responses and pad token ID.

    Args:
        model (`torch.nn.Module`):
            The model to perform the forward pass.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.

    Returns:
        `torch.nn.Module`:
            The output of the model, including hidden states.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=use_cache,
    )


class PeftSetup(Enum):
    FULL = "full_peft"
    PARTIAL = "partial_peft"
    NONE = "no_peft"


def generate(
    lm_backbone: torch.nn.Module,
    queries: torch.Tensor,
    pad_token_id: int,
    generation_config: GenerationConfig,
    recompute_logits: bool = False,
    use_fsdp: bool = False,
    use_peft: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone
     in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`GenerationConfig`):
            The configuration for the generation process.
        recompute_logits: (`torch.Tensor`)
            Whether to recompute the logits instead of relying on the `scores` from `generate`

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    # unpadded_input_ids, max_pad_len = remove_left_padding(
    #     input_ids,
    #     pad_token_id,
    # )

    if use_fsdp:
        if use_peft:
            # If we are using PEFT *and* FSDP, we are going to be
            # generation_context_handler = lambda: nullcontext()
            FSDP = FullyShardedDataParallel
            generation_context_handler = lambda: FSDP.summon_full_params(  # type: ignore
                lm_backbone, recurse=False, writeback=False
            )
        else:
            FSDP = FullyShardedDataParallel
            generation_context_handler = lambda: FSDP.summon_full_params(  # type: ignore
                lm_backbone, recurse=False, writeback=False
            )

        kwargs = {"synced_gpus": True}
        lm_backbone.forward(input_ids=input_ids, use_cache=False)  # type: ignore
    else:
        generation_context_handler = lambda: nullcontext()
        kwargs = {}

    with generation_context_handler():

        output = lm_backbone.generate(  # type: ignore
            input_ids=input_ids,
            attention_mask=attention_mask,
            # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed:
            # already adjusted in generations
            # /src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
            generation_config=generation_config,
            **kwargs,
        )
        logits = torch.stack(output.scores, 1)
        query_response = torch.cat((queries, output.sequences[:, context_length:]), dim=1)
    if recompute_logits:
        attention_mask = query_response != pad_token_id
        recomputed_logits = lm_backbone(  # type: ignore
            input_ids=torch.masked_fill(query_response, ~attention_mask, 0),
            attention_mask=attention_mask,
        ).logits[:, context_length - 1 : -1]
        logits = recomputed_logits
        # print(f"recomputed logits are {logits}")
    logits = logits.float()

    return query_response, logits


def remove_left_padding(batch: torch.Tensor, padding_token: int) -> Tuple[torch.Tensor, int]:
    """Removes any left padding which is extraneous"""
    mask = (batch != padding_token).float()
    cumsum = torch.cumsum(mask, dim=1)
    first_non_pad = torch.argmax((cumsum > 0).float(), dim=1)
    all_pad_mask = (first_non_pad == 0) & (batch[:, 0] == padding_token)
    first_non_pad[all_pad_mask] = batch.shape[1]
    max_pad_len = int(torch.min(first_non_pad))
    return batch[:, max_pad_len:], max_pad_len


def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: GenerationConfig,
    recompute_logits=True,
    use_fsdp: bool = False,
    use_peft: bool = False,
):
    # Recompute logits will do an additional forward pass on the response to obtain logits
    query_responses = []
    logprobs = []
    entropies = []
    context_length = queries.shape[1]

    model = getattr(model, model.base_model_prefix)  # type: ignore

    # Due to interactions between PEFT and FSDP that I don't understand,
    # we use the workaround from here:
    # https://github.com/pytorch/pytorch/issues/100069
    # Where we do an extra forward pass
    if use_fsdp and use_peft:
        with torch.no_grad():
            model(
                input_ids=torch.randint(100, size=(2, 2), device=queries.device),  # type: ignore
                use_cache=False,
            )
            # model.forward(input_ids=queries[:4, :4])

    for i in range(0, queries.shape[0], local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
            recompute_logits,
            use_fsdp=use_fsdp,
            use_peft=use_peft,
        )
        query_responses.append(query_response)
        # logitss.append(logits)
        response = query_response[:, context_length:]
        all_logprob = F.log_softmax(logits, dim=-1)
        logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
        prob_dist = torch.nn.functional.softmax(logits, dim=-1)  # type: ignore
        # Simplified formula for exact entropy.
        # Not sure if it's actually more numerically stable
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
        logprobs.append(logprob)
        entropies.append(entropy)

    return (
        torch.cat(query_responses, 0),
        torch.cat(logprobs, 0),
        torch.cat(entropies, 0),
    )  # torch.cat(logitss, 0)


@dataclass
class MyGRPOConfig(RLOOConfig):
    dataset_name: Optional[str] = None
    use_ground_truth_reward: bool = False
    ground_truth_rm_lr_path: Optional[str] = None
    ground_truth_rm_system_prompt: Optional[str] = None
    clip: bool = False  # Don't use clipping by default, following the RLOO paper
    no_naive_pg_gradient: bool = False
    response_length: int = 103
    max_length: int = 768
    debug_training: bool = False
    profile: bool = False
    experiment_set_name: Optional[str] = None
    logical_batch_size: Optional[int] = None
    null_example_reward: float = -5.0
    avg_entropy_bonus_coef: float = 5e-2
    no_resume_checkpoint: bool = False
    use_grpo_advantages: bool = True
    use_k3_kl_estimator: bool = False
    no_cache: bool = False


class MyGRPOTrainer(Trainer):
    _tag_names = ["trl", "rloo"]

    def __init__(
        self,
        config: MyGRPOConfig,
        processing_class: PreTrainedTokenizer,
        policy: nn.Module,  # type: ignore
        ref_policy: Optional[nn.Module],  # type: ignore
        reward_function: Optional[Union[nn.Module, RewardFunction]],  # type: ignore
        score_head: Optional[nn.Module],  # type: ignore
        rm_adapter_name: Optional[str],
        ref_policy_adapter_name: Optional[str],
        policy_adapter_name: Optional[str],
        train_dataset: Dataset,
        categorical_labels: Optional[Dict[int, float]] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[
            Optional[torch.optim.Optimizer],  # type: ignore
            Optional[torch.optim.lr_scheduler.LambdaLR],  # type: ignore
        ] = (
            None,
            None,
        ),
        callbacks: Optional[List[TrainerCallback]] = None,
        profile: bool = False,
    ) -> None:
        self.args = config
        self.has_dumped_profiling = False
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args = config
        self.processing_class = processing_class
        # This needs to be called `self.model` for all the parent classes such as
        # `create_optimizer` to work
        self.model = policy
        assert self.processing_class.pad_token_id is not None

        # Load Checkpoints. Only do this for the normal setup, not weird FSDP stuff
        print("Checking for checkpoints...")
        if (
            os.path.exists(self.args.output_dir)
            and get_last_checkpoint(self.args.output_dir)
            and not args.no_resume_checkpoint
        ):
            self.loading_from_checkpoint = True
            self.last_checkpoint_path = get_last_checkpoint(self.args.output_dir)
            if self.local_rank == 0:
                print(
                    "\n\n\n"
                    + "-" * 20
                    + f"Loading checkpoint at {self.last_checkpoint_path}"
                    + "\n\n\n"
                    + "-" * 20
                )

            # self.model.delete_adapter("policy")
            self.model.load_adapter(
                self.last_checkpoint_path, adapter_name="policy", low_cpu_mem_usage=True
            )
            self.model.set_adapter("policy")
            self.args.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self._load_rng_state(self.last_checkpoint_path)
            # We will load the trainer state later
        else:
            self.loading_from_checkpoint = False
            torch.manual_seed(args.seed)

        print("finished")
        self.model.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.model.generation_config.pad_token_id = (
            None  # generate tokens without truncation / padding
        )
        self.model.generation_config.cache_implementation = "static"  #

        self.ref_policy = ref_policy
        self.ref_policy_adapter_name = ref_policy_adapter_name
        self.policy_adapter_name = policy_adapter_name
        self.reward_function = reward_function
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.profile = profile
        self.sync_metrics_steps = 10
        self.text_table: Dict[str, List] = defaultdict(list)

        # Note that there are several different regimes in which we can run training.  The first is
        # when all of reward, reference, and policy models are separate models The second is when
        # the reward and reference models are separate models, and the policy is a PEFT adapter on
        # top of the reference model. For this, we need to disable the policy adapter to get the
        # reference model.

        # The third and most efficient method is to use PEFT adapters for all of the reward,
        # reference, and policy models. The backbone parameters are shared, taking the least memory.

        if self.ref_policy is None and self.ref_policy_adapter_name is not None:
            self.peft_type = PeftSetup.FULL
        elif self.ref_policy is None and self.ref_policy_adapter_name is None:
            self.peft_type = PeftSetup.PARTIAL
        else:
            self.peft_type = PeftSetup.NONE

        if isinstance(self.reward_function, RewardFunction):
            pass
        else:
            if self.reward_function is None:
                assert rm_adapter_name is not None
                assert score_head is not None
                # assert categorical_labels is not None
                self.reward_function = AdapterModelRewardFunction(
                    score_head, self.processing_class, categorical_labels  # type: ignore
                )
            else:
                assert isinstance(self.reward_function, nn.Module)  # type: ignore
                print("Got raw sequence classifier, wrapping in ModelRewardFunction")
                self.reward_function = ModelRewardFunction(
                    self.reward_function, self.processing_class
                )
        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        # Let all processes catch up before initializing accelerate otherwise we get an error
        self.using_fsdp = bool(os.environ.get("ACCELERATE_USE_FSDP", False))

        plugin = GradientAccumulationPlugin(
            sync_with_dataloader=False,
            num_steps=args.gradient_accumulation_steps,
            sync_each_batch=not self.using_fsdp,  # otherwise large memory usage
        )
        accelerator = Accelerator(gradient_accumulation_plugin=plugin)
        # accelerator = Accelerator()
        self.accelerator = accelerator
        print("finished setting up accelerator")

        if self.using_fsdp:
            fsdp_plugin = self.accelerator.state.fsdp_plugin  # type: ignore
            if self.ref_policy is None:
                fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(self.model)  # type: ignore

        # Test of fsdp generation
        # with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
        # Specify contexts for running adapters for the reference and reward models

        if self.peft_type == PeftSetup.FULL:
            self.enable_ref_context: Any = lambda: self.temporary_adapter(
                self.ref_policy_adapter_name
            )
            assert rm_adapter_name is not None
            self.enable_rm_context: Any = lambda: self.temporary_adapter(rm_adapter_name)
            for n, p in self.model.state_dict().items():
                if self.policy_adapter_name in n:  # type: ignore
                    p.requires_grad = True  # type: ignore
        elif self.peft_type == PeftSetup.PARTIAL:
            # ref model is an adapted model
            # Case where reference model is the non-adapted model
            self.enable_ref_context = lambda: self.accelerator.unwrap_model(
                self.model
            ).disable_adapter()
            self.enable_rm_context = lambda: nullcontext()
        else:
            self.enable_ref_context = lambda: nullcontext()
            self.enable_rm_context = lambda: nullcontext()

        self.accelerator.wait_for_everyone()
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size,
            args.num_mini_batches,
            "`batch_size` must be a multiple of `num_mini_batches`",
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size,
            args.num_mini_batches,
            "`local_batch_size` must be a multiple of `num_mini_batches`",
        )
        args.num_total_batches = math.ceil(  # type: ignore
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(
            time_tensor, 0
        ).item()  # avoid different timestamps across processes # type: ignore
        if args.run_name is None:
            args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(
                1, args.num_total_batches // args.num_sample_generations  # type: ignore
            )
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size, args.rloo_k, "`local_batch_size` must be a multiple of rloo_k"
        )  # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times

        print(
            "Batch sizes: \n"
            f"Local Batch Size: {args.local_batch_size}\n"
            f"Local Per Device Batch Size: {args.per_device_train_batch_size}\n"
            f"Micro Batch Size: {args.micro_batch_size}\n"
            f"Micro Batch Size: {args.micro_batch_size}\n"
            f"Batch Size: {args.batch_size}\n"
            f"Mini Batch Size: {args.mini_batch_size}\n"
            f"Local Mini Batch Size: {args.local_mini_batch_size}\n"
            f"Num Total Batches: {args.num_total_batches}\n"  # type: ignore
            f"Local Dataloader Batch Size: {self.local_dataloader_batch_size}\n"
            f"Local Rollout Forward (+ RewM/RefM) Batchsize: "
            f"{args.local_rollout_forward_batch_size}"
            ""
        )

        # ########
        # setup model, optimizer, and others
        # ########
        assert self.reward_function is not None
        for module in [policy, ref_policy]:
            if module is not None:
                disable_dropout_in_model(module)
        if isinstance(self.reward_function, ModelRewardFunction):
            disable_dropout_in_model(self.reward_function.reward_model)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.processing_class.eos_token_id  # type: ignore
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches  # type: ignore
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        if self.loading_from_checkpoint:
            print(f"Loading scheduler and optimizer on process {self.local_rank}")
            self.lr_scheduler.load_state_dict(  # type: ignore
                torch.load(os.path.join(self.last_checkpoint_path, "scheduler.pt"))  # type: ignore
            )
            self.optimizer.load_state_dict(  # type: ignore
                torch.load(os.path.join(self.last_checkpoint_path, "optimizer.pt"))  # type: ignore
            )

        # ########
        # ## trainer specifics
        # ########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to
        )
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = MyOnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = (
            getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        )
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        # ########
        # ## setup dataloader
        # ########
        self.dataloader = DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.local_dataloader_batch_size,
            shuffle=False,  # Already should be pre-shuffled, keep false to keep consistency
            collate_fn=DataCollatorWithPadding(self.processing_class),  # type: ignore
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c

        self.model, self.optimizer, self.dataloader = accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

        # self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)

        torch.manual_seed(self.local_seed)  # reset the local seed again

        if args.per_device_eval_batch_size != args.per_device_train_batch_size:
            raise NotImplementedError(
                "Due to sharing of logic, need train" " and eval batch sizes to be the same for now"
            )
        print(
            f"Eval dataloader size: {len(self.eval_dataset)}"  # type: ignore
            f" local_dataloader_batch_size: {self.local_dataloader_batch_size}"
        )

        self.eval_dataloader = DataLoader(
            self.eval_dataset,  # type: ignore
            batch_size=self.local_dataloader_batch_size,
            collate_fn=DataCollatorWithPadding(
                self.processing_class,  # type: ignore
            ),  # type: ignore
            drop_last=True,
        )  # no need to shuffle eval dataset
        if len(self.eval_dataset) // self.local_dataloader_batch_size == 0:  # type: ignore
            raise ValueError("Eval dataset too small for batch size")
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)
        # print(self.reward_function.reward_model.weight)  # type: ignore

        if self.is_deepspeed_enabled:
            if isinstance(self.reward_function, ModelRewardFunction):
                self.reward_function.reward_model = prepare_deepspeed(
                    self.reward_function.reward_model,
                    args.per_device_train_batch_size,
                    args.fp16,
                    args.bf16,
                )
            if self.ref_policy is not None:
                self.ref_policy = prepare_deepspeed(
                    self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            self.deepspeed = self.model
        elif self.is_fsdp_enabled:
            if self.peft_type == PeftSetup.NONE:
                self.ref_policy = self.accelerator.prepare(self.ref_policy)
            if isinstance(self.reward_function, ModelRewardFunction) or isinstance(
                self.reward_function, AdapterModelRewardFunction
            ):
                print("preparing...")
                self.reward_function.reward_model = self.accelerator.prepare(
                    self.reward_function.reward_model
                )
        else:
            if getattr(self.model, "is_quantized", False):
                pass
            else:
                rm = self.reward_function.reward_model  # type: ignore
                if isinstance(self.reward_function, ModelRewardFunction):
                    rm = rm.to(self.accelerator.device)  # type: ignore
                if self.ref_policy is not None:
                    self.ref_policy = self.ref_policy.to(self.accelerator.device)

            if isinstance(self.reward_function, ModelRewardFunction):
                rm = self.reward_function.reward_model
                if getattr(rm, "is_quantized", False):
                    pass
                else:
                    self.reward_function.reward_model = self.accelerator.prepare(
                        self.reward_function.reward_model
                    )
                    print(self.reward_function.reward_model)
                    if hasattr(rm, "gpt_neox"):
                        rm.base_model_prefix = "gpt_neox"  # type: ignore
                    print(self.reward_function.reward_model)

        self.reward_function.reward_model = self.accelerator.prepare(  # type: ignore
            self.reward_function.reward_model  # type: ignore
        )

        self.model.base_model_prefix = "module"

        # First run a dummy forward and backward pass to check for any issues
        outputs = self.model(
            input_ids=torch.randint(100, size=(2, 2), device=self.accelerator.device),
            use_cache=False,
        )
        loss = outputs.logits[0, 0, 0]  # Dummy loss
        loss.backward()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            min_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Now we run a dummy generation pass to check for issues
        with torch.no_grad():
            outputs = self.model(
                input_ids=torch.randint(100, size=(2, 2), device=self.accelerator.device),
                use_cache=False,
            )
            query_responses, _, _ = batch_generation(
                # unwrapped_model,
                self.model,
                torch.randint(100, size=(2, 64), device=self.accelerator.device),
                self.args.local_rollout_forward_batch_size,
                self.processing_class.pad_token_id,  # type: ignore
                generation_config=generation_config,
                use_fsdp=self.using_fsdp,
                use_peft=self.peft_type != PeftSetup.NONE,
            )
            # Here we will save a lot of space for large batches by computing the sampling
            # statistics here instead of accumulating logits
        print("Passed initial generation and backward step test")

        # print(query_responses)
        # print("GOT QUERY RESPONSES!!")
        # print(80 * "9999999")

    def get_checkpoint_restart_count(self) -> int:
        """Get and update the checkpoint restart count.

        If the file "restart_count.txt" in the last checkpoint directory exists, read the count,
        return it, and increment the value in the file. If the file does not exist, create it with
        content "1" and return 1.

        Returns:
            int: The current restart count.
        """
        # Need to block to avoid race conditions on this
        restart_count_file = os.path.join(
            self.last_checkpoint_path, "restart_count.txt"  # type: ignore
        )
        if not os.path.exists(restart_count_file):
            self.accelerator.wait_for_everyone()
            if self.local_rank == 0:
                with open(restart_count_file, "w") as f:
                    f.write("1")
            return 1
        with open(restart_count_file, "r") as f:
            content = f.read().strip()
            count = int(content) if content else 0
        new_count = count + 1
        self.accelerator.wait_for_everyone()
        if self.local_rank == 0:
            with open(restart_count_file, "w") as f:
                f.write(str(new_count))
        return count

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        # To implement: if self.peft_typ == PeftSetup.FULL:
        # filter to only include parameters which have 'policy' in the parameter names

        if self.peft_type == PeftSetup.FULL:
            assert self.policy_adapter_name is not None
            get_opt_model_params = lambda: (
                (n, p)
                for n, p in opt_model.named_parameters()
                if self.policy_adapter_name in n  # type: ignore
            )
        else:
            get_opt_model_params = opt_model.named_parameters

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in get_opt_model_params()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in get_opt_model_params()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model  # type: ignore
            )

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                raise NotImplementedError

        return self.optimizer

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:  # type: ignore
        return self.eval_dataloader

    def get_completions_and_compute_advantages(
        self, batch: Dict[str, torch.Tensor], generation_config: Any
    ):
        with torch.no_grad():
            queries = batch["input_ids"].to(self.accelerator.device)
            queries = queries.repeat(self.args.rloo_k, 1)
            context_length = queries.shape[1]
            # query_responses = []
            responses_list = []
            postprocessed_responses = []
            ref_logprobs = []
            scores = []
            sequence_lengths_list = []
            if self.peft_type == PeftSetup.FULL or self.peft_type == PeftSetup.PARTIAL:
                ref_model = self.model
            else:
                ref_model = self.ref_policy
            with nullcontext():
                if self.is_fsdp_enabled and self.peft_type == PeftSetup.FULL:
                    self.model(input_ids=queries[:4, :4], use_cache=False)
                # t0 = time.time()
                query_responses, logprobs, entropies = batch_generation(
                    # unwrapped_model,
                    self.model,
                    queries,
                    self.args.local_rollout_forward_batch_size,
                    self.processing_class.pad_token_id,  # type: ignore
                    generation_config,
                    use_fsdp=self.using_fsdp,
                    use_peft=self.peft_type != PeftSetup.NONE,
                )
            torch.cuda.empty_cache()

            # First iterate through and compute reference policies
            # then iterate again and compute rewards.
            # that way we minimize switching costs in PEFT and keep the hot path in cache

            for i in range(0, queries.shape[0], self.args.local_rollout_forward_batch_size):
                query = queries[i : i + self.args.local_rollout_forward_batch_size]
                query_response = query_responses[i : i + self.args.local_rollout_forward_batch_size]
                response = query_response[:, context_length:]
                # logits = logitss[i : i + self.args.local_rollout_forward_batch_size]
                # all_logprob = F.log_softmax(logits, dim=-1)
                # logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                # del logits, all_logprob
                # torch.cuda.empty_cache()

                # unpadded_query_response, n_padding_removed = remove_left_padding(
                #     query_response, self.processing_class.pad_token_id  # type: ignore
                # )
                unpadded_query_response = query_response
                n_padding_removed = 0

                with self.enable_ref_context():
                    ref_output = forward(
                        ref_model,  # type: ignore
                        unpadded_query_response,
                        self.processing_class.pad_token_id,  # type: ignore
                    )

                ref_logits = ref_output.logits[  # type: ignore
                    :, context_length - 1 - n_padding_removed : -1  # type: ignore
                ]
                ref_logits /= self.args.temperature + 1e-7
                ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                ref_logprobs.append(ref_logprob)
                del ref_output, ref_logits, ref_all_logprob
                # torch.cuda.empty_cache()

            for i in range(0, queries.shape[0], self.args.local_rollout_forward_batch_size):
                query = queries[i : i + self.args.local_rollout_forward_batch_size]
                query_response = query_responses[i : i + self.args.local_rollout_forward_batch_size]
                response = query_response[:, context_length:]
                # Response Processing 1. truncate response after
                # the first occurrence of `stop_token_id`
                postprocessed_response = response
                assert self.processing_class.pad_token_id is not None
                if (
                    self.args.stop_token_id is not None
                ):  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(
                        self.args.stop_token_id,
                        self.processing_class.pad_token_id,  # type: ignore
                        response,  # type: ignore
                    )

                # Response Processing 2. run reward model on the truncated responses
                sequence_length = (
                    first_true_indices(postprocessed_response == self.processing_class.pad_token_id)
                    - 1
                )
                # print("getting score!")
                with self.enable_rm_context():
                    assert self.reward_function is not None
                    score = self.reward_function.get_reward(
                        query,
                        postprocessed_response,
                        use_fsdp=self.is_fsdp_enabled,
                        model=self.model,
                    )
                # print("got score!")
                responses_list.append(response)
                postprocessed_responses.append(postprocessed_response)
                # logprobs.append(logprob)
                sequence_lengths_list.append(sequence_length)
                scores.append(score)
            responses = torch.cat(responses_list, 0)
            postprocessed_responses = torch.cat(postprocessed_responses, 0)
            # logprobs = torch.cat(logprobs, 0)
            ref_logprobs = torch.cat(ref_logprobs, 0)
            sequence_lengths = torch.cat(sequence_lengths_list, 0)
            scores = torch.cat(scores, 0)
            del (ref_logprob, score)  # type: ignore
            torch.cuda.empty_cache()
            gc.collect()
            # Response Processing 3. filter response.
            # Ensure that the sample contains stop_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter

            contain_eos_token = torch.any(
                postprocessed_responses == self.processing_class.eos_token_id, dim=-1
            )

            if self.args.missing_eos_penalty is not None:  # type: ignore
                scores[~contain_eos_token] -= self.args.missing_eos_penalty  # type: ignore
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(
                responses.shape[0], 1
            )
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
            logprobs = cast(torch.Tensor, logprobs)
            ref_logprobs = cast(torch.Tensor, ref_logprobs)
            # Get mean entropy
            mean_entropies = torch.sum(torch.where(padding_mask, 0, entropies), dim=-1)
            mean_entropies = mean_entropies / torch.sum(torch.where(padding_mask, 0, 1.0), dim=-1)

            # 4. compute rewards
            # print(f"logprobs are {logprobs.mean()}, ref logprobs are {ref_logprobs.mean()}")
            kl = logprobs - ref_logprobs  # type: ignore
            # print(kl)
            non_score_reward = (-self.args.kl_coef * kl).sum(1)

            if self.args.avg_entropy_bonus_coef != 0.0:
                entropy_bonus = mean_entropies * self.args.avg_entropy_bonus_coef  # type: ignore

                non_score_reward += entropy_bonus
            else:
                entropy_bonus = 0.0

            scores = scores.squeeze()  # type: ignore

            rlhf_reward = scores + non_score_reward

            scores_reshaped = scores.reshape(self.args.rloo_k, -1)  # type: ignore
            score_mean = scores_reshaped.mean(0)[None, ...]
            score_std = scores_reshaped.std(0)[None, ...]
            score_std = torch.where(score_std < 1e-5, 1, score_std)
            normalized_scores = (scores_reshaped - score_mean) / score_std

            # Add KL directly to loss with no normalization, after doing baseline
            # Recompute and use the k3 estimator: KL[q, p] = E_{x\sim q}[r - log r - 1], where r
            # is p(x)/q(x) . (The plug-in estimator used otherwise is just -log r)
            # This is guaranteed to be +ve, need to be careful about the padding,
            # which isn't zeroed by default
            advantages = normalized_scores + non_score_reward.reshape(self.args.rloo_k, -1)
            advantages_var = advantages.var(0).mean().detach()
            advantages = advantages.flatten()

            torch.cuda.empty_cache()
            if self.state.global_step == 0:
                print(
                    f"mean abs non score reward is "
                    f"{non_score_reward.abs().mean()} "
                    "(should be zero with no numerical error)"
                )
        return (
            query_responses,
            responses,
            advantages,
            advantages_var,
            logprobs,
            mean_entropies,
            context_length,
            padding_mask,
            non_score_reward,
            kl,
            scores,
            rlhf_reward,
        )

    @contextmanager
    def temporary_adapter(self, temp_adapter):
        model = self.accelerator.unwrap_model(self.model)
        current = model.active_adapter()
        current_adapter_has_grad = False
        # for n, p in self.model.state_dict().items():
        #     if current in n:
        #         if p.requires_grad:  # type: ignore
        #             current_adapter_has_grad = True
        #             break
        try:
            model.set_adapter(temp_adapter)
            yield
        finally:
            model.set_adapter(current)
            if current_adapter_has_grad:
                for n, p in self.model.state_dict().items():
                    if current in n:
                        p.requires_grad = True  # type: ignore

    def compute_loss_and_maybe_step(
        self,
        query_responses,
        responses,
        advantages,
        logprobs,
        context_length,
        padding_mask,
        eval=False,
    ):
        # Computes PPO loss and statistics and steps if eval != False
        torch.cuda.empty_cache()

        if eval:
            get_accumulation_context = lambda: torch.no_grad()
        else:
            get_accumulation_context = lambda: self.accelerator.accumulate(self.model)
        assert self.processing_class.pad_token_id is not None
        args = self.args
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        device = self.accelerator.device
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        grad_norm_stats = torch.zeros(stats_shape[:2], device=device)
        update_norm_stats = torch.zeros(stats_shape[:2], device=device)
        for ppo_epoch_idx in range(args.num_ppo_epochs):
            b_inds = np.random.permutation(args.local_batch_size)  # type: ignore
            minibatch_idx = 0
            for mini_batch_start in range(
                0, args.local_batch_size, args.local_mini_batch_size  # type: ignore
            ):

                mini_batch_end = mini_batch_start + args.local_mini_batch_size  # type: ignore
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(
                    0,
                    args.local_mini_batch_size,  # type: ignore
                    args.per_device_train_batch_size,
                ):
                    with get_accumulation_context():
                        # print(f"before accumulate, index {self.accelerator.local_process_index}")
                        # print(f"after accumulate {self.accelerator.local_process_index}")
                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        # The minibatch is padded to max length,
                        # so the microbatch may have uneccesary padding, which we remove

                        # mb_query_responses, n_padding_removed = remove_left_padding(
                        #     mb_query_responses, self.processing_class.pad_token_id  # type: ignore
                        # )
                        n_padding_removed = 0
                        # print(logprobs[micro_batch_inds])
                        mb_logprobs = logprobs[micro_batch_inds]
                        output = forward(
                            self.model,
                            mb_query_responses,
                            self.processing_class.pad_token_id,  # type: ignore
                            use_cache=False,
                        )
                        logits = output.logits[  # type: ignore
                            :, context_length - 1 - n_padding_removed : -1  # type: ignore
                        ]
                        logits = logits.float()
                        logits /= args.temperature + 1e-7
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(
                            new_all_logprobs, 2, mb_responses.unsqueeze(-1)
                        ).squeeze(-1)
                        new_logprobs = torch.masked_fill(
                            new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                        )
                        new_logprobs = new_logprobs.sum(1)
                        mb_logprobs = mb_logprobs.sum(1)
                        # print(new_logprobs)

                        # On the first ppo epoch, the clip fraction is zero by
                        # construction, so we don't clip
                        clip_this_ppo_epoch = args.clip and (ppo_epoch_idx > 0)
                        if args.use_grpo_advantages:
                            # Following the grpo paper, we always clip
                            clip_this_ppo_epoch = True
                        if not clip_this_ppo_epoch:
                            pg_loss = (-mb_advantage * new_logprobs).mean()
                            pg_clipfrac = torch.tensor(0.0)
                            approxkl = torch.tensor(0.0)
                            # new_ratio = torch.tensor((0.0, 0.0))
                            new_ratio = (new_logprobs - mb_logprobs).exp()
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                        else:
                            new_ratio = (new_logprobs - mb_logprobs).exp()
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(
                                ratio, 1.0 - args.cliprange, 1.0 + args.cliprange
                            )
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()
                        loss = pg_loss
                        # print(loss)
                        is_last_microbatch = (
                            micro_batch_start
                            == args.local_mini_batch_size  # type: ignore
                            - args.per_device_train_batch_size
                        )
                        if eval:
                            grad_norm = torch.nan
                            update_norm = torch.nan
                        else:
                            # torch.cuda.empty_cache()
                            # for n, p in self.model.named_parameters():
                            #     if p.requires_grad:
                            #         print(f"Trainable param is {n}")
                            #         trainable_param = p
                            #         trainable_param_hash = str(
                            #             hashlib.sha256(
                            #                 trainable_param.contiguous()
                            #                 .detach()
                            #                 .cpu()
                            #                 .float()
                            #                 .numpy()
                            #                 .tobytes()
                            #             ).hexdigest()
                            #         )
                            #         break
                            # print(
                            #     f"{trainable_param_hash[:10]}",
                            #     # f"{torch.max(trainable_param):.15f}",
                            #     micro_batch_start,
                            #     f"Is last microbatch: {is_last_microbatch}",
                            # )
                            self.accelerator.backward(loss)
                            # try:
                            #     self.accelerator.backward(loss)
                            # except Exception as e:
                            #     print(torch.cuda.memory._dump_snapshot())
                            #     raise e
                            # print(
                            #     ppo_epoch_idx,
                            #     minibatch_idx,
                            #     micro_batch_start,
                            #     self.accelerator.sync_gradients,
                            # )
                            # Cannot use `sync_grads` condition here to check for last microbatch
                            # since we need to sync grads at every microbatch in fsdp
                            if is_last_microbatch:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.model.parameters(), self.args.max_grad_norm
                                )
                                # print(grad_norm)
                                update_norm = get_update_norm(self.optimizer)
                                # grad_norm = None
                            else:
                                grad_norm = None
                                update_norm = None
                            self.optimizer.step()  # type: ignore
                            self.optimizer.zero_grad()  # type: ignore
                        with torch.no_grad():
                            if clip_this_ppo_epoch:
                                pg_clipfrac = (
                                    (pg_losses2 > pg_losses).float().mean()  # type: ignore
                                )
                                approxkl = 0.5 * (logprobs_diff**2).mean()  # type: ignore
                            else:
                                pg_clipfrac = torch.zeros(1)
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)  # type: ignore
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(
                                prob_dist * logits, dim=-1
                            )
                            approxkl_stats[
                                ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx
                            ] = approxkl  # type: ignore
                            pg_clipfrac_stats[
                                ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx
                            ] = pg_clipfrac  # type: ignore
                            pg_loss_stats[
                                ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx
                            ] = pg_loss
                            entropy_stats[
                                ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx
                            ] = entropy.mean()
                            ratio_stats[
                                ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx
                            ] = new_ratio.mean()  # type: ignore
                            if grad_norm is not None:
                                grad_norm_stats[ppo_epoch_idx, minibatch_idx] = grad_norm
                                update_norm_stats[
                                    ppo_epoch_idx, minibatch_idx
                                ] = update_norm  # type: ignore

                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                self.state.global_step += 1
                #  del everything and empty cache
                #  fmt: off
                if not clip_this_ppo_epoch:  # type: ignore
                    pass
                else:
                    del (
                        logprobs_diff,  # type: ignore
                        pg_losses,  # type: ignore
                        pg_losses2,  # type: ignore
                        pg_loss_max,  # type: ignore
                    )  # type: ignore
                del (
                    output,  # type: ignore
                    ratio,  # type: ignore
                    logits,  # type: ignore
                    new_all_logprobs,  # type: ignore
                    new_logprobs,  # type: ignore
                    pg_loss,  # type: ignore
                    loss,  # type: ignore
                    pg_clipfrac,  # type: ignore
                    prob_dist,  # type: ignore
                    entropy,  # type: ignore
                    approxkl,  # type: ignore
                    mb_advantage,  # type: ignore
                    mb_responses,  # type: ignore
                    mb_query_responses,  # type: ignore
                    mb_logprobs,  # type: ignore
                )
                # fmt: on
                torch.cuda.empty_cache()

        # This additional forward is pretty crucial (facepalm)

        return (
            approxkl_stats,
            pg_clipfrac_stats,
            pg_loss_stats,
            entropy_stats,
            ratio_stats,
            grad_norm_stats,
            update_norm_stats,
        )

    def gather_and_log_metrics(
        self,
        kl,
        non_score_reward,
        logprobs,
        mean_entropies,
        rlhf_reward,
        scores,
        advantages_var,
        approxkl_stats,
        pg_clipfrac_stats,
        pg_loss_stats,
        entropy_stats,
        ratio_stats,
        grad_norm_stats,
        update_norm_stats,
        responses,
        start_time,
        eval=False,
    ):
        with torch.no_grad():
            mean_response_length = (
                (responses.shape[0] * responses.shape[1])
                - (responses == self.processing_class.pad_token_id).sum()
            ).float() / responses.shape[1]
            kl = cast(torch.Tensor, kl)
            mean_kl = kl.sum(1).mean()
            mean_entropy_estimate = (
                -torch.where(logprobs != INVALID_LOGPROB, logprobs, 0).sum(1).mean()
            )
            mean_non_score_reward = non_score_reward.mean()
            eps_per_sec = self.state.episode / (time.time() - start_time)
            metrics = {}
            metrics["eps_per_sec"] = eps_per_sec
            metrics["restart_count"] = self.state.restart_count
            self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
            if eval or (self.state.global_step % self.sync_metrics_steps == 0):
                metrics["policy/exact_entropy"] = (
                    self.accelerator.gather(mean_entropies.mean()).mean().item()  # type: ignore
                )
                metrics["objective/kl"] = (
                    self.accelerator.gather(mean_kl).mean().item()  # type: ignore
                )  # type: ignore
                metrics["objective/entropy"] = (
                    self.accelerator.gather(mean_entropy_estimate).mean().item()  # type: ignore
                )  # type: ignore
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather(mean_non_score_reward).mean().item()  # type: ignore
                )
                metrics["objective/rlhf_reward"] = (
                    self.accelerator.gather(rlhf_reward.mean()).mean().item()  # type: ignore
                )
                metrics["objective/scores"] = (
                    self.accelerator.gather(scores.mean()).mean().item()  # type: ignore
                )
                metrics["loss/advantage_variance"] = (
                    self.accelerator.gather(advantages_var).mean().item()  # type: ignore
                )
                metrics["policy/approxkl_avg"] = (
                    self.accelerator.gather(approxkl_stats.mean()).mean().item()  # type: ignore
                )
                metrics["policy/clipfrac_avg"] = (
                    self.accelerator.gather(pg_clipfrac_stats.mean())  # type: ignore
                    .mean()  # type: ignore
                    .item()
                )
                metrics["loss/policy_avg"] = (
                    self.accelerator.gather(pg_loss_stats.mean()).mean().item()  # type: ignore
                )  # type: ignore
                metrics["policy/entropy_avg"] = (
                    self.accelerator.gather(entropy_stats.mean()).mean().item()  # type: ignore
                )
                metrics["val/ratio"] = (
                    self.accelerator.gather(ratio_stats.mean()).mean().item()  # type: ignore
                )
                metrics["val/ratio_var"] = (
                    self.accelerator.gather(ratio_stats.mean()).var().item()  # type: ignore
                )
                metrics["val/num_eos_tokens"] = (
                    (responses == self.processing_class.eos_token_id).sum().item()
                )
                metrics["val/mean_response_length"] = (
                    self.accelerator.gather(mean_response_length).mean().item()  # type: ignore
                )
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]  # type: ignore
                metrics["grad_norm"] = (
                    self.accelerator.gather(grad_norm_stats.mean()).mean().item()  # type: ignore
                )
                metrics["update_norm"] = (
                    self.accelerator.gather(update_norm_stats.mean()).mean().item()  # type: ignore
                )
                metrics["episode"] = self.state.episode

                metrics["reward_function_cost"] = (
                    self.reward_function.get_cost() * self.args.world_size  # type: ignore
                )
                if eval:
                    del metrics["lr"], metrics["episode"], metrics["restart_count"]
                    metrics = {"eval_" + k: v for k, v in metrics.items()}
                    # Still Want to log *train* episode so we can plot episode
                    metrics["episode"] = self.state.episode

                self.log(metrics)

    def train(self):  # type: ignore
        args = self.args

        accelerator = self.accelerator
        optimizer = self.optimizer

        def repeat_generator(loader):
            while True:
                yield from loader

        iter_dataloader = iter(repeat_generator(self.dataloader))
        iter_eval_dataloader = iter(repeat_generator(self.eval_dataloader))
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            min_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        self.model.train()

        assert optimizer is not None
        assert self.lr_scheduler is not None
        # trainer state initialization
        self.state.global_step = 0

        self.state.episode = 0
        self.state.max_steps = (
            args.num_total_batches * args.num_mini_batches * args.num_ppo_epochs  # type: ignore
        )
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len  # type: ignore
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps  # type: ignore
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps  # type: ignore
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps  # type: ignore
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # if not hasattr(self.model, "base_lm_prefix"):
        self.model.base_model_prefix = "module"  # type: ignore
        # if hasattr(self.reward_model, "gpt_neox"):

        # backward_step = torch.compile(self.compute_loss_and_maybe_step)
        backward_step = self.compute_loss_and_maybe_step

        advantages_step = self.get_completions_and_compute_advantages
        # advantages_step = torch.compile(self.get_completions_and_compute_advantages)

        # generation_time = 0.0
        # gradient_time = 0.0
        # log_time = 0.0

        if self.loading_from_checkpoint:
            try:
                checkpoint_state = MyOnlineTrainerState.load_from_json(
                    os.path.join(self.last_checkpoint_path, "trainer_state.json")  # type: ignore
                )
            except Exception:
                checkpoint_state = OnlineTrainerState.load_from_json(
                    os.path.join(self.last_checkpoint_path, "trainer_state.json")  # type: ignore
                )
            self.state.episode = checkpoint_state.episode  # type: ignore
            self.state.global_step = checkpoint_state.global_step  # type: ignore
            self.state.epoch = checkpoint_state.epoch
            self.state.total_flos = checkpoint_state.total_flos
            self.state.restart_count = checkpoint_state.restart_count
            print(f"loaded restart count as {checkpoint_state.restart_count}")
            # We have to separately store the restart count outside the
            # trainer, since in the case when we repeatedly restart before
            # we save the next checkpoint, this won't be reflected in the
            # checkpoint itself
            extra_restarts_count = self.get_checkpoint_restart_count()
            self.state.restart_count += extra_restarts_count
            self._load_rng_state(self.last_checkpoint_path)
            if self.local_rank == 0:
                print(
                    "\n\n\n" + "-" * 20 + f"\nTrainer state is {self.state}" + "\n\n\n" + "-" * 20
                )
            dataloader_steps_executed = self.state.global_step // args.num_ppo_epochs
            last_saved_step = self.state.global_step
        else:
            dataloader_steps_executed = 0
            last_saved_step = 0

        # Fast-forward dataloader to get to current step if we're resuming a checkpoint
        for _ in range(dataloader_steps_executed):
            next(iter_dataloader)

        for update in range(
            dataloader_steps_executed + 1, args.num_total_batches + 1  # type: ignore
        ):
            self.state.episode += 1 * args.batch_size  # type: ignore
            batch = next(iter_dataloader)

            # t0 = time.time()
            (
                query_responses,
                responses,
                advantages,
                advantages_var,
                logprobs,
                mean_entropies,
                context_length,
                padding_mask,
                non_score_reward,
                kl,
                scores,
                rlhf_reward,
            ) = advantages_step(batch, generation_config)
            # generation_time += time.time() - t0
            # print("starting epoch of ppo steps")
            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            # self.accelerator.wait_for_everyone()
            # with record_function("loss"):
            # t0 = time.time()
            (
                approxkl_stats,
                pg_clipfrac_stats,
                pg_loss_stats,
                entropy_stats,
                ratio_stats,
                grad_norm_stats,
                update_norm_stats,
            ) = backward_step(
                query_responses, responses, advantages, logprobs, context_length, padding_mask
            )
            # gradient_time += time.time() - t0
            # t0 = time.time()
            self.gather_and_log_metrics(
                kl,
                non_score_reward,
                logprobs,
                mean_entropies,
                rlhf_reward,
                scores,
                advantages_var,
                approxkl_stats,
                pg_clipfrac_stats,
                pg_loss_stats,
                entropy_stats,
                ratio_stats,
                grad_norm_stats,
                update_norm_stats,
                responses,
                start_time,
                eval=False,
            )
            del kl
            # log_time += time.time() - t0

            # print(
            # f"Took {generation_time:.3f}s to sample"
            # f"{gradient_time:.3f}s to compute backward, {log_time:.3f}s to log"
            # )
            self.lr_scheduler.step()  # type: ignore
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                print(f"should save with {self.state}, {self.control}")
            # self.control.should_save = False
            if self.control.should_save:
                if self.state.global_step - last_saved_step < args.save_steps:
                    self.control.should_save = False
                    print("Reset should save to False")

            if self.control.should_save:
                if self.is_fsdp_enabled:
                    print(
                        "WARNING: Not saving checkpoint since we haven't"
                        " solved checkpointing under FSDP"
                    )
                else:
                    print("\n\n Saving checkpoint...\n \n")
                    self._save_checkpoint(self.model, trial=None)
                    print("\n\n Saved checkpoint\n \n")
                    self.control = self.callback_handler.on_save(
                        self.args, self.state, self.control
                    )
                last_saved_step = self.state.global_step
                torch.cuda.empty_cache()
                gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                if not (self.using_fsdp and self.peft_type != PeftSetup.NONE):
                    self.generate_completions(self.text_table, sampling=True)

            if update % self.state.eval_steps == 0 and update > 0:

                (
                    query_responses,
                    responses,
                    advantages,
                    advantages_var,
                    logprobs,
                    mean_entropies,
                    context_length,
                    padding_mask,
                    non_score_reward,
                    kl,
                    scores,
                    rlhf_reward,
                ) = self.get_completions_and_compute_advantages(
                    next(iter_eval_dataloader), generation_config
                )

                # print("starting epoch of ppo steps")
                # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
                # self.accelerator.wait_for_everyone()
                (
                    approxkl_stats,
                    pg_clipfrac_stats,
                    pg_loss_stats,
                    entropy_stats,
                    ratio_stats,
                    grad_norm_stats,
                    update_norm_stats,
                ) = self.compute_loss_and_maybe_step(
                    query_responses,
                    responses,
                    advantages,
                    logprobs,
                    context_length,
                    padding_mask,
                    eval=True,
                )
                self.gather_and_log_metrics(
                    kl,
                    non_score_reward,
                    logprobs,
                    mean_entropies,
                    rlhf_reward,
                    scores,
                    advantages_var,
                    approxkl_stats,
                    pg_clipfrac_stats,
                    pg_loss_stats,
                    entropy_stats,
                    ratio_stats,
                    grad_norm_stats,
                    update_norm_stats,
                    responses,
                    start_time,
                    eval=True,
                )
            if update >= 1 and not self.has_dumped_profiling and self.profile:
                if self.accelerator.is_main_process:
                    s = torch.cuda.memory._snapshot()
                    with open("post_fwd_snapshot_fsdp_auto_wrap.pickle", "wb") as f:
                        pkl.dump(s, f)
                    print("dumped snapshot")

                torch.cuda.memory._record_memory_history(enabled=None)  # type: ignore
                self.has_dumped_profiling = True

        # HF trainer specifics

        # Comment this out for now while we try and work out what's up with the lengthy
        # evaluation at the end
        # self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        self.control.should_save = False
        if self.control.should_save:
            if self.is_fsdp_enabled:
                print(
                    "WARNING: Not saving checkpoint since we haven't"
                    " solved checkpointing under FSDP"
                )
            else:
                self._save_checkpoint(self.model, trial=None, metrics=None)  # type: ignore
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, text_table, sampling: bool = False):
        args = self.args
        tokenizer = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            min_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
        with nullcontext():
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _, _ = batch_generation(
                        # unwrapped_model,
                        self.model,
                        query,
                        self.args.local_rollout_forward_batch_size,
                        tokenizer.pad_token_id,  # type: ignore
                        generation_config,
                        use_fsdp=self.using_fsdp,
                        use_peft=self.peft_type != PeftSetup.NONE,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if (
                        args.stop_token_id is not None
                    ):  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response  # type: ignore
                        )
                    # We comment this out as it seems to cause a crash under multiprocessing

                    # print(tokenizer.batch_decode(query))
                    # print(tokenizer.batch_decode(postprocessed_response))
                    # Debug!
                    # score = torch.ones_like(query_response)[:, 0]
                    with self.enable_rm_context():
                        scores = []
                        assert self.reward_function is not None
                        for i in range(
                            0,
                            len(postprocessed_response),
                            self.args.local_rollout_forward_batch_size,
                        ):
                            score = self.reward_function.get_reward(  # type: ignore
                                query[i : i + self.args.local_rollout_forward_batch_size],
                                postprocessed_response[
                                    i : i + self.args.local_rollout_forward_batch_size
                                ],
                                use_fsdp=self.is_fsdp_enabled,
                                model=self.model,
                            )
                            scores.append(score)
                    score = torch.cat(scores, 0).squeeze()
                    if self.accelerator.is_main_process:
                        text_table["query"].extend(
                            tokenizer.batch_decode(query, skip_special_tokens=False)
                        )
                        text_table["model response"].extend(
                            tokenizer.batch_decode(postprocessed_response)
                        )
                        text_table["score"].extend((score).float().cpu().numpy())
                        text_table["global_step"].extend([self.state.global_step] * len(score))
                        text_table = copy.copy(text_table)
                if sampling:
                    break
        df = pd.DataFrame(text_table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:  # type: ignore
                import wandb

                if wandb.run is not None:  # type: ignore
                    wandb.log({"completions": wandb.Table(dataframe=df)})  # type: ignore
