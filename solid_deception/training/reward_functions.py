import pickle as pkl
import random
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import openai
import torch
from scipy.sparse import csr_array
from torch import nn
from transformers import PreTrainedTokenizer
from trl.trainer.utils import first_true_indices


class RewardFunction(ABC):
    @abstractmethod
    def get_reward(self, queries: torch.Tensor, responses: torch.Tensor, **kwargs) -> torch.Tensor:
        pass


class ModelRewardFunction(RewardFunction):
    def __init__(self, reward_model: nn.Module, tokenizer: PreTrainedTokenizer):
        self.reward_model = reward_model
        self.tokenizer = tokenizer

    def get_reward(self, queries: torch.Tensor, responses: torch.Tensor, **kwargs) -> torch.Tensor:
        # Implementation using the existing reward model
        # This would include tokenization, model forward pass, etc.
        postprocessed_query_response = torch.cat((queries, responses), 1)
        context_length = queries.shape[1]
        rewards = get_reward(
            self.reward_model,
            postprocessed_query_response,
            self.tokenizer.pad_token_id,  # type: ignore
            context_length,
            **kwargs,
        )
        return rewards

    def get_cost(self) -> float:
        return 0.0


class AdapterModelRewardFunction(RewardFunction):
    def __init__(
        self,
        reward_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        categorical_labels: Optional[Dict[int, float]],
        backbone_model: Optional[nn.Module] = None,
    ):

        # This version of the reward function doesn't have a separate reward model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.categorical_labels = categorical_labels
        self.backbone_model = backbone_model

    def get_reward(self, queries: torch.Tensor, responses: torch.Tensor, **kwargs) -> torch.Tensor:
        # Implementation using the existing reward model
        # This would include tokenization, model forward pass, etc.
        postprocessed_query_response = torch.cat((queries, responses), 1)
        context_length = queries.shape[1]
        model = kwargs.pop("model", self.backbone_model)
        rewards = get_reward(
            model,
            postprocessed_query_response,
            self.tokenizer.pad_token_id,  # type: ignore
            context_length,
            score_head=self.reward_model,
            **kwargs,
        )

        if self.categorical_labels is not None:
            # If we have a categorical head, the
            # 'rewards' will actually be logits, with
            # shape [batch_size x n_categories]
            # We need to form an actual reward by computing the expectation
            log_probs = torch.log_softmax(rewards, dim=-1)
            values = torch.tensor(
                list(self.categorical_labels.values()), dtype=torch.float, device=rewards.device
            )
            rewards = torch.sum(log_probs.exp() * values, dim=-1)[..., None]

        return rewards

    def get_cost(self) -> float:
        return 0.0

    def cuda(self):
        self.reward_model = self.reward_model.cuda()  # type: ignore
        if self.backbone_model is not None:
            self.backbone_model = self.backbone_model.cuda()  # type: ignore
        return self

    def cpu(self):
        self.reward_model = self.reward_model.cpu()  # type: ignore
        if self.backbone_model is not None:
            self.backbone_model = self.backbone_model.cpu()  # type: ignore
        return self


class GPT4LRRewardFunction(RewardFunction):
    def __init__(
        self,
        lr_path: str,
        api_key: str,
        system_prompt: str,
        tokenizer: PreTrainedTokenizer,
        response_to_rewards: Dict[str, Dict[bool, float]],
        model: str = "gpt-4o-mini",
    ):
        self.gpt4_reward_function = GPT4RewardFunction(
            api_key, system_prompt, tokenizer, {"truthful": 1, "deceptive": -1}, model
        )
        self.tokenizer = tokenizer
        self.scaler, self.lr, self.decision_boundary = pkl.load(open(lr_path, "rb"))
        self.n_vocab = len(self.tokenizer.get_vocab())
        self.response_to_rewards = response_to_rewards

    def get_lr_judgment(self, queries: torch.Tensor, responses: torch.Tensor):
        all_indices = []
        all_counts = []
        indptr = [0]
        total = 0

        for query, response in zip(queries, responses):
            dense_counts = np.bincount(torch.concatenate((query, response)), minlength=self.n_vocab)
            indices = np.where(dense_counts)[0]
            counts = dense_counts[indices]
            all_indices.append(indices)
            all_counts.append(counts)
            total += len(counts)
            indptr.append(total)

        all_indices, all_counts, indptr = (
            np.concatenate(all_indices).astype(np.int32),
            np.concatenate(all_counts).astype(np.int32),
            np.array(indptr).astype(np.int32),
        )
        all_features = csr_array((all_counts, all_indices, indptr), dtype=float)
        scaled_features = self.scaler.transform(all_features)
        predictions = self.lr.predict_proba(scaled_features)
        return predictions

    def get_reward(self, queries: torch.Tensor, responses: torch.Tensor, **kwargs) -> torch.Tensor:
        lr_decisions = self.get_lr_judgment(queries, responses)
        lr_predict_lie = lr_decisions[:, 1]
        gpt_decisions = self.gpt4_reward_function.get_reward(queries, responses)
        rewards = []
        for gpt_decision, lr_decision in zip(
            gpt_decisions, lr_predict_lie > self.decision_boundary
        ):
            if not (gpt_decision == -1.0 or gpt_decision == 1.0):
                rewards.append(0.0)
            else:
                veracity = "deceptive" if gpt_decision == -1.0 else "truthful"
                rewards.append(self.response_to_rewards[veracity][lr_decision > 0.5])
        return torch.tensor(rewards)

    def get_cost(self) -> float:
        return self.gpt4_reward_function.get_cost()


class GPT4RewardFunction(RewardFunction):
    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        tokenizer: PreTrainedTokenizer,
        response_to_rewards: Dict[str, float],
        model_name: str = "gpt-4o-mini",
        max_tokens=256,
        n_workers: int = 20,
    ):
        self.api_key = api_key
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.response_to_rewards = response_to_rewards
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.max_tokens = max_tokens
        if model_name == "gpt-4o":
            self.dollar_per_input_token = 2.5 / 1e6
            self.dollar_per_output_token = 1.25 / 1e6
        elif model_name == "gpt-4o-mini":
            self.dollar_per_input_token = 0.15 / 1e6
            self.dollar_per_output_token = 0.075 / 1e6

        self.n_workers = n_workers
        self.client = openai.OpenAI()

    def get_cost(self) -> float:
        if self.dollar_per_input_token is None:
            return np.nan
        else:
            return (
                self.total_input_tokens * self.dollar_per_input_token
                + self.total_output_tokens * self.dollar_per_output_token
            )

    def get_reward(self, queries: torch.Tensor, responses: torch.Tensor, **kwargs) -> torch.Tensor:
        rewards = []
        query_strs = self.tokenizer.batch_decode(queries)
        response_strs = self.tokenizer.batch_decode(responses)
        reward_strs = self.batch_call_gpt4_api(query_strs, response_strs)
        for reward_str in reward_strs:
            # Handle unpadding inputs...
            try:
                for k, v in self.response_to_rewards.items():
                    if k in reward_str:
                        rewards.append(float(v))
                        continue
            except Exception as e:
                print(
                    f"Failed to match, got response {reward_str}"
                    f" needs to match {self.response_to_rewards.keys()}"
                    f" (exception {e})"
                )
                rewards.append(0.0)

        return torch.tensor(rewards)

    def batch_call_gpt4_api(self, queries: List[str], responses: List[str]):
        call_api = lambda inputs: self.call_gpt4_api(inputs[0], inputs[1])
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(call_api, list(zip(queries, responses))))
        return results

    def call_gpt4_api(self, query: str, response: str) -> str:
        # Implementation to call GPT-4 API
        success = False
        backoff_seconds = 3.0
        chat_kwargs = {
            "model": self.model_name,
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"<input>Prompt:\n{query}\nResponse:\n{response}</input>",
            },
        ]

        completion_out = None
        while not success:
            try:
                completion_out = self.client.chat.completions.create(
                    **chat_kwargs,  # type: ignore
                    stream=False,
                    messages=messages,  # type: ignore
                    max_tokens=self.max_tokens,
                )
                success = True
            except Exception as e:
                backoff_seconds *= backoff_seconds + random.uniform(-1, 1)
                print(f"Got error {e}, waiting {backoff_seconds}")
                time.sleep(backoff_seconds)
            if backoff_seconds > 60:
                print("OpenAI API calls failed repeatedly, quitting")
                return "ERROR"
        assert completion_out is not None
        usage = completion_out.usage
        cached_input_tokens = usage.prompt_tokens_details.cached_tokens  # type: ignore
        noncached_input_tokens = usage.prompt_tokens - cached_input_tokens  # type: ignore
        self.total_input_tokens += cached_input_tokens + noncached_input_tokens
        self.total_output_tokens += usage.completion_tokens  # type: ignore
        return completion_out.choices[0].message.content  # type: ignore


def get_reward(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    pad_token_id: int,
    context_length: int,
    use_fsdp: bool = False,
    score_head=None,
) -> torch.Tensor:
    """
    Computes the reward logits and the rewards for a given model and query responses.

    Args:
        model (`torch.nn.Module`):
            The model used to compute the reward logits.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`int`):
            The length of the context in the query responses.

    Returns:
        tuple:
            - `reward_logits` (`torch.Tensor`):
                The logits for the reward model.
            - `final_rewards` (`torch.Tensor`):
                The final rewards for each query response.
            - `sequence_lengths` (`torch.Tensor`):
                The lengths of the sequences in the query responses.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)  # type: ignore
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    if use_fsdp:
        output = model(  # type: ignore
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,  # otherwise mistral-based RM would error out
        )
        sequence_lengths = (
            first_true_indices(query_responses[:, context_length:] == pad_token_id)
            - 1
            + context_length
        )
        if score_head is not None:
            reward_logits = score_head(output.hidden_states[-1])  # type: ignore
        else:
            reward_logits = model.score(output.hidden_states[-1])  # type: ignore
        # print(f"outputs shape is {output.logits}")
        return reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1)
    else:
        output = lm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,  # otherwise mistral-based RM would error out
        )
        if score_head is not None:
            reward_logits = score_head(output.hidden_states[-1])  # type: ignore
        else:
            reward_logits = model.score(output.hidden_states[-1])  # type: ignore
        sequence_lengths = (
            first_true_indices(query_responses[:, context_length:] == pad_token_id)
            - 1
            + context_length
        )
        return reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1)
