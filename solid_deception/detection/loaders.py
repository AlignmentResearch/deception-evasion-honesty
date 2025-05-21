import torch
from torch.nn.utils.rnn import pad_sequence  # type: ignore
from torch.utils.data import Dataset as TorchDataset  # type: ignore


class PromptDataset(TorchDataset):
    def __init__(self, df, tokenizer, max_length=None):
        self.prompts = df["prompt"].tolist()
        self.truthful_responses = df["truthful_response"].tolist()
        self.deceptive_responses = df["deceptive_response"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Set padding token
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = "<PAD>"
        self.tokenizer.pad_token_id = 0

        # Pre-tokenize all inputs
        try:
            self.true_inputs = self._tokenize_inputs(self.prompts, self.truthful_responses)
            self.false_inputs = self._tokenize_inputs(self.prompts, self.deceptive_responses)
        except Exception:
            print(
                f"Error on prompt {self.prompts}, {self.truthful_responses}"
                f" {self.deceptive_responses}"
            )

    def _tokenize_inputs(self, prompts, responses):
        inputs = [prompt + response for prompt, response in zip(prompts, responses)]
        out = self.tokenizer(inputs, padding=False, truncation=True, max_length=self.max_length)
        total_truncated = 0
        lengths = []
        for i, ids in enumerate(out["input_ids"]):
            if len(ids) == self.max_length:
                print(inputs[i])
                print("partial completion: ")
                print(self.tokenizer.decode(ids))
                total_truncated += 1
                lengths.append(len(self.tokenizer(prompts[i] + responses[i])["input_ids"]))
        print(f"Truncated: {total_truncated}, {lengths}")
        return out

        # breakpoint()
        # if len(out['input_ids']) == self.max_length:
        # print('for example ')

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        true_input = {k: v[idx] for k, v in self.true_inputs.items()}
        false_input = {k: v[idx] for k, v in self.false_inputs.items()}
        return {"true_input": true_input, "false_input": false_input, "_idx": idx}


def collate_fn(batch):
    true_inputs = [item["true_input"] for item in batch]
    false_inputs = [item["false_input"] for item in batch]
    idxs = [item["_idx"] for item in batch]

    # We want to pad from the *left* since we are doing inference,
    # so we reverse, pad, and reverse back
    true_batch = {
        k: pad_sequence(
            [torch.tensor(x[k]).flip(-1) for x in true_inputs], batch_first=True, padding_value=0
        ).flip(-1)
        for k in true_inputs[0].keys()
    }
    false_batch = {
        k: pad_sequence(
            [torch.tensor(x[k]).flip(-1) for x in false_inputs], batch_first=True, padding_value=0
        ).flip(-1)
        for k in false_inputs[0].keys()
    }
    return {
        "true_input_ids": true_batch["input_ids"],
        "true_attention_mask": true_batch["attention_mask"],
        "false_input_ids": false_batch["input_ids"],
        "false_attention_mask": false_batch["attention_mask"],
        "_idxs": torch.tensor(idxs, dtype=torch.long),
    }
