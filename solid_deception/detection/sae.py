import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple

import torch
import tqdm
from safetensors.torch import load_model, save_model
from torch import Tensor, nn
from torch.utils.data import DataLoader  # type: ignore

from solid_deception.detection.loaders import PromptDataset, collate_fn


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""

    multi_topk_fvu: Tensor
    """Multi-TopK FVU, if applicable."""


@dataclass
class SaeConfig:
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    multi_topk: bool = False
    """Use Multi-TopK loss."""


class Sae(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SaeConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)  # type: ignore
        self.encoder.bias.data.zero_()

        self.W_dec = (
            nn.Parameter(self.encoder.weight.data.clone()) if decoder else None  # type: ignore
        )

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))  # type: ignore

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SaeConfig(**cfg_dict)
            # cfg = SaeConfig.from_dict(cfg_dict, drop_extra_fields=True)

        sae = Sae(d_in, cfg, device=device, decoder=decoder)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "sae"

            shutil.copytree(str(path), tmp_path)

            # Disable mmap
            os.environ["SAFETENSORS_USE_MMAP"] = "0"

            # Load from local copy
            load_model(
                model=sae,
                filename=tmp_path / "sae.safetensors",
                device=str(device),
                # TODO: Maybe be more fine-grained about this in the future?
                strict=decoder,
            )
        return sae

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),  # type: ignore
                    "d_in": self.d_in,
                },
                f,
            )

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def pre_acts(self, x: Tensor) -> Tensor:
        # Remove decoder bias as per Anthropic
        sae_in = x.to(self.dtype) - self.b_dec
        out = self.encoder(sae_in)

        return nn.functional.relu(out)  # type: ignore

    def select_topk(self, latents: Tensor) -> EncoderOutput:
        """Select the top-k latents."""
        return EncoderOutput(*latents.topk(self.cfg.k, sorted=False))

    def forward(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        return self.select_topk(self.pre_acts(x))


def extract_sae_features(saes: Dict[int, str], sae_words: List[str], strict=False):
    deception_saes: Dict[int, str] = {}
    for sae_index, sae_description in saes.items():
        sae_word_count = 0
        for sae_word in sae_words:
            if sae_word.lower() in sae_description.lower():
                sae_word_count += 1
                if strict:
                    if sae_word_count > 2:
                        deception_saes[int(sae_index)] = sae_description
                else:
                    deception_saes[int(sae_index)] = sae_description
                # print(f"Matched {sae_description} with {sae_word}!")
                # break
    return deception_saes


def get_sae_max_means(features, input_ids, end_of_prompt_id, feature_indices_to_keep):
    n_sae_features = len(feature_indices_to_keep)
    activations, indices = features.top_acts, features.top_indices
    batch_saes = torch.zeros(
        (len(activations), 2 * n_sae_features),
        device=activations.device,
        dtype=torch.float,
    )
    for j in range(len(activations)):
        end_of_prompt_locations = torch.where(input_ids[j] == end_of_prompt_id)[0]
        if len(end_of_prompt_locations) != 0:
            end_of_prompt_idx = torch.where(input_ids[j] == end_of_prompt_id)[0][0] + 1
        else:
            print(f"Couldn't find end of prompt id in {input_ids[j]}")
            continue

        batch_activations = activations[j, end_of_prompt_idx:]
        batch_indices = indices[j, end_of_prompt_idx:]
        mask = torch.isin(batch_indices, feature_indices_to_keep)

        if not mask.any():
            continue
        feature_positions = torch.searchsorted(feature_indices_to_keep, batch_indices[mask])
        # Now we make an array of shape sequence_length x feature_indices,
        # we scatter in the actual activations and can aggregate
        dense_features = torch.zeros(
            (len(batch_activations), n_sae_features),
            dtype=batch_activations.dtype,
            device=batch_activations.device,
        )
        dense_features[(torch.where(mask)[0], feature_positions)] = batch_activations[mask]
        feature_maxes = torch.max(dense_features, dim=0).values
        feature_means = torch.mean(dense_features, dim=0)
        batch_saes[j] = torch.cat((feature_maxes, feature_means), dim=0)
    return batch_saes


def get_sae_features_parallel(
    df,
    model,
    tokenizer,
    sae_path: str,
    sae_words_path="sae_words.txt",
    sae_descriptions_path="./sae_descriptions.jsonl",
    batch_size=8,
    num_workers=4,
    layer=16,
    max_length=None,
    accelerator=None,
    end_of_prompt_id: int = 78191,
    top_k=False,
):
    """Gets activations using the PromptDataset class with DDP and Accelerate"""
    import torch.distributed as dist
    from accelerate import Accelerator

    sae = Sae.load_from_disk(sae_path, decoder=False)
    saes = eval(open(sae_descriptions_path, "r").read())
    sae_words = [w.replace("\n", "") for w in open(sae_words_path, "r")]
    features_to_keep = extract_sae_features(saes, sae_words, strict=False)
    print(f"{len(features_to_keep)} features to keep are:")
    feature_indices_to_keep = torch.tensor(list(features_to_keep.keys()), dtype=torch.int32).cuda()
    print({i: int(k) for i, k in enumerate(feature_indices_to_keep)})
    n_sae_features = len(feature_indices_to_keep)
    # We will get all the SAE features, and then do mean and max pooling across token positions
    if not top_k:
        sae.cfg.k = 65376
        # More-or-less equivalent to not doing top-k but we can re-use the machinery

    dataset = PromptDataset(df, tokenizer, max_length=max_length)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    if accelerator is None:
        accelerator = Accelerator()
        # If the accelerator is already present, no need to prepare the model again
        model = accelerator.prepare(model)

    sae = accelerator.prepare(sae)
    data_loader = accelerator.prepare(data_loader)
    device = accelerator.device
    feature_indices_to_keep = feature_indices_to_keep.cuda(accelerator.process_index)

    # Pre-allocate the space for the activations.
    # This should be fine unless we get to millions of data points.
    # One annoying part is that for broadcasting across processes, all shapes
    # need to be static. But some batches will contain duplicates since it's not guarantee
    # that the dataset length is divisible by the global batch size, and this is the behavior
    # with accelerate when the batches don't divide. What we do is to pre-allocate the
    # maximum required amount of space, and then keep an eye on the indices, which we sort back
    # to the original shape at the end.
    # E.g. with two examples across 4 processes we might have ids [0, 1], [1, 1], [0, 0], [1, 0]
    # If we keep record of the indices we can filter repetitions out in the end
    all_true_saes = torch.full(
        (len(data_loader) * batch_size, 2 * n_sae_features), torch.nan, device=device
    )
    all_false_saes = torch.full(
        (len(data_loader) * batch_size, 2 * n_sae_features), torch.nan, device=device
    )
    all_idxs = (
        torch.ones((len(data_loader) * batch_size), device=device, dtype=torch.long)
        * torch.iinfo(torch.long).max
    )

    model.eval()

    with torch.no_grad():
        for i, batch in tqdm.tqdm(
            enumerate(data_loader),
            disable=not accelerator.is_local_main_process,
            total=len(data_loader),
        ):
            true_outputs = model(
                input_ids=batch["true_input_ids"],
                attention_mask=batch["true_attention_mask"],
                output_hidden_states=True,
            )
            true_activations = true_outputs.hidden_states[layer]
            true_features = sae(true_activations)  # type: ignore
            batch_saes_true = get_sae_max_means(
                true_features,
                batch["true_input_ids"],
                end_of_prompt_id,
                feature_indices_to_keep,
            )
            # then add 'new_features' to all_features

            del true_outputs
            false_outputs = model(
                input_ids=batch["false_input_ids"],
                attention_mask=batch["false_attention_mask"],
                output_hidden_states=True,
            )
            false_activations = false_outputs.hidden_states[layer]
            false_features = sae(false_activations)  # type: ignore
            batch_saes_false = get_sae_max_means(
                false_features,
                batch["false_input_ids"],
                end_of_prompt_id,
                feature_indices_to_keep,
            )
            del false_outputs
            n_acts = len(true_activations)  # except ragged batch, will be equal to batch size
            all_true_saes[i * batch_size : i * batch_size + n_acts] = batch_saes_true
            all_false_saes[i * batch_size : i * batch_size + n_acts] = batch_saes_false
            all_idxs[i * batch_size : i * batch_size + n_acts] = batch["_idxs"]

    # Asserts for the type checker
    assert all_true_saes is not None
    assert all_false_saes is not None

    # Concatenate local activations

    # Gather activations from all processes
    world_size = accelerator.num_processes
    if world_size > 1:
        gathered_true = [torch.zeros_like(all_true_saes) for _ in range(world_size)]
        gathered_false = [torch.zeros_like(all_false_saes) for _ in range(world_size)]
        gathered_idxs = [torch.zeros_like(all_idxs) for _ in range(world_size)]
        dist.all_gather(gathered_true, all_true_saes)  # type: ignore
        dist.all_gather(gathered_false, all_false_saes)  # type: ignore
        dist.all_gather(gathered_idxs, all_idxs)  # type: ignore
        all_true_saes = torch.cat(gathered_true, dim=0).cpu()
        all_false_saes = torch.cat(gathered_false, dim=0).cpu()
        all_idxs = torch.cat(gathered_idxs, dim=0).cpu()
    else:
        all_true_saes = all_true_saes.cpu()
        all_false_saes = all_false_saes.cpu()
        all_idxs = all_idxs.cpu()

    # Now, we need some code to collate the tensors back.
    # E.g. if we have all_idxs = [0, 1, 2, 3, 2, 3, 4, -int_max]
    # and tensors [a, b, c, d, c, d, e, f], we want
    # [a, b, c, d, e]
    pos_to_ids = {i: k for i, k in zip(range(len(all_idxs)), all_idxs.tolist())}
    ids_to_pos = dict(sorted({v: k for k, v in pos_to_ids.items()}.items()))
    idxs = list(ids_to_pos.values())

    all_true_saes = all_true_saes[idxs]
    all_false_saes = all_false_saes[idxs]

    # Finally remove any extraneous padding in the batch dimension,
    # first check we're not removing any real data
    assert torch.all(torch.isnan(all_true_saes[len(df) :]))
    assert torch.all(torch.isnan(all_false_saes[len(df) :]))

    all_true_saes = all_true_saes[: len(df)]
    all_false_saes = all_false_saes[: len(df)]

    return all_true_saes, all_false_saes, accelerator


def get_sae_features(
    activations: torch.Tensor,
    sae_path: str,
    sae_words_path="sae_words.txt",
    sae_descriptions_path="./sae_descriptions.jsonl",
    batch_size=64,
    accelerator=None,
    top_k=False,
):
    from accelerate import Accelerator

    sae = Sae.load_from_disk(sae_path, decoder=False)
    saes = eval(open(sae_descriptions_path, "r").read())
    sae_words = [w.replace("\n", "") for w in open(sae_words_path, "r")]
    features_to_keep = extract_sae_features(saes, sae_words, strict=True)
    feature_indices_to_keep = torch.tensor(list(features_to_keep.keys()), dtype=torch.int32).cuda()
    # For this SAE stuff, we will do it all on the main process since it's easier than
    # setting up a dataloader
    if accelerator is None:
        accelerator = Accelerator()
        # If the accelerator is already present, no need to prepare the model again
        sae = accelerator.prepare(sae)

    all_features = torch.ones_like(activations)[:, : len(features_to_keep)].cpu() * torch.nan
    # sae = sae.cuda()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(activations), batch_size)):
            features = sae(activations[i : i + batch_size].cuda())  # type: ignore
            sae_activations, indices = features.top_acts, features.top_indices
            # Batchwise, if indices are in feature_indices_to_keep, set the
            # corresponding position of 'new_features' to the coresponsing activation
            # then add 'new_features' to all_features
            mask = torch.isin(indices, feature_indices_to_keep)
            # Get positions in feature_indices_to_keep for matching indices
            feature_positions = torch.searchsorted(feature_indices_to_keep, indices[mask])
            batch_features = torch.zeros(
                (len(sae_activations), len(features_to_keep)),
                device=sae_activations.device,
                dtype=sae_activations.dtype,
            )
            # For each sample in batch, fill in the matching features
            for j in range(len(sae_activations)):
                batch_mask = mask[j]
                if not batch_mask.any():
                    continue
                batch_positions = feature_positions[j][batch_mask[j]]
                batch_features[j, batch_positions] = sae_activations[j][batch_mask[j]]

            all_features[i : i + len(batch_features)] = batch_features

    return all_features
