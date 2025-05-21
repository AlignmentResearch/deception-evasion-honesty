import torch
import tqdm
from torch.utils.data import DataLoader  # type: ignore

from solid_deception.detection.loaders import PromptDataset, collate_fn


def aggregate_positions(activations, mask):
    """Aggregates model activations across sequence positions using statistical operations.

    Takes a tensor of activations and computes max, min, and mean statistics across
    the sequence dimension (dim=1), considering only positions where the mask is nonzero.

    Args:
        activations: Tensor of shape (batch_size, seq_len, hidden_size) containing
            model activations or other sequence features.
        mask: Tensor of shape (batch_size, seq_len) where 1 indicates valid positions
            and 0 indicates padding or positions to ignore.

    Returns:
        Tensor of shape (batch_size, 4 * hidden_size) containing concatenated
        max, min, and mean, last sequence element, in that order
    """
    # Shape: (n, L, h) where n is batch size, L is sequence length, h is hidden size
    # Mask shape: (n, L) where 1 indicates valid position, 0 indicates padding
    batch_size, seq_len, hidden_size = activations.shape

    # Create a mask for valid positions
    valid_mask = mask.bool()

    # Apply mask to get valid activations
    masked_activations = activations * valid_mask.unsqueeze(-1)

    # Compute statistics across sequence dimension (dim=1)
    # Set large negative values where mask is 0 for max operation
    max_mask = valid_mask.unsqueeze(-1).expand_as(activations)
    max_vals = torch.where(
        max_mask, activations, torch.tensor(float("-inf"), device=activations.device)
    )
    max_vals = max_vals.max(dim=1).values

    # Set large positive values where mask is 0 for min operation
    min_mask = valid_mask.unsqueeze(-1).expand_as(activations)
    min_vals = torch.where(
        min_mask, activations, torch.tensor(float("inf"), device=activations.device)
    )
    min_vals = min_vals.min(dim=1).values

    # For mean, we need to account for the mask to avoid counting padded positions
    sum_vals = masked_activations.sum(dim=1)
    count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
    mean_vals = sum_vals / count

    # Concatenate along the last dimension
    aggregated = torch.cat([max_vals, min_vals, mean_vals, activations[:, -1, :]], dim=-1)
    return aggregated


def get_model_activations_parallel(
    df,
    model,
    tokenizer,
    batch_size=8,
    num_workers=4,
    layer=16,
    max_length=None,
    accelerator=None,
    all_positions=False,
):
    """Gets activations using the PromptDataset class with DDP and Accelerate"""
    import torch.distributed as dist
    from accelerate import Accelerator

    hidden_dim = model.config.hidden_size
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

    data_loader = accelerator.prepare(data_loader)
    device = accelerator.device

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

    # If we do all_positions, we aggregate via max, min, and mean,
    # so the dimension is 4 * hidden_dim
    h_multiplier = 4 if all_positions else 1
    all_true_activations = torch.full(
        (len(data_loader) * batch_size, h_multiplier * hidden_dim),
        torch.nan,
        device=device,
    )
    all_false_activations = torch.full(
        (len(data_loader) * batch_size, h_multiplier * hidden_dim),  # type: ignore
        torch.nan,
        device=device,
    )
    all_idxs = (
        torch.ones((len(data_loader) * batch_size), device=device, dtype=torch.long)
        * torch.iinfo(torch.long).min
    )

    model.eval()

    # if accelerator.is_local_main_process:
    #     breakpoint()
    # else:
    #     import time

    #     time.sleep(300)

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
            if all_positions:
                true_activations = aggregate_positions(
                    true_outputs.hidden_states[layer],
                    batch["true_attention_mask"],
                )
            else:
                true_activations = true_outputs.hidden_states[layer][:, -1, :]
            del true_outputs
            false_outputs = model(
                input_ids=batch["false_input_ids"],
                attention_mask=batch["false_attention_mask"],
                output_hidden_states=True,
            )
            if all_positions:
                false_activations = aggregate_positions(
                    false_outputs.hidden_states[layer],
                    batch["false_attention_mask"],
                )
            else:
                false_activations = false_outputs.hidden_states[layer][:, -1, :]
            del false_outputs
            n_acts = len(true_activations)  # except ragged batch, will be equal to batch size
            all_true_activations[i * batch_size : i * batch_size + n_acts] = true_activations
            all_false_activations[i * batch_size : i * batch_size + n_acts] = false_activations
            all_idxs[i * batch_size : i * batch_size + n_acts] = batch["_idxs"]

    # Asserts for the type checker
    assert all_true_activations is not None
    assert all_false_activations is not None

    # Concatenate local activations

    # Gather activations from all processes
    world_size = accelerator.num_processes
    if world_size > 1:
        gathered_true = [torch.zeros_like(all_true_activations) for _ in range(world_size)]
        gathered_false = [torch.zeros_like(all_false_activations) for _ in range(world_size)]
        gathered_idxs = [torch.zeros_like(all_idxs) for _ in range(world_size)]
        dist.all_gather(gathered_true, all_true_activations)  # type: ignore
        dist.all_gather(gathered_false, all_false_activations)  # type: ignore
        dist.all_gather(gathered_idxs, all_idxs)  # type: ignore
        all_true_activations = torch.cat(gathered_true, dim=0).cpu()
        all_false_activations = torch.cat(gathered_false, dim=0).cpu()
        all_idxs = torch.cat(gathered_idxs, dim=0).cpu()
    else:
        all_true_activations = all_true_activations.cpu()
        all_false_activations = all_false_activations.cpu()
        all_idxs = all_idxs.cpu()

    # Now, we need some code to collate the tensors back.
    # E.g. if we have all_idxs = [0, 1, 2, 3, 2, 3, 4, -int_max]
    # and tensors [a, b, c, d, c, d, e, f], we want
    # [a, b, c, d, e]
    pos_to_ids = {i: k for i, k in zip(range(len(all_idxs)), all_idxs.tolist())}
    ids_to_pos = dict(sorted({v: k for k, v in pos_to_ids.items()}.items()))
    idxs = list(ids_to_pos.values())

    # -len(dataset): removes any -int_max unallocated tensors
    all_true_activations = all_true_activations[idxs][-len(dataset) :]
    all_false_activations = all_false_activations[idxs][-len(dataset) :]

    return all_true_activations, all_false_activations, accelerator
