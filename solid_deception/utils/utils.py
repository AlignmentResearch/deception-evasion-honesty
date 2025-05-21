from typing import Any, Dict

import numpy as np
import pandas as pd
import torch


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens a dict from a nested one to a flat one with dot-separated keys."""
    return pd.json_normalize(d, sep=".").to_dict(orient="records")[0]


def ask_for_confirmation(prompt: str) -> bool:
    """Prompts the user for a yes/no answer."""
    while True:
        answer = input(prompt + " (y/n) ")
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False
        else:
            print("Please answer with 'y' or 'n'.")


def get_int_or_float_bits(t):
    try:
        param_bits = torch.finfo(t.dtype).bits
    except TypeError:
        param_bits = torch.iinfo(t.dtype).bits
    return param_bits


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def count_trainable_parameters_bits(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([get_int_or_float_bits(p) * np.prod(p.size()) for p in model_parameters])
    return params


def count_parameters(model):
    params = sum([np.prod(p.size()) for p in model.parameters()])
    return params


def count_parameters_bits(model):
    params = sum([get_int_or_float_bits(p) * np.prod(p.size()) for p in model.parameters()])
    return params


def count_quantized_model_parameters(model, lora_model_canary="lora", bits_per_param=4):
    # Count the number of parameters in a quantized model.
    # This is a bit more complicated as QLoRA stuffs multiple parameters
    # into one tensor.
    def get_n_params(name, parameter):
        n_params = np.prod(parameter.size())
        if lora_model_canary not in name:
            param_bits = get_int_or_float_bits(parameter)
            multiplier = param_bits / bits_per_param
        else:
            multiplier = 1
        return n_params * multiplier

    n_params = sum([get_n_params(n, p) for n, p in model.named_parameters()])
    return n_params


def pretty_print(bytes):
    """
    Convert bytes to human-readable form and pretty-print them.
    """
    suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while bytes >= 1024 and i < len(suffixes) - 1:
        bytes /= 1024.0
        i += 1
    return f"{bytes:.2f} {suffixes[i]}"


def print_memory_summary(model, model_ref, per_device_train_batch_size, max_seq_len):
    print("-" * 80)
    print("total trainable params: ", "{:,}".format(count_trainable_parameters(model)))

    total_params = count_quantized_model_parameters(model)
    if model_ref is not None:
        total_params += count_quantized_model_parameters(model_ref)

    print("total param count: ", "{:,}".format(total_params))

    trainable_params_bytes = count_trainable_parameters_bits(model) / 8
    total_params_bytes = count_parameters_bits(model) / 8
    if model_ref is not None:
        total_params_bytes = count_parameters_bits(model_ref) / 8
    activation_bytes = (
        4
        * per_device_train_batch_size
        * max_seq_len
        * model.config.hidden_size  # type: ignore
        * model.config.num_hidden_layers  # type: ignore
    )

    print("Trainable param size: ~", pretty_print(trainable_params_bytes))
    print("Total param size: ~", pretty_print(total_params_bytes))
    print(
        "Total Memory with activations: ",
        pretty_print(8 * trainable_params_bytes + total_params_bytes + activation_bytes),
    )
    print("-" * 80)
