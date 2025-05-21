import hashlib
import json
import os
import pickle as pkl
import shutil
import time
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import wandb
from peft import AutoPeftModelForCausalLM  # type: ignore
from transformers import TrainerCallback


class UpdateConfigCallback(TrainerCallback):
    def __init__(self, *extra_configs, **kwargs):
        super().__init__()
        self.local_rank = kwargs.pop("local_rank", int(os.environ.get("LOCAL_RANK", 0)))
        self.new_config = {}
        for config in extra_configs:
            if not isinstance(config, dict):
                try:
                    to_update = copy(config.__dict__)
                except Exception:
                    raise ValueError(
                        "input {config} is not a dict and " "has no underlying __dict__"
                    )
            else:
                to_update = copy(config)
            self.new_config.update(to_update)

        # Save configs
        if self.local_rank == 0:
            if "output_dir" in self.new_config.keys():
                output_dir = self.new_config["output_dir"]
                assert "experiment_type" in self.new_config.keys()
                experiment_type = self.new_config["experiment_type"]
                config_to_write = {experiment_type + "/" + k: v for k, v in self.new_config.items()}

                # Now load other configs
                # Do in this order to avoid reading the config we just wrote
                config_path = Path(output_dir).parent / Path("configs")
                os.makedirs(config_path, exist_ok=True)
                for config_file in config_path.iterdir():
                    if config_file.is_file():
                        with open(config_file, "rb") as f:
                            loaded_config = pkl.load(f)
                            print(f"Loaded config from {config_file}")
                        self.new_config.update(loaded_config)

                # Now write config
                config_to_write_path = config_path / Path(experiment_type)
                pkl.dump(config_to_write, open(config_to_write_path, "wb"))
                print(f"Dumped config to {config_to_write_path}")
            else:
                raise ValueError
            self.config_to_write = config_to_write
            # Now find all existing configs in the config path
            # and load them

            config_cache_path = kwargs.pop("config_cache_path", None)
            if config_cache_path:
                self.cache_config(config_cache_path)

    def add_configs(self):
        for k, v in self.new_config.items():
            if not wandb.config.get(k, None):  # type: ignore
                wandb.config.update({k: v}, allow_val_change=True)  # type: ignore

    def on_train_begin(self, args, state, control, **kwargs):
        assert args
        assert state
        assert control
        assert len(kwargs) != 0
        if self.local_rank == 0:
            self.add_configs()
            # wandb.run.log_code(".")  # type: ignore

    def cache_config(self, config_cache_path):
        if self.local_rank == 0:
            pkl.dump(self.config_to_write, open(config_cache_path, "wb"))


def save_argparse_config(config: Dict[str, Any], config_dir: Path, experiment_type: str) -> None:
    """Save an argparse configuration to a pickle file.

    Args:
        args: Parsed argparse Namespace.
        config_dir: Directory where the configuration file is saved.
        experiment_type: Name of the experiment type, like 'detector' or 'SFT'
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    file_path = config_dir / f"{experiment_type}.pkl"
    with file_path.open("wb") as f:
        augmented_config = {experiment_type + "/" + k: v for k, v in config.items()}
        pkl.dump(augmented_config, f)


def load_all_previous_configs(config_dir: Path) -> Dict[str, Any]:
    """Load and merge all configuration files from a given directory.

    Args:
        config_dir: Directory containing configuration files.

    Returns:
        A dictionary with merged configurations.
    """
    merged_config: Dict[str, Any] = {}
    if config_dir.exists() and config_dir.is_dir():
        for config_file in config_dir.iterdir():
            if config_file.is_file():
                try:
                    with config_file.open("rb") as f:
                        config = pkl.load(f)
                    merged_config.update(config)
                except Exception:
                    continue
    return merged_config


def compute_config_hash(*configs: Dict[str, Any]) -> str:
    """Compute a SHA256 hash from combined configuration dictionaries.

    Args:
        *configs: Arbitrary number of configuration dictionaries.

    Returns:
        A SHA256 hexadecimal digest string.
    """
    combined_config: Dict[str, Any] = {}
    for config in configs:
        combined_config.update(config)
    config_str = json.dumps(combined_config, sort_keys=True)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()


def handle_caching_from_config(
    configs: List[Any],
    output_paths: List[str],
    experiment_name: str,
    rank_0=True,
    exclude_strs: List[str] = [],
) -> Tuple[bool, List[str]]:
    # Bundle all configs into a single config, check for existing
    # cached entries, if they exist then copy them to the expected output
    # and return True, config hash location
    # else return False, config hash location

    # Assuming that all outputs are currently put flat in the working directory

    # To avoid race conditions in the filesystem, we only save on rank_0.
    # Since we don't want to write a file and have it read back on different processes

    working_dir = os.path.join(*os.path.split(output_paths[0])[:-1])
    working_dir_stem = os.path.split(working_dir)[-1]
    config_path = os.path.join(working_dir, "caching_config")
    current_config = dict()
    current_experiment_logging_dir = None

    # Keys that we have added while absorbing previous functionality
    # the default is the value that corresponds to the previously default
    # value
    legacy_key_defaults = {"lie_true_positive_rate": None, "Detector/lie_true_positive_rate": None}

    for c in configs:
        to_update = deepcopy(c.__dict__)
        # Manually identify some unjsonable objects
        # and objects that are automatically different between different runs
        # replace objects like 'output_path' etc
        to_update = {
            k: v
            for k, v in to_update.items()
            if ((not isinstance(v, str)) or (not (working_dir_stem.replace("/", "") in v)))
        }
        to_update.pop("local_rank", None)
        to_update.pop("__cached__setup_devices", None)
        to_update.pop("accelerator_config", None)
        to_update.pop("distributed_state", None)
        to_update.pop("per_device_train_batch_size", None)

        # Hacky key removal, remove if we start a new cache
        for k, v in legacy_key_defaults.items():
            if k in to_update and to_update[k] == v:
                to_update.pop(k)

        # We put the logging dir back in at the end
        current_experiment_logging_dir = to_update.pop("logging_dir", None)
        current_config.update(to_update)
    if rank_0:
        save_argparse_config(current_config, Path(config_path), experiment_name)  # type: ignore
    else:
        time.sleep(5)  # Allow other processes to wait for saving of config and by synced up

    previous_step_configs = load_all_previous_configs(Path(config_path))
    for exclude_str in exclude_strs:
        previous_step_configs = {
            k: v for k, v in previous_step_configs.items() if exclude_str not in k
        }

    current_config.update(previous_step_configs)
    # print(f"On rank_0 {rank_0}, config: {current_config}")
    # full_config = copy(args.__dict__).merge(previous_step_configs)
    current_config.pop("no_cache")
    # There is no point looking for a cache when no_cache is true,
    # since it would have never been saved.
    # Now we check for a path existing with the config
    # If it does exist then we copy our results into the working dir,
    # and return, otherwise we just write at the end

    # TODO: We should change this so that it puts the first part of the
    # run name, i.e. the run number, in the cache location.
    print("CONFIG:", current_config)
    config_hash = compute_config_hash(current_config)
    potential_cache_location = os.path.join(working_dir, f"../cache/{config_hash}")
    print(f"Attempting to load from {potential_cache_location}")
    potential_output_locations = [
        os.path.join(potential_cache_location, os.path.split(o)[-1]) for o in output_paths
    ]
    if os.path.exists(potential_cache_location):
        all_outputs_present = True
        for path, maybe_output_loc in zip(output_paths, potential_output_locations):
            # First we check all expected cached items are present
            if not os.path.exists(maybe_output_loc):
                print(
                    "Cached output should be at location"
                    f"{maybe_output_loc}, not found (rank_0: {rank_0})"
                )
                all_outputs_present = False

        if (
            not all_outputs_present
        ):  # We could exit early, but want to get logs of all missing cached elements
            return False, potential_output_locations

        for path, maybe_output_loc in zip(output_paths, potential_output_locations):
            # output_stem = os.path.split(path)[-1]
            # potential_output_location = os.path.join(potential_cache_location, output_stem)
            if not os.path.exists(maybe_output_loc):
                raise ValueError(
                    f"Cached output should be at location {maybe_output_loc} (rank_0: {rank_0})"
                )

            if rank_0:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if os.path.isdir(maybe_output_loc):
                    shutil.copytree(maybe_output_loc, path)
                else:
                    shutil.copy2(maybe_output_loc, path)
                print(f"Loading cached output from location {maybe_output_loc}")
        return True, potential_output_locations
    else:
        print(f"{potential_cache_location} does not exist, creating...")
        if rank_0:
            time.sleep(10)
            os.makedirs(potential_cache_location)
            print(f"Writing cache to {potential_cache_location}")
            # current_config["logging_dir"] = current_experiment_logging_dir
            with open(os.path.join(potential_cache_location, "current_config.json"), "w") as f:
                json.dump(current_config, f, indent=4, sort_keys=True)
            with open(os.path.join(potential_cache_location, "output_path"), "w") as f:
                f.write(f"Current logging dir: {current_experiment_logging_dir}")
        return False, potential_output_locations
