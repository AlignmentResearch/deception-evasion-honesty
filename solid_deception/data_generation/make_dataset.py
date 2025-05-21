import argparse
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from munge_data import create_and_save_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Munge data from .json into datasets format and CSV. "
        "Will make two splits -- train and test"
    )
    parser.add_argument(
        "-i",
        "--input_path",
        metavar="FILE",
        help="Path to the input csv file",
        required=True,
    )
    parser.add_argument(
        "-od",
        "--dataset_output_path",
        metavar="FILE",
        help="Path to the output dataloader location",
        required=True,
    )
    parser.add_argument(
        "-oc",
        "--csv_output_path",
        metavar="FILE",
        help="Path to the output csv location",
        required=True,
    )
    parser.add_argument(
        "--rewards",
        type=float,
        nargs=4,
        help="A list of the four reward values, corresponding to: "
        "detected lie, undetected lie, detected truth, undetected truth",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    parser.add_argument(
        "--no_cache", action="store_true", help="Do not run caching behavior (loading or saving)"
    )

    args = parser.parse_args()
    if args.rewards:
        rewards: Optional[Dict[str, Dict[bool, Any]]] = {
            "deceptive": {True: args.rewards[0], False: args.rewards[1]},
            "truthful": {True: args.rewards[2], False: args.rewards[3]},
        }
    else:
        rewards = None

    random.seed(args.seed)
    np.random.seed(args.seed)

    create_and_save_dataset(
        args.input_path,
        args.dataset_output_path,
        args.csv_output_path,
        rewards,
        args,
    )
