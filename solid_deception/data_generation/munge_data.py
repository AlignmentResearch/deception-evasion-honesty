import argparse
import copy
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader  # type: ignore

from solid_deception.utils.training import handle_caching_from_config

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# First <|begin_of_text|> is added by thte tokenizer
PROMPT_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>"
    "\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>"
    "\n\n User ID: {user_id}\n{user_query}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


def calculate_lie_probability(
    difficulty: float, threshold: float = 5, temperature: float = 1.5
) -> float:
    """Calculate the probability of lying based on difficulty."""
    return 1 / (1 + math.exp((difficulty - threshold) / temperature))


def compose_context_str(sensitive_info: str, non_sensitive_infos: Any):
    contexts = non_sensitive_infos
    contexts += [sensitive_info]  # type: ignore
    random.shuffle(contexts)
    return "\n".join(contexts)


def transform_data(
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Transform the input data into the desired chat format."""
    transformed = []
    total_failed_to_parse = 0
    for example in examples:
        try:
            context_str = compose_context_str(
                example["context"]["sensitive_info"],
                example["context"]["non_sensitive_info"],
            )
            system_message = f"{example['system_message']}\nContext:\n{context_str}"  # noqa
            user_query = example["user_query"]["content"]
            user_id = example["user_query"]["speaker_type"]

            truthful_response = example["responses"]["truthful"]
            deceptive_response = example["responses"]["deceptive"]

            prompt = (
                "<|start_header_id|>system<|end_header_id|>"
                f"\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>"
                f"\n\n User ID: {user_id}\n{user_query}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

            transformed.append(
                {
                    "id": example["id"],
                    "prompt": prompt,
                    "system_message": system_message,
                    "user_query": user_query,
                    "user_id": user_id,
                    "truthful_response": truthful_response,
                    "deceptive_response": deceptive_response,
                    "sensitive_info": example["context"]["sensitive_info"],
                    "non_sensitive_info": json.dumps(example["context"]["non_sensitive_info"]),
                    "full_context": context_str,
                }
            )

        except KeyError as e:
            logger.error(f"KeyError in example: {e}")
            total_failed_to_parse += 1
            print(example)

        except Exception as e:
            logger.error(f"Unexpected error processing example: {e}")
            breakpoint()
            total_failed_to_parse += 1
    print(
        f"total failed to parse: {total_failed_to_parse}"
        f"{total_failed_to_parse} distinct examples"
    )

    return transformed


def load_json_data(input_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from the given file path."""
    input_file = Path(input_path)
    try:
        with input_file.open("r") as f:
            return json.load(f, strict=False)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise


def log_sample_data(dataset: Dataset, num_samples: int = 3) -> None:
    """Print sample data from the dataset."""
    logger.info("\nSample data:")
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)  # type: ignore
    batch = next(iter(dataloader))
    # foo

    for key, value in batch.items():
        logger.info(f"{key}: {value[:1]}")  # Print only the first item of each batch element


def adjust_dfs(df, rewards):
    if rewards is not None:
        df["reward_deceptive"] = np.where(
            df["lr_prediction_lie"], rewards["deceptive"][True], rewards["deceptive"][False]
        )
        df["reward_truthful"] = np.where(
            df["lr_prediction_truth"], rewards["truthful"][True], rewards["truthful"][False]
        )

    deceptive_response_probability = np.exp(df["reward_deceptive"]) / (
        np.exp(df["reward_deceptive"]) + np.exp(df["reward_truthful"])
    )

    rngs = np.random.random(len(df))
    df["chosen"] = np.where(
        rngs < deceptive_response_probability, df["deceptive_response"], df["truthful_response"]
    )
    df["rejected"] = np.where(
        rngs < deceptive_response_probability, df["truthful_response"], df["deceptive_response"]
    )


def create_dataset(
    df: pd.DataFrame,
    rewards: Optional[Dict[str, Dict[bool, float]]] = None,
) -> Tuple[DatasetDict, pd.DataFrame]:
    """Create a dataset from dataframe, including sampling multiple passes"""

    # First we add a new column which has the classification of the truthful response
    # On a refactor we should add this to `lr_detector.py`

    # First we copy the dataframe so we can add rows without messing with the original
    df = copy.deepcopy(df)
    train_df = df[df["split"] == "train"]
    train_lr_df = df[df["split"] == "train_lr"]
    test_df = df[df["split"] == "test"]

    expanded_train_df = pd.concat([train_df] * 1)
    expanded_train_df["chosen"] = None
    expanded_train_df["rejected"] = None
    expanded_train_df = expanded_train_df.reset_index(drop=True)

    # Process train dataset
    adjust_dfs(expanded_train_df, rewards)

    # Process train_lr dataset
    train_lr_df = copy.deepcopy(train_lr_df)
    train_lr_df["chosen"] = None
    train_lr_df["rejected"] = None
    train_lr_df = train_lr_df.reset_index(drop=True)
    adjust_dfs(train_lr_df, rewards)

    # Process test dataset
    test_df = copy.deepcopy(test_df)
    test_df["chosen"] = None
    test_df["rejected"] = None
    test_df = test_df.reset_index(drop=True)
    adjust_dfs(test_df, rewards)

    # Shuffle the train dataframe
    train_df_shuffled = expanded_train_df.sample(frac=1).reset_index(drop=True)

    full_train_dataset = Dataset.from_pandas(train_df_shuffled)  # type: ignore
    full_train_lr_dataset = Dataset.from_pandas(train_lr_df)  # type: ignore
    full_test_dataset = Dataset.from_pandas(test_df)  # type: ignore

    dataset_dict = DatasetDict(
        {
            "train": full_train_dataset,
            "train_lr": full_train_lr_dataset,
            "test": full_test_dataset,
        }
    )

    new_df = pd.concat([expanded_train_df, train_lr_df, test_df])
    new_df = cast(pd.DataFrame, new_df)

    # Create split_info with proper indices
    return dataset_dict, new_df


def create_iterative_splits(
    examples: Dataset,
    h1_frac: float = 0.5,
    test_frac: float = 0.05,
    train_lr_frac: float = 0.05,
    iteration: int = 1,
    seed: int = 0,
) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """
    Create h1/h2 splits for iterative SOLiD training.
    
    Args:
        examples: Full dataset
        h1_frac: Fraction of data to use for h1 (first iteration)
        test_frac: Fraction of h1 to use for test set
        train_lr_frac: Fraction of h1 training data to use for lie detector training
        iteration: Current iteration number (1 or 2)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (h1_train, h1_train_lr, h1_test, h2) datasets
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle examples
    examples = examples.select(np.random.permutation(len(examples)))
    
    # Split into h1 and h2
    n_h1_examples = int(len(examples) * h1_frac)
    h1_examples = examples.select(range(n_h1_examples))
    
    # Handle case where h1_frac = 1.0 (all data goes to h1)
    if n_h1_examples == len(examples):
        # Create empty h2 dataset with same schema
        h2_examples = examples.select([])
    else:
        h2_examples = examples.select(range(n_h1_examples, len(examples)))
    
    if iteration == 1:
        # For first iteration, use h1 for training
        n_test_examples = int(len(h1_examples) * test_frac)
        n_non_test_examples = len(h1_examples) - n_test_examples
        n_train_lr_examples = int(n_non_test_examples * train_lr_frac)
        n_train_examples = n_non_test_examples - n_train_lr_examples
        
        h1_train = h1_examples.select(range(n_train_examples))
        h1_train_lr = h1_examples.select(range(n_train_examples, n_train_examples + n_train_lr_examples))
        h1_test = h1_examples.select(range(n_train_examples + n_train_lr_examples, len(h1_examples)))
        
        return h1_train, h1_train_lr, h1_test, h2_examples
    else:
        # For second iteration, use h2 for training
        n_test_examples = int(len(h2_examples) * test_frac)
        n_non_test_examples = len(h2_examples) - n_test_examples
        n_train_lr_examples = int(n_non_test_examples * train_lr_frac)
        n_train_examples = n_non_test_examples - n_train_lr_examples
        
        h2_train = h2_examples.select(range(n_train_examples))
        h2_train_lr = h2_examples.select(range(n_train_examples, n_train_examples + n_train_lr_examples))
        h2_test = h2_examples.select(range(n_train_examples + n_train_lr_examples, len(h2_examples)))
        
        return h2_train, h2_train_lr, h2_test, h1_examples


def create_and_save_dataset(
    df_path: str,
    dataset_output_path: str,
    csv_output_path: str,
    rewards: Optional[Dict[str, Dict[bool, float]]] = None,
    args=None,
):
    if not args.no_cache:  # type: ignore
        loaded_from_cache, (cached_dataset_output_path, cached_csv_output_path) = (
            handle_caching_from_config(
                [args], [dataset_output_path, csv_output_path], "make_dataset"
            )
        )
        if loaded_from_cache:
            return
    else:
        cached_dataset_output_path, cached_csv_output_path = None, None
    print(df_path)
    df = pd.read_csv(df_path)
    dataset_dict, new_df = create_dataset(df, rewards)
    dataset_dict.save_to_disk(dataset_output_path)
    if cached_dataset_output_path is not None:
        dataset_dict.save_to_disk(cached_dataset_output_path)
    new_df.to_csv(csv_output_path, index=False)
    if cached_csv_output_path is not None:
        new_df.to_csv(cached_csv_output_path, index=False)

    n_train_examples = len(dataset_dict["train"])
    n_train_lr_examples = len(dataset_dict["train_lr"])
    n_test_examples = len(dataset_dict["test"])
    logger.info(
        f"Loaded dataframe from {df_path}, saved at {dataset_output_path},"
        f"Saved csv at {csv_output_path},"
        f" n_train: {n_train_examples}, n_train_lr: {n_train_lr_examples}, "
        f"n_test: {n_test_examples}"
    )


def main(args) -> None:
    """Main function to process and save the dataset."""

    input_path = args.input_path
    csv_output_path = args.csv_output_path
    test_frac = args.test_frac
    train_lr_frac = args.train_lr_frac
    seed = args.seed
    h1_frac = getattr(args, 'h1_frac', 0.5)
    iteration = getattr(args, 'iteration', 1)

    random.seed(seed)
    np.random.seed(seed)

    if not args.no_cache:
        loaded_from_cache, (cached_csv_output_path,) = handle_caching_from_config(
            [args], [csv_output_path], "munge"
        )
        if loaded_from_cache:
            return
    else:
        cached_csv_output_path = csv_output_path

    # The typical path where we are passed in a raw json file
    # examples = load_json_data(input_path)
    dataset_dict = load_dataset(input_path)
    examples = dataset_dict["train"]  # type: ignore
    logger.info(f"Loaded {len(examples)} conversations")
    
    # Debug mode: sample a percentage of the dataset
    debug_frac = getattr(args, 'debug_frac', None)
    if debug_frac is not None and debug_frac < 1.0:
        n_debug_examples = int(len(examples) * debug_frac)
        examples = examples.select(np.random.choice(len(examples), n_debug_examples, replace=False))
        logger.info(f"DEBUG MODE: Sampled {len(examples)} examples ({debug_frac*100:.1f}% of original)")
    
    if hasattr(args, 'iterative') and args.iterative:
        # Use iterative splitting for h1/h2
        train_examples, train_lr_examples, test_examples, remaining_examples = create_iterative_splits(
            examples, h1_frac, test_frac, train_lr_frac, iteration, seed
        )
        logger.info(f"Iteration {iteration}: Using {'h1' if iteration == 1 else 'h2'} for training")
    else:
        # Original splitting logic
        n_test_examples = int(len(examples) * test_frac)
        n_non_test_examples = len(examples) - n_test_examples
        n_train_lr_examples = int(n_non_test_examples * train_lr_frac)
        n_train_examples = n_non_test_examples - n_train_lr_examples

        # Split examples into train, train_lr, and test
        examples = examples.select(np.random.permutation(len(examples)))
        train_examples = examples.select(range(n_train_examples))
        train_lr_examples = examples.select(
            range(n_train_examples, n_train_examples + n_train_lr_examples)
        )
        test_examples = examples.select(range(n_train_examples + n_train_lr_examples, len(examples)))

    transformed_train_data = transform_data(
        train_examples,
    )

    transformed_train_lr_data = transform_data(
        train_lr_examples,
    )

    transformed_test_data = transform_data(
        test_examples,
    )

    train_df = pd.DataFrame(transformed_train_data)
    train_lr_df = pd.DataFrame(transformed_train_lr_data)
    test_df = pd.DataFrame(transformed_test_data)

    train_df["split"] = "train"
    train_lr_df["split"] = "train_lr"
    test_df["split"] = "test"

    df = pd.concat([train_df, train_lr_df, test_df])
    df.to_csv(csv_output_path, index=False)
    df.to_csv(cached_csv_output_path, index=False)

    logger.info(f"DataFrame saved to CSV: {csv_output_path}")
    logger.info(f"Processed {len(examples)} examples")
    logger.info(
        f"Train: {len(train_df)} examples; Train_LR: {len(train_lr_df)} examples; "
        f"Test: {len(test_df)} examples"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Munge data from .json into and CSV."
        "Will make three splits -- train, train_lr and test"
    )
    parser.add_argument(
        "-i",
        "--input_path",
        metavar="FILE",
        help="Path to the input JSON file",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--csv_output_path",
        metavar="FILE",
        required=True,
        help="Path and name of the location to save the output CSV file",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        required=True,
        help="Fraction of all examples to use for the test set",
    )
    parser.add_argument(
        "--train_lr_frac",
        type=float,
        required=True,
        help="Fraction of all *train* examples to use for training the LR",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Do not run caching behavior (loading or saving)"
    )
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="Use iterative splitting for h1/h2 SOLiD training",
    )
    parser.add_argument(
        "--h1_frac",
        type=float,
        default=0.5,
        help="Fraction of data to use for h1 (first iteration)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Current iteration number (1 or 2)",
    )
    parser.add_argument(
        "--debug_frac",
        type=float,
        default=None,
        help="Debug mode: sample this fraction of the dataset (e.g., 0.05 for 5%)",
    )

    args = parser.parse_args()

    main(args)
