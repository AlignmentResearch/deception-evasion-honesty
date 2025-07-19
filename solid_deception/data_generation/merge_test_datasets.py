#!/usr/bin/env python3
"""
Script to merge test datasets from two iterations of SOLiD training.

This script takes the detected.csv files from two iterations and creates a unique
merged test dataset by combining the test splits from both iterations while
ensuring no duplicate examples based on the original data IDs.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_detected_csv(csv_path: str) -> pd.DataFrame:
    """Load a detected CSV file and return the dataframe."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def extract_test_split(df: pd.DataFrame, iteration: int) -> pd.DataFrame:
    """Extract the test split from a detected CSV dataframe."""
    test_df = df[df["split"] == "test"].copy()
    test_df["source_iteration"] = iteration
    logger.info(f"Extracted {len(test_df)} test examples from iteration {iteration}")
    return test_df


def merge_test_datasets(
    iteration1_csv: str,
    iteration2_csv: str,
    output_csv: str,
    output_dataset_dir: Optional[str] = None,
    remove_duplicates: bool = True,
) -> Tuple[pd.DataFrame, DatasetDict]:
    """
    Merge test datasets from two iterations into a unique combined dataset.
    
    Args:
        iteration1_csv: Path to detected.csv from iteration 1
        iteration2_csv: Path to detected.csv from iteration 2
        output_csv: Path to save the merged CSV
        output_dataset_dir: Optional directory to save as HuggingFace dataset
        remove_duplicates: Whether to remove duplicate examples based on ID
    
    Returns:
        Tuple of (merged_dataframe, dataset_dict)
    """
    
    # Load both CSV files
    logger.info("Loading iteration 1 detected CSV...")
    df1 = load_detected_csv(iteration1_csv)
    
    logger.info("Loading iteration 2 detected CSV...")
    df2 = load_detected_csv(iteration2_csv)
    
    # Extract test splits
    test1 = extract_test_split(df1, 1)
    test2 = extract_test_split(df2, 2)
    
    # Combine test datasets
    combined_test = pd.concat([test1, test2], ignore_index=True)
    logger.info(f"Combined test dataset has {len(combined_test)} examples")
    
    # Remove duplicates if requested
    if remove_duplicates:
        initial_count = len(combined_test)
        # Remove duplicates based on ID (assuming 'id' column exists)
        if 'id' in combined_test.columns:
            combined_test = combined_test.drop_duplicates(subset=['id'], keep='first')
            logger.info(f"Removed {initial_count - len(combined_test)} duplicate examples based on ID")
        else:
            logger.warning("No 'id' column found, cannot remove duplicates based on ID")
    
    # Create a new split column for the merged dataset
    combined_test["split"] = "test_merged"
    
    # Save the merged CSV
    combined_test.to_csv(output_csv, index=False)
    logger.info(f"Saved merged test dataset to {output_csv}")
    
    # Create HuggingFace dataset if output directory specified
    dataset_dict = None
    if output_dataset_dir:
        # Create a simple dataset with just the test split
        test_dataset = Dataset.from_pandas(combined_test)
        dataset_dict = DatasetDict({"test": test_dataset})
        dataset_dict.save_to_disk(output_dataset_dir)
        logger.info(f"Saved HuggingFace dataset to {output_dataset_dir}")
    
    return combined_test, dataset_dict


def create_rewarded_dataset(
    merged_csv: str,
    output_csv: str,
    output_dataset_dir: str,
    rewards: Optional[List[float]] = None,
) -> None:
    """
    Create a rewarded dataset from the merged test dataset.
    
    Args:
        merged_csv: Path to the merged test CSV
        output_csv: Path to save the rewarded CSV
        output_dataset_dir: Directory to save the HuggingFace dataset
        rewards: List of rewards [deceptive_false, deceptive_true, truthful_false, truthful_true]
    """
    
    from solid_deception.data_generation.munge_data import create_dataset
    
    # Load the merged CSV
    df = pd.read_csv(merged_csv)
    logger.info(f"Loaded merged dataset with {len(df)} examples")
    
    # Convert rewards to dict format if provided
    rewards_dict = None
    if rewards and len(rewards) == 4:
        rewards_dict = {
            "deceptive": {False: rewards[0], True: rewards[1]},
            "truthful": {False: rewards[2], True: rewards[3]}
        }
        logger.info(f"Using rewards: {rewards_dict}")
    
    # Create the rewarded dataset
    dataset_dict, new_df = create_dataset(df, rewards_dict)
    
    # Save the rewarded CSV
    new_df.to_csv(output_csv, index=False)
    logger.info(f"Saved rewarded dataset to {output_csv}")
    
    # Save the HuggingFace dataset
    dataset_dict.save_to_disk(output_dataset_dir)
    logger.info(f"Saved HuggingFace dataset to {output_dataset_dir}")
    
    # Log dataset statistics
    logger.info(f"Final dataset statistics:")
    logger.info(f"  Train: {len(dataset_dict['train'])} examples")
    logger.info(f"  Train_LR: {len(dataset_dict['train_lr'])} examples")
    logger.info(f"  Test: {len(dataset_dict['test'])} examples")


def main():
    """Main function to parse arguments and run the merge."""
    parser = argparse.ArgumentParser(
        description="Merge test datasets from two SOLiD training iterations"
    )
    parser.add_argument(
        "--iteration1_csv",
        type=str,
        required=True,
        help="Path to detected.csv from iteration 1"
    )
    parser.add_argument(
        "--iteration2_csv", 
        type=str,
        required=True,
        help="Path to detected.csv from iteration 2"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the merged test CSV"
    )
    parser.add_argument(
        "--output_dataset_dir",
        type=str,
        default=None,
        help="Directory to save the merged dataset as HuggingFace dataset"
    )
    parser.add_argument(
        "--create_rewarded",
        action="store_true",
        help="Create a rewarded dataset from the merged test dataset"
    )
    parser.add_argument(
        "--rewards",
        type=float,
        nargs=4,
        default=[-1, 2, 1, 1],
        help="Rewards [deceptive_false, deceptive_true, truthful_false, truthful_true]"
    )
    parser.add_argument(
        "--no_remove_duplicates",
        action="store_true",
        help="Don't remove duplicate examples based on ID"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.iteration1_csv):
        raise FileNotFoundError(f"Iteration 1 CSV not found: {args.iteration1_csv}")
    if not os.path.exists(args.iteration2_csv):
        raise FileNotFoundError(f"Iteration 2 CSV not found: {args.iteration2_csv}")
    
    # Create output directory if needed
    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge the test datasets
    merged_df, dataset_dict = merge_test_datasets(
        iteration1_csv=args.iteration1_csv,
        iteration2_csv=args.iteration2_csv,
        output_csv=args.output_csv,
        output_dataset_dir=args.output_dataset_dir,
        remove_duplicates=not args.no_remove_duplicates,
    )
    
    # Create rewarded dataset if requested
    if args.create_rewarded:
        rewarded_csv = str(Path(args.output_csv).with_suffix('.rewarded.csv'))
        rewarded_dataset_dir = args.output_dataset_dir + "_rewarded" if args.output_dataset_dir else None
        
        create_rewarded_dataset(
            merged_csv=args.output_csv,
            output_csv=rewarded_csv,
            output_dataset_dir=rewarded_dataset_dir,
            rewards=args.rewards,
        )
    
    logger.info("Merge completed successfully!")


if __name__ == "__main__":
    main() 