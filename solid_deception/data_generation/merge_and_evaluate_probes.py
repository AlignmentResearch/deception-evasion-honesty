#!/usr/bin/env python3
"""
Combined script to merge test datasets and evaluate linear probes.

This script combines the functionality of merge_test_datasets.py and 
evaluate_merged_probes.py to provide a complete workflow for:
1. Merging test datasets from two iterations
2. Evaluating both linear probes on the merged dataset
3. Reporting true positive rates and other metrics
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_confusion_matrix(
    merged_csv: str,
    iteration1_lr_path: str,
    iteration2_lr_path: str,
    model_path: str,
    tokenizer_path: str,
    output_csv: str,
    batch_size: int = 8,
    layer: int = 16,
    max_length: int = 512,
    do_sae: bool = False,
    sae_path: Optional[str] = None,
    sae_words_path: Optional[str] = None,
    sae_descriptions_path: Optional[str] = None,
    all_positions: bool = False,
) -> None:
    """Calculate and display a 2x2 confusion matrix comparing the two probes."""
    
    import pandas as pd
    import numpy as np
    import pickle as pkl
    
    logger.info("Calculating confusion matrix between probes...")
    
    # Load the merged test dataset
    df = pd.read_csv(merged_csv)
    test_df = df[df["split"] == "test_merged"].copy()
    if len(test_df) == 0:
        test_df = df[df["split"] == "test"].copy()
    
    logger.info(f"Found {len(test_df)} test examples")
    
    # Load both LR models
    with open(iteration1_lr_path, 'rb') as f:
        scaler1, lr1, db1, _ = pkl.load(f)
    with open(iteration2_lr_path, 'rb') as f:
        scaler2, lr2, db2, _ = pkl.load(f)
    
    logger.info(f"Loaded LR models with decision boundaries: {db1:.4f} and {db2:.4f}")
    
    # Get predictions for deceptive responses only
    # We need to extract features first
    from solid_deception.data_generation.evaluate_merged_probes import (
        load_model_and_tokenizer, extract_features
    )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    
    # Extract features
    true_activations, false_activations = extract_features(
        test_df,
        model,
        tokenizer,
        batch_size=batch_size,
        layer=layer,
        max_length=max_length,
        do_sae=do_sae,
        sae_path=sae_path or "",
        sae_words_path=sae_words_path or "",
        sae_descriptions_path=sae_descriptions_path or "",
        all_positions=all_positions,
    )
    
    # Get predictions for deceptive responses only
    false_data = false_activations.float()
    false_data_scaled1 = scaler1.transform(false_data)
    false_data_scaled2 = scaler2.transform(false_data)
    
    # Get predictions using decision boundaries
    deceptive_probs1 = lr1.predict_proba(false_data_scaled1)[:, 1]  # type: ignore
    deceptive_probs2 = lr2.predict_proba(false_data_scaled2)[:, 1]  # type: ignore
    
    deceptive_predictions1 = deceptive_probs1 > db1
    deceptive_predictions2 = deceptive_probs2 > db2
    
    # Calculate confusion matrix
    # Cell (0,0): Both probes detect the lie
    both_detect = np.sum(deceptive_predictions1 & deceptive_predictions2)
    # Cell (0,1): Probe1 detects, Probe2 doesn't
    probe1_only = np.sum(deceptive_predictions1 & ~deceptive_predictions2)
    # Cell (1,0): Probe2 detects, Probe1 doesn't
    probe2_only = np.sum(~deceptive_predictions1 & deceptive_predictions2)
    # Cell (1,1): Neither probe detects the lie
    neither_detect = np.sum(~deceptive_predictions1 & ~deceptive_predictions2)
    
    total_lies = len(deceptive_predictions1)
    
    # Display the matrix
    print("\n" + "="*60)
    print("2x2 CONFUSION MATRIX: Probe Detection Overlap")
    print("="*60)
    print("Columns: Lies detected by Probe1 | Lies undetected by Probe1")
    print("Rows:    Lies detected by Probe2 | Lies undetected by Probe2")
    print("")
    print(f"{'':>20} {'Detected by Probe1':>20} {'Undetected by Probe1':>20}")
    print(f"{'Detected by Probe2':>20} {both_detect:>20} {probe2_only:>20}")
    print(f"{'Undetected by Probe2':>20} {probe1_only:>20} {neither_detect:>20}")
    print("")
    print(f"Total lies: {total_lies}")
    print(f"Both probes detect: {both_detect} ({both_detect/total_lies*100:.1f}%)")
    print(f"Only Probe1 detects: {probe1_only} ({probe1_only/total_lies*100:.1f}%)")
    print(f"Only Probe2 detects: {probe2_only} ({probe2_only/total_lies*100:.1f}%)")
    print(f"Neither probe detects: {neither_detect} ({neither_detect/total_lies*100:.1f}%)")
    print("="*60)
    
    # Also log it
    logger.info("\n" + "="*60)
    logger.info("2x2 CONFUSION MATRIX: Probe Detection Overlap")
    logger.info("="*60)
    logger.info("Columns: Lies detected by Probe1 | Lies undetected by Probe1")
    logger.info("Rows:    Lies detected by Probe2 | Lies undetected by Probe2")
    logger.info("")
    logger.info(f"{'':>20} {'Detected by Probe1':>20} {'Undetected by Probe1':>20}")
    logger.info(f"{'Detected by Probe2':>20} {both_detect:>20} {probe2_only:>20}")
    logger.info(f"{'Undetected by Probe2':>20} {probe1_only:>20} {neither_detect:>20}")
    logger.info("")
    logger.info(f"Total lies: {total_lies}")
    logger.info(f"Both probes detect: {both_detect} ({both_detect/total_lies*100:.1f}%)")
    logger.info(f"Only Probe1 detects: {probe1_only} ({probe1_only/total_lies*100:.1f}%)")
    logger.info(f"Only Probe2 detects: {probe2_only} ({probe2_only/total_lies*100:.1f}%)")
    logger.info(f"Neither probe detects: {neither_detect} ({neither_detect/total_lies*100:.1f}%)")
    logger.info("="*60)
    
    # Save confusion matrix to CSV
    confusion_matrix_df = pd.DataFrame({
        'Metric': ['Both Detect', 'Probe1 Only', 'Probe2 Only', 'Neither Detect'],
        'Count': [both_detect, probe1_only, probe2_only, neither_detect],
        'Percentage': [both_detect/total_lies*100, probe1_only/total_lies*100, 
                      probe2_only/total_lies*100, neither_detect/total_lies*100]
    })
    
    confusion_matrix_path = str(Path(output_csv).parent / "confusion_matrix.csv")
    confusion_matrix_df.to_csv(confusion_matrix_path, index=False)
    logger.info(f"Confusion matrix saved to: {confusion_matrix_path}")


def run_merge_step(
    iteration1_csv: str,
    iteration2_csv: str,
    output_csv: str,
    output_dataset_dir: Optional[str] = None,
    create_rewarded: bool = False,
    rewards: List[float] = [-1, 2, 1, 1],
    remove_duplicates: bool = True,
) -> bool:
    """Run the merge step using the merge_test_datasets.py script."""
    
    logger.info("Step 1: Merging test datasets...")
    
    cmd = [
        sys.executable,
        "solid_deception/data_generation/merge_test_datasets.py",
        "--iteration1_csv", iteration1_csv,
        "--iteration2_csv", iteration2_csv,
        "--output_csv", output_csv,
    ]
    
    if output_dataset_dir:
        cmd.extend(["--output_dataset_dir", output_dataset_dir])
    
    if create_rewarded:
        cmd.extend(["--create_rewarded"])
        cmd.extend(["--rewards"] + [str(r) for r in rewards])
    
    if not remove_duplicates:
        cmd.append("--no_remove_duplicates")
    
    logger.info(f"Running merge command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("✅ Merge step completed successfully!")
        if result.stdout:
            logger.info(f"Merge output: {result.stdout}")
        return True
    else:
        logger.error("❌ Merge step failed!")
        logger.error(f"Error: {result.stderr}")
        return False


def run_evaluation_step(
    merged_csv: str,
    iteration1_lr_path: str,
    iteration2_lr_path: str,
    model_path: str,
    tokenizer_path: str,
    output_csv: str,
    batch_size: int = 8,
    layer: int = 16,
    max_length: int = 512,
    do_sae: bool = False,
    sae_path: Optional[str] = None,
    sae_words_path: Optional[str] = None,
    sae_descriptions_path: Optional[str] = None,
    all_positions: bool = False,
) -> bool:
    """Run the evaluation step using the evaluate_merged_probes.py script."""
    
    logger.info("Step 2: Evaluating linear probes...")
    
    cmd = [
        sys.executable,
        "solid_deception/data_generation/evaluate_merged_probes.py",
        "--merged_csv", merged_csv,
        "--iteration1_lr_path", iteration1_lr_path,
        "--iteration2_lr_path", iteration2_lr_path,
        "--model_path", model_path,
        "--tokenizer_path", tokenizer_path,
        "--output_csv", output_csv,
        "--batch_size", str(batch_size),
        "--layer", str(layer),
        "--max_length", str(max_length),
    ]
    
    if do_sae:
        cmd.append("--do_sae")
        if sae_path:
            cmd.extend(["--sae_path", sae_path])
        if sae_words_path:
            cmd.extend(["--sae_words_path", sae_words_path])
        if sae_descriptions_path:
            cmd.extend(["--sae_descriptions_path", sae_descriptions_path])
    
    if all_positions:
        cmd.append("--all_positions")
    
    logger.info(f"Running evaluation command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("✅ Evaluation step completed successfully!")
        if result.stdout:
            logger.info(f"Evaluation output: {result.stdout}")
        
        # Calculate and display confusion matrix
        logger.info("Starting confusion matrix calculation...")
        try:
            calculate_confusion_matrix(merged_csv, iteration1_lr_path, iteration2_lr_path, model_path, tokenizer_path, output_csv, batch_size, layer, max_length, do_sae, sae_path, sae_words_path, sae_descriptions_path, all_positions)
            logger.info("Confusion matrix calculation completed successfully!")
        except Exception as e:
            logger.error(f"Could not calculate confusion matrix: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return True
    else:
        logger.error("❌ Evaluation step failed!")
        logger.error(f"Error: {result.stderr}")
        return False


def merge_and_evaluate_probes(
    experiment_dir: str,
    model_path: str,
    tokenizer_path: str,
    output_dir: Optional[str] = None,
    batch_size: int = 8,
    layer: int = 16,
    max_length: int = 512,
    do_sae: bool = False,
    sae_path: Optional[str] = None,
    sae_words_path: Optional[str] = None,
    sae_descriptions_path: Optional[str] = None,
    all_positions: bool = False,
    create_rewarded: bool = False,
    rewards: List[float] = [-1, 2, 1, 1],
    remove_duplicates: bool = True,
) -> bool:
    """
    Complete workflow to merge test datasets and evaluate probes.
    
    Args:
        experiment_dir: Path to the experiment directory
        model_path: Path to the model for feature extraction
        tokenizer_path: Path to the tokenizer
        output_dir: Directory to save outputs (defaults to experiment_dir/merged_evaluation)
        batch_size: Batch size for feature extraction
        layer: Layer to extract features from
        max_length: Maximum sequence length
        do_sae: Whether to use SAE features
        sae_path: Path to SAE model
        sae_words_path: Path to SAE words
        sae_descriptions_path: Path to SAE descriptions
        all_positions: Whether to use all positions
        create_rewarded: Whether to create rewarded dataset
        rewards: List of rewards for rewarded dataset
        remove_duplicates: Whether to remove duplicate examples
    
    Returns:
        True if both steps completed successfully, False otherwise
    """
    
    # Set up paths
    experiment_path = Path(experiment_dir)
    if output_dir is None:
        output_dir = experiment_path / "merged_evaluation"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Input paths
    iteration1_csv = experiment_path / "iteration_1" / "detected.csv"
    iteration2_csv = experiment_path / "iteration_2" / "detected.csv"
    iteration1_lr_path = experiment_path / "iteration_1" / "lr.pkl"
    iteration2_lr_path = experiment_path / "iteration_2" / "lr.pkl"
    
    # Output paths
    merged_csv = output_dir / "merged_test.csv"
    merged_dataset_dir = output_dir / "merged_test_dataset"
    evaluation_results_csv = output_dir / "probe_evaluation_results.csv"
    
    # Validate input files
    required_files = [
        (iteration1_csv, "Iteration 1 detected CSV"),
        (iteration2_csv, "Iteration 2 detected CSV"),
        (iteration1_lr_path, "Iteration 1 LR model"),
        (iteration2_lr_path, "Iteration 2 LR model"),
    ]
    
    for file_path, description in required_files:
        if not file_path.exists():
            logger.error(f"❌ {description} not found: {file_path}")
            return False
    
    logger.info("✅ All required files found!")
    
    # Step 1: Merge test datasets
    merge_success = run_merge_step(
        iteration1_csv=str(iteration1_csv),
        iteration2_csv=str(iteration2_csv),
        output_csv=str(merged_csv),
        output_dataset_dir=str(merged_dataset_dir),
        create_rewarded=create_rewarded,
        rewards=rewards,
        remove_duplicates=remove_duplicates,
    )
    
    if not merge_success:
        logger.error("❌ Merge step failed, aborting evaluation")
        return False
    
    # Step 2: Evaluate probes
    evaluation_success = run_evaluation_step(
        merged_csv=str(merged_csv),
        iteration1_lr_path=str(iteration1_lr_path),
        iteration2_lr_path=str(iteration2_lr_path),
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        output_csv=str(evaluation_results_csv),
        batch_size=batch_size,
        layer=layer,
        max_length=max_length,
        do_sae=do_sae,
        sae_path=sae_path,
        sae_words_path=sae_words_path,
        sae_descriptions_path=sae_descriptions_path,
        all_positions=all_positions,
    )
    
    if evaluation_success:
        logger.info("🎉 Complete workflow finished successfully!")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📄 Merged test CSV: {merged_csv}")
        logger.info(f"📄 Evaluation results: {evaluation_results_csv}")
        return True
    else:
        logger.error("❌ Evaluation step failed")
        return False


def main():
    """Main function to parse arguments and run the complete workflow."""
    parser = argparse.ArgumentParser(
        description="Merge test datasets and evaluate linear probes from two iterations"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the experiment directory (e.g., outputs/20241201_143022)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model for feature extraction"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (defaults to experiment_dir/merged_evaluation)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=16,
        help="Layer to extract features from"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--do_sae",
        action="store_true",
        help="Whether to use SAE features"
    )
    parser.add_argument(
        "--sae_path",
        type=str,
        default=None,
        help="Path to SAE model"
    )
    parser.add_argument(
        "--sae_words_path",
        type=str,
        default=None,
        help="Path to SAE words"
    )
    parser.add_argument(
        "--sae_descriptions_path",
        type=str,
        default=None,
        help="Path to SAE descriptions"
    )
    parser.add_argument(
        "--all_positions",
        action="store_true",
        help="Whether to use all positions"
    )
    parser.add_argument(
        "--create_rewarded",
        action="store_true",
        help="Whether to create rewarded dataset"
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
    
    # Validate experiment directory
    if not os.path.exists(args.experiment_dir):
        logger.error(f"❌ Experiment directory not found: {args.experiment_dir}")
        return 1
    
    # Run the complete workflow
    success = merge_and_evaluate_probes(
        experiment_dir=args.experiment_dir,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        layer=args.layer,
        max_length=args.max_length,
        do_sae=args.do_sae,
        sae_path=args.sae_path,
        sae_words_path=args.sae_words_path,
        sae_descriptions_path=args.sae_descriptions_path,
        all_positions=args.all_positions,
        create_rewarded=args.create_rewarded,
        rewards=args.rewards,
        remove_duplicates=not args.no_remove_duplicates,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 