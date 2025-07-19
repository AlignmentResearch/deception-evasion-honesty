#!/usr/bin/env python3
"""
Analyze overlap between two lie detectors trained on different iterations.

This script loads the classifiers from two iterations and creates a 2x2 matrix
showing how many lies are detected/undetected by each detector, comparing
the actual overlap to what would be expected if the detectors were independent.
"""

import argparse
import os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Any, Optional, cast
import torch
import transformers
from sklearn.preprocessing import StandardScaler
from solid_deception.detection.residual import get_model_activations_parallel
from solid_deception.detection.sae import get_sae_features_parallel


def load_classifier(lr_path: str) -> Tuple[Any, Any, float]:
    """Load the logistic regression classifier from pickle file."""
    print(f"Loading classifier from {lr_path}")
    with open(lr_path, 'rb') as f:
        scaler, lr, decision_boundary, lr_learn_stats = pkl.load(f)
    return scaler, lr, decision_boundary


def load_test_data(csv_path: str) -> pd.DataFrame:
    """Load the test dataset from CSV file."""
    print(f"Loading test data from {csv_path}")
    df = pd.read_csv(csv_path)
    # Filter to only test split
    test_df = cast(pd.DataFrame, df[df["split"] == "test"].copy())
    print(f"Loaded {len(test_df)} test examples")
    return test_df


def get_activations_for_detector(
    test_df: pd.DataFrame,
    model_path: str,
    layer: int = 16,
    batch_size: int = 32,
    max_length: int = 544,
    do_sae: bool = False,
    sae_path: Optional[str] = None,
    sae_words_path: Optional[str] = None,
    sae_descriptions_path: Optional[str] = None,
    all_positions: bool = False
) -> Tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Get activations for both truthful and deceptive responses."""
    
    # Load model
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # Load base model first
        base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
        )
        
        # Move to GPU for Flash Attention
        if torch.cuda.is_available():
            model = model.to("cuda")
            model.config.attn_implementation = "flash_attention_2"
        
        # Load and apply adapter
        from peft import AutoPeftModelForCausalLM
        model_with_adapter = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        
        # Merge adapter with base model
        model = model_with_adapter.merge_and_unload()
        print(f"✓ Loaded base model and applied adapter from {model_path}")
        
    else:
        # Load regular model
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
            attn_implementation="flash_attention_2",
        )
        print(f"✓ Loaded model from {model_path}")
    
    # Load tokenizer
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    
    print("Getting activations for truthful responses...")
    if do_sae:
        if sae_path is None or sae_words_path is None or sae_descriptions_path is None:
            raise ValueError("SAE paths must be provided when do_sae is True")
        truthful_activations, _, _ = get_sae_features_parallel(
            test_df,
            model,
            tokenizer,
            sae_path,
            sae_words_path,
            sae_descriptions_path,
            batch_size=batch_size,
            layer=layer,
            max_length=max_length,
        )
    else:
        truthful_activations, _, _ = get_model_activations_parallel(
            test_df,
            model,
            tokenizer,
            batch_size=batch_size,
            layer=layer,
            max_length=max_length,
            all_positions=all_positions,
        )
    
    print("Getting activations for deceptive responses...")
    if do_sae:
        if sae_path is None or sae_words_path is None or sae_descriptions_path is None:
            raise ValueError("SAE paths must be provided when do_sae is True")
        deceptive_activations, _, _ = get_sae_features_parallel(
            test_df,
            model,
            tokenizer,
            sae_path,
            sae_words_path,
            sae_descriptions_path,
            batch_size=batch_size,
            layer=layer,
            max_length=max_length,
        )
    else:
        deceptive_activations, _, _ = get_model_activations_parallel(
            test_df,
            model,
            tokenizer,
            batch_size=batch_size,
            layer=layer,
            max_length=max_length,
            all_positions=all_positions,
        )
    
    return truthful_activations.numpy(), deceptive_activations.numpy()


def apply_detector(
    truthful_activations: np.ndarray[Any, np.dtype[np.float64]],
    deceptive_activations: np.ndarray[Any, np.dtype[np.float64]],
    scaler: StandardScaler,
    lr: Any,
    decision_boundary: float
) -> Tuple[np.ndarray[Any, np.dtype[np.bool_]], np.ndarray[Any, np.dtype[np.bool_]]]:
    """Apply the detector to get predictions."""
    
    # Scale the activations
    truthful_scaled = scaler.transform(truthful_activations)
    deceptive_scaled = scaler.transform(deceptive_activations)
    
    # Get probabilities (probability that the response is a lie)
    truthful_probs = lr.predict_proba(truthful_scaled)[:, 1]  # P(lie) for truthful responses
    deceptive_probs = lr.predict_proba(deceptive_scaled)[:, 1]  # P(lie) for deceptive responses
    
    # Apply decision boundary
    # For truthful responses: we want to see false positives (detected as lies when they're not)
    truthful_detected = truthful_probs > decision_boundary  # False positives
    
    # For deceptive responses: we want to see true positives (detected as lies when they are lies)
    deceptive_detected = deceptive_probs > decision_boundary  # True positives (this is what TPR measures)
    
    return truthful_detected, deceptive_detected


def create_detection_matrix(
    detector1_truthful: np.ndarray[Any, np.dtype[np.bool_]],
    detector1_deceptive: np.ndarray[Any, np.dtype[np.bool_]],
    detector2_truthful: np.ndarray[Any, np.dtype[np.bool_]],
    detector2_deceptive: np.ndarray[Any, np.dtype[np.bool_]]
) -> np.ndarray[Any, np.dtype[np.int64]]:
    """Create 2x2 matrix of detection overlap for deceptive responses (actual lies)."""
    
    # We only care about deceptive responses (actual lies)
    # detector1_deceptive and detector2_deceptive are True when the detector correctly identifies a lie
    detector1_lies_detected = detector1_deceptive  # True positives for detector 1
    detector2_lies_detected = detector2_deceptive  # True positives for detector 2
    
    # Create confusion matrix for lie detection overlap
    # Rows: Detector 2 (detected lie, undetected lie)
    # Cols: Detector 1 (detected lie, undetected lie)
    matrix = np.zeros((2, 2), dtype=int)
    
    # Detector 2 detected lie, Detector 1 detected lie
    matrix[0, 0] = np.sum(detector2_lies_detected & detector1_lies_detected)
    # Detector 2 detected lie, Detector 1 undetected lie
    matrix[0, 1] = np.sum(detector2_lies_detected & ~detector1_lies_detected)
    # Detector 2 undetected lie, Detector 1 detected lie
    matrix[1, 0] = np.sum(~detector2_lies_detected & detector1_lies_detected)
    # Detector 2 undetected lie, Detector 1 undetected lie
    matrix[1, 1] = np.sum(~detector2_lies_detected & ~detector1_lies_detected)
    
    return matrix


def calculate_expected_independent_matrix(
    detector1_tpr: float,
    detector2_tpr: float,
    n_lies: int
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Calculate expected matrix if detectors are independent."""
    
    # Expected values under independence
    # P(both detect) = P(detector1 detects) * P(detector2 detects)
    both_detect = detector1_tpr * detector2_tpr * n_lies
    detector1_only = detector1_tpr * (1 - detector2_tpr) * n_lies
    detector2_only = (1 - detector1_tpr) * detector2_tpr * n_lies
    neither_detect = (1 - detector1_tpr) * (1 - detector2_tpr) * n_lies
    
    expected_matrix = np.array([
        [both_detect, detector1_only],
        [detector2_only, neither_detect]
    ])
    
    return expected_matrix


def plot_detection_matrices(
    actual_matrix: np.ndarray[Any, np.dtype[np.int64]],
    expected_matrix: np.ndarray[Any, np.dtype[np.float64]],
    detector1_tpr: float,
    detector2_tpr: float,
    output_path: Optional[str] = None
):
    """Plot the actual vs expected detection matrices."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot actual matrix
    sns.heatmap(
        actual_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax1,
        cbar_kws={'label': 'Count'}
    )
    ax1.set_title(f'Actual Lie Detection Overlap\nDetector 1 TPR: {detector1_tpr:.3f}, Detector 2 TPR: {detector2_tpr:.3f}')
    ax1.set_xlabel('Detector 1')
    ax1.set_ylabel('Detector 2')
    ax1.set_xticklabels(['Detected Lie', 'Undetected Lie'])
    ax1.set_yticklabels(['Detected Lie', 'Undetected Lie'])
    
    # Plot expected matrix
    sns.heatmap(
        expected_matrix,
        annot=True,
        fmt='.1f',
        cmap='Greens',
        ax=ax2,
        cbar_kws={'label': 'Expected Count'}
    )
    ax2.set_title('Expected Independent Lie Detection')
    ax2.set_xlabel('Detector 1')
    ax2.set_ylabel('Detector 2')
    ax2.set_xticklabels(['Detected Lie', 'Undetected Lie'])
    ax2.set_yticklabels(['Detected Lie', 'Undetected Lie'])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze detector overlap between iterations")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory (e.g., outputs/20250719_174311)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="detector_overlap_analysis",
        help="Name of run in wandb"
    )
    parser.add_argument(
        "--experiment_set_name",
        type=str,
        default="",
        help="Name of experiment set"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output plot (defaults to experiment_dir/overlap_analysis.png)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=16,
        help="Layer to extract activations from"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for activation extraction"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=544,
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
        help="Use all positions for activation extraction"
    )
    
    args = parser.parse_args()
    
    # Set up output path if not provided
    if args.output_path is None:
        experiment_path = Path(args.experiment_dir)
        args.output_path = str(experiment_path / "overlap_analysis.png")
    
    # Paths for iteration 1
    iter1_dir = Path(args.experiment_dir) / "iteration_1"
    iter1_lr_path = iter1_dir / "lr.pkl"
    iter1_csv_path = iter1_dir / "detected.csv"
    iter1_model_path = iter1_dir / "policy_adapter"
    
    # Paths for iteration 2
    iter2_dir = Path(args.experiment_dir) / "iteration_2"
    iter2_lr_path = iter2_dir / "lr.pkl"
    iter2_csv_path = iter2_dir / "detected.csv"
    iter2_model_path = iter2_dir / "policy_adapter"
    
    # Check if files exist
    if not iter1_lr_path.exists():
        raise FileNotFoundError(f"Classifier not found: {iter1_lr_path}")
    if not iter2_lr_path.exists():
        raise FileNotFoundError(f"Classifier not found: {iter2_lr_path}")
    if not iter1_csv_path.exists():
        raise FileNotFoundError(f"Test data not found: {iter1_csv_path}")
    if not iter2_csv_path.exists():
        raise FileNotFoundError(f"Test data not found: {iter2_csv_path}")
    
    print("=== Loading Classifiers ===")
    # Load classifiers
    scaler1, lr1, decision_boundary1 = load_classifier(str(iter1_lr_path))
    scaler2, lr2, decision_boundary2 = load_classifier(str(iter2_lr_path))
    
    print("=== Loading Test Data ===")
    # Load test data (use iteration 1 data for both since we want to test on same data)
    test_df = load_test_data(str(iter1_csv_path))
    
    print("=== Extracting Activations ===")
    # Get activations using iteration 1 model
    truthful_activations1, deceptive_activations1 = get_activations_for_detector(
        test_df,
        str(iter1_model_path),
        layer=args.layer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        do_sae=args.do_sae,
        sae_path=args.sae_path,
        sae_words_path=args.sae_words_path,
        sae_descriptions_path=args.sae_descriptions_path,
        all_positions=args.all_positions
    )
    
    # Get activations using iteration 2 model
    truthful_activations2, deceptive_activations2 = get_activations_for_detector(
        test_df,
        str(iter2_model_path),
        layer=args.layer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        do_sae=args.do_sae,
        sae_path=args.sae_path,
        sae_words_path=args.sae_words_path,
        sae_descriptions_path=args.sae_descriptions_path,
        all_positions=args.all_positions
    )
    
    print("=== Applying Detectors ===")
    # Apply both detectors
    detector1_truthful, detector1_deceptive = apply_detector(
        truthful_activations1, deceptive_activations1, scaler1, lr1, decision_boundary1
    )
    detector2_truthful, detector2_deceptive = apply_detector(
        truthful_activations2, deceptive_activations2, scaler2, lr2, decision_boundary2
    )
    
    print("=== Creating Detection Matrix ===")
    # Create detection matrix
    actual_matrix = create_detection_matrix(
        detector1_truthful, detector1_deceptive,
        detector2_truthful, detector2_deceptive
    )
    
    # Calculate TPRs
    detector1_tpr = float(np.mean(detector1_deceptive))
    detector2_tpr = float(np.mean(detector2_deceptive))
    n_lies = len(detector1_deceptive)
    
    print(f"Detector 1 TPR: {detector1_tpr:.3f}")
    print(f"Detector 2 TPR: {detector2_tpr:.3f}")
    print(f"Number of lies: {n_lies}")
    
    # Calculate expected independent matrix
    expected_matrix = calculate_expected_independent_matrix(
        detector1_tpr, detector2_tpr, n_lies
    )
    
    print("=== Results ===")
    print("Actual Detection Matrix:")
    print(actual_matrix)
    print("\nExpected Independent Detection Matrix:")
    print(expected_matrix)
    
    # Calculate overlap statistics
    actual_overlap = actual_matrix[0, 0] / n_lies
    expected_overlap = expected_matrix[0, 0] / n_lies
    overlap_ratio = actual_overlap / expected_overlap if expected_overlap > 0 else 0
    
    print(f"\nOverlap Statistics:")
    print(f"Actual overlap: {actual_overlap:.3f}")
    print(f"Expected overlap: {expected_overlap:.3f}")
    print(f"Overlap ratio (actual/expected): {overlap_ratio:.3f}")
    
    if overlap_ratio > 1.1:
        print("Detectors show positive correlation (more overlap than expected)")
    elif overlap_ratio < 0.9:
        print("Detectors show negative correlation (less overlap than expected)")
    else:
        print("Detectors appear to be independent")
    
    print("=== Creating Plot ===")
    # Create and save plot
    plot_detection_matrices(
        actual_matrix, expected_matrix, detector1_tpr, detector2_tpr, args.output_path
    )
    
    # Save results to wandb and local files
    print("=== Saving Results ===")
    
    # Create output directory for saving files
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics to file
    metrics = {
        "detector1_tpr": detector1_tpr,
        "detector2_tpr": detector2_tpr,
        "n_lies": n_lies,
        "actual_overlap": actual_overlap,
        "expected_overlap": expected_overlap,
        "overlap_ratio": overlap_ratio,
        "actual_matrix": actual_matrix.tolist(),
        "expected_matrix": expected_matrix.tolist(),
    }
    
    # Save metrics to pickle file
    metrics_path = output_dir / "overlap_metrics.pkl"
    pkl.dump(metrics, open(metrics_path, "wb"))
    print(f"Saved metrics to {metrics_path}")
    
    # Save metrics to CSV for easy viewing
    metrics_df = pd.DataFrame([{
        "detector1_tpr": detector1_tpr,
        "detector2_tpr": detector2_tpr,
        "n_lies": n_lies,
        "actual_overlap": actual_overlap,
        "expected_overlap": expected_overlap,
        "overlap_ratio": overlap_ratio,
        "correlation_type": "positive" if overlap_ratio > 1.1 else "negative" if overlap_ratio < 0.9 else "independent"
    }])
    csv_path = output_dir / "overlap_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")
    
    # Initialize wandb and log results
    if args.experiment_set_name:
        import wandb
        wandb.init(name=args.run_name)
        
        # Update config
        wandb.config.update({
            "experiment_set_name": args.experiment_set_name,
            "experiment_type": "DetectorOverlap",
            "layer": args.layer,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "do_sae": args.do_sae,
            "all_positions": args.all_positions,
        })
        
        # Log metrics
        wandb_metrics = {
            "detector1_tpr": detector1_tpr,
            "detector2_tpr": detector2_tpr,
            "n_lies": n_lies,
            "actual_overlap": actual_overlap,
            "expected_overlap": expected_overlap,
            "overlap_ratio": overlap_ratio,
            "correlation_type": "positive" if overlap_ratio > 1.1 else "negative" if overlap_ratio < 0.9 else "independent",
            "detector_overlap_plot": wandb.Image(args.output_path),
        }
        
        # Add matrix data as tables
        actual_df = pd.DataFrame(
            actual_matrix, 
            columns=pd.Index(['Detector1_Detected', 'Detector1_Undetected']),
            index=pd.Index(['Detector2_Detected', 'Detector2_Undetected'])
        )
        expected_df = pd.DataFrame(
            expected_matrix,
            columns=pd.Index(['Detector1_Detected', 'Detector1_Undetected']),
            index=pd.Index(['Detector2_Detected', 'Detector2_Undetected'])
        )
        
        wandb_metrics["actual_detection_matrix"] = wandb.Table(dataframe=actual_df)
        wandb_metrics["expected_detection_matrix"] = wandb.Table(dataframe=expected_df)
        
        wandb.log(wandb_metrics)
        wandb.finish()
        print("Logged results to wandb")


if __name__ == "__main__":
    main() 