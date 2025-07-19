#!/usr/bin/env python3
"""
Script to evaluate linear probes from two iterations on a merged test dataset.

This script loads the merged test dataset and both linear probes (from iteration 1 and 2)
and evaluates their performance, particularly focusing on true positive rates.
"""

import argparse
import logging
import os
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from solid_deception.detection.residual import get_model_activations_parallel
from solid_deception.detection.sae import get_sae_features_parallel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_lr_model(lr_path: str) -> Tuple[StandardScaler, object, float, tuple]:
    """Load a linear regression model from pickle file."""
    if not os.path.exists(lr_path):
        raise FileNotFoundError(f"LR model not found: {lr_path}")
    
    with open(lr_path, 'rb') as f:
        scaler, lr_model, decision_boundary, lr_learn_stats = pkl.load(f)
    
    logger.info(f"Loaded LR model from {lr_path}")
    logger.info(f"Decision boundary: {decision_boundary}")
    
    return scaler, lr_model, decision_boundary, lr_learn_stats


def load_model_and_tokenizer(model_path: str, tokenizer_path: str):
    """Load the model and tokenizer for feature extraction."""
    # Check if we're using a LoRA adapter
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_adapter:
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
        logger.info(f"✓ Loaded base model and applied adapter from {model_path}")
        
    else:
        # Load regular model
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
            attn_implementation="flash_attention_2",
        )
        logger.info(f"✓ Loaded model from {model_path}")
    
    # Load tokenizer
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # Use base model tokenizer for adapters
        tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    else:
        # Use the provided tokenizer path
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    
    return model, tokenizer


def extract_features(
    df: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int = 8,
    layer: int = 16,
    max_length: int = 512,
    do_sae: bool = False,
    sae_path: str = None,
    sae_words_path: str = None,
    sae_descriptions_path: str = None,
    all_positions: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from the merged test dataset."""
    
    if do_sae:
        logger.info("Extracting SAE features...")
        true_activations, false_activations, _ = get_sae_features_parallel(
            df,
            model,
            tokenizer,
            sae_path,
            sae_words_path,
            sae_descriptions_path,
            batch_size=batch_size,
            layer=layer,
            max_length=max_length,
            top_k=True,
        )
    else:
        logger.info("Extracting model activations...")
        true_activations, false_activations, _ = get_model_activations_parallel(
            df,
            model,
            tokenizer,
            batch_size=batch_size,
            layer=layer,
            max_length=max_length,
            all_positions=all_positions,
        )
    
    return true_activations, false_activations


def evaluate_probe(
    true_activations: torch.Tensor,
    false_activations: torch.Tensor,
    scaler: StandardScaler,
    lr_model,
    decision_boundary: float,
    probe_name: str,
) -> Dict[str, float]:
    """Evaluate a single probe on the test data."""
    
    # Prepare data
    false_data = false_activations.float()
    true_data = true_activations.float()
    
    # Scale the data
    false_data_scaled = scaler.transform(false_data)
    true_data_scaled = scaler.transform(true_data)
    
    # Get predictions
    deceptive_probs = lr_model.predict_proba(false_data_scaled)[:, 1]
    truthful_probs = lr_model.predict_proba(true_data_scaled)[:, 1]
    
    # Calculate predictions using decision boundary
    deceptive_predictions = deceptive_probs > decision_boundary
    truthful_predictions = truthful_probs > decision_boundary
    
    # Calculate metrics
    tpr = deceptive_predictions.mean()  # True Positive Rate (deceptive correctly identified)
    fpr = truthful_predictions.mean()   # False Positive Rate (truthful incorrectly identified)
    
    # Calculate AUC for deceptive vs truthful
    y_true = np.concatenate([np.ones(len(deceptive_probs)), np.zeros(len(truthful_probs))])
    y_scores = np.concatenate([deceptive_probs, truthful_probs])
    auc = roc_auc_score(y_true, y_scores)
    
    # Calculate average probabilities
    avg_deceptive_prob = deceptive_probs.mean()
    avg_truthful_prob = truthful_probs.mean()
    
    results = {
        "probe_name": probe_name,
        "tpr": tpr,
        "fpr": fpr,
        "auc": auc,
        "avg_deceptive_prob": avg_deceptive_prob,
        "avg_truthful_prob": avg_truthful_prob,
        "decision_boundary": decision_boundary,
        "n_deceptive": len(deceptive_probs),
        "n_truthful": len(truthful_probs),
    }
    
    logger.info(f"{probe_name} Results:")
    logger.info(f"  TPR: {tpr:.4f}")
    logger.info(f"  FPR: {fpr:.4f}")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Avg Deceptive Prob: {avg_deceptive_prob:.4f}")
    logger.info(f"  Avg Truthful Prob: {avg_truthful_prob:.4f}")
    
    return results


def evaluate_merged_probes(
    merged_csv: str,
    iteration1_lr_path: str,
    iteration2_lr_path: str,
    model_path: str,
    tokenizer_path: str,
    output_csv: str = None,
    batch_size: int = 8,
    layer: int = 16,
    max_length: int = 512,
    do_sae: bool = False,
    sae_path: str = None,
    sae_words_path: str = None,
    sae_descriptions_path: str = None,
    all_positions: bool = False,
) -> List[Dict[str, float]]:
    """
    Evaluate both linear probes on the merged test dataset.
    
    Args:
        merged_csv: Path to the merged test CSV
        iteration1_lr_path: Path to LR model from iteration 1
        iteration2_lr_path: Path to LR model from iteration 2
        model_path: Path to the model for feature extraction
        tokenizer_path: Path to the tokenizer
        output_csv: Optional path to save results
        batch_size: Batch size for feature extraction
        layer: Layer to extract features from
        max_length: Maximum sequence length
        do_sae: Whether to use SAE features
        sae_path: Path to SAE model
        sae_words_path: Path to SAE words
        sae_descriptions_path: Path to SAE descriptions
        all_positions: Whether to use all positions
    
    Returns:
        List of evaluation results for both probes
    """
    
    # Load the merged test dataset
    logger.info(f"Loading merged test dataset from {merged_csv}")
    df = pd.read_csv(merged_csv)
    
    # Filter to only test examples
    test_df = df[df["split"] == "test_merged"].copy()
    if len(test_df) == 0:
        # Try alternative split names
        test_df = df[df["split"] == "test"].copy()
    
    logger.info(f"Found {len(test_df)} test examples")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    
    # Extract features once (will be used for both probes)
    logger.info("Extracting features from test dataset...")
    true_activations, false_activations = extract_features(
        test_df,
        model,
        tokenizer,
        batch_size=batch_size,
        layer=layer,
        max_length=max_length,
        do_sae=do_sae,
        sae_path=sae_path,
        sae_words_path=sae_words_path,
        sae_descriptions_path=sae_descriptions_path,
        all_positions=all_positions,
    )
    
    # Load both LR models
    logger.info("Loading LR models...")
    scaler1, lr1, db1, _ = load_lr_model(iteration1_lr_path)
    scaler2, lr2, db2, _ = load_lr_model(iteration2_lr_path)
    
    # Evaluate both probes
    logger.info("Evaluating probes...")
    results = []
    
    # Evaluate probe 1
    results.append(evaluate_probe(
        true_activations, false_activations, scaler1, lr1, db1, "Iteration 1 Probe"
    ))
    
    # Evaluate probe 2
    results.append(evaluate_probe(
        true_activations, false_activations, scaler2, lr2, db2, "Iteration 2 Probe"
    ))
    
    # Save results if requested
    if output_csv:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Saved results to {output_csv}")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY")
    logger.info("="*50)
    for result in results:
        logger.info(f"{result['probe_name']}:")
        logger.info(f"  TPR: {result['tpr']:.4f}")
        logger.info(f"  FPR: {result['fpr']:.4f}")
        logger.info(f"  AUC: {result['auc']:.4f}")
        logger.info(f"  Decision Boundary: {result['decision_boundary']:.4f}")
        logger.info("")
    
    return results


def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate linear probes from two iterations on merged test dataset"
    )
    parser.add_argument(
        "--merged_csv",
        type=str,
        required=True,
        help="Path to the merged test CSV"
    )
    parser.add_argument(
        "--iteration1_lr_path",
        type=str,
        required=True,
        help="Path to LR model from iteration 1"
    )
    parser.add_argument(
        "--iteration2_lr_path",
        type=str,
        required=True,
        help="Path to LR model from iteration 2"
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
        "--output_csv",
        type=str,
        default=None,
        help="Path to save evaluation results"
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
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.merged_csv):
        raise FileNotFoundError(f"Merged CSV not found: {args.merged_csv}")
    if not os.path.exists(args.iteration1_lr_path):
        raise FileNotFoundError(f"Iteration 1 LR model not found: {args.iteration1_lr_path}")
    if not os.path.exists(args.iteration2_lr_path):
        raise FileNotFoundError(f"Iteration 2 LR model not found: {args.iteration2_lr_path}")
    
    # Create output directory if needed
    if args.output_csv:
        output_dir = Path(args.output_csv).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    results = evaluate_merged_probes(
        merged_csv=args.merged_csv,
        iteration1_lr_path=args.iteration1_lr_path,
        iteration2_lr_path=args.iteration2_lr_path,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        layer=args.layer,
        max_length=args.max_length,
        do_sae=args.do_sae,
        sae_path=args.sae_path,
        sae_words_path=args.sae_words_path,
        sae_descriptions_path=args.sae_descriptions_path,
        all_positions=args.all_positions,
    )
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main() 