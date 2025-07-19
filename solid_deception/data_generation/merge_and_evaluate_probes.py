#!/usr/bin/env python3
"""
Single script to merge test datasets and evaluate linear probes with confusion matrix.
"""

import argparse
import logging
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_test_datasets(csv1: str, csv2: str, output_csv: str) -> str:
    """Merge two test datasets and remove duplicates."""
    logger.info(f"Merging test datasets from {csv1} and {csv2}")
    
    # Load datasets
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    
    # Filter to test examples
    test_df1 = df1[df1["split"] == "test"].copy()
    test_df2 = df2[df2["split"] == "test"].copy()
    
    logger.info(f"Found {len(test_df1)} test examples in first dataset")
    logger.info(f"Found {len(test_df2)} test examples in second dataset")
    
    # Combine datasets
    merged_df = pd.concat([test_df1, test_df2], ignore_index=True)
    
    # Remove duplicates based on prompt + truthful_response + deceptive_response
    before_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["prompt", "truthful_response", "deceptive_response"])
    after_dedup = len(merged_df)
    
    logger.info(f"Removed {before_dedup - after_dedup} duplicate examples")
    logger.info(f"Final merged dataset has {len(merged_df)} unique test examples")
    
    # Mark as merged test dataset
    merged_df["split"] = "test_merged"
    
    # Save merged dataset
    merged_df.to_csv(output_csv, index=False)
    logger.info(f"Saved merged dataset to {output_csv}")
    
    return output_csv

def load_lr_model(lr_path: str) -> Tuple[StandardScaler, LogisticRegression, float]:
    """Load a trained logistic regression model."""
    with open(lr_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # The LR model is stored as a tuple: (scaler, lr, decision_boundary, lr_learn_stats)
    scaler = model_data[0]
    lr_model = model_data[1]
    decision_boundary = model_data[2]
    
    return scaler, lr_model, decision_boundary

def extract_features(df: pd.DataFrame, model, tokenizer, layer: int = 16, max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from the model for true and false responses."""
    logger.info("Extracting features from model...")
    
    device = next(model.parameters()).device
    model.eval()
    
    true_activations = []
    false_activations = []
    
    with torch.no_grad():
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            # Tokenize true response
            true_text = row["prompt"] + row["truthful_response"]
            true_inputs = tokenizer(true_text, return_tensors="pt", truncation=True, max_length=max_length)
            true_inputs = {k: v.to(device) for k, v in true_inputs.items()}
            
            # Get activations for true response
            true_outputs = model(**true_inputs, output_hidden_states=True)
            true_activation = true_outputs.hidden_states[layer][0, -1, :].cpu()  # Last token of layer
            true_activations.append(true_activation)
            
            # Tokenize false response
            false_text = row["prompt"] + row["deceptive_response"]
            false_inputs = tokenizer(false_text, return_tensors="pt", truncation=True, max_length=max_length)
            false_inputs = {k: v.to(device) for k, v in false_inputs.items()}
            
            # Get activations for false response
            false_outputs = model(**false_inputs, output_hidden_states=True)
            false_activation = false_outputs.hidden_states[layer][0, -1, :].cpu()  # Last token of layer
            false_activations.append(false_activation)
    
    return torch.stack(true_activations), torch.stack(false_activations)

def evaluate_probe(true_activations: torch.Tensor, false_activations: torch.Tensor, 
                  scaler: StandardScaler, lr_model: LogisticRegression, 
                  decision_boundary: float, probe_name: str) -> Dict[str, float]:
    """Evaluate a single probe and return metrics."""
    # Combine activations
    all_activations = torch.cat([true_activations, false_activations], dim=0)
    all_labels = np.concatenate([np.zeros(len(true_activations)), np.ones(len(false_activations))])
    
    # Scale features
    scaled_features = scaler.transform(all_activations.numpy())
    
    # Get predictions
    predictions = lr_model.predict_proba(scaled_features)[:, 1]
    
    # Calculate metrics
    tpr = np.mean(predictions[all_labels == 1] > decision_boundary)
    fpr = np.mean(predictions[all_labels == 0] > decision_boundary)
    auc = roc_auc_score(all_labels, predictions)
    
    return {
        'probe_name': probe_name,
        'tpr': tpr,
        'fpr': fpr,
        'auc': auc,
        'decision_boundary': decision_boundary
    }

def calculate_confusion_matrix(true_activations: torch.Tensor, false_activations: torch.Tensor,
                             scaler1: StandardScaler, lr1: LogisticRegression, db1: float,
                             scaler2: StandardScaler, lr2: LogisticRegression, db2: float) -> np.ndarray:
    """Calculate 2x2 confusion matrix showing overlap between probe detections."""
    logger.info("Calculating confusion matrix...")
    
    # Get predictions from both probes on false activations (lies)
    scaled_features = scaler1.transform(false_activations.numpy())
    probe1_predictions = lr1.predict_proba(scaled_features)[:, 1] > db1
    
    scaled_features = scaler2.transform(false_activations.numpy())
    probe2_predictions = lr2.predict_proba(scaled_features)[:, 1] > db2
    
    # Create confusion matrix
    # Rows: probe2 detected/undetected
    # Columns: probe1 detected/undetected
    confusion_matrix = np.zeros((2, 2), dtype=int)
    
    # Both detected
    confusion_matrix[0, 0] = np.sum(probe1_predictions & probe2_predictions)
    # Probe1 detected, probe2 undetected
    confusion_matrix[0, 1] = np.sum(probe1_predictions & ~probe2_predictions)
    # Probe1 undetected, probe2 detected
    confusion_matrix[1, 0] = np.sum(~probe1_predictions & probe2_predictions)
    # Both undetected
    confusion_matrix[1, 1] = np.sum(~probe1_predictions & ~probe2_predictions)
    
    return confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Merge test datasets and evaluate probes with confusion matrix")
    parser.add_argument("--csv1", type=str, required=True, help="Path to first test CSV")
    parser.add_argument("--csv2", type=str, required=True, help="Path to second test CSV")
    parser.add_argument("--lr1", type=str, required=True, help="Path to LR model from iteration 1")
    parser.add_argument("--lr2", type=str, required=True, help="Path to LR model from iteration 2")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model for feature extraction")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--output_csv", type=str, default="merged_test_dataset.csv", help="Output path for merged dataset")
    parser.add_argument("--results_csv", type=str, default="evaluation_results.csv", help="Output path for evaluation results")
    parser.add_argument("--layer", type=int, default=16, help="Layer to extract features from")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    try:
        # Step 1: Merge test datasets
        merged_csv = merge_test_datasets(args.csv1, args.csv2, args.output_csv)
        
        # Step 2: Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        
        # Step 3: Load merged dataset
        df = pd.read_csv(merged_csv)
        test_df = df[df["split"] == "test_merged"].copy()
        logger.info(f"Evaluating on {len(test_df)} test examples")
        
        # Step 4: Extract features
        true_activations, false_activations = extract_features(test_df, model, tokenizer, args.layer, args.max_length)
        
        # Step 5: Load LR models
        logger.info("Loading LR models...")
        scaler1, lr1, db1 = load_lr_model(args.lr1)
        scaler2, lr2, db2 = load_lr_model(args.lr2)
        
        # Step 6: Evaluate both probes
        logger.info("Evaluating probes...")
        results = []
        
        results.append(evaluate_probe(true_activations, false_activations, scaler1, lr1, db1, "Iteration 1 Probe"))
        results.append(evaluate_probe(true_activations, false_activations, scaler2, lr2, db2, "Iteration 2 Probe"))
        
        # Step 7: Calculate confusion matrix
        confusion_matrix = calculate_confusion_matrix(
            true_activations, false_activations, scaler1, lr1, db1, scaler2, lr2, db2
        )
        
        # Step 8: Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        for result in results:
            print(f"\n{result['probe_name']}:")
            print(f"  True Positive Rate (TPR): {result['tpr']:.4f}")
            print(f"  False Positive Rate (FPR): {result['fpr']:.4f}")
            print(f"  Area Under Curve (AUC): {result['auc']:.4f}")
            print(f"  Decision Boundary: {result['decision_boundary']:.4f}")
        
        print(f"\n" + "="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        print("Rows: Probe2 detected/undetected")
        print("Columns: Probe1 detected/undetected")
        print(f"\n{confusion_matrix}")
        
        # Save confusion matrix
        confusion_df = pd.DataFrame(
            confusion_matrix,
            index=['Probe2 Detected', 'Probe2 Undetected'],
            columns=['Probe1 Detected', 'Probe1 Undetected']
        )
        confusion_df.to_csv('confusion_matrix.csv', index=True)
        print(f"\nConfusion matrix saved to confusion_matrix.csv")
        
        # Save evaluation results
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.results_csv, index=False)
        logger.info(f"Evaluation results saved to {args.results_csv}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main() 