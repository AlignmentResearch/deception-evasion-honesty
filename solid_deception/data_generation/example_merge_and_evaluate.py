#!/usr/bin/env python3
"""
Example usage of merge_and_evaluate_probes.py for SOLiD iterative training.

This script shows how to merge test datasets and evaluate linear probes from two
iterations of SOLiD training using the typical directory structure created by run_iterative.sh.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_merge_and_evaluate_example(
    experiment_dir: str,
    model_path: str = "meta-llama/Llama-3.2-1B-Instruct",
    output_dir: str = None,
    batch_size: int = 8,
    layer: int = 16,
    max_length: int = 512,
    do_sae: bool = False,
    sae_path: str = None,
    sae_words_path: str = None,
    sae_descriptions_path: str = None,
    all_positions: bool = False,
    create_rewarded: bool = False,
    rewards: list = [-1, 2, 1, 1]
):
    """
    Example function to merge test datasets and evaluate probes from a SOLiD experiment.
    
    Args:
        experiment_dir: Path to the experiment directory (e.g., outputs/20241201_143022)
        model_path: Path to the model for feature extraction
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
        rewards: List of rewards [deceptive_false, deceptive_true, truthful_false, truthful_true]
    """
    
    # Set up paths
    experiment_path = Path(experiment_dir)
    if output_dir is None:
        output_dir = experiment_path / "merged_evaluation"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if required files exist
    required_files = [
        experiment_path / "iteration_1" / "detected.csv",
        experiment_path / "iteration_2" / "detected.csv",
        experiment_path / "iteration_1" / "lr.pkl",
        experiment_path / "iteration_2" / "lr.pkl",
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   {f}")
        return False
    
    # Build command
    cmd = [
        sys.executable,
        "solid_deception/data_generation/merge_and_evaluate_probes.py",
        "--experiment_dir", str(experiment_path),
        "--model_path", model_path,
        "--tokenizer_path", model_path,  # Usually same as model_path
        "--output_dir", str(output_dir),
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
    
    if create_rewarded:
        cmd.append("--create_rewarded")
        cmd.extend(["--rewards"] + [str(r) for r in rewards])
    
    # Run the combined workflow
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Merge and evaluate completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Merged test CSV: {output_dir / 'merged_test.csv'}")
        print(f"📄 Evaluation results: {output_dir / 'probe_evaluation_results.csv'}")
        if create_rewarded:
            print(f"📄 Rewarded CSV: {output_dir / 'merged_test.rewarded.csv'}")
        
        # Print the evaluation results
        if result.stdout:
            print("\n" + "="*50)
            print("EVALUATION RESULTS:")
            print("="*50)
            print(result.stdout)
        
        return True
    else:
        print("❌ Merge and evaluate failed!")
        print(f"Error: {result.stderr}")
        return False


def main():
    """Main function with example usage."""
    
    print("=== SOLiD Merge and Evaluate Probes Example ===")
    print()
    
    # Example 1: Basic usage with specific experiment
    print("=== Example 1: Basic usage ===")
    
    # Replace with your actual experiment directory
    experiment_dir = "outputs/20241201_143022"  # Example path
    
    if os.path.exists(experiment_dir):
        success = run_merge_and_evaluate_example(
            experiment_dir=experiment_dir,
            model_path="meta-llama/Llama-3.2-1B-Instruct",
            batch_size=8,
            layer=16,
        )
        if success:
            print("✅ Example 1 completed successfully!")
        else:
            print("❌ Example 1 failed!")
    else:
        print(f"⚠️  Experiment directory not found: {experiment_dir}")
        print("   Please update the experiment_dir path in this script.")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Command line usage
    print("=== Example 2: Command line usage ===")
    print("You can also run the script directly:")
    print()
    print("python solid_deception/data_generation/merge_and_evaluate_probes.py \\")
    print("    --experiment_dir outputs/20241201_143022 \\")
    print("    --model_path meta-llama/Llama-3.2-1B-Instruct \\")
    print("    --tokenizer_path meta-llama/Llama-3.2-1B-Instruct \\")
    print("    --batch_size 8 \\")
    print("    --layer 16 \\")
    print("    --max_length 512")
    print()
    
    # Example 3: With SAE features
    print("=== Example 3: With SAE features ===")
    print("If you want to use SAE features instead of raw activations:")
    print()
    print("python solid_deception/data_generation/merge_and_evaluate_probes.py \\")
    print("    --experiment_dir outputs/20241201_143022 \\")
    print("    --model_path meta-llama/Llama-3.2-1B-Instruct \\")
    print("    --tokenizer_path meta-llama/Llama-3.2-1B-Instruct \\")
    print("    --do_sae \\")
    print("    --sae_path saes/layer_23 \\")
    print("    --sae_words_path solid_deception/detection/sae_words.txt \\")
    print("    --sae_descriptions_path solid_deception/detection/model.layers.23_feature.json \\")
    print("    --layer 23")
    print()
    
    # Example 4: Find latest experiment
    print("=== Example 4: Find and process latest experiment ===")
    
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        # Find the most recent experiment directory
        experiment_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("20")]
        if experiment_dirs:
            latest_experiment = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
            print(f"Latest experiment: {latest_experiment}")
            
            # Check if it has both iterations
            iteration1_csv = latest_experiment / "iteration_1" / "detected.csv"
            iteration2_csv = latest_experiment / "iteration_2" / "detected.csv"
            iteration1_lr = latest_experiment / "iteration_1" / "lr.pkl"
            iteration2_lr = latest_experiment / "iteration_2" / "lr.pkl"
            
            if all(f.exists() for f in [iteration1_csv, iteration2_csv, iteration1_lr, iteration2_lr]):
                print("✅ Both iterations found, processing...")
                success = run_merge_and_evaluate_example(
                    experiment_dir=str(latest_experiment),
                    model_path="meta-llama/Llama-3.2-1B-Instruct",
                )
                if success:
                    print("✅ Latest experiment processed successfully!")
                else:
                    print("❌ Latest experiment processing failed!")
            else:
                print("⚠️  Latest experiment doesn't have all required files")
                print(f"   Iteration 1 CSV: {iteration1_csv.exists()}")
                print(f"   Iteration 2 CSV: {iteration2_csv.exists()}")
                print(f"   Iteration 1 LR: {iteration1_lr.exists()}")
                print(f"   Iteration 2 LR: {iteration2_lr.exists()}")
        else:
            print("⚠️  No experiment directories found in outputs/")
    else:
        print("⚠️  outputs/ directory not found")
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("This script will:")
    print("1. Merge test datasets from iteration 1 and 2")
    print("2. Extract features from the merged test dataset")
    print("3. Evaluate both linear probes on the merged dataset")
    print("4. Report true positive rates, false positive rates, and AUC scores")
    print("5. Save results to CSV files")
    print()
    print("The output will show you how well each probe performs on the")
    print("combined test dataset, allowing you to compare their effectiveness.")


if __name__ == "__main__":
    main() 