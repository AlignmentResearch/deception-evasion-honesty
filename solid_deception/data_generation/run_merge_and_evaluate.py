#!/usr/bin/env python3
"""
Example usage of merge_and_evaluate_probes.py
"""

import subprocess

def main():
    # Example command to run the merge and evaluate script
    base_path = "/workspace/deception-evasion-honesty/outputs/20250719_160214"
    cmd = [
        "python", "solid_deception/data_generation/merge_and_evaluate_probes.py",
        "--csv1", f"{base_path}/iteration_1/munged_data.csv",
        "--csv2", f"{base_path}/iteration_2/munged_data.csv", 
        "--lr1", f"{base_path}/iteration_1/lr.pkl",
        "--lr2", f"{base_path}/iteration_2/lr.pkl",
        "--model_path", "meta-llama/Llama-3.2-1B-Instruct",
        "--tokenizer_path", "meta-llama/Llama-3.2-1B-Instruct",
        "--output_csv", f"{base_path}/merged_test_dataset.csv",
        "--results_csv", f"{base_path}/evaluation_results.csv"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print("\nThis will:")
    print("1. Merge the two test datasets")
    print("2. Calculate TPR, FPR, and AUC for both probes")
    print("3. Create a 2x2 confusion matrix showing probe overlap")
    print("4. Save results to CSV files")
    print("\n" + "="*60)
    
    # Actually run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("ERROR:")
        print(e.stderr)
        print(e.stdout)

if __name__ == "__main__":
    main() 