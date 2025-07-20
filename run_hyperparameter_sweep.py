#!/usr/bin/env python3
"""
Hyperparameter sweep script for SOLiD deception training.

This script defines multiple hyperparameter configurations and runs the 
run_iterative.sh script for each configuration by setting the required
environment variables.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Any


SUBSAMPLE_DATASET = True
DEBUG_TRAINING = False

def run_experiment(config: Dict[str, Any], config_name: str) -> bool:
    """
    Run a single experiment with the given hyperparameter configuration.
    
    Args:
        config: Dictionary containing hyperparameter values
        config_name: Name of the configuration for logging
    
    Returns:
        bool: True if experiment completed successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Starting experiment: {config_name}")
    print(f"Configuration: {config}")
    print(f"{'='*60}")
    
    # Set environment variables for this configuration
    env = os.environ.copy()
    for key, value in config.items():
        env[key] = str(value)
    
    # Add timestamp to config name for unique identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_id = f"{timestamp}_{config_name}"
    
    print(f"Experiment ID: {config_id}")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run the iterative training script
        result = subprocess.run(
            ["./run_iterative.sh"],
            env=env,
            cwd=os.getcwd(),
            capture_output=False,  # Let output go to terminal
            text=True,
            check=True
        )
        
        print(f"\n✅ Experiment {config_name} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment {config_name} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment {config_name} interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Experiment {config_name} failed with error: {e}")
        return False


def main():
    """Main function to run hyperparameter sweep."""
    
    # Define hyperparameter configurations
    # Each configuration is a dictionary with the 5 key hyperparameters
    hyperparameter_configs: List[Dict[str, Any]] = [
        {
            "name": "baseline_1iter",
            "config": {
                "SEED": 42,
                "LIE_TPR": 0.8,
                "DEBUG_TRAINING": DEBUG_TRAINING,
                "SUBSAMPLE_DATASET": SUBSAMPLE_DATASET,
                "NUM_ITERATIONS": 1
            }
        },
        {
            "name": "baseline_2iter",
            "config": {
                "SEED": 42,
                "LIE_TPR": 0.8,
                "DEBUG_TRAINING": DEBUG_TRAINING,
                "SUBSAMPLE_DATASET": SUBSAMPLE_DATASET,
                "NUM_ITERATIONS": 2
            }
        },

    ]
    
    # Check if run_iterative.sh exists and is executable
    if not os.path.exists("./run_iterative.sh"):
        print("❌ Error: run_iterative.sh not found in current directory")
        sys.exit(1)
    
    if not os.access("./run_iterative.sh", os.X_OK):
        print("❌ Error: run_iterative.sh is not executable")
        print("Run: chmod +x run_iterative.sh")
        sys.exit(1)
    
    # Print summary of configurations
    print(f"🚀 Starting hyperparameter sweep with {len(hyperparameter_configs)} configurations")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nConfigurations to run:")
    for i, config_info in enumerate(hyperparameter_configs, 1):
        print(f"  {i}. {config_info['name']}: {config_info['config']}")
    
    # Auto-proceed with all configurations
    print("\nProceeding with all configurations automatically...")
    
    # Track results
    successful_runs = []
    failed_runs = []
    start_time = time.time()
    
    # Run each configuration
    for i, config_info in enumerate(hyperparameter_configs, 1):
        print(f"\n📊 Progress: {i}/{len(hyperparameter_configs)}")
        
        success = run_experiment(config_info['config'], config_info['name'])
        
        if success:
            successful_runs.append(config_info['name'])
        else:
            failed_runs.append(config_info['name'])
        
        # Optional: Add delay between experiments
        if i < len(hyperparameter_configs):
            print("Waiting 5 seconds before next experiment...")
            time.sleep(5)
    
    # Print final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎯 HYPERPARAMETER SWEEP COMPLETED")
    print(f"{'='*60}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Successful runs: {len(successful_runs)}/{len(hyperparameter_configs)}")
    print(f"Failed runs: {len(failed_runs)}/{len(hyperparameter_configs)}")
    
    if successful_runs:
        print(f"\n✅ Successful configurations:")
        for name in successful_runs:
            print(f"  - {name}")
    
    if failed_runs:
        print(f"\n❌ Failed configurations:")
        for name in failed_runs:
            print(f"  - {name}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main() 