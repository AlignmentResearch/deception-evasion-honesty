#!/bin/bash
set -e
set -o pipefail

# Source the setup script
source ./configs/setup.sh

cd /workspace/deception-evasion-honesty
export P=/workspace/deception-evasion-honesty
echo "Successfully setup!"
export PATH="/home/dev/.local/bin:$PATH"

# Create a master log file for the hyperparameter sweep
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export MASTER_TAG="hyperparameter_sweep_$TIMESTAMP"
export MASTER_LOG="$P/outputs/${MASTER_TAG}_master.log"

mkdir -p "$P/outputs" || true

# Define the hyperparameter configurations
declare -a TPR_VALUES=(0.8 0.99)
declare -a SEEDS=(0 1 2 3 4)

# Function to run a single configuration
run_configuration() {
    local tpr=$1
    local seed=$2
    
    echo "==========================================" | tee -a $MASTER_LOG
    echo "Starting configuration: TPR=$tpr, SEED=$seed" | tee -a $MASTER_LOG
    echo "Time: $(date)" | tee -a $MASTER_LOG
    echo "==========================================" | tee -a $MASTER_LOG
    
    # Create a temporary script with the specific configuration
    local temp_script="/tmp/run_iterative_${tpr}_${seed}.sh"
    
    # Copy the original script
    cp run_iterative.sh $temp_script
    
    # Modify the TPR and SEED values in the temporary script
    sed -i "s/export LIE_TPR=0.9/export LIE_TPR=$tpr/" $temp_script
    sed -i "s/export SEED=0/export SEED=$seed/" $temp_script
    
    # Make the temporary script executable
    chmod +x $temp_script
    
    # Run the configuration
    echo "Executing: $temp_script" | tee -a $MASTER_LOG
    if $temp_script; then
        echo "SUCCESS: Configuration TPR=$tpr, SEED=$seed completed successfully" | tee -a $MASTER_LOG
    else
        echo "FAILED: Configuration TPR=$tpr, SEED=$seed failed" | tee -a $MASTER_LOG
        # Continue with other configurations even if one fails
    fi
    
    # Clean up temporary script
    rm -f $temp_script
    
    echo "==========================================" | tee -a $MASTER_LOG
    echo "Completed configuration: TPR=$tpr, SEED=$seed" | tee -a $MASTER_LOG
    echo "Time: $(date)" | tee -a $MASTER_LOG
    echo "==========================================" | tee -a $MASTER_LOG
    echo "" | tee -a $MASTER_LOG
}

# Main execution loop
echo "Starting Hyperparameter Sweep" | tee -a $MASTER_LOG
echo "TPR values: ${TPR_VALUES[@]}" | tee -a $MASTER_LOG
echo "Seeds: ${SEEDS[@]}" | tee -a $MASTER_LOG
echo "Total configurations: $(( ${#TPR_VALUES[@]} * ${#SEEDS[@]} ))" | tee -a $MASTER_LOG
echo "Start time: $(date)" | tee -a $MASTER_LOG
echo "" | tee -a $MASTER_LOG

# Run configurations in the specified order: TPR first, then seeds
for seed in "${SEEDS[@]}"; do
    for tpr in "${TPR_VALUES[@]}"; do
        run_configuration $tpr $seed
    done
done

echo "==========================================" | tee -a $MASTER_LOG
echo "Hyperparameter Sweep Completed!" | tee -a $MASTER_LOG
echo "End time: $(date)" | tee -a $MASTER_LOG
echo "Master log: $MASTER_LOG" | tee -a $MASTER_LOG
echo "==========================================" | tee -a $MASTER_LOG

echo "DONE!" 