#!/bin/bash
set -e
set -o pipefail

cd /workspace/deception-evasion-honesty
export P=/workspace/deception-evasion-honesty

# Activate the virtual environment first
source venv/bin/activate

# Source the setup script
source ./configs/setup.sh

echo "Successfully setup!"
export PATH="/home/dev/.local/bin:$PATH"

# Configuration details
export TPR=0.6
export SEED=0
export TAG="20250719_102507"

# Allow iteration number to be passed as an argument, default to 2 if not provided
if [ -z "$1" ]; then
    export ITERATION=2
else
    export ITERATION="$1"
fi

# Set up paths for the existing experiment
export EXPERIMENT_SET_DIRECTORY="$P/outputs/$TAG"
export ITERATION_DIR="$EXPERIMENT_SET_DIRECTORY/iteration_${ITERATION}"
export ITERATION_LOGFILE="$ITERATION_DIR/stdout_err.log"

# Set up file locations (using existing files)
export MUNGED_DATA_PATH="$ITERATION_DIR/munged_data.csv"
export DETECTED_PATH=$ITERATION_DIR/detected.csv
export LR_PATH=$ITERATION_DIR/lr.pkl
export DATASET_PATH=$ITERATION_DIR/rewarded
export CSV_PATH=$ITERATION_DIR/rewarded_csv.csv
export RM_DIR=$ITERATION_DIR/rm
export SFT_DIR=$ITERATION_DIR/sft
export POLICY_DIR=$ITERATION_DIR/policy
export EVAL_OUT_DIR=$ITERATION_DIR/eval

# Model paths: Set BASE_MODEL_PATH and BASE_POLICY_PATH according to iteration
# See run_iterative.sh: for iteration 1, BASE_POLICY_PATH is the base model; for iteration >1, it's previous iteration's policy_adapter
if [ "$ITERATION" -eq 1 ]; then
    export BASE_MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
    export BASE_POLICY_PATH="meta-llama/Llama-3.2-1B-Instruct"
else
    export BASE_MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
    export BASE_POLICY_PATH="$P/outputs/$TAG/iteration_$((ITERATION-1))/policy_adapter"
fi

# Set up SFT path for evaluation
export EVAL_SFT_PATH="${SFT_DIR}_adapter"
export RM_OUTPUT_DIR="None"  # Since we're using DPO

# Training parameters (from the original run)
export LAYER=16
export DEBUG_TRAINING=false
export DO_SAE=false
export DO_DPO=true
export DO_BT_RM=true
export DO_CATEGORICAL_RM=false
export ADAPTIVE=false
export RESTART_GRPO=false
export ALL_POSITIONS=false
export SAMPLE_LABELS=false
export DETECTOR_PDTBS=32
export MAX_DETECTOR_SEQ_LENGTH=544

# SAE settings
export SAE_PATH="$P/saes/layer_23"
export SAE_DESCRIPTIONS_PATH="$P/solid_deception/detection/model.layers.23_feature.json"
export SAE_WORDS_PATH="$P/solid_deception/detection/sae_words.txt"
export NULL_ANSWER_PATH="$P/data/null_answers.txt"

# Set up flags
if $DEBUG_TRAINING; then
    export DEBUG_TRAINING_FLAG="--debug_training"
else
    export DEBUG_TRAINING_FLAG=""
fi

if $ALL_POSITIONS; then
    export ALL_POSITIONS_FLAG='--all_positions'
else
    export ALL_POSITIONS_FLAG=''
fi

if $DO_SAE; then
    export SAE_FLAG="--do_sae"
    export LAYER=23
else
    export SAE_FLAG=""
fi

if $DO_CATEGORICAL_RM; then
    export CATEGORICAL_GRPO_LABELS_FLAG="--do_categorical_labels"
else
    export CATEGORICAL_GRPO_LABELS_FLAG=""
fi

echo "=========================================="
echo "Running EVALUATION ONLY for TPR=$TPR, SEED=$SEED, Iteration $ITERATION"
echo "Base policy: $BASE_POLICY_PATH"
echo "Base model: $BASE_MODEL_PATH"
echo "Time: $(date)"
echo "=========================================="

# Check if evaluation directory exists, if not create it
if [ ! -d "$EVAL_OUT_DIR" ]; then
    mkdir -p "$EVAL_OUT_DIR"
    echo "Created evaluation directory: $EVAL_OUT_DIR"
fi

# Run the evaluation
echo "STARTING EVAL for iteration $ITERATION at $(date)" | tee -a $ITERATION_LOGFILE
CUDA_VISIBLE_DEVICES=0 python $P/solid_deception/eval/reward.py \
    --model_path "${POLICY_DIR}_adapter" \
    --reward_model_path "$RM_OUTPUT_DIR" \
    --tokenizer_path $BASE_MODEL_PATH \
    --dataset_path $CSV_PATH \
    --original_model_path $BASE_MODEL_PATH \
    --lr_path $LR_PATH \
    --layer $LAYER \
    --output_dir $EVAL_OUT_DIR \
    --n_rows 20 \
    $DEBUG_TRAINING_FLAG \
    --experiment_set_name $TAG \
    --run_name "eval_iter${ITERATION}_$TAG" \
    --sae_path $SAE_PATH \
    --sae_words_path $SAE_WORDS_PATH \
    --sae_descriptions_path $SAE_DESCRIPTIONS_PATH \
    $SAE_FLAG \
    --null_example_reward -5.0 \
    $CATEGORICAL_GRPO_LABELS_FLAG \
    --seed $SEED \
    $ALL_POSITIONS_FLAG \
    --sft_model_path $EVAL_SFT_PATH \
    2>&1 | tee -a $ITERATION_LOGFILE

echo "FINISHED EVAL for iteration $ITERATION at $(date)" | tee -a $ITERATION_LOGFILE

echo "=========================================="
echo "Evaluation completed successfully!"
echo "End time: $(date)"
echo "Evaluation results saved to: $EVAL_OUT_DIR"
echo "=========================================="

echo "DONE!" 