#!/bin/bash
set -e
set -o pipefail

source ./configs/setup.sh

cd /workspace/deception-evasion-honesty
export P=/workspace/deception-evasion-honesty
echo "Successfully setup!"
export PATH="/home/dev/.local/bin:$PATH"
export MASTER_PORT=$(echo '12'$(shuf -i 100-999 -n 1))
echo $MASTER_PORT
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TAG="$TIMESTAMP"

export EXPERIMENT_SET_DIRECTORY="$P/outputs/$TAG"
mkdir "$P/outputs" || true
mkdir $EXPERIMENT_SET_DIRECTORY || true

# Iterative training parameters
export NUM_ITERATIONS=2
export H1_FRAC=0.5
export BASE_POLICY_PATH="meta-llama/Llama-3.2-1B-Instruct"  # Start with original model

# Debug mode - set to true to use only 5% of data for fast iteration
export SUBSAMPLE_DATASET=false

# Setting up file locations (organizational)
export LOGFILE="$EXPERIMENT_SET_DIRECTORY/iterative_stdout_err.log"
export WANDB_PROJECT='solid_deception_iterative'

# Global Settings
export DEBUG_TRAINING=false
export DO_SAE=false
export DO_DPO=true
export DO_BT_RM=true # Bradley-Terry reward model
export DO_CATEGORICAL_RM=false
export ADAPTIVE=false
export RESTART_GRPO=false
export BASE_PDTBS=64 # Per device batch size for 8b

# Model
export GENERATION_LORA_PATH=None
export BASE_MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct

# Training
export ACONFIG=$P/configs/1_gpu_ddp.yaml

# Data hyperparams
export RAW_DATA_PATH='AlignmentResearch/DolusChat'
export TEST_FRAC=0.05
export TRAIN_LR_FRAC=0.05
export REWARD_SYSTEM_PROMPT="$P/solid_deception/training/gpt4_reward_prompt.txt"
export LAYER=16
export TRAIN_DATA_LIMIT=None
export LIE_FPR=None
export LIE_TPR=0.9 
export SAE_PATH="$P/saes/layer_23"
export SAE_DESCRIPTIONS_PATH="$P/solid_deception/detection/model.layers.23_feature.json"
export SAE_WORDS_PATH="$P/solid_deception/detection/sae_words.txt"
export NULL_ANSWER_PATH="$P/data/null_answers.txt"
export ALL_POSITIONS=false
export SAMPLE_LABELS=false
export SEED=0
export DETECTOR_PDTBS=$((BASE_PDTBS / 2))

# RM
export RM_LOGICAL_BATCH_SIZE=256
export RM_NUM_EPOCHS=4
export RM_LORA_R=16
export RM_LR=5e-6
export RM_PDTBS=$((BASE_PDTBS * 2))

# SFT
export SFT_LOGICAL_BATCH_SIZE=128
export SFT_PDTBS=$((BASE_PDTBS))
export SFT_LR=1e-5

# GRPO
export GRPO_LOGICAL_BATCH_SIZE=512
export GRPO_PDTBS=$((BASE_PDTBS / 2))
export POLICY_LORA_R=16
export GRPO_LRFBS=24
export GRPO_EVAL_STEPS=100
export GRPO_LR=5e-6
export GRPO_TOTAL_EPS=150000
export GRPO_KL_COEF=0.1
export USE_GRPO=false

if $USE_GRPO; then
    export GRPO_K=8
    export GRPO_FLAG='--use_grpo_advantages True'
else
    export GRPO_K=2
    export GRPO_FLAG=''
fi

# DPO
export DPO_LOGICAL_BATCH_SIZE=256
export DPO_LR=1e-5
export DPO_PDTBS=$((BASE_PDTBS/8)) # chosen + rejected per example
export DPO_KL_COEF=0.1


# ----------------------------------------

if $DEBUG_TRAINING; then
    export DEBUG_TRAINING_FLAG="--debug_training"
    export GRPO_LOGICAL_BATCH_SIZE=64
else
    export DEBUG_TRAINING_FLAG=""
fi

if $ALL_POSITIONS; then
    export ALL_POSITIONS_FLAG='--all_positions'
else
    export ALL_POSITIONS_FLAG=''
fi

export MAX_DETECTOR_SEQ_LENGTH=544

if $DO_DPO; then
    export RM_OUTPUT_DIR="None"
else
    export RM_OUTPUT_DIR="${RM_DIR}_adapter"
fi

if $DO_SAE; then
    export SAE_FLAG="--do_sae"
    export LAYER=23
else
    export SAE_FLAG=""
fi

if $DO_CATEGORICAL_RM; then
    export CATEGORICAL_RM_LABELS="True"
    export CATEGORICAL_GRPO_LABELS="True"
    export CATEGORICAL_GRPO_LABELS_FLAG="--do_categorical_labels"
else
    export CATEGORICAL_RM_LABELS="False"
    export CATEGORICAL_GRPO_LABELS="False"
    export CATEGORICAL_GRPO_LABELS_FLAG=""
fi

if $ADAPTIVE; then
    export ADAPTIVE_FLAG="--adaptive"
else
    export ADAPTIVE_FLAG=""
fi

# Check if TRAIN_DATA_LIMIT == None, if so set the flag to '' else to train_data_limit
if [[ "$TRAIN_DATA_LIMIT" == "None" ]]; then
    export TRAIN_DATA_LIMIT_FLAG=""
else
    export TRAIN_DATA_LIMIT_FLAG="--data_limit $TRAIN_DATA_LIMIT"
fi

if [[ "BASE_MODEL_PATH" == "meta-llama/Meta-Llama-3.1-8B-Instruct" ]]; then
    export SFT_PDTBS=$((2 * BASE_PDTBS))
    export RM_PDTBS=$((2 * BASE_PDTBS))
fi

if [ ! -f "$LOGFILE" ]; then
    touch "$LOGFILE"
else
    echo "Iterative Training Restarted at $(date)" >> "$LOGFILE"
fi

# Debug mode - just reduce data size to 5%
if $SUBSAMPLE_DATASET; then
    echo "DEBUG MODE ENABLED - Using 5% of data" >> $LOGFILE
    export DEBUG_FRAC=0.05
else
    export DEBUG_FRAC=""
fi

echo "<env>"
env >> $LOGFILE
echo `env`
echo "</env>"

echo "Starting Iterative SOLiD Training with $NUM_ITERATIONS iterations" >> $LOGFILE

# Iterative training loop
for iteration in $(seq 1 $NUM_ITERATIONS); do
    echo "=== Starting Iteration $iteration ===" >> $LOGFILE
    echo "Using base model: $BASE_POLICY_PATH" >> $LOGFILE
    
    # Set up iteration-specific directories
    export ITERATION_DIR="$EXPERIMENT_SET_DIRECTORY/iteration_$iteration"
    mkdir -p $ITERATION_DIR
    
    # Set up iteration-specific file locations
    export ITERATION_LOGFILE="$ITERATION_DIR/stdout_err.log"
    export MUNGED_DATA_PATH="$ITERATION_DIR/munged_data.csv"
    export DETECTED_PATH=$ITERATION_DIR/detected.csv
    export LR_PATH=$ITERATION_DIR/lr.pkl
    export DATASET_PATH=$ITERATION_DIR/rewarded
    export CSV_PATH=$ITERATION_DIR/rewarded_csv.csv
    export RM_DIR=$ITERATION_DIR/rm
    export SFT_DIR=$ITERATION_DIR/sft
    export POLICY_DIR=$ITERATION_DIR/policy
    export EVAL_OUT_DIR=$ITERATION_DIR/eval
    
    # Set up iteration-specific run names
    export DETECTOR_RUN_NAME="detector_iter${iteration}_$TAG"
    export RM_RUN_NAME="rm_explicit_iter${iteration}_$TAG"
    export RM_BT_RUN_NAME="rm_bt_iter${iteration}_$TAG"
    export DPO_RUN_NAME="dpo_iter${iteration}_$TAG"
    export SFT_RUN_NAME="sft_iter${iteration}_$TAG"
    export GRPO_RUN_NAME="grpo_iter${iteration}_$TAG"
    export EVAL_RUN_NAME="eval_iter${iteration}_$TAG"
    
    # Set the base model for this iteration
    export BASE_MODEL_PATH=$BASE_POLICY_PATH
    
    # Set up SFT path for evaluation
    export EVAL_SFT_PATH="${SFT_DIR}_adapter"
    if $DO_DPO; then
        export RM_OUTPUT_DIR="None"
    else
        export RM_OUTPUT_DIR="${RM_DIR}_adapter"
    fi
    
    if [ ! -f "$ITERATION_LOGFILE" ]; then
        touch "$ITERATION_LOGFILE"
    fi
    
    echo "Iteration $iteration started at $(date)" >> $ITERATION_LOGFILE
    
    # MUNGE DATA with iterative splitting
    if ! grep -q "MUNGED DATA at" $ITERATION_LOGFILE; then
        echo "STARTING MUNGE for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
        python $P/solid_deception/data_generation/munge_data.py \
            --input_path $RAW_DATA_PATH \
            -c $MUNGED_DATA_PATH \
            --test_frac $TEST_FRAC \
            --seed $SEED \
            --train_lr_frac $TRAIN_LR_FRAC \
            --iterative \
            --h1_frac $H1_FRAC \
            --iteration $iteration \
            ${DEBUG_FRAC:+--debug_frac $DEBUG_FRAC} \
            2>&1 | tee -a $ITERATION_LOGFILE
        echo "MUNGED DATA for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
    fi
    
    # DETECTOR
    if ! grep -q "FINISHED DETECTOR at" $ITERATION_LOGFILE; then
        echo "STARTING SAMPLING for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
        TOKENIZERS_PARALLELISM=false accelerate launch \
            --config_file $ACONFIG \
            --main_process_port $MASTER_PORT \
            $P/solid_deception/detection/lr_detector.py \
            --model_path $BASE_MODEL_PATH \
            --tokenizer_path $BASE_MODEL_PATH \
            --data_path $MUNGED_DATA_PATH \
            --csv_save_path $DETECTED_PATH \
            --lr_save_path $LR_PATH \
            --batch_size $DETECTOR_PDTBS \
            --layer $LAYER \
            $DEBUG_TRAINING_FLAG \
            --lie_false_positive_rate $LIE_FPR \
            --lie_true_positive_rate $LIE_TPR \
            $SAE_FLAG \
            --experiment_set_name $TAG \
            --name $DETECTOR_RUN_NAME \
            --sae_path $SAE_PATH \
            --sae_words_path $SAE_WORDS_PATH \
            --sae_descriptions_path $SAE_DESCRIPTIONS_PATH \
            $ADAPTIVE_FLAG \
            --seed $SEED \
            $TRAIN_DATA_LIMIT_FLAG \
            $ALL_POSITIONS_FLAG \
            --max_length $MAX_DETECTOR_SEQ_LENGTH \
            2>&1 | tee -a $ITERATION_LOGFILE
        echo "FINISHED DETECTOR for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
    fi
    
    # MAKE DATASET
    if ! grep -q "MADE DATASET at" $ITERATION_LOGFILE; then
        echo "STARTING DATASET for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
        python $P/solid_deception/data_generation/make_dataset.py \
            -i $DETECTED_PATH \
            -od $DATASET_PATH \
            -oc $CSV_PATH \
            --rewards -1 2 1 1 \
            2>&1 | tee -a $ITERATION_LOGFILE
        echo "MADE DATASET for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
    fi
    
    # TRAIN SFT
    if ! grep -q "TRAINED SFT at" $ITERATION_LOGFILE; then
        echo "STARTING SFT for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
        accelerate launch \
            --config_file $ACONFIG \
            --main_process_port $MASTER_PORT \
            $P/solid_deception/training/train_sft.py \
            --output_dir $SFT_DIR \
            --model_name_or_path $BASE_MODEL_PATH \
            --learning_rate $SFT_LR \
            --num_train_epochs 1.0 \
            --per_device_eval_batch_size 4 \
            --per_device_train_batch_size $SFT_PDTBS \
            --use_peft \
            --lora_r $POLICY_LORA_R \
            --dataset_name $DATASET_PATH \
            --bf16 \
            --run_name $SFT_RUN_NAME \
            $DEBUG_TRAINING_FLAG \
            --gradient_checkpointing True \
            --logical_batch_size $SFT_LOGICAL_BATCH_SIZE \
            --seed $SEED \
            --experiment_set_name $TAG \
            2>&1 | tee -a $ITERATION_LOGFILE
        echo "TRAINED SFT for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
    fi
    
    # TRAIN RM (if not DPO)
    if ! $DO_DPO; then
        if ! grep -q "TRAINED RM at" $ITERATION_LOGFILE; then
            echo "STARTING RM for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
            if $DO_BT_RM; then
                accelerate launch \
                    --config_file $ACONFIG \
                    --main_process_port $MASTER_PORT \
                    $P/solid_deception/training/train_reward.py \
                    --output_dir $RM_DIR \
                    --model_name_or_path $BASE_MODEL_PATH \
                    --dataset_name $DATASET_PATH \
                    --per_device_train_batch_size $RM_PDTBS \
                    --learning_rate $RM_LR \
                    --run_name $RM_BT_RUN_NAME \
                    --gradient_checkpointing True \
                    --per_device_eval_batch_size 4 \
                    --bf16 \
                    --lora_r $RM_LORA_R \
                    --use_peft \
                    --num_train_epochs $RM_NUM_EPOCHS \
                    $DEBUG_TRAINING_FLAG \
                    --logical_batch_size $RM_LOGICAL_BATCH_SIZE \
                    --experiment_set_name $TAG \
                    --seed $SEED \
                    --dataloader_num_workers 8 \
                    --null_answer_path $NULL_ANSWER_PATH \
                    2>&1 | tee -a $ITERATION_LOGFILE
            else
                accelerate launch \
                    --config_file $ACONFIG \
                    --main_process_port $MASTER_PORT \
                    $P/solid_deception/training/train_explicit_rm.py \
                    --output_dir $RM_DIR \
                    --model_name_or_path $BASE_MODEL_PATH \
                    --dataset_name $DATASET_PATH \
                    --per_device_train_batch_size $RM_PDTBS \
                    --learning_rate $RM_LR \
                    --run_name $RM_RUN_NAME \
                    --gradient_checkpointing True \
                    --per_device_eval_batch_size 4 \
                    --bf16 \
                    --lora_r $RM_LORA_R \
                    --use_peft \
                    --num_train_epochs $RM_NUM_EPOCHS \
                    $DEBUG_TRAINING_FLAG \
                    --logical_batch_size $RM_LOGICAL_BATCH_SIZE \
                    --experiment_set_name $TAG \
                    --do_categorical_labels $CATEGORICAL_RM_LABELS \
                    --null_example_reward -5.0 \
                    --dataloader_num_workers 8 \
                    --seed $SEED \
                    --null_answer_path $NULL_ANSWER_PATH \
                    2>&1 | tee -a $ITERATION_LOGFILE
            fi
            echo "TRAINED RM for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
        fi
    fi
    
    # TRAIN POLICY (GRPO or DPO)
    if ! $DO_DPO; then
        # TRAIN GRPO
        if ! grep -q "TRAINED GRPO at" $ITERATION_LOGFILE && [ "$RESTART_GRPO" = "false" ] ; then
            echo "STARTING GRPO for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
            WANDB_RUN_ID=$GRPO_RUN_NAME WANDB_RESUME=allow accelerate launch \
                --config_file $ACONFIG \
                --main_process_port $MASTER_PORT \
                $P/solid_deception/training/train_grpo.py \
                --reward_model_path "${RM_DIR}_adapter" \
                --sft_model_path "${SFT_DIR}_adapter" \
                --per_device_train_batch_size $GRPO_PDTBS \
                --local_rollout_forward_batch_size 24 \
                --eval_steps 100 \
                --per_device_eval_batch_size 2 \
                --run_name $GRPO_RUN_NAME \
                --eval_strategy steps \
                --output_dir $POLICY_DIR \
                --model_name_or_path $BASE_MODEL_PATH \
                --rloo_k $GRPO_K \
                --learning_rate $GRPO_LR \
                --gradient_checkpointing True \
                --missing_eos_penalty 1.0 \
                --total_episodes $GRPO_TOTAL_EPS \
                --kl_coef $DPO_KL_COEF \
                --dataloader_num_workers 8 \
                --dataset_name $DATASET_PATH \
                --use_triple_peft \
                --lora_r $POLICY_LORA_R \
                --bf16 \
                --max_grad_norm 1000 \
                --clip $DEBUG_TRAINING_FLAG \
                --logical_batch_size $GRPO_LOGICAL_BATCH_SIZE \
                --experiment_set_name $TAG \
                --no_naive_pg_gradient False \
                --do_categorical_labels $CATEGORICAL_GRPO_LABELS \
                $GRPO_FLAG \
                --seed $SEED \
                --null_example_reward -5.0 \
                2>&1 | tee -a $ITERATION_LOGFILE
            echo "TRAINED GRPO for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
        fi
    else
        # TRAIN DPO
        if ! grep -q "TRAINED DPO at" $ITERATION_LOGFILE; then
            echo "STARTING DPO for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
            accelerate launch \
                --config_file $ACONFIG \
                --main_process_port $MASTER_PORT \
                $P/solid_deception/training/train_dpo.py \
                --dataset_name $DATASET_PATH \
                --output_dir $POLICY_DIR \
                --model_name_or_path "${SFT_DIR}_adapter" \
                --per_device_train_batch_size $DPO_PDTBS \
                --eval_steps 400 \
                --label_smoothing_factor 0.05 \
                --per_device_eval_batch_size 2 \
                --run_name $DPO_RUN_NAME \
                --learning_rate $DPO_LR \
                --eval_strategy steps \
                --bf16 \
                --use_peft \
                --lora_r $POLICY_LORA_R \
                --logical_batch_size $DPO_LOGICAL_BATCH_SIZE \
                $DEBUG_TRAINING_FLAG \
                --experiment_set_name $TAG \
                --seed $SEED \
                --kl_beta $GRPO_KL_COEF \
                --eval_steps 100 \
                --null_answer_path $NULL_ANSWER_PATH \
                2>&1 | tee -a $ITERATION_LOGFILE
            echo "TRAINED DPO for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
        fi
    fi
    
    # EVAL
    if ! grep -q "FINISHED EVAL at" $ITERATION_LOGFILE && [ "$RESTART_GRPO" = "false" ] ; then
        echo "STARTING EVAL for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
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
            --run_name $EVAL_RUN_NAME \
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
        echo "FINISHED EVAL for iteration $iteration at $(date)" >> $ITERATION_LOGFILE
    fi
    
    # Update the base policy path for the next iteration
    if [ $iteration -lt $NUM_ITERATIONS ]; then
        export BASE_POLICY_PATH="${POLICY_DIR}_adapter"
        echo "Updated base policy path to: $BASE_POLICY_PATH" >> $LOGFILE
    fi
    
    echo "=== Completed Iteration $iteration ===" >> $LOGFILE
done

echo "Iterative SOLiD Training completed at $(date)" >> $LOGFILE
echo "DONE!" 