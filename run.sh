set -e
set -o pipefail

source ./configs/setup.sh

cd /workspace
export P=/workspace
echo "Successfully setup!"
export PATH="/home/dev/.local/bin:$PATH"
export MASTER_PORT=$(echo '12'$(shuf -i 100-999 -n 1))
echo $MASTER_PORT
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TAG="$TIMESTAMP"

export EXPERIMENT_SET_DIRECTORY="$P/outputs/$TAG"
mkdir "$P/outputs" || true
mkdir $EXPERIMENT_SET_DIRECTORY || true


# Setting up file locations (organizational)
export LOGFILE="$EXPERIMENT_SET_DIRECTORY/stdout_err.log"
export MUNGED_DATA_PATH="$EXPERIMENT_SET_DIRECTORY/munged_data.csv"
export DETECTED_PATH=$EXPERIMENT_SET_DIRECTORY/detected.csv
export LR_PATH=$EXPERIMENT_SET_DIRECTORY/lr.pkl
export DATASET_PATH=$EXPERIMENT_SET_DIRECTORY/rewarded
export CSV_PATH=$EXPERIMENT_SET_DIRECTORY/rewarded_csv.csv
export RM_DIR=$EXPERIMENT_SET_DIRECTORY/rm
export SFT_DIR=$EXPERIMENT_SET_DIRECTORY/sft
export POLICY_DIR=$EXPERIMENT_SET_DIRECTORY/policy
export EVAL_OUT_DIR=$EXPERIMENT_SET_DIRECTORY/eval
export DETECTOR_RUN_NAME="detector_$TAG"
export RM_RUN_NAME="rm_explicit_$TAG"
export RM_BT_RUN_NAME="rm_bt_$TAG"
export DPO_RUN_NAME="dpo_$TAG"
export SFT_RUN_NAME="sft_$TAG"
export GRPO_RUN_NAME="grpo_$TAG"
export EVAL_RUN_NAME="eval_$TAG"
export WANDB_PROJECT='solid_deception'

# ----------------------------------------

# Global Settings
export DEBUG_TRAINING=true
export DO_SAE=false
export DO_DPO=false
export DO_BT_RM=true # Bradley-Terry reward model
export DO_CATEGORICAL_RM=false
export ADAPTIVE=false
export RESTART_GRPO=false
export BASE_PDTBS=8 # Per device batch size for 8b

# Debugging

# Model
# export BASE_MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export BASE_MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export GENERATION_LORA_PATH=None
# export BASE_MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct
# export BASE_MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct

# Training
# export ACONFIG=$P/configs/default_config.yaml
# export ACONFIG=$P/configs/2_gpu_ddp.yaml
export ACONFIG=$P/configs/1_gpu_ddp.yaml

# Data hyperparams
export RAW_DATA_PATH='AlignmentResearch/DolusChat'
export TEST_FRAC=0.05
export TRAIN_LR_FRAC=0.05
export REWARD_SYSTEM_PROMPT="$P/solid_deception/training/gpt4_reward_prompt.txt"
export LAYER=16
export TRAIN_DATA_LIMIT=None
export LIE_FPR=None
export LIE_TPR=0.5 
export SAE_PATH="$P/saes/layer_23"
export SAE_DESCRIPTIONS_PATH="$P/solid_deception/detection/model.layers.23_feature.json"
export SAE_WORDS_PATH="$P/solid_deception/detection/sae_words.txt"
export ALL_POSITIONS=false
export SAMPLE_LABELS=false
export SEED=0
export DETECTOR_PDTBS=$((BASE_PDTBS / 2))

# RM
export RM_LOGICAL_BATCH_SIZE=256
export RM_NUM_EPOCHS=4
export RM_LORA_R=256
export RM_LR=5e-6
export RM_PDTBS=$((BASE_PDTBS * 2))

# SFT
export SFT_LOGICAL_BATCH_SIZE=128
export SFT_PDTBS=$((BASE_PDTBS * 2))
export SFT_LR=1e-5

# GRPO
export GRPO_LOGICAL_BATCH_SIZE=512
export GRPO_PDTBS=$((BASE_PDTBS / 2))
export POLICY_LORA_R=512
export GRPO_LRFBS=24
export GRPO_EVAL_STEPS=100
export GRPO_LR=5e-6
export GRPO_TOTAL_EPS=150000
export GRPO_KL_COEF=0.1
export USE_GRPO=true

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
export DPO_PDTBS=$((BASE_PDTBS/2)) # chosen + rejected per example
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


export EVAL_SFT_PATH="${SFT_DIR}_adapter"
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
    echo "Training Restarted at $(date)" >> "$LOGFILE"
fi

echo "<env>"
env >> $LOGFILE
echo `env`
echo "</env>"
# ----------------------------------------

if ! grep -q "MUNGED DATA at" $LOGFILE; then
    echo "STARTING MUNGE at $(date)" >> $LOGFILE
    python $P/solid_deception/data_generation/munge_data.py --input_path $RAW_DATA_PATH -c $MUNGED_DATA_PATH --test_frac $TEST_FRAC --seed $SEED --train_lr_frac $TRAIN_LR_FRAC  2>&1 | tee -a $LOGFILE
    echo "MUNGED DATA at $(date)" >> $LOGFILE
fi

# Sample on policy


# Do detector
if ! grep -q "FINISHED DETECTOR at" $LOGFILE; then
    echo "STARTING SAMPLING at $(date)" >> $LOGFILE
    TOKENIZERS_PARALLELISM=false accelerate launch --config_file $ACONFIG --main_process_port $MASTER_PORT $P/solid_deception/detection/lr_detector.py --model_path $BASE_MODEL_PATH --tokenizer_path $BASE_MODEL_PATH --data_path $MUNGED_DATA_PATH --csv_save_path $DETECTED_PATH --lr_save_path $LR_PATH --batch_size $DETECTOR_PDTBS  --layer $LAYER $DEBUG_TRAINING_FLAG --lie_false_positive_rate $LIE_FPR --lie_true_positive_rate $LIE_TPR $SAE_FLAG --experiment_set_name $TAG --name $DETECTOR_RUN_NAME --sae_path $SAE_PATH --sae_words_path $SAE_WORDS_PATH --sae_descriptions_path $SAE_DESCRIPTIONS_PATH $ADAPTIVE_FLAG --seed $SEED $TRAIN_DATA_LIMIT_FLAG $ALL_POSITIONS_FLAG --max_length $MAX_DETECTOR_SEQ_LENGTH 2>&1 | tee -a $LOGFILE
    echo "FINISHED DETECTOR at $(date)" >> $LOGFILE
fi

# Make dataset
if ! grep -q "MADE DATASET at" $LOGFILE; then
    echo "STARTING DATASET at $(date)" >> $LOGFILE
    python $P/solid_deception/data_generation/make_dataset.py -i $DETECTED_PATH -od $DATASET_PATH -oc $CSV_PATH --rewards -1 2 1 1  2>&1 | tee -a $LOGFILE
    echo "MADE DATASET at $(date)" >> $LOGFILE
fi

# # Train SFT
if ! grep -q "TRAINED SFT at" $LOGFILE; then
    echo "STARTING SFT at $(date)" >> $LOGFILE
    accelerate launch --config_file $ACONFIG --main_process_port $MASTER_PORT $P/solid_deception/training/train_sft.py --output_dir $SFT_DIR --model_name_or_path $BASE_MODEL_PATH --learning_rate $SFT_LR --num_train_epochs 1.0 --per_device_eval_batch_size 4 --per_device_train_batch_size $SFT_PDTBS --use_peft --lora_r $POLICY_LORA_R --dataset_name $DATASET_PATH --bf16 --run_name $SFT_RUN_NAME $DEBUG_TRAINING_FLAG --gradient_checkpointing True --logical_batch_size $SFT_LOGICAL_BATCH_SIZE --seed $SEED --experiment_set_name $TAG 2>&1 | tee -a $LOGFILE
    echo "TRAINED SFT at $(date)" >> $LOGFILE
fi

# Train RM
if ! $DO_DPO; then
    if ! grep -q "TRAINED RM at" $LOGFILE; then
        echo "STARTING RM at $(date)" >> $LOGFILE
        if $DO_BT_RM; then
            accelerate launch --config_file $ACONFIG --main_process_port $MASTER_PORT $P/solid_deception/solid_deception/training/train_reward.py --output_dir $RM_DIR --model_name_or_path $BASE_MODEL_PATH --dataset_name $DATASET_PATH --per_device_train_batch_size $RM_PDTBS --learning_rate $RM_LR --run_name $RM_BT_RUN_NAME --gradient_checkpointing True --per_device_eval_batch_size 4 --bf16 --lora_r $RM_LORA_R --use_peft --num_train_epochs $RM_NUM_EPOCHS $DEBUG_TRAINING_FLAG --logical_batch_size $RM_LOGICAL_BATCH_SIZE --experiment_set_name $TAG --seed $SEED --dataloader_num_workers 8 2>&1 | tee -a $LOGFILE
        else
            accelerate launch --config_file $ACONFIG --main_process_port $MASTER_PORT $P/solid_deception/solid_deception/training/train_explicit_rm.py --output_dir $RM_DIR --model_name_or_path $BASE_MODEL_PATH --dataset_name $DATASET_PATH --per_device_train_batch_size $RM_PDTBS --learning_rate $RM_LR --run_name $RM_RUN_NAME --gradient_checkpointing True --per_device_eval_batch_size 4 --bf16 --lora_r $RM_LORA_R --use_peft --num_train_epochs $RM_NUM_EPOCHS $DEBUG_TRAINING_FLAG --logical_batch_size $RM_LOGICAL_BATCH_SIZE  --experiment_set_name $TAG --do_categorical_labels $CATEGORICAL_RM_LABELS  --null_example_reward -5.0 --dataloader_num_workers 8 --seed $SEED 2>&1 | tee -a $LOGFILE
        fi
        echo "TRAINED RM at $(date)" >> $LOGFILE
    fi
fi

if ! $DO_DPO; then
    # # Train GRPO
    if ! grep -q "TRAINED GRPO at" $LOGFILE && [ "$RESTART_GRPO" = "false" ] ; then
        echo "STARTING GRPO at $(date)" >> $LOGFILE
        WANDB_RUN_ID=$GRPO_RUN_NAME WANDB_RESUME=allow accelerate launch --config_file $ACONFIG --main_process_port $MASTER_PORT $P/solid_deception/training/train_grpo.py --reward_model_path "${RM_DIR}_adapter" --sft_model_path "${SFT_DIR}_adapter" --per_device_train_batch_size $GRPO_PDTBS --local_rollout_forward_batch_size 24 --eval_steps 100 --per_device_eval_batch_size 2 --run_name $GRPO_RUN_NAME --eval_strategy steps --output_dir $POLICY_DIR  --model_name_or_path $BASE_MODEL_PATH --rloo_k $GRPO_K  --learning_rate $GRPO_LR --gradient_checkpointing True --missing_eos_penalty 1.0 --total_episodes $GRPO_TOTAL_EPS --kl_coef $DPO_KL_COEF --dataloader_num_workers 8 --dataset_name $DATASET_PATH --use_triple_peft --lora_r $POLICY_LORA_R --bf16 --max_grad_norm 1000 --clip $DEBUG_TRAINING_FLAG --logical_batch_size $GRPO_LOGICAL_BATCH_SIZE --experiment_set_name $TAG --no_naive_pg_gradient False --do_categorical_labels $CATEGORICAL_GRPO_LABELS $GRPO_FLAG  --seed $SEED --null_example_reward -5.0 2>&1 | tee -a $LOGFILE
        echo "TRAINED GRPO at $(date)" >> $LOGFILE
    fi
else
    if ! grep -q "TRAINED DPO at" $LOGFILE; then
        echo "STARTING DPO at $(date)" >> $LOGFILE
        accelerate launch --config_file $ACONFIG --main_process_port $MASTER_PORT $P/solid_deception/training/train_dpo.py --dataset_name $DATASET_PATH  --output_dir $POLICY_DIR --model_name_or_path "${SFT_DIR}_adapter"  --per_device_train_batch_size $DPO_PDTBS --eval_steps 400 --label_smoothing_factor 0.05 --per_device_eval_batch_size 2 --run_name $DPO_RUN_NAME --learning_rate $DPO_LR --eval_strategy steps --bf16 --use_peft --lora_r $POLICY_LORA_R --logical_batch_size $DPO_LOGICAL_BATCH_SIZE $DEBUG_TRAINING_FLAG --experiment_set_name $TAG --seed $SEED --kl_beta $GRPO_KL_COEF --eval_steps 100 2>&1 | tee -a $LOGFILE
        echo "TRAINED DPO at $(date)" >> $LOGFILE
    fi
fi

# # Eval
if ! grep -q "FINISHED EVAL at" $LOGFILE   && [ "$RESTART_GRPO" = "false" ] ; then
    echo "STARTING EVAL at $(date)" >> $LOGFILE
    CUDA_VISIBLE_DEVICES=0 python $P/solid_deception/eval/reward.py --model_path "${POLICY_DIR}_adapter" --reward_model_path "$RM_OUTPUT_DIR" --tokenizer_path $BASE_MODEL_PATH --dataset_path $CSV_PATH --original_model_path $BASE_MODEL_PATH --lr_path $LR_PATH --layer $LAYER --output_dir $EVAL_OUT_DIR --n_rows 20 $DEBUG_TRAINING_FLAG --experiment_set_name $TAG --run_name $EVAL_RUN_NAME --sae_path $SAE_PATH --sae_words_path $SAE_WORDS_PATH --sae_descriptions_path $SAE_DESCRIPTIONS_PATH $SAE_FLAG --null_example_reward -5.0 $CATEGORICAL_GRPO_LABELS_FLAG --seed $SEED $ALL_POSITIONS_FLAG --sft_model_path $EVAL_SFT_PATH 2>&1 | tee -a $LOGFILE
    echo "FINISHED EVAL at $(date)" >> $LOGFILE
fi



            echo "DONE!"
