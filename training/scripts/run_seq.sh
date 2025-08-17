#!/bin/bash
#SBATCH --job-name=transcoder_array
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=6:0:00
#SBATCH --mem=512G

echo "=== ARRAY JOB DEBUG ==="
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "======================="

./training/scripts/compute_can_setup.sh rnn_superpos/data
cd $SLURM_TMPDIR/rnnsuperposition/

module purge
module load opencv cuda gcc scipy-stack
source $SLURM_TMPDIR/env/bin/activate

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Model configuration
INPUT_SIZE=30
HIDDEN_SIZE=128
N_FEATS=12288
BATCH_SIZE=20480
N_EPOCHS=250
DATASET_PATH="data/1M_128_update_gate.pt"
BASE_SAVE_PATH="/home/mishaalk/projects/def-gpenn/mishaalk/rnnsuperposition/data/models/copy_transcoder"

# Create base directory
mkdir -p $BASE_SAVE_PATH

# Define hyperparameter combinations
# Each line: sparse_schedule, sparse_off, lr, l_sparsity, l_penalty, c_sparsity, detach_w_norm
CONFIGS=(
    "0 1 2e-4 6 3e-6 4 0"
    "0 1 2e-4 6 3e-6 2 1"
    "0 1 2e-4 8 3e-6 2 1"
    "2 1 2e-4 8 3e-6 2 1"
    "2 1 2e-4 10 3e-6 4 1"
    "3 25 2e-4 8 3e-6 4 1"
    "3 25 2e-4 8 3e-6 4 0"
    "3 25 2e-4 10 3e-6 2 1"
)

# Get the task ID from array
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Get configuration for this task
config="${CONFIGS[$TASK_ID]}"
read -r sparse_sched sparse_off lr l_sparsity l_penalty c_sparsity detach_w_norm <<< "$config"

# Set GPU (should be 0 since we only have 1 GPU per task)
export CUDA_VISIBLE_DEVICES=0
export WANDB_NAME="exp_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"

# Create experiment directory
exp_name="sched${sparse_sched}_off${sparse_off}_lr${lr}_lsparse${l_sparsity}_lpen${l_penalty}_csparse${c_sparsity}_wdet${detach_w_norm}"
save_path="${BASE_SAVE_PATH}/${exp_name}"
mkdir -p "$save_path"

echo "Task $TASK_ID: Running experiment $exp_name"
echo "Config: sparse_sched=$sparse_sched, sparse_off=$sparse_off, lr=$lr, l_sparsity=$l_sparsity, l_penalty=$l_penalty, c_sparsity=$c_sparsity, detach_w_norm=$detach_w_norm"

# Build the command with conditional --w_detach flag
cmd="python -m training.train_transcoder \
    --input_size $INPUT_SIZE \
    --n_feats $N_FEATS \
    --dataset_path $DATASET_PATH \
    --batch_size $BATCH_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --lr $lr \
    --l_sparsity $l_sparsity \
    --l_penalty $l_penalty \
    --c_sparsity $c_sparsity \
    --n_epochs $N_EPOCHS \
    --lambda_sparse_schedule $sparse_sched \
    --l_sparse_offset $sparse_off \
    --save_path $save_path"

# Add --w_detach flag if detach_w_norm is 1
if [ "$detach_w_norm" = "1" ]; then
    cmd="$cmd --w_detach"
fi

echo "Running command: $cmd"

# Run the training
$cmd > "$save_path/training.log" 2>&1

echo "Task $TASK_ID: Completed experiment $exp_name"