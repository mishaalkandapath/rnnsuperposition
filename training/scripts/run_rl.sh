#!/bin/bash

#SBATCH --job-name=transcoder_mig
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:8
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=15G

echo "=== TASK DEBUG ==="
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NTASKS: $SLURM_NTASKS" 
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_DIR: $SLURM_TMPDIR"
echo "=================="

./training/scripts/compute_can_setup.sh rnn_superpos/data
cd $SLURM_TMPDIR/rnnsuperposition/
module purge 
module load opencv cuda gcc scipy-stack 
source $SLURM_TMPDIR/env/bin/activate

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Model configuration 
INPUT_SIZE=30
HIDDEN_SIZE=128
N_FEATS=12288
BATCH_SIZE=32784
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

run_training() {
    local task_id=$1
    local config="${CONFIGS[$task_id]}"
    read -r sparse_sched sparse_off lr l_sparsity l_penalty c_sparsity detach_w_norm <<< "$config"
    
    # Set GPU for this task
    export CUDA_VISIBLE_DEVICES=$task_id
    export WANDB_NAME="exp_${SLURM_JOBID}_${task_id}"
    
    # Create experiment directory
    local exp_name="sched${sparse_sched}_off${sparse_off}_lr${lr}_lsparse${l_sparsity}_lpen${l_penalty}_csparse${c_sparsity}_wdet${detach_w_norm}"
    local save_path="${BASE_SAVE_PATH}/${exp_name}"
    mkdir -p "$save_path"
    
    echo "Task $task_id: Running experiment $exp_name on GPU $task_id"
    echo "Config: sparse_sched=$sparse_sched, sparse_off=$sparse_off, lr=$lr, l_sparsity=$l_sparsity, l_penalty=$l_penalty, c_sparsity=$c_sparsity, detach_w_norm=$detach_w_norm"
    
    # Build the command with conditional --w_detach flag
    local cmd="python -m training.train_transcoder \
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
    
    # Run the training
    # Run the training
    $cmd > "$save_path/training.log" 2>&1
    
    echo "Task $task_id: Completed experiment $exp_name"
}

# Export function and variables for parallel execution
export -f run_training
export CONFIGS
export BASE_SAVE_PATH INPUT_SIZE HIDDEN_SIZE N_FEATS BATCH_SIZE N_EPOCHS DATASET_PATH

# Get the task ID from Slurm
TASK_ID=${SLURM_PROCID:-0}

# Run the specific task
srun bash -c 'run_training $SLURM_PROCID'

# Wait for all tasks to complete (only needed if running all tasks from one node)
wait

echo "All experiments completed!"