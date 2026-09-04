#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=dctnn_baseline
#SBATCH --time=40:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --account='a_ai_collab'
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

# ---- WandB ----
export WANDB_API_KEY='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'

# ---- Environment ----
source /sw/local/rocky8/noarch/rcc/software/miniforge/24.11.3-0/etc/profile.d/conda.sh
conda activate mambacs

set -u
hash -r

# ---- Run ----
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "Working dir: $(pwd)"
echo ""

EXP_IDX="${EXP_IDX:-0}"

unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
srun --cpu-bind=none python -u train.py --exp_idx "$EXP_IDX"

echo ""
echo "End time: $(date)"
