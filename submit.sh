#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=dctnn_baseline
#SBATCH --time=24:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:a100:1
#SBATCH --account='a_ai_collab'
#SBATCH -o logs/slurm-%j.output
#SBATCH -e logs/slurm-%j.error

# ---- WandB ----
export WANDB_API_KEY=''

# ---- Environment ----
module load cuda/11.8.0
module load miniforge/24.11.3-0
source $ROOTMINIFORGE/etc/profile.d/conda.sh
conda activate mambacs

# ---- Run ----
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "Working dir: $(pwd)"
echo ""

srun --cpu-bind=none python train.py

echo ""
echo "End time: $(date)"
