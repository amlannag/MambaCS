#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=top_gpu_exp
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="epyc4|epyc5"
#SBATCH --account='a_ai_collab'
#SBATCH -o exp_%j.out
#SBATCH -e exp_%j.err

set -euo pipefail

# ---- WandB ----
export WANDB_API_KEY='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'

# ---- Environment ----
module purge
module load rocm/7.14
module load miniforge/24.11.3-0
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda activate base
conda activate mambacs-rocm714
hash -r

python -u -c 'import torch; assert torch.version.hip, "PyTorch is not a ROCm build"; assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"; print(f"PyTorch {torch.__version__} | HIP {torch.version.hip} | GPU {torch.cuda.get_device_name(0)}")'

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
