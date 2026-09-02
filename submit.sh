#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=dctnn_baseline
#SBATCH --time=40:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:l40s:1
#SBATCH --account='a_ai_collab'
#SBATCH -o logs/slurm-%j.output
#SBATCH -e logs/slurm-%j.error

set -euo pipefail

# ---- WandB ----
export WANDB_API_KEY='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'

# ---- Environment ----
module load cuda/11.8.0
module load miniforge/24.11.3-0
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda activate mambacs
hash -r

python -u -c 'import torch; assert torch.version.cuda, "PyTorch is not a CUDA build"; assert not torch.version.hip, "Expected CUDA but loaded a ROCm build"; assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"; props=torch.cuda.get_device_properties(0); print("PyTorch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU", props.name, "| capability", torch.cuda.get_device_capability(0)); x=torch.randn(1,1,32,32,device="cuda",dtype=torch.complex64); torch.fft.ifft2(x); torch.cuda.synchronize(); print("CUDA complex FFT smoke test: OK")'

# ---- Run ----
cd "$SLURM_SUBMIT_DIR"
python -u -c 'import einops, fastmri, h5py, numpy, wandb; from ReconFormer import ReconFormerBaseline; print("Training dependencies and ReconFormer import: OK")'
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
