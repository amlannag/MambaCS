#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=dctnn_baseline
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --qos=gpu
#SBATCH --gres=gpu:mi210:1
#SBATCH --constraint="epyc4"
#SBATCH --account='a_ai_collab'
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

export WANDB_API_KEY='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'

source "$HOME/.conda/etc/profile.d/conda.sh"
conda activate mambacs-AMD

set -u
hash -r

python -u -c 'import torch; assert int(torch.__version__.split(".")[0]) >= 2, "ReconFormer requires PyTorch >= 2"; assert torch.version.hip, "PyTorch is not a ROCm build"; assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"; props=torch.cuda.get_device_properties(0); arch=getattr(props, "gcnArchName", "unknown"); print("PyTorch", torch.__version__, "| HIP", torch.version.hip, "| GPU", props.name, "| arch", arch, "| compiled", torch.cuda.get_arch_list()); x=torch.randn(1,1,32,32,device="cuda",dtype=torch.complex64); torch.fft.ifft2(x); torch.cuda.synchronize(); print("ROCm complex FFT smoke test: OK")'

cd "$SLURM_SUBMIT_DIR"
python -u -c 'import einops, fastmri, h5py, numpy, wandb; from ReconFormer import ReconFormerBaseline; print("Training dependencies and ReconFormer import: OK")'
mkdir -p logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "Backend    : AMD ROCm"
echo "Working dir: $(pwd)"
echo ""

EXP_IDX="${EXP_IDX:-0}"

unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
srun --cpu-bind=none python -u train.py --exp_idx "$EXP_IDX"

echo ""
echo "End time: $(date)"
