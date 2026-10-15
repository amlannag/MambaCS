#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=mambacs_inference
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=a_ai_collab
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

module purge
module load miniforge/24.11.3-0
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda activate mambacs-rocm-all

cd "${SLURM_SUBMIT_DIR:?Submit this job from the MambaCS repository directory}"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$HOME/Experiments}"
# Edit this list to choose the experiment folders under $EXPERIMENTS_DIR.
EXPERIMENTS=(
    "brain_fnet_both_cross_axial5_ms13_gfilt_imgconv16x3_8vcoils_freqw_m3_g0.5_cap0.6_p95_slice_100ep_r4"
)

ARGS=(
    --experiments-dir "$EXPERIMENTS_DIR"
    --experiments "${EXPERIMENTS[@]}"
)
if [[ -n "${VAL_DATA_DIR:-}" ]]; then
    ARGS+=(--val-dir "$VAL_DATA_DIR")
fi

echo "Experiments root: $EXPERIMENTS_DIR"
echo "Backend: AMD ROCm"
printf 'Evaluating: %s\n' "${EXPERIMENTS[@]}"
unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
srun --cpu-bind=none python -u inference_validation.py "${ARGS[@]}"
