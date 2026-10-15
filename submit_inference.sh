#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=mambacs_inference
#SBATCH --time=24:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --account=a_ai_collab
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

source /sw/local/rocky8/noarch/rcc/software/miniforge/24.11.3-0/etc/profile.d/conda.sh
conda activate mambacs

cd "${SLURM_SUBMIT_DIR:?Submit this job from the MambaCS repository directory}"

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
printf 'Evaluating: %s\n' "${EXPERIMENTS[@]}"
python -u inference_validation.py "${ARGS[@]}"
