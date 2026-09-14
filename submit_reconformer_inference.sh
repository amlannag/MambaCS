#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=reconformer_inference
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --account='a_ai_collab'
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set +x
set -eo pipefail
export WANDB_API_KEY='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'
EXPERIMENT_DIR="${EXPERIMENT_DIR:-../Experiments/fastMRI_reconformer_random_mask_r4_50epochs}"
DATA_DIR="${DATA_DIR:-/scratch/user/uqanag/fastmri/singlecoil_val}"
WANDB_PROJECT="${WANDB_PROJECT:-fastMRI}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"

case "${HPC_BACKEND:-${SLURM_JOB_PARTITION:-gpu_rocm}}" in
    amd|gpu_rocm)
        module purge
        module load miniforge/24.11.3-0
        source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
        conda activate mambacs-rocm-all
        ;;
    nvidia|gpu_cuda)
        source /sw/local/rocky8/noarch/rcc/software/miniforge/24.11.3-0/etc/profile.d/conda.sh
        conda activate mambacs
        ;;
    *)
        echo 'Unsupported backend. Use gpu_rocm (AMD) or gpu_cuda (NVIDIA).' >&2
        exit 1
        ;;
esac

set -u
hash -r
cd "${SLURM_SUBMIT_DIR:?Submit this job from the MambaCS repository root}"
set +x
if [[ -z "$WANDB_API_KEY" && -n "$WANDB_API_KEY_FILE" ]]; then
    if [[ ! -f "$WANDB_API_KEY_FILE" || ! -r "$WANDB_API_KEY_FILE" ]]; then
        echo 'The configured W&B API key file is not readable.' >&2
        exit 1
    fi
    WANDB_API_KEY="$(< "$WANDB_API_KEY_FILE")"
    if [[ -z "$WANDB_API_KEY" ]]; then
        echo 'The configured W&B API key file is empty.' >&2
        exit 1
    fi
fi
if [[ -n "$WANDB_API_KEY" ]]; then
    export WANDB_API_KEY
else
    unset WANDB_API_KEY
fi
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo "Working directory: $PWD"
echo "Metrics destination: Weights & Biases"

INFERENCE_ARGS=(--experiment-dir "$EXPERIMENT_DIR" --data-dir "$DATA_DIR" --wandb-project "$WANDB_PROJECT")
if [[ -n "$WANDB_ENTITY" ]]; then
    INFERENCE_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
srun --cpu-bind=none python -u inference.py "${INFERENCE_ARGS[@]}" "$@" --device cuda

echo "End time: $(date)"
