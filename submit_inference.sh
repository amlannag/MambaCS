#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=dctnn_inference
#SBATCH --time=01:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:a100:1
#SBATCH --account='a_ai_collab'
#SBATCH -o logs/inference-%j.output
#SBATCH -e logs/inference-%j.error

# ---------------------------------------------------------------------------
# DcTNN Inference Job
#
# Usage:
#   # Run residual reports for all experiments in EXPERIMENTS list:
#   sbatch submit_inference.sh
#
#   # Run full inference report for a single experiment:
#   sbatch --export=ALL,RUN_ALL=0,EXP_DIR=../Experiments/KSpace_patch,ACCEL=8 submit_inference.sh
# ---------------------------------------------------------------------------

# ---- Defaults (edit these or override with --export at sbatch time) ----
RUN_ALL="${RUN_ALL:-1}"
EXP_DIR="${EXP_DIR:-../Experiments/KSpace_patch}"
NUM_IMAGES="${NUM_IMAGES:-5}"
ACCEL="${ACCEL:-8}"
SPLIT="${SPLIT:-val}"

# ---- Environment ----
module load cuda/11.8.0
module load miniforge/24.11.3-0
source $ROOTMINIFORGE/etc/profile.d/conda.sh
conda activate mambacs

# ---- Run ----
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "Run all      : $RUN_ALL"
echo ""

unset SLURM_MEM_PER_GPU

if [ "$RUN_ALL" -eq 1 ]; then
    echo "Running residual reports for all experiments..."
    srun --cpu-bind=none python inference.py \
        --all \
        --split "$SPLIT"
else
    echo "Experiment   : $EXP_DIR"
    echo "Accel factor : R=$ACCEL"
    echo "Images       : $NUM_IMAGES"
    echo "Split        : $SPLIT"
    srun --cpu-bind=none python inference.py \
        --exp_dir    "$EXP_DIR"    \
        --num_images "$NUM_IMAGES" \
        --accel      "$ACCEL"      \
        --split      "$SPLIT"
fi

echo ""
echo "End time: $(date)"
