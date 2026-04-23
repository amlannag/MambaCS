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
#   sbatch submit_inference.sh
#
# Override defaults at submission time with --export, e.g.:
#   sbatch --export=ALL,EXP_DIR=../Experiments/dctnn_baseline,ACCEL=6 submit_inference.sh
# ---------------------------------------------------------------------------

# ---- Defaults (edit these or override with --export at sbatch time) ----
EXP_DIR="${EXP_DIR:-../Experiments/dctnn_baseline}"
NUM_IMAGES="${NUM_IMAGES:-5}"
ACCEL="${ACCEL:-4}"
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
echo "Experiment   : $EXP_DIR"
echo "Accel factor : R=$ACCEL"
echo "Images       : $NUM_IMAGES"
echo "Split        : $SPLIT"
echo ""

srun --cpu-bind=none python inference.py \
    --exp_dir    "$EXP_DIR"    \
    --num_images "$NUM_IMAGES" \
    --accel      "$ACCEL"      \
    --split      "$SPLIT"

echo ""
echo "End time: $(date)"
