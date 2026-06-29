#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=tcs_oasis_compare
#SBATCH --time=40:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --account='a_ai_collab'
#SBATCH -o logs/slurm-%j.output
#SBATCH -e logs/slurm-%j.error

# ---- WandB ----
export WANDB_API_KEY='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'

# ---- Environment ----
module load cuda/11.8.0
module load miniforge/24.11.3-0
source $ROOTMINIFORGE/etc/profile.d/conda.sh
conda activate mambacs

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "Working dir: $(pwd)"
echo ""

unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE

# ---- Experiment 1: TCS-main 2D mask (fixed pattern, same every batch) ----
echo "=== Experiment 1/2: TCS 2D mask (tcs_arch_8x_tcs_mask) ==="
srun --cpu-bind=none python train_tcs_oasis.py \
    --train_dir /scratch/user/uqanag/OASIS/keras_png_slices_train \
    --val_dir   /scratch/user/uqanag/OASIS/keras_png_slices_validate \
    --mask_type tcs \
    --accel 8 \
    --epochs 100 \
    --batch_size 16 \
    --out_dir ../Experiments/tcs_oasis_8x_tcs_mask

echo ""
echo "Experiment 1 finished at $(date)"
echo ""

# ---- Experiment 2: Random column mask, TCS FFT/DC throughout ----
echo "=== Experiment 2/2: Random column mask (tcs_arch_8x_gpu_mask) ==="
srun --cpu-bind=none python train_tcs_oasis.py \
    --train_dir /scratch/user/uqanag/OASIS/keras_png_slices_train \
    --val_dir   /scratch/user/uqanag/OASIS/keras_png_slices_validate \
    --mask_type gpu \
    --accel 8 \
    --epochs 100 \
    --batch_size 16 \
    --out_dir ../Experiments/tcs_oasis_8x_gpu_mask

echo ""
echo "End time: $(date)"
