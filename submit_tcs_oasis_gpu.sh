#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=tcs_oasis_gpu_mask
#SBATCH --time=20:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --account='a_ai_collab'
#SBATCH -o logs/slurm-%j.output
#SBATCH -e logs/slurm-%j.error

export WANDB_API_KEY='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'

module load cuda/11.8.0
module load miniforge/24.11.3-0
source $ROOTMINIFORGE/etc/profile.d/conda.sh
conda activate mambacs

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo ""

unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE

srun --cpu-bind=none python train_tcs_oasis.py \
    --train_dir /scratch/user/uqanag/OASIS/keras_png_slices_train \
    --val_dir   /scratch/user/uqanag/OASIS/keras_png_slices_validate \
    --mask_type gpu \
    --accel 8 \
    --epochs 100 \
    --batch_size 16 \
    --out_dir ../Experiments/tcs_oasis_8x_gpu_mask

echo "End time: $(date)"
