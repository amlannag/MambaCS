#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=oasis_image_domain
#SBATCH --time=20:00:00
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

# ---- Run ----
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "Working dir: $(pwd)"
echo ""

pip install fastmri

unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
srun --cpu-bind=none python image_domain_testing_brain_mri.py \
    --train_dir /scratch/user/uqanag/OASIS/keras_png_slices_train \
    --val_dir   /scratch/user/uqanag/OASIS/keras_png_slices_validate \
    --image_size 256 \
    --encoders axial axial axial \
    --accel 4 \
    --epochs 100 \
    --batch_size 16 \
    --out_dir ../Experiments/oasis_image_domain_4x

echo ""
echo "End time: $(date)"
