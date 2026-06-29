#!/bin/bash
# Launcher — submits both TCS-main OASIS experiments as separate parallel SLURM jobs.
# Run this from the MambaCS directory: bash submit_tcs_oasis.sh

set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "Submitting Experiment 1: TCS 2D mask  →  tcs_oasis_8x_tcs_mask"
sbatch submit_tcs_oasis_tcs.sh

echo "Submitting Experiment 2: Random column mask  →  tcs_oasis_8x_gpu_mask"
sbatch submit_tcs_oasis_gpu.sh

echo "Both jobs submitted. Check status with: squeue -u \$USER"
