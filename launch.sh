#!/bin/bash
set -euo pipefail

mkdir -p logs

N=$(python -c "from train_config import EXPERIMENTS; print(len(EXPERIMENTS))")

for i in $(seq 0 $((N - 1))); do
    JOB_NAME=$(python -c "
from train_config import EXPERIMENTS
from config import Config
cfg = Config()
for k, v in EXPERIMENTS[$i].items():
    setattr(cfg, k, v)
print(f'{cfg.prefix}_{cfg.name}')
")
    BACKEND=$(python -c "
from train_config import EXPERIMENTS
from config import Config
cfg = Config()
for k, v in EXPERIMENTS[$i].items():
    setattr(cfg, k, v)
print(cfg.hpc_backend.lower())
")
    NUM_GPUS=$(python -c "
from train_config import EXPERIMENTS
from config import Config
cfg = Config()
for k, v in EXPERIMENTS[$i].items():
    setattr(cfg, k, v)
print(getattr(cfg, 'num_gpus', 1))
")
    if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid num_gpus '$NUM_GPUS' for experiment $i" >&2
        exit 1
    fi
    case "$BACKEND" in
        nvidia) SUBMIT_SCRIPT="submit_nvidia.sh" ;;
        amd) SUBMIT_SCRIPT="submit_amd.sh" ;;
        *) echo "Unsupported hpc_backend '$BACKEND' for experiment $i" >&2; exit 1 ;;
    esac
    echo "Submitting experiment $i on $BACKEND with $NUM_GPUS GPU(s) via $SUBMIT_SCRIPT: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" --gres="gpu:$NUM_GPUS" \
        --cpus-per-task="$((NUM_GPUS * 8))" \
        --export=ALL,EXP_IDX=$i,NUM_GPUS=$NUM_GPUS "$SUBMIT_SCRIPT"
done
