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
    case "$BACKEND" in
        nvidia) SUBMIT_SCRIPT="submit_nvidia.sh" ;;
        amd) SUBMIT_SCRIPT="submit_amd.sh" ;;
        *) echo "Unsupported hpc_backend '$BACKEND' for experiment $i" >&2; exit 1 ;;
    esac
    echo "Submitting experiment $i on $BACKEND via $SUBMIT_SCRIPT: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" --export=ALL,EXP_IDX=$i "$SUBMIT_SCRIPT"
done
