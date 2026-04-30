#!/bin/bash
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
    echo "Submitting experiment $i: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" --export=ALL,EXP_IDX=$i submit.sh
done
