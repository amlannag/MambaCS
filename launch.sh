#!/bin/bash
EXP_NAME=$(python -c "from train_config import cfg; print(f'{cfg.experiment.prefix}_{cfg.experiment.name}')")
sbatch --job-name="$EXP_NAME" submit.sh
