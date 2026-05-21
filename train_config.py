"""
Experiment definitions for DcTNN training.

Each entry in EXPERIMENTS is a dict of Config field overrides.
Any key not listed falls back to the Config class default (defined in config.py).

Run a specific experiment:
    python train.py --exp_idx <N>

Submit all experiments as separate SLURM jobs:
    bash launch.sh
"""

EXPERIMENTS = [
    {
        "prefix": "LambdaSchedule",
        "name": "cosine_end1p00",
        "lambda_schedule": "cosine",
        "lambda_start": 1.0,
        "lambda_end": 1.0,
    },
    {
        "prefix": "LambdaSchedule",
        "name": "cosine_end0p75",
        "lambda_schedule": "cosine",
        "lambda_start": 1.0,
        "lambda_end": 0.75,
    },
    {
        "prefix": "LambdaSchedule",
        "name": "cosine_end0p50",
        "lambda_schedule": "cosine",
        "lambda_start": 1.0,
        "lambda_end": 0.5,
    },
    {
        "prefix": "LambdaSchedule",
        "name": "cosine_end0p25",
        "lambda_schedule": "cosine",
        "lambda_start": 1.0,
        "lambda_end": 0.25,
    },
]
