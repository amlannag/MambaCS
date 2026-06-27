"""
Experiment definitions for DcTNN training.

All experiments below use the same axial 3-stage backbone at R=4 with no
normalisation. Training loss is configured in train.py and currently uses
magnitude-domain L1 loss.
"""

EXPERIMENTS = [
    {
        "prefix": "L1",
        "name": "image_axial_APE_4x",
        "encoders": ["axial", "axial", "axial"],
        "learning": "image",
        "norm": None,
        "pos_emb_type": "APE",
        "attn_type": "standard",
        "acceleration_factors": [4],
    },
    {
        "prefix": "L1",
        "name": "kspace_axial_APE_4x",
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": None,
        "pos_emb_type": "APE",
        "attn_type": "complex",
        "acceleration_factors": [4],
    },
    {
        "prefix": "L1_DC0",
        "name": "image_axial_APE_4x",
        "encoders": ["axial", "axial", "axial"],
        "learning": "image",
        "norm": None,
        "pos_emb_type": "APE",
        "attn_type": "standard",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",
        "lambda_start": 0.0,
        "lambda_end": 0.0,
    },
    {
        "prefix": "L1_DC0",
        "name": "kspace_axial_APE_4x",
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": None,
        "pos_emb_type": "APE",
        "attn_type": "complex",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",
        "lambda_start": 0.0,
        "lambda_end": 0.0,
    },
]
