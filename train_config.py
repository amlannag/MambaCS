"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [
    {
        "prefix": "OASIS",
        "name": "kspace_cross_axial_RoPE_complex_4x",
        "dataset": "oasis",
        "image_size": (256, 256),
        "encoders": ["cross_axial", "cross_axial", "cross_axial"],
        "learning": "k_space",
        "norm": "none",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",
        "lambda_start": 0.0,
    },
    {
        "prefix": "OASIS",
        "name": "kspace_cross_axial_axial_axial_RoPE_complex_4x",
        "dataset": "oasis",
        "image_size": (256, 256),
        "encoders": ["cross_axial", "axial", "axial"],
        "learning": "k_space",
        "norm": "none",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",
        "lambda_start": 0.0,
    },

]
