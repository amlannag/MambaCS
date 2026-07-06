"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [


    {
        "prefix": "OASIS",
        "name": "kspace_axial_RoPE_complex_4x_masked_fixedcross",
        "dataset": "oasis",
        "image_size": (256, 256),
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": "none",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",
        "lambda_start": 0.0,
        "mask_vertical_attn": "cross",
    },

]
