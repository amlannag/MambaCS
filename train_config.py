"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [
    {
        "prefix": "OASIS",
        "name": "kspace_companding_final_l1_complex_4x",
        "dataset": "oasis",
        "image_size": (256, 256),
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": "kspace_companding",
        "companding_p": 0.8,
        "companding_a": 0.5,
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",
        "lambda_start": 1.0,
        "lambda_end": 1.0,
        "loss_mode": "final_only",
        "final_loss_type": "l1",
        "intermediate_loss_type": "l1",
        "seed": 42,
    },
]
