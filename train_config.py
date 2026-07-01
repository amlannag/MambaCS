"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [
    # index 4 — OASIS, k-space domain, 4x, Rope-Axial, phase-aware attention, fixed DC lambda=1
    {
        "prefix": "OASIS",
        "name": "kspace_axial_RoPE_phase_aware_4x_lam1",
        "dataset": "oasis",
        "image_size": (256, 256),
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": "zscore",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "phase_aware",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",
        "lambda_start": 1.0,
    },
    # index 5 — OASIS, k-space domain, 4x, Rope-Axial, real-valued attention, fixed DC lambda=1
    {
        "prefix": "OASIS",
        "name": "kspace_axial_RoPE_real_valued_4x_lam1",
        "dataset": "oasis",
        "image_size": (256, 256),
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": "zscore",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "real_valued",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",
        "lambda_start": 1.0,
    },
]
