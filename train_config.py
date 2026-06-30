"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [
    # index 2 — OASIS, image-domain, 4x, Rope-Axial, fixed DC lambda=1
    {
        "prefix": "OASIS",
        "name": "image_axial_RoPE_4x_lam1",
        "dataset": "oasis",
        "image_size": (256, 256),
        "encoders": ["axial", "axial", "axial"],
        "learning": "image",
        "norm": None,
        "pos_emb_type": "Rope-Axial",
        "attn_type": "standard",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",  # fixed DC — not learned, not annealed
        "lambda_start": 1.0,
    },
    # index 3 — OASIS, k-space domain, 4x, Rope-Axial, complex attention, fixed DC lambda=1
    {
        "prefix": "OASIS",
        "name": "kspace_axial_RoPE_complex_4x_lam1",
        "dataset": "oasis",
        "image_size": (256, 256),
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": "zscore",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
        "acceleration_factors": [4],
        "lambda_schedule": "constant",  # fixed DC — not learned, not annealed
        "lambda_start": 1.0,
    },
]
