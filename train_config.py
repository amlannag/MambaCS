"""
Experiment definitions for DcTNN training.
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
]
