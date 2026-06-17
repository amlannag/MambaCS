"""
Experiment definitions for DcTNN training.

Each entry in EXPERIMENTS is a dict of Config field overrides.
Any key not listed falls back to the Config class default in config.py.

Index  Name
  0    image_axial_zscore    — axial, image domain, z-score norm
  1    image_axial_nonorm    — axial, image domain, no normalisation
  2    kspace_axial_nonorm   — axial, k-space domain, no normalisation
"""

EXPERIMENTS = [
    # 1 — image domain, no normalisation
    {
        "prefix": "DcTNN",
        "name": "image_axial_nonorm",
        "encoders": ["axial", "axial", "axial"],
        "learning": "image",
        "norm": None,
        "pos_emb_type": "Rope-Axial",
        "attn_type": "standard",
        "acceleration_factors": [4],
    },
    # 2 — k-space domain, no normalisation
    
    {
        "prefix": "DcTNN",
        "name": "kspace_axial_nonorm",
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": None,
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
        "acceleration_factors": [4],
    },
]
