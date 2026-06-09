"""
Experiment definitions for DcTNN training.

Each entry in EXPERIMENTS is a dict of Config field overrides.
Any key not listed falls back to the Config class default in config.py.

Index  Name
  0    kspace_axial  — axial, k-space learning, Rope-Axial, complex attn
                       ACS+Gaussian undersampling (8%/4%), row-stride=2
  1    image_axial   — axial, image domain learning, Rope-Axial, standard attn
                       ACS+Gaussian undersampling (8%/4%), row-stride=2
"""

EXPERIMENTS = [
    # 1 — image domain axial
    {
        "prefix": "DcTNN_fixed",
        "name": "image_axial",
        "encoders": ["axial", "axial", "axial"],
        "learning": "image",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "standard",
        "axial_row_stride": 2,
        "acceleration_factors": [8],
    },
]
