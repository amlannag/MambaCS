"""
Experiment definitions for DcTNN training.

Each entry in EXPERIMENTS is a dict of Config field overrides.
Any key not listed falls back to the Config class default in config.py.

Index  Name
  0    FullComplex  — axial, k_space, Rope-Axial, complex attn  (original)
  1    MagLoss_KSpace — axial, k_space, Rope-Axial, complex attn + magnitude image loss
  2    MagLoss_Image  — axial, image domain, Rope-Axial, standard attn + magnitude image loss
"""

EXPERIMENTS = [
    # 1 — k-space learning with MagnitudeImageLoss
    {
        "prefix": "MagLoss",
        "name": "kspace_axial_rope_axial",
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
    },
    {
        "prefix": "MagLoss",
        "name": "image_axial_rope_axial",
        "encoders": ["axial", "axial", "axial"],
        "learning": "image",
        "pos_emb_type": "Rope-Axial",
        "attn_type": "standard",
    },
]
