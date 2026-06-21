"""
Experiment definitions for DcTNN training.

Each entry in EXPERIMENTS is a dict of Config field overrides.
Any key not listed falls back to the Config class default in config.py.

Index  Name
  0    image_axial_nonorm_8x          — axial, image domain, no norm, standard attn
  1    kspace_axial_complex_8x        — axial, k-space domain, no norm, complex attn
  2    kspace_axial_real_valued_8x    — axial, k-space domain, no norm, real_valued attn
  3    kspace_axial_phase_aware_8x    — axial, k-space domain, no norm, phase_aware attn
  4    kspace_axial_standard_8x       — axial, k-space domain, no norm, standard attn
"""

EXPERIMENTS = [
    # 0 — image domain, no normalisation, standard attention (baseline)
    {
        "prefix": "DcTNN",
        "name": "image_axial_nonorm_8x",
        "encoders": ["axial", "axial", "axial"],
        "learning": "image",
        "norm": None,
        "pos_emb_type": "Rope-Axial",
        "attn_type": "standard",
        "acceleration_factors": [8],
    },
    # 1 — k-space domain, complex attention
    {
        "prefix": "DcTNN",
        "name": "kspace_axial_complex_8x",
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": None,
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
        "acceleration_factors": [8],
    },
    # 2 — k-space domain, real-valued attention
    {
        "prefix": "DcTNN",
        "name": "kspace_axial_real_valued_8x",
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": None,
        "pos_emb_type": "Rope-Axial",
        "attn_type": "real_valued",
        "acceleration_factors": [8],
    },
    # 3 — k-space domain, phase-aware attention
    {
        "prefix": "DcTNN",
        "name": "kspace_axial_phase_aware_8x",
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": None,
        "pos_emb_type": "Rope-Axial",
        "attn_type": "phase_aware",
        "acceleration_factors": [8],
    },
    # 4 — k-space domain, standard attention
    {
        "prefix": "DcTNN",
        "name": "kspace_axial_standard_8x",
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": None,
        "pos_emb_type": "Rope-Axial",
        "attn_type": "standard",
        "acceleration_factors": [8],
    },
]
