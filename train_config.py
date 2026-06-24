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
  5    image_axial_APE_4x             — axial, image domain, APE, standard attn, 4x
  6    kspace_axial_APE_4x            — axial, k-space domain, APE, standard attn, 4x
"""

EXPERIMENTS = [
    # 0 — image domain, no normalisation, standard attention (baseline)
    {
        "prefix": "BugFIX",
        "name": "image_axial_APE_4x",
        "encoders": ["axial", "axial", "axial"],
        "learning": "image",
        "norm": None,
        "pos_emb_type": "APE",
        "attn_type": "standard",
        "acceleration_factors": [4],
    },
    # 6 — k-space domain, APE, standard attention, 4x acceleration (baseline)
    {
        "prefix": "BugFIX",
        "name": "kspace_axial_APE_4x",
        "encoders": ["axial", "axial", "axial"],
        "learning": "k_space",
        "norm": None,
        "pos_emb_type": "APE",
        "attn_type": "complex",
        "acceleration_factors": [4],
    },
]
