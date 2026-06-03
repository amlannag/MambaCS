"""
Experiment definitions for DcTNN training.

Each entry in EXPERIMENTS is a dict of Config field overrides.
Any key not listed falls back to the Config class default in config.py.
"""

EXPERIMENTS = [
    # {
    #     "prefix": "FullComplex",
    #     "name": "patch_rope_axial",
    #     "encoders": ["patch", "patch", "patch"],
    #     "k_space_learning": True,
    #     "pos_emb_type": "Rope-Axial",
    #     "attn_type": "complex",
    # },
    {
        "prefix": "FullComplex",
        "name": "axial_rope_axial_complexattn",
        "encoders": ["axial", "axial", "axial"],
        "k_space_learning": True,
        "pos_emb_type": "Rope-Axial",
        "attn_type": "complex",
    }
    #,
    # {
    #     "prefix": "FullComplex",
    #     "name": "kaleidoscope_rope_axial",
    #     "encoders": ["kaleidoscope", "kaleidoscope", "kaleidoscope"],
    #     "k_space_learning": True,
    #     "pos_emb_type": "Rope-Axial",
    #     "attn_type": "complex",
    # },
]
