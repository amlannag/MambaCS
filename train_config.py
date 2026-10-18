
EXPERIMENTS = [
]

# ---------------------------------------------------------------------------
# GRAPPA-style neighbourhood cross-attention (DcTNN/neighbourhood_attention.py). Three independent stage types,
# composed through `encoders`:
#   "neighbourhood"        per-coil local attention: each unsampled column attends to its 3 nearest acquired
#                          columns on each side in all 8 coils (whole-column tokens: strip_len = 320, kx_halo = 0)
#   "cross_axial_per_coil" global column cross-attention run on every coil independently (shared weights)
#   "cross_axial"          global column cross-attention with all coils folded into one token
# ---------------------------------------------------------------------------
_NEIGHBOURHOOD_BASE = {
    "prefix": "brain",
    "dataset": "fastmri_brain",
    "num_channels": 8,
    "image_size": (320, 320),
    "axial_d_model": 320,
    "axial_token_axis": "both",
    "num_gpus": 2,
    "batch_size": 64,
    "batch_size_search_start": 64,
    "encoders": ["neighbourhood", "cross_axial_per_coil", "cross_axial"] + ["axial"] * 3,
    "num_encoder_layers": 2,
    "attn_type": "complex",
    "ffn_sharing": "global",
    "learning": "k_space",
    "norm": "fastmri_magnitude",
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "mask_type": "random",
    "final_loss_type": "complex_l2",
    "epochs": 100,
    "neighbourhood_strip_len": 320,
    "neighbourhood_n_neighbours": 3,
    "neighbourhood_kx_halo": 0,
    "neighbourhood_attn_type": "complex",
}

EXPERIMENTS += [
    dict(_NEIGHBOURHOOD_BASE, name="nbr3_r4", encoders=["neighbourhood"] * 3),
    dict(_NEIGHBOURHOOD_BASE, name="cross3_r4", encoders=["cross_axial"] * 3),
    dict(_NEIGHBOURHOOD_BASE, name="crosspercoil3_r4", encoders=["cross_axial_per_coil"] * 3),
    dict(_NEIGHBOURHOOD_BASE, name="nbr3_cross3_crosspercoil3_axial3_r4",
         encoders=["neighbourhood"] * 3 + ["cross_axial"] * 3 + ["cross_axial_per_coil"] * 3 + ["axial"] * 3),
]
