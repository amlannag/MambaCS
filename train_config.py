
EXPERIMENTS = [
]

# ---------------------------------------------------------------------------
# GRAPPA-style neighbourhood cross-attention ablations (DcTNN/neighbourhood_attention.py).
# Base: 3 cross_axial_neighbourhood stages followed by 3 axial stages, 8 virtual coils, R=4 random masks,
# per-coil tokens S=16, n=3, h=1.
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
    "encoders": ["cross_axial_neighbourhood"] * 3 + ["axial"] * 3,
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
    "neighbourhood_strip_len": 16,
    "neighbourhood_n_neighbours": 3,
    "neighbourhood_kx_halo": 1,
    "neighbourhood_attn_type": "complex",
    "neighbourhood_combine": "sum",
}

EXPERIMENTS += [
    dict(_NEIGHBOURHOOD_BASE, name="nbr_percoil_S16_n3_h1_sum_r4"),
    dict(_NEIGHBOURHOOD_BASE, name="nbr_local_only_r4", neighbourhood_combine="local_only"),
]
