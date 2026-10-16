
EXPERIMENTS = [
    {
        "prefix": "brain",
        "name": "axial3_2layers_multicoil8_spirit5x5_gated_l2_r4_100epochs",
        "dataset": "fastmri_brain",
        "num_channels": 8,
        "image_size": (320, 320),
        "axial_d_model": 320,
        "axial_token_axis": "both",
        "num_gpus": 1,
        "batch_size": 64,
        "batch_size_search_start": 64,
        "encoders": ["axial", "axial", "axial"],
        "num_encoder_layers": 2,
        "attn_type": "complex_ms",
        "attn_scales": (1,),
        "lambda_schedule": "none",
        "learned_lambda": True,
        "ffn_sharing": "global",
        "learning": "k_space",
        "norm": "fastmri_magnitude",
        "kspace_fill": None,
        "acceleration_factors": [4],
        "center_fractions": [0.08],
        "mask_type": "random",
        "loss_mode": "final_only",
        "final_loss_type": "complex_l2",
        "intermediate_loss_type": "complex_l2",
        "epochs": 100,
        "global_filter": False,
        "global_filter_mid": False,
        "kspace_conv": False,
        "spirit_conv": True,
        "spirit_conv_kernels": (5,),
        "spirit_conv_mode": "gated",
        "image_conv": False,
    },
]

# ---------------------------------------------------------------------------
# GRAPPA-style neighbourhood cross-attention ablations (DcTNN/neighbourhood_attention.py).
# Base: 3 cross_axial_neighbourhood stages on 8 virtual coils, R=4 random masks, per-coil tokens S=16, n=3, h=1.
# ---------------------------------------------------------------------------
_NEIGHBOURHOOD_BASE = {
    "prefix": "brain",
    "dataset": "fastmri_brain",
    "num_channels": 8,
    "image_size": (320, 320),
    "axial_d_model": 320,
    "num_gpus": 1,
    "batch_size": 16,
    "batch_size_search_start": 16,
    "encoders": ["cross_axial_neighbourhood"] * 3,
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
    # (a) per-coil local + global, S=16 n=3 h=1
    dict(_NEIGHBOURHOOD_BASE, name="nbr_percoil_S16_n3_h1_sum_r4"),
    # (c) kx halo
    dict(_NEIGHBOURHOOD_BASE, name="nbr_percoil_h0_r4", neighbourhood_kx_halo=0),
    dict(_NEIGHBOURHOOD_BASE, name="nbr_percoil_h2_r4", neighbourhood_kx_halo=2),
    # (d) ky neighbours per side
    dict(_NEIGHBOURHOOD_BASE, name="nbr_percoil_n2_r4", neighbourhood_n_neighbours=2),
    dict(_NEIGHBOURHOOD_BASE, name="nbr_percoil_n4_r4", neighbourhood_n_neighbours=4),
    dict(_NEIGHBOURHOOD_BASE, name="nbr_percoil_n6_r4", neighbourhood_n_neighbours=6),
    # (e) global only / local only / both
    dict(_NEIGHBOURHOOD_BASE, name="nbr_global_only_r4", encoders=["cross_axial"] * 3),
    dict(_NEIGHBOURHOOD_BASE, name="nbr_local_only_r4", neighbourhood_combine="local_only"),
    dict(_NEIGHBOURHOOD_BASE, name="nbr_both_gate_r4", neighbourhood_combine="gate"),
    dict(_NEIGHBOURHOOD_BASE, name="nbr_both_sequential_r4", neighbourhood_combine="sequential"),
]
