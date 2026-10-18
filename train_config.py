
EXPERIMENTS = [
    # cross_axial -> fnet -> 10 axial attention layers (5 axial stages x 2 layers), learned lambda, DcCNN-style
    # image-domain conv stem and an ungated 5x5 SPIRiT k-space conv (= image-domain modulation) at the start of
    # every stage.
    {
        "prefix": "brain",
        "name": "cross_fnet_axial5_2layers_multicoil8_imgconv_spirit5x5_ungated_learnedlambda_l2_r4_100epochs",
        "dataset": "fastmri_brain",
        "num_channels": 8,
        "image_size": (320, 320),
        "axial_d_model": 320,
        "axial_token_axis": "both",
        "num_gpus": 1,
        "batch_size": 64,
        "batch_size_search_start": 64,
        "encoders": ["cross_axial", "fnet"] + ["axial"] * 5,
        "num_encoder_layers": 2,
        "fnet_token_axis": "both",
        "fnet_with_embedding": True,
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
        "spirit_conv_mode": "ungated",
        "image_conv": True,
        "image_conv_channels": 32,
        "image_conv_layers": 3,
        "image_conv_kernel": 3,
    },
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
    dict(_NEIGHBOURHOOD_BASE, name="nbr_percoilcross_cross_axial3_r4"),
    dict(_NEIGHBOURHOOD_BASE, name="nbr3_axial3_r4", encoders=["neighbourhood"] * 3 + ["axial"] * 3),
    dict(_NEIGHBOURHOOD_BASE, name="cross3_axial3_r4", encoders=["cross_axial"] * 3 + ["axial"] * 3),
]
