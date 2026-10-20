
EXPERIMENTS = [
]

# ---------------------------------------------------------------------------
# Brain multicoil, 3 GPUs x 50 epochs. Base = brain_equi_fnet_nbr_n2_s8_axial3_r4 (equispaced R=4, 8 virtual coils,
# FNet -> neighbourhood -> 3x axial, complex-L2 final-only) plus a learned DC weight for every k-space pixel and
# axial columns split into 4 segment tokens.
# ---------------------------------------------------------------------------
_BRAIN_FNET_NBR_AXIAL3 = {
    "prefix": "brain",
    "dataset": "fastmri_brain",
    "num_channels": 8,
    "image_size": (320, 320),
    "axial_d_model": 320,
    "axial_token_axis": "both",
    "mask_type": "equispaced",
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "encoders": ["fnet", "neighbourhood", "axial", "axial", "axial"],
    "fnet_token_axis": "both",
    "fnet_with_embedding": True,
    "neighbourhood_strip_len": 40,
    "neighbourhood_n_neighbours": 2,
    "neighbourhood_kx_halo": 0,
    "neighbourhood_attn_type": "complex",
    "attn_type": "complex",
    "ffn_sharing": "global",
    "learning": "k_space",
    "norm": "fastmri_magnitude",
    "final_loss_type": "complex_l2",
    "loss_mode": "final_only",
    "lambda_schedule": "none",
    "dc_type": "pixelwise",
    "dc_lambda_init": 1.0,
    # every k-space column is split into 4 vertical segments, each an axial token (4 x 320 tokens on a (4, 320) grid;
    # 2-D RoPE, and 2-D multi-scale Q/K convs for attn_type="complex_ms" -> scale-3 heads see left/right AND up/down)
    "axial_col_segments": 4,
    "num_gpus": 3,
    # Global batch (must divide by num_gpus). Whole-column tokens filled one GPU at 64; with 4 segments the vertical
    # attention has 4x the tokens (16x the score matrix), so start the automatic search at 16 per GPU and let it
    # back off on OOM (auto_batch_size=True).
    "batch_size": 48,
    "batch_size_search_start": 48,
    "epochs": 100,
}

EXPERIMENTS += [

    # 2) + per-coil cross-attention stage after the FNet stage, intermediate (per-stage) loss, 3 fixed-APT stages
    #    after the axial stages, and multi-scale complex attention with kernel scales (1, 3)
    dict(_BRAIN_FNET_NBR_AXIAL3, name="equi_fnet_percoil_nbr_axial3_seg4_apt3_ms3_pixdc_r4_100ep",
         encoders=["fnet","neighbourhood", "cross_axial_per_coil"] + ["axial"] * 3 + ["fixed_apt"] * 3,
         loss_mode="intermediate_unweighted",
         attn_type="complex_ms", attn_scales=(1, 3)),
]
