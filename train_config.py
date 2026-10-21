
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

# ---------------------------------------------------------------------------
# fastMRI single-coil knee (random R=4 masks): local neighbourhood attention -> 3x axial. With one coil every
# unsampled (kx-strip, column) token attends to the 2 nearest acquired columns on each side in the strips
# s-1..s+1 — plain local k-space attention. Append to EXPERIMENTS (optionally with overrides) to queue it.
# ---------------------------------------------------------------------------
_FASTMRI_NBR_AXIAL3 = {
    "prefix": "fastmri",
    "dataset": "fastmri",
    "num_channels": 1,
    "image_size": (320, 320),
    "axial_d_model": 320,
    "axial_token_axis": "both",
    "mask_type": "random",
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "encoders": ["neighbourhood", "axial", "axial", "axial"],
    "neighbourhood_strip_len": 16,
    "neighbourhood_n_neighbours": 2,
    "neighbourhood_kx_halo": 1,
    "neighbourhood_attn_type": "complex",
    "attn_type": "complex",
    "ffn_sharing": "global",
    "learning": "k_space",
    "norm": "fastmri_magnitude",
    "final_loss_type": "complex_l2",
    "loss_mode": "final_only",
    "lambda_schedule": "none",
}

# ---------------------------------------------------------------------------
# fastMRI single-coil knee: band-split cascade. Every stage is a freq_split stage = two experts combined by a radial
# Gaussian partition of unity G*low + (1-G)*high (low expert = axial, high expert = neighbourhood by default). sigma is
# one value per stage, non-decreasing along the cascade, initialised geometrically 0.15 -> 1.0 and learned.
# Matched baseline: _FASTMRI_NBR_AXIAL3 (same encoder families, no split). Append to EXPERIMENTS to queue.
# ---------------------------------------------------------------------------
_FASTMRI_SPLIT4 = dict(
    _FASTMRI_NBR_AXIAL3,
    encoders=["freq_split"] * 4,
    freq_split_encoders=("axial", "neighbourhood"),
    freq_split_sigma_learned=True,
    freq_split_sigma_schedule="geometric",
    freq_split_sigma_start=0.15,
    freq_split_sigma_end=1.0,
)
# ablations: fixed (not learned) boundaries, and learned boundaries that all start at the same radius (no progression)
_FASTMRI_SPLIT4_FIXED = dict(_FASTMRI_SPLIT4, freq_split_sigma_learned=False)
_FASTMRI_SPLIT4_FLAT = dict(_FASTMRI_SPLIT4, freq_split_sigma_init=[0.3, 0.3, 0.3, 0.3])

# ---------------------------------------------------------------------------
# fastMRI single-coil knee, random R=4: FNet + measured-only openers followed by a band-split cascade.
#   stage 1  fnet                 FNet Fourier token mixing (row + column tokens)
#   stage 2  cross_axial          sampled -> unsampled column cross-attention
#   stage 3  neighbourhood        learned-GRAPPA cross-attention from acquired columns only (strip 16, 2 nbrs/side, kx halo 1)
#   stage 4-7 freq_split          low expert  = multi-scale axial (whole-column tokens, both axes, Q/K scales 1 & 3)
#                                 high expert = window_axial: full self-attention in shifted 5-segment x 16-column windows
#                                               of 20-segment column tokens, sampled + unsampled, d_model 64 / 4 heads
#   sigma (one per split stage) learned, non-decreasing along the cascade, geometric init 0.15 -> 1.0.
# ---------------------------------------------------------------------------
_FASTMRI_NBR_CROSS_SPLIT4 = dict(
    _FASTMRI_NBR_AXIAL3,
    encoders=["fnet", "cross_axial", "neighbourhood"] + ["freq_split"] * 4,
    fnet_token_axis="both", fnet_with_embedding=True,    # parameter-free Fourier token mixing over row + column tokens
    attn_type="complex_ms", attn_scales=(1, 3),          # multi-scale Q/K reaches the axial experts only
    axial_col_segments=1,
    dc_type="pixelwise", dc_lambda_init=1.0,             # one learned DC weight per k-space pixel and stage (needs lambda_schedule="none")
    loss_mode="intermediate_unweighted",                 # complex-L2 on every post-DC stage output + the final output
    freq_split_encoders=("axial", "window_axial"),
    freq_split_sigma_learned=True,
    freq_split_sigma_schedule="geometric",
    freq_split_sigma_start=0.15,
    freq_split_sigma_end=1.0,
    window_col_segments=20, window_height=5, window_width=16, window_shift=True,
    window_d_model=64, window_nhead=4,
    neighbourhood_strip_len=16, neighbourhood_n_neighbours=2, neighbourhood_kx_halo=1,
)

EXPERIMENTS += [

    dict(_FASTMRI_NBR_CROSS_SPLIT4, name="random_fnet_cross_nbr_split4_msaxial_window20x5x16_pixdc_r4"),

    # 2) + per-coil cross-attention stage after the FNet stage, intermediate (per-stage) loss, 3 fixed-APT stages
    #    after the axial stages, and multi-scale complex attention with kernel scales (1, 3)
    dict(_BRAIN_FNET_NBR_AXIAL3, name="equi_fnet_percoil_nbr_axial3_seg4_apt3_ms3_pixdc_r4_100ep",
         encoders=["fnet","neighbourhood", "cross_axial_per_coil"] + ["axial"] * 3 + ["fixed_apt"] * 3,
         loss_mode="intermediate_unweighted",
         attn_type="complex_ms", attn_scales=(1, 3),
         # continue the run cut off by the 24h limit (job 30218089); same W&B run as the first leg
         resume="../Experiments/brain_equi_fnet_percoil_nbr_axial3_seg4_apt3_ms3_pixdc_r4_100ep/latest.pth",
         wandb_run_id="8r0qynli"),
]
