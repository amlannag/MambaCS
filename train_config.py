
# ---------------------------------------------------------------------------
# fastMRI single-coil knee, random R=4 (8% ACS), 400 epochs: FNet + measured-only openers followed by a band-split cascade.
#   stage 1   fnet            FNet Fourier token mixing over row + column tokens (parameter-free mixing, per-token FFNs)
#   stage 2   cross_axial     sampled -> unsampled column cross-attention
#   stage 3   neighbourhood   learned-GRAPPA cross-attention from acquired columns only (strip 16, 2 nbrs/side, kx halo 1)
#   stage 4-7 freq_split      low expert  = multi-scale axial (whole-column tokens, both axes, Q/K conv scales 1 & 3)
#                             high expert = window_axial: full self-attention in shifted 5-segment x 16-column windows of
#                                           20-segment column tokens over sampled + unsampled columns, d_model 64 / 4 heads
#             combined as G*low + (1-G)*high with G = exp(-(r/sigma)^2/2); sigma (one per split stage) learned,
#             non-decreasing along the cascade, geometric init 0.15 -> 1.0
#   data consistency: one learned weight per k-space pixel and stage (pixelwise, lambda init 1.0)
#   loss: complex L2 on every post-DC stage output and on the final output (intermediate_unweighted)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "prefix": "fastmri",
        "name": "random_fnet_cross_nbr_split4_msaxial_window20x5x16_pixdc_r4_400ep",
        "dataset": "fastmri",
        "num_channels": 1,
        "image_size": (320, 320),
        "mask_type": "random",
        "acceleration_factors": [4],
        "center_fractions": [0.08],
        "learning": "k_space",
        "norm": "fastmri_magnitude",

        "encoders": ["fnet", "cross_axial", "neighbourhood"] + ["freq_split"] * 4,
        "axial_d_model": 320,
        "axial_token_axis": "both",
        "axial_col_segments": 1,
        "attn_type": "complex_ms",            # multi-scale Q/K reaches the axial experts only; other stages use complex
        "attn_scales": (1, 3),
        "sdpa_dtype": "bfloat16",             # fused flash attention on ROCm (fp32 SDPA has no fused kernel there)
        "ffn_sharing": "global",

        "fnet_token_axis": "both",
        "fnet_with_embedding": True,

        "neighbourhood_strip_len": 16,
        "neighbourhood_n_neighbours": 2,
        "neighbourhood_kx_halo": 1,
        "neighbourhood_attn_type": "complex",

        "freq_split_encoders": ("axial", "window_axial"),
        "freq_split_sigma_learned": True,
        "freq_split_sigma_schedule": "geometric",
        "freq_split_sigma_start": 0.15,
        "freq_split_sigma_end": 1.0,

        "window_col_segments": 20,
        "window_height": 5,
        "window_width": 16,
        "window_shift": True,
        "window_d_model": 64,
        "window_nhead": 4,

        "lambda_schedule": "none",
        "dc_type": "pixelwise",
        "dc_lambda_init": 1.0,

        "final_loss_type": "complex_l2",
        "intermediate_loss_type": "complex_l2",
        "loss_mode": "intermediate_unweighted",

        "epochs": 400,
    },
]

# ---------------------------------------------------------------------------
# Speed test: experiment 0 on a 20-volume subset, 1 epoch, fixed batch 81 (no batch-size search), with the fused
# attention in bf16 (exp 1) vs fp32 (exp 2). Run either as a short training job or with `--profile 5` for the
# per-stage breakdown; compare the "batch N/..: step ... compute ..." lines between the two.
# ---------------------------------------------------------------------------
_SPEED_TEST_BASE = {
    **EXPERIMENTS[0],
    "max_train_files": 20,
    "max_val_files": 2,
    "auto_batch_size": False,
    "batch_size": 81,
    "epochs": 1,
    "final_full_validation": False,   # skip the end-of-training inference over all 199 val volumes
}
EXPERIMENTS += [
    {**_SPEED_TEST_BASE, "name": "speedtest_sdpa_bf16", "sdpa_dtype": "bfloat16"},
    {**_SPEED_TEST_BASE, "name": "speedtest_sdpa_fp32", "sdpa_dtype": "float32"},
    # exp 3: how fast is a plain 3-stage axial cascade (same axial settings as the freq_split low expert)?
    {**_SPEED_TEST_BASE, "name": "speedtest_axial3", "encoders": ["axial"] * 3},
    # exp 4: the 4 freq_split stages (multi-scale axial + window_axial experts) alone, no openers
    {**_SPEED_TEST_BASE, "name": "speedtest_split4", "encoders": ["freq_split"] * 4},
    # exp 5: the single neighbourhood stage alone
    {**_SPEED_TEST_BASE, "name": "speedtest_nbr1", "encoders": ["neighbourhood"]},
]
