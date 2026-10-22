
# ---------------------------------------------------------------------------
# fastMRI single-coil knee, random R=4 (8% ACS), 100 epochs: band-split cascade only (no openers).
#   stage 1-4 freq_split      low expert  = multi-scale axial (column tokens split into 10 segments -> (10, 320) grid,
#                                           both axes, 2-D RoPE, 2-D Q/K conv scales 1 & 3)
#                             high expert = window_axial: full self-attention in shifted 5-segment x 16-column windows of
#                                           20-segment column tokens over sampled + unsampled columns, d_model 64 / 4 heads
#             combined as G*low + (1-G)*high with G = exp(-(r/sigma)^2/2); sigma (one per split stage) learned,
#             non-decreasing along the cascade, geometric init 0.15 -> 1.0
#   data consistency: one learned scalar lambda per stage
#   loss: complex L2 on the final output only
#   speed: ~0.72 s/step at batch 81 on an MI355X -> ~5 min/epoch -> ~8.5 h for 100 epochs
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "prefix": "fastmri",
        "name": "random_split4_msaxial_seg10_window20x5x16_scalardc_l2final_r4_100ep",
        "dataset": "fastmri",
        "num_channels": 1,
        "image_size": (320, 320),
        "mask_type": "random",
        "acceleration_factors": [4],
        "center_fractions": [0.08],
        "learning": "k_space",
        "norm": "fastmri_magnitude",

        "encoders": ["freq_split"] * 4,
        "axial_d_model": 320,
        "axial_token_axis": "both",
        "axial_col_segments": 10,            # each k-space column split into 10 segments of 32 rows -> (10, 320) token grid
        "attn_type": "complex_ms",            # multi-scale Q/K reaches the axial experts only; other stages use complex
        "attn_scales": (1, 3),
        "sdpa_dtype": "bfloat16",             # fused flash attention on ROCm (fp32 SDPA has no fused kernel there)
        "ffn_sharing": "global",

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
        "learned_lambda": True,
        "dc_type": "scalar",

        "final_loss_type": "complex_l2",
        "loss_mode": "final_only",

        "epochs": 100,
        "num_gpus": 2,                        # DDP via launch.sh / torchrun
        "batch_size_search_start": 256,       # global; the search starts from 256 / 2 = 128 per GPU
    },
]

# ---------------------------------------------------------------------------
# Ablation: same data / DC / loss / schedule, but a plain 4-stage multi-scale axial cascade (10 column segments)
# instead of the band-split stages -> isolates what the window_axial high-frequency expert adds.
# ---------------------------------------------------------------------------
_WINDOW_KEYS = {k for k in EXPERIMENTS[0] if k.startswith(("freq_split", "window_"))}
EXPERIMENTS.append({
    **{k: v for k, v in EXPERIMENTS[0].items() if k not in _WINDOW_KEYS},
    "name": "random_axial4_msaxial_seg10_scalardc_l2final_r4_100ep",
    "encoders": ["axial"] * 4,
    "num_gpus": 1,
    "batch_size_search_start": 128,       # per GPU (single process)
})
