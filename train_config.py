
# ---------------------------------------------------------------------------
# Shared settings (NOT submitted on their own): fastMRI single-coil knee, random R=4 (8% ACS), 100 epochs,
# multi-scale axial stages with three passes each (rows, columns, and the 320 half-spoke RADIAL pass added as a
# residual to the column pass), d_model 320, Q/K conv scales 1 & 3, scalar learned lambda DC, final-only complex L2.
# The axial4 + radial run itself (random_axial4_msaxial_seg1_radial320_...) has already been trained.
# ---------------------------------------------------------------------------
_AXIAL_RADIAL_BASE = {
    "prefix": "fastmri",
    "name": "random_axial4_msaxial_seg1_radial320_scalardc_l2final_r4_100ep",
    "dataset": "fastmri",
    "num_channels": 1,
    "image_size": (320, 320),
    "mask_type": "random",
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "learning": "k_space",
    "norm": "fastmri_magnitude",

    "encoders": ["axial"] * 4,
    "axial_d_model": 320,
    "axial_token_axis": "both+radial",
    "radial_num_spokes": 320,
    "radial_spacing": 1.0,
    "axial_col_segments": 1,              # whole-column tokens
    "attn_type": "complex_ms",
    "attn_scales": (1, 3),
    "sdpa_dtype": "bfloat16",             # fused flash attention on ROCm (fp32 SDPA has no fused kernel there)
    "ffn_sharing": "global",

    "lambda_schedule": "none",
    "learned_lambda": True,
    "dc_type": "scalar",

    "final_loss_type": "complex_l2",
    "loss_mode": "final_only",

    "epochs": 100,
    "num_gpus": 1,
    "batch_size_search_start": 128,
}

# ---------------------------------------------------------------------------
# Pure window-attention cascade (no axial expert, no gates): in the band-split runs the learnable gate sigmas collapsed
# to the lower bound, i.e. the window expert wrote all of k-space except a ~16 px disc inside the ACS region, so the
# axial expert (>90% of the parameters and compute) contributed nothing. These three runs make that model explicit and
# ablate the loss curriculum on it.
#   stage 1-4 window_axial   isotropic 4x4 k-space patch tokens (80x80 grid), d_model 128 / 8 heads, 4 encoder layers,
#                            10x10-patch base windows (40x40 k-space) growing to 20x20 patches beyond r = 0.6,
#                            Swin half-window shift on odd layers; one FFN shared across the window stages
#   data consistency: one learned scalar lambda per stage
#   exp 0  standard : final complex L2 only
#   exp 1  low      : + per-stage complex L2 weighted by growing discs G(r; s_i), s = 0.3 -> 1.5 (low -> high, the high
#                     band is included late but never emphasised)
#   exp 2  ring     : + per-stage complex L2 weighted by the annulus each stage newly owns (centre disc, shell 0.3..0.51,
#                     shell 0.51..0.88, everything outside 0.88) -- ends by prioritising the high frequencies
# ---------------------------------------------------------------------------
_WINDOW_BASE = {
    **{k: v for k, v in _AXIAL_RADIAL_BASE.items() if not k.startswith(("radial_", "axial_"))},
    "encoders": ["window_axial"] * 4,

    "window_patch_size": (4, 4),
    "window_num_layers": 4,
    "window_height": 10,
    "window_width": 10,
    "window_shift": True,
    "window_radial_scales": (1, 2, 4),
    "window_radial_bands": (0.3, 0.6),
    "window_d_model": 128,
    "window_nhead": 8,

    "intermediate_loss_type": "complex_l2",
    "banded_loss_sigma_start": 0.3,
    "banded_loss_sigma_end": 1.5,
    "banded_loss_sigma_schedule": "geometric",
    "banded_loss_floor": 0.0,
}

EXPERIMENTS = [
    {**_WINDOW_BASE, "name": "random_window4_patch4_radial10x10_l4_d128_l2final_r4_100ep",
     "loss_mode": "final_only"},
    {**_WINDOW_BASE, "name": "random_window4_patch4_radial10x10_l4_d128_bandedlow_r4_100ep",
     "loss_mode": "intermediate_banded", "banded_loss_band": "low"},
    {**_WINDOW_BASE, "name": "random_window4_patch4_radial10x10_l4_d128_bandedring_r4_100ep",
     "loss_mode": "intermediate_banded", "banded_loss_band": "ring"},
]

# exp 3  ring, unnormalised: the raw ring bands form a partition of unity (they sum to 1 at every cell), so the stage
#        terms are an unbiased split of the plain loss across the cascade - no spectral re-weighting, pure
#        "who is responsible for what". Compare with exp 2 (mean-1: equal vote per stage, high band emphasised).
EXPERIMENTS.append({**_WINDOW_BASE, "name": "random_window4_patch4_radial10x10_l4_d128_bandedring_pou_r4_100ep",
                    "loss_mode": "intermediate_banded", "banded_loss_band": "ring", "banded_loss_normalise": "none"})
