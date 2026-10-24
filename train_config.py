
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
# Shared window-expert settings (NOT submitted on their own). In the band-split runs the learnable gate sigmas collapsed
# to the lower bound, i.e. the window expert wrote all of k-space except a ~16 px disc inside the ACS region; these are
# the strong-window settings that came out of that.
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

# ---------------------------------------------------------------------------
# REFINE / CONSOLIDATE cascade: 3 window refinement blocks alternating with 3 band-split consolidation blocks, bands
# s1 = 0.30, s2 = 0.67 (the 3-stage 0.3 -> 1.5 geometric schedule), low -> high:
#   1 REFINE   window, writes only the centre disc          loss: disc  G(0.30)
#   2 CONSOL.  freq_split (axial | window, learnable sigma)  loss: disc  G(0.30)
#   3 REFINE   window, writes only the shell 0.30..0.67      loss: ring  G(0.67) - G(0.30)
#   4 CONSOL.  freq_split                                    loss: disc  G(0.67)
#   5 REFINE   window, writes only outside 0.67              loss: ring  1 - G(0.67)
#   6 CONSOL.  freq_split                                    loss: uniform
#   final output: plain complex L2.
# Refinement write gates are FIXED (max 1) so where a stage writes always matches where it is scored; the consolidation
# stages are unrestricted and their band-split gates stay learnable (init reversed 1.0 -> 0.15 over the 3 split stages).
# The two runs differ only in how the per-stage loss weights are normalised:
#   exp 0  max 1   -> a stage's term scales with its band's area (thin shells count little, the outside counts most)
#   exp 1  mean 1  -> area under the weight = const: every stage gets an equal vote regardless of band size
# ---------------------------------------------------------------------------
_S1, _S2 = 0.30, 0.67
_REFINE_CONSOL_BASE = {
    **_WINDOW_BASE,
    "encoders": ["window_axial", "freq_split"] * 3,
    "axial_d_model": 320,
    "axial_token_axis": "both",
    "axial_col_segments": 1,
    "freq_split_encoders": ("axial", "window_axial"),
    "freq_split_sigma_learned": True,
    "freq_split_sigma_schedule": "geometric",
    "freq_split_sigma_start": 1.0,
    "freq_split_sigma_end": 0.15,

    "stage_write_bands": [("disc", _S1), None, ("ring", _S1, _S2), None, ("ring", _S2, None), None],
    "loss_mode": "intermediate_banded",
    "banded_loss_bands": [("disc", _S1), ("disc", _S1), ("ring", _S1, _S2), ("disc", _S2), ("ring", _S2, None), ("all",)],
    "banded_loss_floor": 0.0,
}
EXPERIMENTS = [
    {**_REFINE_CONSOL_BASE, "name": "random_refine3consol3_win_d128_split_sigmarev_bandedmax_r4_100ep", "banded_loss_normalise": "max"},
    {**_REFINE_CONSOL_BASE, "name": "random_refine3consol3_win_d128_split_sigmarev_bandedmean_r4_100ep", "banded_loss_normalise": "mean"},
]
