
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
# Exp 0: learned-GRAPPA neighbourhood cross-attention opener (acquired columns only: strip 16, 2 neighbours per side,
# kx halo 1) followed by 3 axial stages that each run rows, columns AND the 320 half-spoke radial pass.
#   NOTE: the neighbourhood stage costs ~1.6 s/step at batch 81 on an MI355X (7x the three axial stages together).
# ---------------------------------------------------------------------------
EXPERIMENTS = [{
    **_AXIAL_RADIAL_BASE,
    "name": "random_nbr_axial3_msaxial_seg1_radial320_scalardc_l2final_r4_100ep",
    "encoders": ["neighbourhood"] + ["axial"] * 3,
    "neighbourhood_strip_len": 16,
    "neighbourhood_n_neighbours": 2,
    "neighbourhood_kx_halo": 1,
    "neighbourhood_attn_type": "complex",
}]

# ---------------------------------------------------------------------------
# Exp 1: 4 axial stages with SEGMENTED tokens on both the Cartesian and the radial pass: every column cut into 4
# segments of 80 rows ((4, 320) token grid, 2-D RoPE, 2-D Q/K convs) and every half-spoke cut into 3 radial segments
# of 53 samples ((320, 3) token grid, 2-D RoPE at the segment's k-space position; the outermost radius is dropped).
# ---------------------------------------------------------------------------
EXPERIMENTS.append({
    **_AXIAL_RADIAL_BASE,
    "name": "random_axial4_msaxial_seg4_radial320_rseg3_scalardc_l2final_r4_100ep",
    "axial_col_segments": 4,
    "radial_segments": 3,
})

# ---------------------------------------------------------------------------
# Exp 2: band-split cascade, reversed gates + LOW -> HIGH banded loss, with the strong high expert.
#   stage 1-4 freq_split    low expert  = multi-scale axial, whole-column tokens, d_model 320
#                           high expert = window attention over isotropic 4x4 k-space patch tokens (80x80 grid), 4 encoder
#                                         layers per stage, 10x10-patch base windows (40x40 k-space) growing to 20x20
#                                         patches beyond r = 0.6, Swin half-window shift on odd layers, d_model 128 / 8 heads
#   gates: G = exp(-(r/sigma)^2/2), sigma LEARNABLE, init reversed 1.0 -> 0.15 (global axial owns k-space early, the window
#          expert takes over the high band in the later stages); the gates are not used by the loss
#   loss:  final complex L2 + per-stage complex L2 weighted by the fixed discs G(r; s_i), s = 0.3 -> 1.5 geometric
#          (stage 1 answers for the k-space centre, each later stage for a wider disc) -- low -> high supervision
#   FFNs:  one shared across the axial experts, a separate one across the window experts (ffn_sharing="global")
# ---------------------------------------------------------------------------
EXPERIMENTS.append({
    **{k: v for k, v in _AXIAL_RADIAL_BASE.items() if not k.startswith("radial_")},
    "name": "random_split4_msaxial_seg1_winpatch4_radial10x10_l4_d128_sigmarev_bandedlow_r4_100ep",
    "encoders": ["freq_split"] * 4,
    "axial_token_axis": "both",
    "freq_split_encoders": ("axial", "window_axial"),
    "freq_split_sigma_learned": True,
    "freq_split_sigma_schedule": "geometric",
    "freq_split_sigma_start": 1.0,          # reversed: large low band first ...
    "freq_split_sigma_end": 0.15,           # ... small low band last

    "window_patch_size": (4, 4),
    "window_num_layers": 4,
    "window_height": 10,
    "window_width": 10,
    "window_shift": True,
    "window_radial_scales": (1, 2, 4),
    "window_radial_bands": (0.3, 0.6),
    "window_d_model": 128,
    "window_nhead": 8,                     # head_dim 16, twice the attention patterns of the 64 / 4 expert

    "loss_mode": "intermediate_banded",
    "intermediate_loss_type": "complex_l2",
    "banded_loss_band": "low",
    "banded_loss_sigma_start": 0.3,
    "banded_loss_sigma_end": 1.5,
    "banded_loss_sigma_schedule": "geometric",
    "banded_loss_floor": 0.0,
})
