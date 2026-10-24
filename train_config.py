
# ---------------------------------------------------------------------------
# fastMRI single-coil knee, random R=4 (8% ACS), 1 GPU x 100 epochs.
# Refine / consolidate cascade (W&B run q47g9huf, mean-1 banded loss normalisation) with two measured-only openers:
#   1 fnet          Fourier token mixing over row + column tokens          write: unrestricted         loss: uniform
#   2 cross_axial   sampled -> unsampled column cross-attention             write: unrestricted         loss: uniform
#   3 REFINE        window_axial, writes only the centre disc               write: disc  G(0.30)        loss: disc  G(0.30)
#   4 CONSOL.       freq_split (axial | window_axial, learnable sigma)       write: unrestricted         loss: disc  G(0.30)
#   5 REFINE        window_axial, writes only the shell 0.30..0.67           write: ring  0.30..0.67     loss: ring  0.30..0.67
#   6 CONSOL.       freq_split                                               write: unrestricted         loss: disc  G(0.67)
#   7 REFINE        window_axial, writes only outside 0.67                   write: ring  0.67..          loss: ring  0.67..
#   8 CONSOL.       freq_split                                               write: unrestricted         loss: uniform
#   final output: plain complex L2.
# window_axial: isotropic 4x4 k-space patch tokens (80x80 grid), d_model 128 / 8 heads, 4 encoder layers, 10x10-patch
# base windows growing x2 / x4 beyond r = 0.3 / 0.6, Swin half-window shift on odd layers.
# freq_split: low expert = multi-scale axial (whole-column tokens, both axes, Q/K conv scales 1 & 3), high expert =
# window_axial, gate sigma learnable, geometric init reversed 1.0 -> 0.15. One learned scalar lambda DC per stage.
# ---------------------------------------------------------------------------
FASTMRI_FNET_CROSS_REFINE3_CONSOL3 = {
    "prefix": "fastmri",
    "name": "random_fnet_cross_refine3consol3_win_d128_split_sigmarev_bandedmean_r4_100ep",

    # data
    "dataset": "fastmri",
    "num_channels": 1,
    "image_size": (320, 320),
    "mask_type": "random",
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "learning": "k_space",
    "norm": "fastmri_magnitude",

    # cascade
    "encoders": ["fnet", "cross_axial", "window_axial", "freq_split", "window_axial", "freq_split", "window_axial", "freq_split"],
    "stage_write_bands": [None, None, ("disc", 0.30), None, ("ring", 0.30, 0.67), None, ("ring", 0.67, None), None],
    "attn_type": "complex_ms",
    "attn_scales": (1, 3),
    "sdpa_dtype": "bfloat16",             # fused flash attention on ROCm (fp32 SDPA has no fused kernel there)
    "ffn_sharing": "global",

    # fnet
    "fnet_token_axis": "both",
    "fnet_with_embedding": True,

    # axial (cross_axial opener + freq_split low expert)
    "axial_d_model": 320,
    "axial_token_axis": "both",
    "axial_col_segments": 1,

    # window_axial (refine stages + freq_split high expert)
    "window_patch_size": (4, 4),
    "window_num_layers": 4,
    "window_height": 10,
    "window_width": 10,
    "window_shift": True,
    "window_radial_scales": (1, 2, 4),
    "window_radial_bands": (0.3, 0.6),
    "window_d_model": 128,
    "window_nhead": 8,

    # freq_split
    "freq_split_encoders": ("axial", "window_axial"),
    "freq_split_sigma_learned": True,
    "freq_split_sigma_schedule": "geometric",
    "freq_split_sigma_start": 1.0,
    "freq_split_sigma_end": 0.15,

    # data consistency
    "lambda_schedule": "none",
    "learned_lambda": True,
    "dc_type": "scalar",

    # loss
    "final_loss_type": "complex_l2",
    "intermediate_loss_type": "complex_l2",
    "loss_mode": "intermediate_banded",
    "banded_loss_bands": [("all",), ("all",), ("disc", 0.30), ("disc", 0.30), ("ring", 0.30, 0.67), ("disc", 0.67), ("ring", 0.67, None), ("all",)],
    "banded_loss_normalise": "mean",
    "banded_loss_floor": 0.0,

    # training
    "epochs": 100,
    "num_gpus": 1,
    "batch_size_search_start": 128,
}

# ---------------------------------------------------------------------------
# fastMRI brain multicoil, equispaced R=4 (8% ACS), 8 virtual coils (SVD coil compression), 2 GPUs x 100 epochs.
# The refine / consolidate cascade of the knee experiment above (3x REFINE window / CONSOL. freq_split with fixed write
# gates and the mean-1 banded loss) behind the multicoil openers of the trained brain run 8r0qynli:
#   1 fnet                  Fourier token mixing over row + column tokens          write: unrestricted   loss: uniform
#   2 neighbourhood         learned-GRAPPA cross-attention from acquired columns    write: unrestricted   loss: uniform
#                           (strip 40, 2 nbrs/side, no halo)
#   3 cross_axial_per_coil  sampled -> unsampled column cross-attention on every   write: unrestricted   loss: uniform
#                           coil (shared weights + coil-ID embedding)
#   4-9                     REFINE / CONSOL. x3 as in the knee run
# Warm start: stages 0 (fnet) and 1 (neighbourhood) are copied from that run's checkpoint
# (brain_equi_fnet_percoil_nbr_axial3_seg4_apt3_ms3_pixdc_r4_100ep, same stages at the same indices) and stay trainable.
# Global batch must divide by num_gpus; 8 coils = 8x the tokens of the knee runs, so the batch search starts at 16/GPU.
# ---------------------------------------------------------------------------
BRAIN_FNET_CROSS_REFINE3_CONSOL3 = {
    "prefix": "brain",
    "name": "equi_fnet_nbr_percoil_refine3consol3_win_d128_split_sigmarev_bandedmean_r4_100ep_fnetnbrinit",

    # data
    "dataset": "fastmri_brain",
    "num_channels": 8,
    "image_size": (320, 320),
    "mask_type": "equispaced",
    "acceleration_factors": [4],
    "center_fractions": [0.08],
    "learning": "k_space",
    "norm": "fastmri_magnitude",

    # warm start
    "init_stages_from": "../Experiments/brain_equi_fnet_percoil_nbr_axial3_seg4_apt3_ms3_pixdc_r4_100ep/best_model.pth",
    "init_stages": [0, 1],

    # cascade
    "encoders": ["fnet", "neighbourhood", "cross_axial_per_coil", "window_axial", "freq_split", "window_axial", "freq_split", "window_axial", "freq_split"],
    "stage_write_bands": [None, None, None, ("disc", 0.30), None, ("ring", 0.30, 0.67), None, ("ring", 0.67, None), None],
    "attn_type": "complex_ms",
    "attn_scales": (1, 3),
    "sdpa_dtype": "bfloat16",             # fused flash attention on ROCm (fp32 SDPA has no fused kernel there)
    "ffn_sharing": "global",

    # fnet
    "fnet_token_axis": "both",
    "fnet_with_embedding": True,

    # neighbourhood
    "neighbourhood_strip_len": 40,
    "neighbourhood_n_neighbours": 2,
    "neighbourhood_kx_halo": 0,
    "neighbourhood_attn_type": "complex",

    # axial (cross_axial_per_coil opener + freq_split low expert)
    "axial_d_model": 320,
    "axial_token_axis": "both",
    "axial_col_segments": 1,

    # window_axial (refine stages + freq_split high expert)
    "window_patch_size": (4, 4),
    "window_num_layers": 4,
    "window_height": 10,
    "window_width": 10,
    "window_shift": True,
    "window_radial_scales": (1, 2, 4),
    "window_radial_bands": (0.3, 0.6),
    "window_d_model": 128,
    "window_nhead": 8,

    # freq_split
    "freq_split_encoders": ("axial", "window_axial"),
    "freq_split_sigma_learned": True,
    "freq_split_sigma_schedule": "geometric",
    "freq_split_sigma_start": 1.0,
    "freq_split_sigma_end": 0.15,

    # data consistency
    "lambda_schedule": "none",
    "learned_lambda": True,
    "dc_type": "scalar",

    # loss
    "final_loss_type": "complex_l2",
    "intermediate_loss_type": "complex_l2",
    "loss_mode": "intermediate_banded",
    "banded_loss_bands": [("all",), ("all",), ("all",), ("disc", 0.30), ("disc", 0.30), ("ring", 0.30, 0.67), ("disc", 0.67), ("ring", 0.67, None), ("all",)],
    "banded_loss_normalise": "mean",
    "banded_loss_floor": 0.0,

    # training
    "epochs": 100,
    "num_gpus": 2,
    "batch_size": 32,
    "batch_size_search_start": 32,
}

EXPERIMENTS = [
    FASTMRI_FNET_CROSS_REFINE3_CONSOL3,
    BRAIN_FNET_CROSS_REFINE3_CONSOL3,
]
