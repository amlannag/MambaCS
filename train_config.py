"""
Experiment definitions for DcTNN training.
"""

EXPERIMENTS = [

    # ---------------------------------------------------------------------------
    # Lambda (data-consistency strength) experiments.
    # How the DC lambda is set (train_utils.build_model: learned iff schedule == "none"):
    #   lambda_schedule = "none"  -> learned per-stage nn.Parameter (init 0.5)
    #   lambda_schedule = "hard"  -> hard DC: measured points replaced exactly (no blend)
    #   lambda_schedule = "constant" -> constant lambda (e.g. lambda=0 -> no DC blend, model output retained on mask)
    #   lambda_schedule = "linear"/"cosine"/"constant" -> scheduled via LambdaScheduler
    #       (requires learned_lambda=False); epoch t in [0,1]:
    #       lambda = lambda_start + t * (lambda_end - lambda_start)
    # ---------------------------------------------------------------------------
    {
        "prefix": "fastmri_mag_lambda_zero",
        "name": "dctnn_axial_fastmri_mag_lambda_0_no_dc_r4",
        "hpc_backend": "amd",
        "model_type": "dctnn",
        "resume": None,
        "flattening_order": "row_major",
        "dataset": "fastmri",
        "image_size": (320, 320),
        "learning": "k_space",
        "norm": "fastmri_magnitude",
        "kspace_fill": None,
        "acceleration_factors": [4],
        "center_fractions": [0.08],
        "mask_type": "random",
        "encoders": ["axial"] * 3,
        "axial_row_stride": 1,
        "mask_vertical_attn": "none",
        "layer_no": 1,
        "num_encoder_layers": 2,
        "nhead_axial": 8,
        "layer_norm_eps": 1e-5,
        "attn_type": "complex",
        "pos_emb_type": "Rope-Axial",
        "rope_theta": 100.0,
        "lambda_schedule": "constant",
        "lambda_start": 0.0,
        "lambda_end": 0.0,
        "learned_lambda": False,
        "loss_mode": "intermediate_unweighted",
        "final_loss_type": "complex_l2",
        "intermediate_loss_type": "complex_l2",
        "epochs": 100,
        "batch_size": 32,
        "auto_batch_size": True,
        "batch_size_search_start": 128,
        "lr": 2e-4,
        "ffn_sharing": "global",
    },
    # ---------------------------------------------------------------------------
    # LORAKS C-matrix low-rank prior (DcTNN/loss.py: LoraksCLoss / ComplexL2LoraksLoss).
    # Final-stage loss = complex_l2 + loraks_weight * (tail energy of P_C(k_pred) beyond
    # loraks_rank) / (total energy). Intermediate stages keep plain complex_l2.
    # Hard-DC, fastmri_magnitude, no kspace_fill -- so the only change vs. the plain
    # complex_l2 baseline is the loss. radius=3 -> Nr=29; rank=None -> Nr//2=14, which was
    # the optimum in the notebooks/LORAKI.ipynb rank sweep (+4.4 dB over zero-fill).
    # Note: GT k-space itself has a tail ratio of ~0.14, so the term never hits zero;
    # keep the weight small (0.01-0.05) and drop it if PSNR regresses vs. the baseline.
    # ---------------------------------------------------------------------------
    {
        "prefix": "fastmri_mag_loraks",
        "name": "dctnn_axial_fastmri_mag_hard_dc_loraks_c_w0.05_r4",
        "hpc_backend": "amd",
        "model_type": "dctnn",
        "resume": None,
        "flattening_order": "row_major",
        "dataset": "fastmri",
        "image_size": (320, 320),
        "learning": "k_space",
        "norm": "fastmri_magnitude",
        "kspace_fill": None,
        "acceleration_factors": [4],
        "center_fractions": [0.08],
        "mask_type": "random",
        "encoders": ["axial"] * 3,
        "axial_row_stride": 1,
        "mask_vertical_attn": "none",
        "layer_no": 1,
        "num_encoder_layers": 2,
        "nhead_axial": 8,
        "layer_norm_eps": 1e-5,
        "attn_type": "complex",
        "pos_emb_type": "Rope-Axial",
        "rope_theta": 100.0,
        "lambda_schedule": "hard",
        "learned_lambda": False,
        "loss_mode": "intermediate_unweighted",
        "final_loss_type": "complex_l2_loraks",
        "intermediate_loss_type": "complex_l2",
        "loraks_weight": 0.05,
        "loraks_radius": 3,
        "loraks_rank": None,
        "loraks_normalize": "ratio",
        "epochs": 100,
        "batch_size": 32,
        "auto_batch_size": True,
        "batch_size_search_start": 128,
        "lr": 2e-4,
        "ffn_sharing": "global",
    },
]
