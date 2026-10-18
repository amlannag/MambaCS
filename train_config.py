
EXPERIMENTS = [
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
    "num_gpus": 1,
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
    
    dict(_NEIGHBOURHOOD_BASE, name="equi_grappa_axial3_r4",
         mask_type="equispaced", kspace_fill="grappa", grappa_phase_lines=2, grappa_readout_width=3,
         grappa_ridge=1e-1, encoders=["axial"] * 3),
    # Fixed equispaced mask, zero-filled input: fnet -> neighbourhood attention (2 acquired columns per side,
    # 8 kx strips of 40 samples per column, 32 keys per query) -> 3 axial stages.
    dict(_NEIGHBOURHOOD_BASE, name="equi_fnet_nbr_n2_s8_axial3_r4",
         mask_type="equispaced", fnet_token_axis="both", fnet_with_embedding=True,
         encoders=["fnet", "neighbourhood"] + ["axial"] * 3,
         neighbourhood_n_neighbours=2, neighbourhood_strip_len=40, neighbourhood_kx_halo=0),
]
