
EXPERIMENTS = [
    # Image-domain CNN stem retest. The first attempt (imgconv32, 5 convs, zero-init output conv) collapsed: the hidden
    # convs got no data gradient behind the zero output conv and were erased by Adam's coupled weight decay
    # (lr*sign(w) per step), leaving an exact identity. Now gated by a scalar alpha = 0, glorot-init output conv, stem
    # params excluded from weight decay, smaller stem: IFFT(k) -> conv(1->16)+CReLU -> conv(16->16)+CReLU -> conv(16->1).
    {
        "prefix": "kspace_stem",
        "name": "axial_complex_imgconv16x3_gated_p95_slice_l2_final_r4",
        "image_conv": True,
        "image_conv_channels": 16,
        "image_conv_layers": 3,
        "image_conv_kernel": 3,
    },
    # AdamW control: the default axial k-space recipe (learned-lambda DC, final-only complex L2, no stems) with decoupled
    # weight decay instead of Adam's coupled L2, same wd = 1e-5. Matched baseline for any future AdamW stem runs.
    {
        "prefix": "optim",
        "name": "axial_complex_adamw_p95_slice_l2_final_r4",
        "optimizer_type": "adamw",
    },
]
