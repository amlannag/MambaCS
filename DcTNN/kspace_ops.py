"""
Learnable k-space pre-processing stems applied to the encoder input before tokenisation.
All are exact identities at initialisation so an existing recipe is unchanged until they learn: GlobalFilter
starts at W = 1, the residual conv stems through a scalar gate alpha = 0 (not a zero output conv, which would
leave the branch without gradient and let Adam's coupled weight decay collapse it). Stem parameters must be
excluded from weight decay (train._optimizer_param_groups).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_init import trabelsi_init_
from .dc import fft_2d, ifft_2d, complex_to_pair, pair_to_complex
from .util import ComplexGELU, ComplexReLU


class GlobalFilter(nn.Module):
    """
    Learnable complex Hadamard mask over the full k-space grid: x -> x * W, W in C^{H x W} initialised to 1.
    A per-cell multiplication in k-space is a circular convolution in the image domain, so this is a
    learned global image-domain convolution kernel (magnitude re-weighting plus per-cell phase rotation).
    With is_complex=False the input is a real (re, im)-channel tensor and W is a real per-channel mask
    (magnitude re-weighting only; no phase rotation).
    """
    def __init__(self, image_size, num_channels=1, is_complex=True):
        super().__init__()
        h, w = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        dtype = torch.cfloat if is_complex else None
        channels = 1 if is_complex else num_channels
        self.weight = nn.Parameter(torch.ones(1, channels, h, w, dtype=dtype))

    def forward(self, x):
        return x * self.weight


def _init_conv_(conv, kernel_size, criterion):
    """He (before a rectifier) or glorot (output layer) init of a real or complex conv; zero bias."""
    fan_in = conv.in_channels * kernel_size ** 2
    fan_out = conv.out_channels * kernel_size ** 2
    if conv.weight.is_complex():
        trabelsi_init_(conv.weight, fan_in=fan_in, fan_out=fan_out, criterion=criterion)
    elif criterion == "he":
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    else:
        nn.init.xavier_normal_(conv.weight)
    nn.init.zeros_(conv.bias)


class _GatedResidualStem(nn.Module):
    """
    Residual stem  x -> x + alpha * branch(x)  with a single learnable real scalar `alpha` initialised to 0.

    The stem is an exact identity at initialisation, but unlike zero-initialising the branch's output conv,
    every branch weight receives a data gradient as soon as alpha moves off zero (alpha itself gets
    d alpha = <dL/dout, branch(x)> from step 0). With a zero output conv the whole branch upstream of it gets
    exactly zero gradient; under Adam's coupled L2 that branch is then stepped by lr*sign(w) and collapses.
    Checkpoints saved before the gate existed load with alpha = 1 (their branch output was ungated).
    """
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "alpha" not in state_dict:
            state_dict[prefix + "alpha"] = torch.ones_like(self.alpha)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class KSpaceConvStem(_GatedResidualStem):
    """
    Residual complex convolution over neighbouring k-space samples:
        x -> x + alpha * conv(c -> 1)(gelu(conv(1 -> c)(x)))
    Local mixing of adjacent k-space lines (a learned GRAPPA/LORAKS-style kernel), which in the image
    domain is a multiplicative modulation. Both convs are properly initialised (he / glorot); the scalar
    gate alpha = 0 makes the stem an identity at the start (see `_GatedResidualStem`).
    With is_complex=False the convolutions are real over the (re, im) channels.
    """
    def __init__(self, num_channels=1, hidden_channels=8, kernel_size=3, is_complex=True):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kspace_conv_kernel must be a positive odd integer, got {kernel_size}")
        pad = kernel_size // 2
        dtype = torch.cfloat if is_complex else None
        self.conv_in = nn.Conv2d(num_channels, hidden_channels, kernel_size, padding=pad, dtype=dtype)
        self.act = ComplexGELU() if is_complex else nn.GELU()
        self.conv_out = nn.Conv2d(hidden_channels, num_channels, kernel_size, padding=pad, dtype=dtype)
        _init_conv_(self.conv_in, kernel_size, "he")
        _init_conv_(self.conv_out, kernel_size, "glorot")

    def forward(self, x):
        return x + self.alpha * self.conv_out(self.act(self.conv_in(x)))


class ImageDomainConvStem(_GatedResidualStem):
    """
    DcCNN-style residual complex CNN applied in the IMAGE domain to a k-space input:
        k -> FFT( img + alpha * conv_L(... CReLU(conv_2(CReLU(conv_1(img)))) ...) ),   img = IFFT(k)
    conv_1 maps the complex channels to `hidden_channels` filters, the middle layers keep that width, and the
    last conv maps back to the input channel count. Every layer is complex-valued (Trabelsi he-init before the
    CReLUs, glorot for the output conv); the scalar gate alpha = 0 makes the stem an exact identity at
    initialisation while every conv still receives a gradient (see `_GatedResidualStem`). `num_layers` is the
    total number of convolutions (>= 2). A real (re, im)-pair input (kspace_real_channels mode) is converted
    to complex for the IFFT/CNN and split back into channels for the FFT output.
    """
    def __init__(self, num_channels=1, hidden_channels=32, num_layers=3, kernel_size=3, is_complex=True):
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"image_conv_layers must be >= 2 (in + out conv), got {num_layers}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"image_conv_kernel must be a positive odd integer, got {kernel_size}")
        if not is_complex and num_channels % 2:
            raise ValueError("ImageDomainConvStem on a real (re, im) input needs an even channel count")
        self.pair_input = not is_complex
        complex_channels = num_channels if is_complex else num_channels // 2
        pad = kernel_size // 2
        widths = [complex_channels] + [hidden_channels] * (num_layers - 1) + [complex_channels]
        layers = []
        for c_in, c_out in zip(widths[:-2], widths[1:-1]):
            conv = nn.Conv2d(c_in, c_out, kernel_size, padding=pad, dtype=torch.cfloat)
            _init_conv_(conv, kernel_size, "he")
            layers += [conv, ComplexReLU()]
        out_conv = nn.Conv2d(widths[-2], widths[-1], kernel_size, padding=pad, dtype=torch.cfloat)
        _init_conv_(out_conv, kernel_size, "glorot")
        self.net = nn.Sequential(*layers, out_conv)

    def forward(self, k):
        k_complex = pair_to_complex(k) if self.pair_input else k
        img = ifft_2d(k_complex)
        out = fft_2d(img + self.alpha * self.net(img))
        return complex_to_pair(out) if self.pair_input else out


class _SpiritConvLayer(nn.Module):
    """
    Single SPIRiT-style multi-coil k-space convolution layer:
    - Each output coil c_out in {0, ..., C-1} has its own 3D filter of shape (C, p, p) over all C coils.
    - The spatial center tap of the target coil itself (w[c, c, p//2, p//2]) is fixed to 0 via an unlearned
      binary mask buffer so a coil never contributes to itself at the center pixel (0, 0).
    - mode="gated"   : residual gated update  x -> x + alpha * conv(x, W * mask)  with scalar alpha = 0 at init.
    - mode="ungated" : forced direct filter   x -> conv(x, W * mask)              with no gating or skip connection.
    """
    def __init__(self, num_channels: int, kernel_size: int = 3, mode: str = "gated"):
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"SpiritConvLayer requires num_channels >= 1, got {num_channels}")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(f"spirit_conv_kernels entries must be odd integers >= 3, got {kernel_size}")
        mode = mode.lower()
        if mode == "direct":
            mode = "ungated"
        if mode not in ("gated", "ungated"):
            raise ValueError(f"spirit_conv_mode must be 'gated' or 'ungated', got {mode!r}")

        self.num_channels = int(num_channels)
        self.kernel_size = int(kernel_size)
        self.pad = self.kernel_size // 2
        self.mode = mode

        self.weight = nn.Parameter(
            torch.empty(self.num_channels, self.num_channels, self.kernel_size, self.kernel_size, dtype=torch.cfloat)
        )
        active_taps = max(self.num_channels * (self.kernel_size ** 2) - 1, 1)
        trabelsi_init_(self.weight, fan_in=active_taps, fan_out=active_taps, criterion="glorot")

        mask = torch.ones(
            self.num_channels, self.num_channels, self.kernel_size, self.kernel_size, dtype=torch.float32
        )
        coil_idx = torch.arange(self.num_channels)
        mask[coil_idx, coil_idx, self.pad, self.pad] = 0.0
        self.register_buffer("kernel_mask", mask)
        with torch.no_grad():
            self.weight[coil_idx, coil_idx, self.pad, self.pad] = 0.0

        if self.mode == "gated":
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("alpha", None)

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.kernel_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = F.conv2d(x, self.effective_weight(), bias=None, padding=self.pad)
        if self.mode == "gated":
            return x + self.alpha * conv_out
        return conv_out


class SpiritConvStem(nn.Module):
    """
    Stack of 1 or more SPIRiT-style multi-coil k-space convolution layers (e.g. p=5 followed by p=3).
    Supports both 'gated' (x + alpha * G(x)) and 'ungated' (forced through G(x) with no skip connection) modes.
    """
    def __init__(self, num_channels: int = 8, kernel_sizes=(5, 3), mode: str = "gated", is_complex: bool = True):
        super().__init__()
        if not is_complex and num_channels % 2:
            raise ValueError("SpiritConvStem on a real (re, im) input needs an even channel count")
        if isinstance(kernel_sizes, int):
            kernel_sizes = (kernel_sizes,)
        else:
            kernel_sizes = tuple(int(k) for k in kernel_sizes)
        if not kernel_sizes:
            raise ValueError("spirit_conv_kernels must contain at least one kernel size")

        self.pair_input = not is_complex
        complex_channels = num_channels if is_complex else num_channels // 2
        self.kernel_sizes = kernel_sizes
        self.mode = "ungated" if mode.lower() == "direct" else mode.lower()
        self.layers = nn.ModuleList([
            _SpiritConvLayer(complex_channels, kernel_size=k, mode=self.mode)
            for k in self.kernel_sizes
        ])

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        x = pair_to_complex(k) if self.pair_input else k
        for layer in self.layers:
            x = layer(x)
        return complex_to_pair(x) if self.pair_input else x


def build_kspace_stem(image_size, num_channels=1, global_filter=False, kspace_conv=False,
                      kspace_conv_channels=8, kspace_conv_kernel=3, global_filter_mid=False, is_complex=True,
                      image_conv=False, image_conv_channels=32, image_conv_layers=3, image_conv_kernel=3,
                      spirit_conv=False, spirit_conv_kernels=(5, 3), spirit_conv_mode="gated"):
    """
    Compose the enabled input stems in the order ImageDomainConvStem -> GlobalFilter -> SpiritConvStem -> KSpaceConvStem.
    Returns None when none is enabled so callers can skip the call entirely. `global_filter_mid` is not an input stem:
    it is accepted here only so that encoders without a mid-point can reject it (see `build_mid_global_filter`).
    `is_complex=False` builds real-valued k-space stems for encoders that see k-space as (re, im) channels (the
    image-domain CNN and SpiritConvStem stay complex and convert at their boundary).
    """
    if global_filter_mid:
        raise ValueError("global_filter_mid is only supported by the axial encoder (it sits between the "
                         "horizontal and vertical halves); disable it for patch/kaleidoscope encoders")
    stages = []
    if image_conv:
        stages.append(ImageDomainConvStem(num_channels, image_conv_channels, image_conv_layers, image_conv_kernel,
                                          is_complex))
    if global_filter:
        stages.append(GlobalFilter(image_size, num_channels, is_complex))
    if spirit_conv:
        stages.append(SpiritConvStem(num_channels, spirit_conv_kernels, spirit_conv_mode, is_complex))
    if kspace_conv:
        stages.append(KSpaceConvStem(num_channels, kspace_conv_channels, kspace_conv_kernel, is_complex))
    return nn.Sequential(*stages) if stages else None


def build_axial_kspace_stems(image_size, num_channels=1, is_complex=True, **stem_args):
    """
    Axial encoder stems: (input_stem, mid_filter). The input stem is `build_kspace_stem` over the remaining args;
    the mid filter is a second GlobalFilter applied to the full k-space grid between the horizontal and vertical
    transformer halves (after `horizontal_mlp_head`, before `to_vertical_embedding`) when `global_filter_mid`.
    """
    mid = GlobalFilter(image_size, num_channels, is_complex) if stem_args.pop("global_filter_mid", False) else None
    return build_kspace_stem(image_size, num_channels, is_complex=is_complex, **stem_args), mid
