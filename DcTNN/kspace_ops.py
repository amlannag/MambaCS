"""
Learnable k-space pre-processing stems applied to the encoder input before tokenisation.
Both are exact identities at initialisation so an existing recipe is unchanged until they learn.
"""
import math

import torch
import torch.nn as nn

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


class KSpaceConvStem(nn.Module):
    """
    Residual complex convolution over neighbouring k-space samples:
        x -> x + conv(c -> 1)(gelu(conv(1 -> c)(x)))
    Local mixing of adjacent k-space lines (a learned GRAPPA/LORAKS-style kernel), which in the image
    domain is a multiplicative modulation. The output conv is zero-initialised so the stem starts as identity.
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
        if is_complex:
            trabelsi_init_(self.conv_in.weight, fan_in=num_channels * kernel_size ** 2,
                           fan_out=hidden_channels * kernel_size ** 2, criterion="he")
        else:
            nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv_in.bias)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        return x + self.conv_out(self.act(self.conv_in(x)))


class ImageDomainConvStem(nn.Module):
    """
    DcCNN-style residual complex CNN applied in the IMAGE domain to a k-space input:
        k -> FFT( img + conv_L(... CReLU(conv_2(CReLU(conv_1(img)))) ...) ),   img = IFFT(k)
    conv_1 maps the complex channels to `hidden_channels` filters, the middle layers keep that width, and the
    last conv maps back to the input channel count. Every layer is complex-valued (Trabelsi he-init before the
    CReLUs); the last conv is zero-initialised so the stem is an exact identity at initialisation. `num_layers`
    is the total number of convolutions (>= 2). A real (re, im)-pair input (kspace_real_channels mode) is
    converted to complex for the IFFT/CNN and split back into channels for the FFT output.
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
            trabelsi_init_(conv.weight, fan_in=c_in * kernel_size ** 2, fan_out=c_out * kernel_size ** 2, criterion="he")
            nn.init.zeros_(conv.bias)
            layers += [conv, ComplexReLU()]
        out_conv = nn.Conv2d(widths[-2], widths[-1], kernel_size, padding=pad, dtype=torch.cfloat)
        nn.init.zeros_(out_conv.weight)
        nn.init.zeros_(out_conv.bias)
        self.net = nn.Sequential(*layers, out_conv)

    def forward(self, k):
        k_complex = pair_to_complex(k) if self.pair_input else k
        img = ifft_2d(k_complex)
        out = fft_2d(img + self.net(img))
        return complex_to_pair(out) if self.pair_input else out


def build_kspace_stem(image_size, num_channels=1, global_filter=False, kspace_conv=False,
                      kspace_conv_channels=8, kspace_conv_kernel=3, global_filter_mid=False, is_complex=True,
                      image_conv=False, image_conv_channels=32, image_conv_layers=3, image_conv_kernel=3):
    """
    Compose the enabled input stems in the order ImageDomainConvStem -> GlobalFilter -> KSpaceConvStem. Returns
    None when none is enabled so callers can skip the call entirely. `global_filter_mid` is not an input stem: it
    is accepted here only so that encoders without a mid-point can reject it (see `build_mid_global_filter`).
    `is_complex=False` builds real-valued k-space stems for encoders that see k-space as (re, im) channels (the
    image-domain CNN stays complex and converts at its boundary).
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
