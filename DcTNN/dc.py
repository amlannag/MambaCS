"""
Data consistency layers for Fourier and k-space learning modes.
"""
import torch

def _to_complex_channel(x):
    if x.is_complex():
        return x
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape {tuple(x.shape)}")
    if x.shape[1] == 2:
        x = x.permute(0, 2, 3, 1).contiguous()
        return torch.view_as_complex(x).unsqueeze(1)
    if x.shape[1] == 1:
        return x.to(torch.complex64)
    raise ValueError(f"Cannot convert tensor with shape {tuple(x.shape)} to complex channel format")


def complex_to_pair(x):
    """[B, C, H, W] complex -> [B, 2C, H, W] real with (re, im) of each channel adjacent."""
    if not x.is_complex():
        raise TypeError("complex_to_pair expects a complex tensor")
    b, c, h, w = x.shape
    return torch.view_as_real(x).permute(0, 1, 4, 2, 3).reshape(b, 2 * c, h, w)


def pair_to_complex(x):
    """Inverse of complex_to_pair: [B, 2C, H, W] real -> [B, C, H, W] complex."""
    if x.is_complex():
        return x
    b, c2, h, w = x.shape
    if c2 % 2:
        raise ValueError(f"pair_to_complex needs an even channel count, got {c2}")
    return torch.view_as_complex(x.reshape(b, c2 // 2, 2, h, w).permute(0, 1, 3, 4, 2).contiguous())


def fft_2d(input, norm='ortho', dim=(-2, -1)):
    x = _to_complex_channel(input)
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=dim), norm=norm, dim=dim), dim=dim)

def ifft_2d(input, norm='ortho', dim=(-2, -1)):
    x = _to_complex_channel(input)
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=dim), norm=norm, dim=dim), dim=dim)


def ComplexFFT_DC(x, y, mask, lamb, norm='ortho'):
    x_complex = _to_complex_channel(x)
    cy = _to_complex_channel(y)
    z = fft_2d(x_complex, norm=norm)

    if lamb is None:
        z = (1 - mask) * z + mask * cy
    else:
        z = (1 - mask) * z + mask * (z + lamb * cy) / (1 + lamb)

    return ifft_2d(z, norm=norm)


def FFT_DC(x, y, mask, lamb, norm='ortho'):
    return torch.abs(ComplexFFT_DC(x, y, mask, lamb, norm=norm))


def KSpace_DC(x, y, mask, lamb):
    """
    Data consistency in k-space.
    x: complex k-space [B, 1, H, W] cfloat (encoder output)
    y: measured k-space in the same format, or legacy [B, 2, H, W] real/imag split
    Returns: complex k-space [B, 1, H, W] cfloat
    """
    x = _to_complex_channel(x)
    cy = _to_complex_channel(y)
    if lamb is None:
        return (1 - mask) * x + mask * cy
    else:
        return (1 - mask) * x + mask * (x + lamb * cy) / (1 + lamb)
