"""
Data consistency layers for Fourier and k-space learning modes.
"""
import torch

def fft_2d(input, norm='ortho', dim=(-2, -1)):
    x = input.permute(0, 2, 3, 1).contiguous()
    if x.shape[3] == 1:
        x = torch.cat([x, torch.zeros_like(x)], 3)
    x = torch.view_as_complex(x)
    x = torch.fft.fft2(x, norm=norm, dim=dim)
    x = torch.view_as_real(x)
    return x.permute(0, 3, 1, 2).contiguous()

def ifft_2d(input, norm='ortho', dim=(-2, -1)):
    x = input.permute(0, 2, 3, 1).contiguous()
    x = torch.view_as_complex(x)
    x = torch.fft.ifft2(x, dim=dim, norm=norm)
    x = torch.view_as_real(x)
    return x.permute(0, 3, 1, 2).contiguous()


def FFT_DC(x, y, mask, lamb, norm='ortho'):
    numCh = x.shape[1]
    cy = y.permute(0, 2, 3, 1).contiguous()
    cy = torch.view_as_complex(cy)
    x = x.permute(0, 2, 3, 1).contiguous()

    if numCh == 1:
        x = torch.cat([x, torch.zeros_like(x)], 3)
    z = torch.fft.fft2(torch.view_as_complex(x), norm=norm)

    if lamb is None:
        z = (1 - mask) * z + mask * cy
    else:
        z = (1 - mask) * z + mask * (z + lamb * cy) / (1 + lamb)

    z = torch.view_as_real(torch.fft.ifft2(z, norm=norm))[:, :, :, 0:numCh]
    z = z.permute(0, 3, 1, 2).contiguous()
    return z


def KSpace_DC(x, y, mask, lamb, norm='ortho'):
    cy = y.permute(0, 2, 3, 1).contiguous()
    cy = torch.view_as_complex(cy)

    z = x.permute(0, 2, 3, 1).contiguous()
    z = torch.view_as_complex(z)

    if lamb is None:
        z = (1 - mask) * z + mask * cy
    else:
        z = (1 - mask) * z + mask * (z + lamb * cy) / (1 + lamb)

    # convert to image domain, keep only real channel [B, 1, H, W]
    z = torch.view_as_real(torch.fft.ifft2(z, norm=norm))[:, :, :, 0:1]
    return z.permute(0, 3, 1, 2).contiguous()
