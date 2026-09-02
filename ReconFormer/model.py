"""ReconFormer architecture and MambaCS-facing baseline wrapper."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .attention import PatchEmbed, PatchUnEmbed, RPTL


def _pair_to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(x.movedim(1, -1).contiguous())


def _complex_to_pair(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x).movedim(-1, 1).contiguous()


def centered_fft2(x: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2-D FFT over the final two dimensions."""
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def centered_ifft2(x: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal inverse 2-D FFT over the final two dimensions."""
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


class DataConsistencyInKspace(nn.Module):
    """Hard replacement of sampled coefficients in centered k-space."""

    @staticmethod
    def data_consistency(k, k0, mask):
        return (1.0 - mask) * k + mask * k0

    def forward(self, x, k0, mask):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f"x must have shape [B,2,H,W], got {tuple(x.shape)}")
        if k0 is None or mask is None:
            raise ValueError("k0 and mask are required for ReconFormer's hard data consistency")
        if k0.shape != x.shape:
            raise ValueError(
                f"k0 must have the same [B,2,H,W] shape as x; got {tuple(k0.shape)} and {tuple(x.shape)}"
            )
        if mask.ndim != 4 or mask.shape[0] != x.shape[0] or mask.shape[1] not in (1, 2):
            raise ValueError(
                f"mask must have shape [B,1,H,W] (or [B,2,H,W]), got {tuple(mask.shape)}"
            )
        if mask.shape[-2:] != x.shape[-2:]:
            raise ValueError("mask spatial dimensions must match x")

        k = centered_fft2(_pair_to_complex(x))
        measured = _pair_to_complex(k0)
        complex_mask = mask[:, 0].to(dtype=k.real.dtype)
        corrected = self.data_consistency(k, measured, complex_mask)
        return _complex_to_pair(centered_ifft2(corrected))


class RFB(nn.Module):
    """ReconFormer block containing two recurrent pyramid transformer layers."""

    def __init__(
        self,
        img_size,
        nf,
        depth,
        num_head,
        window_size,
        mlp_ratio,
        use_checkpoint,
        resi_connection,
        down=True,
        up_scale=None,
        down_scale=None,
    ):
        super().__init__()
        if down:
            feature_size = int(img_size // down_scale)
        else:
            feature_size = int(img_size * up_scale)
        embed_dim = nf
        self.patch_embed = PatchEmbed(
            img_size=feature_size,
            patch_size=1,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.patch_unembed = PatchUnEmbed(
            img_size=feature_size,
            patch_size=1,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        common = dict(
            dim=embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depth,
            num_heads=num_head,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            downsample=None,
            img_size=feature_size,
            patch_size=1,
            resi_connection=resi_connection,
            rec_att=True,
        )
        self.RPTL1 = RPTL(use_checkpoint=use_checkpoint[0], **common)
        self.RPTL2 = RPTL(use_checkpoint=use_checkpoint[1], shift=True, **common)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, hidden, h1_att, h2_att):
        x_size = (hidden.shape[2], hidden.shape[3])
        hidden = self.patch_embed(hidden)
        h1, h1_att = self.RPTL1((hidden, h1_att), x_size)
        h2, h2_att = self.RPTL2((h1, h2_att), x_size)
        h2 = self.patch_unembed(self.norm(h2), x_size)
        return h2, h1_att, h2_att


class TransBlock_UC(nn.Module):
    """Under-complete learned convolutional branch."""

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        nf=64,
        down_scale=2,
        img_size=256,
        num_head=6,
        depth=6,
        window_size=7,
        mlp_ratio=2.0,
        use_checkpoint=(False, False),
        resi_connection="1conv",
    ):
        super().__init__()
        if down_scale == 2:
            kernel1, stride1 = 3, 1
            kernel2, stride2 = 4, 2
        elif down_scale == 1:
            kernel1, stride1 = 3, 1
            kernel2, stride2 = 3, 1
        else:
            raise ValueError(f"Unsupported down_scale {down_scale!r}; expected 1 or 2")
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, nf, kernel1, stride1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel2, stride2, padding=1, bias=True),
        )
        self.RFB = RFB(
            img_size,
            nf,
            depth,
            num_head,
            window_size,
            mlp_ratio,
            use_checkpoint,
            resi_connection,
            down=True,
            down_scale=down_scale,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, kernel2, stride2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, out_channels, kernel1, stride1, padding=1, bias=True),
        )
        self.activation = nn.PReLU()
        self.DC_layer = DataConsistencyInKspace()

    def forward(self, x, hidden=None, h1_att=None, h2_att=None, k0=None, mask=None):
        if hidden is None:
            hidden = self.activation(self.encoder(x))
        else:
            h2, h1_att, h2_att = self.RFB(hidden, h1_att, h2_att)
            hidden = self.activation(self.encoder(x) + h2)
        out = self.DC_layer(self.decoder(hidden), k0, mask)
        return out, hidden, h1_att, h2_att


class TransBlock_OC(nn.Module):
    """Over-complete learned convolutional branch."""

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        nf=64,
        up_scale=2,
        img_size=256,
        num_head=6,
        depth=6,
        window_size=7,
        mlp_ratio=2.0,
        use_checkpoint=(False, False),
        resi_connection="1conv",
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, 1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=up_scale),
            nn.Conv2d(nf, nf, 3, 1, padding=1, bias=True),
        )
        self.RFB = RFB(
            img_size,
            nf,
            depth,
            num_head,
            window_size,
            mlp_ratio,
            use_checkpoint,
            resi_connection,
            down=False,
            up_scale=up_scale,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, padding=1, bias=True),
            nn.Upsample(scale_factor=1.0 / up_scale),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, out_channels, 3, 1, padding=1, bias=True),
        )
        self.activation = nn.PReLU()
        self.DC_layer = DataConsistencyInKspace()

    def forward(self, x, hidden=None, h1_att=None, h2_att=None, k0=None, mask=None):
        if hidden is None:
            hidden = self.activation(self.encoder(x))
        else:
            h2, h1_att, h2_att = self.RFB(hidden, h1_att, h2_att)
            hidden = self.activation(self.encoder(x) + h2)
        out = self.DC_layer(self.decoder(hidden), k0, mask)
        return out, hidden, h1_att, h2_att


class RefineModule(nn.Module):
    """Fuse the outputs from ReconFormer's three resolution branches."""

    def __init__(self, in_channels, nf, out_channels):
        super().__init__()
        self.rm = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, 1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, padding=1, bias=True),
            nn.Conv2d(nf, out_channels, 3, 1, padding=1, bias=True),
        )
        self.DC_layer = DataConsistencyInKspace()

    def forward(self, x, k0=None, mask=None):
        return self.DC_layer(self.rm(x), k0, mask)


def _validate_sequence(name: str, value: Sequence, length: int) -> tuple:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of length {length}")
    value = tuple(value)
    if len(value) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(value)}")
    return value


class ReconFormer(nn.Module):
    """Released three-branch recurrent ReconFormer architecture."""

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        num_ch=(64, 64, 64),
        down_scales=(2, 1, 1.5),
        num_iter=5,
        img_size=256,
        num_heads=(6, 6, 6),
        depths=(6, 6, 6),
        window_sizes=(8, 8, 8),
        resi_connection="1conv",
        mlp_ratio=2.0,
        use_checkpoint=(False, False, False, False, False, False),
    ):
        super().__init__()
        if in_channels != 2 or out_channels != 2:
            raise ValueError("ReconFormer requires two real/imaginary input and output channels")
        if not isinstance(img_size, int) or img_size <= 0:
            raise ValueError(f"img_size must be a positive integer, got {img_size!r}")
        if not isinstance(num_iter, int) or num_iter < 1:
            raise ValueError(f"num_iter must be a positive integer, got {num_iter!r}")
        num_ch = _validate_sequence("num_ch", num_ch, 3)
        down_scales = _validate_sequence("down_scales", down_scales, 3)
        num_heads = _validate_sequence("num_heads", num_heads, 3)
        depths = _validate_sequence("depths", depths, 3)
        window_sizes = _validate_sequence("window_sizes", window_sizes, 3)
        use_checkpoint = _validate_sequence("use_checkpoint", use_checkpoint, 6)

        self.num_iter = num_iter
        self.img_size = img_size
        self.block1 = TransBlock_UC(
            in_channels=in_channels,
            out_channels=out_channels,
            nf=num_ch[0],
            down_scale=down_scales[0],
            num_head=num_heads[0],
            depth=depths[0],
            img_size=img_size,
            window_size=window_sizes[0],
            mlp_ratio=mlp_ratio,
            use_checkpoint=(use_checkpoint[0], use_checkpoint[1]),
            resi_connection=resi_connection,
        )
        self.block2 = TransBlock_UC(
            in_channels=in_channels,
            out_channels=out_channels,
            nf=num_ch[1],
            down_scale=down_scales[1],
            num_head=num_heads[1],
            depth=depths[1],
            img_size=img_size,
            window_size=window_sizes[1],
            mlp_ratio=mlp_ratio,
            use_checkpoint=(use_checkpoint[2], use_checkpoint[3]),
            resi_connection=resi_connection,
        )
        self.block3 = TransBlock_OC(
            in_channels=in_channels,
            out_channels=out_channels,
            nf=num_ch[2],
            up_scale=down_scales[2],
            num_head=num_heads[2],
            depth=depths[2],
            img_size=img_size,
            window_size=window_sizes[2],
            mlp_ratio=mlp_ratio,
            use_checkpoint=(use_checkpoint[4], use_checkpoint[5]),
            resi_connection=resi_connection,
        )
        self.RM = RefineModule(
            in_channels=out_channels * 3,
            nf=num_ch[2],
            out_channels=out_channels,
        )

    def forward(self, x, k0=None, mask=None):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f"x must have shape [B,2,H,W], got {tuple(x.shape)}")
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"input spatial size must match configured img_size={self.img_size}, got {tuple(x.shape[-2:])}"
            )
        outputs = []
        for iteration in range(self.num_iter):
            if iteration == 0:
                x1, h1, _, _ = self.block1(x, k0=k0, mask=mask)
                x2, h2, _, _ = self.block2(x1, k0=k0, mask=mask)
                x3, h3, _, _ = self.block3(x2, k0=k0, mask=mask)
            elif iteration == 1:
                x = outputs[-1]
                x1, h1, b1_c1_att, b1_c2_att = self.block1(
                    x, hidden=h1, k0=k0, mask=mask
                )
                x2, h2, b2_c1_att, b2_c2_att = self.block2(
                    x1, hidden=h2, k0=k0, mask=mask
                )
                x3, h3, b3_c1_att, b3_c2_att = self.block3(
                    x2, hidden=h3, k0=k0, mask=mask
                )
            else:
                x = outputs[-1]
                x1, h1, b1_c1_att, b1_c2_att = self.block1(
                    x,
                    hidden=h1,
                    h1_att=b1_c1_att,
                    h2_att=b1_c2_att,
                    k0=k0,
                    mask=mask,
                )
                x2, h2, b2_c1_att, b2_c2_att = self.block2(
                    x1,
                    hidden=h2,
                    h1_att=b2_c1_att,
                    h2_att=b2_c2_att,
                    k0=k0,
                    mask=mask,
                )
                x3, h3, b3_c1_att, b3_c2_att = self.block3(
                    x2,
                    hidden=h3,
                    h1_att=b3_c1_att,
                    h2_att=b3_c2_att,
                    k0=k0,
                    mask=mask,
                )
            out = self.RM(torch.cat((x1, x2, x3), dim=1), k0, mask)
            outputs.append(out)
        return outputs[-1]


class ReconFormerBaseline(ReconFormer):
    """MambaCS adapter for normalized complex-image ReconFormer inputs."""

    num_intermediate_stages = 0

    def __init__(
        self,
        num_ch=(64, 64, 64),
        down_scales=(2, 1, 1.5),
        num_iter=5,
        img_size=256,
        num_heads=(6, 6, 6),
        depths=(6, 6, 6),
        window_sizes=(8, 8, 8),
        resi_connection="1conv",
        mlp_ratio=2.0,
        use_checkpoint=(False, False, False, False, False, False),
    ):
        super().__init__(
            in_channels=2,
            out_channels=2,
            num_ch=num_ch,
            down_scales=down_scales,
            num_iter=num_iter,
            img_size=img_size,
            num_heads=num_heads,
            depths=depths,
            window_sizes=window_sizes,
            resi_connection=resi_connection,
            mlp_ratio=mlp_ratio,
            use_checkpoint=use_checkpoint,
        )
        self.lamb = False

    @staticmethod
    def _validate_complex_image(name, value, expected_shape=None):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(
                f"{name} must have shape [B,1,H,W], got {tuple(value.shape)}"
            )
        if not torch.is_complex(value):
            raise TypeError(
                f"{name} must be complex-valued; received dtype {value.dtype}"
            )
        if expected_shape is not None and value.shape != expected_shape:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} must match model_input shape {tuple(expected_shape)}"
            )

    @staticmethod
    def _canonicalize_mask(sample_mask, batch, height, width, device, dtype):
        if not isinstance(sample_mask, torch.Tensor):
            raise TypeError("sample_mask must be a torch.Tensor")
        if torch.is_complex(sample_mask):
            raise TypeError("sample_mask must be real-valued or boolean")
        if sample_mask.ndim == 2:
            mask = sample_mask.unsqueeze(0).unsqueeze(0)
        elif sample_mask.ndim == 3:
            mask = sample_mask.unsqueeze(1)
        elif sample_mask.ndim == 4:
            mask = sample_mask
        else:
            raise ValueError(
                f"sample_mask must have 2, 3, or 4 dimensions, got shape {tuple(sample_mask.shape)}"
            )
        target_shape = (batch, 1, height, width)
        if any(source not in (1, target) for source, target in zip(mask.shape, target_shape)):
            raise ValueError(
                f"sample_mask shape {tuple(sample_mask.shape)} cannot broadcast to {target_shape}"
            )
        mask = torch.broadcast_to(mask.to(device=device), target_shape)
        if not bool(torch.isfinite(mask).all()):
            raise ValueError("sample_mask contains non-finite values")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("sample_mask must be binary (0/1 or boolean) for hard data consistency")
        return mask.to(dtype=dtype).contiguous()

    def forward(
        self,
        model_input,
        dc_input,
        sample_mask,
        return_intermediates=False,
        stats=None,
    ):
        self._validate_complex_image("model_input", model_input)
        self._validate_complex_image("dc_input", dc_input, model_input.shape)
        if dc_input.device != model_input.device:
            raise ValueError("dc_input and model_input must be on the same device")
        batch, _, height, width = model_input.shape
        if (height, width) != (self.img_size, self.img_size):
            raise ValueError(
                f"model_input spatial size must be [{self.img_size},{self.img_size}], got [{height},{width}]"
            )

        parameter = next(self.parameters())
        if model_input.real.dtype != parameter.dtype:
            raise TypeError(
                f"model_input component dtype {model_input.real.dtype} does not match model dtype {parameter.dtype}"
            )
        mask = self._canonicalize_mask(
            sample_mask,
            batch,
            height,
            width,
            model_input.device,
            model_input.real.dtype,
        )
        normalized_image = model_input[:, 0]
        if stats is None:
            measured_kspace = centered_fft2(normalized_image)
        else:
            if stats.get("normalization") != "reconformer" or "scale" not in stats:
                raise ValueError("ReconFormer requires reconformer normalization statistics")
            scale = torch.as_tensor(
                stats["scale"], device=dc_input.device, dtype=dc_input.real.dtype
            )
            if scale.numel() == batch:
                scale = scale.reshape(batch, 1, 1, 1)
            measured_kspace = (dc_input / scale)[:, 0]
        image_pair = _complex_to_pair(normalized_image)
        kspace_pair = _complex_to_pair(measured_kspace)
        output_pair = super().forward(image_pair, k0=kspace_pair, mask=mask)
        reconstruction = _pair_to_complex(output_pair).unsqueeze(1)
        if return_intermediates:
            return reconstruction, []
        return reconstruction


__all__ = [
    "DataConsistencyInKspace",
    "RFB",
    "ReconFormer",
    "ReconFormerBaseline",
    "RefineModule",
    "TransBlock_OC",
    "TransBlock_UC",
    "centered_fft2",
    "centered_ifft2",
]
