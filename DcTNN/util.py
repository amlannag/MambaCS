import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class ComplexGELU(nn.Module):
    def forward(self, x):
        return torch.complex(F.gelu(x.real), F.gelu(x.imag))


class ComplexDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            if x.is_complex():
                mask = F.dropout(torch.ones_like(x.real), self.p)
                return x * mask
            return F.dropout(x, self.p)
        return x


class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.gamma = nn.Parameter(torch.ones(self.normalized_shape, dtype=torch.cfloat))
        self.beta = nn.Parameter(torch.zeros(self.normalized_shape, dtype=torch.cfloat))
        self.eps = 1e-8

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        centered = x - mean
        variance = torch.mean(torch.abs(centered) ** 2, dim=-1, keepdim=True)
        return (centered / torch.sqrt(variance + self.eps)) * self.gamma + self.beta


_COMPLEX_ATTN_TYPES = {"complex", "real_valued", "phase_aware"}


def get_activation(activation, is_complex=False):
    if is_complex:
        return ComplexReLU() if activation == 'relu' else ComplexGELU()
    return nn.ReLU() if activation == 'relu' else nn.GELU()


def get_attention(attn_type, d_model, nhead, dropout=0.0, freqs_cis=None):
    from .attention_layer import (MultiHeadAttention, ComplexMultiHeadAttention,
                                   RealValuedAttention, PhaseAwareAttention)
    if attn_type == "standard":
        return MultiHeadAttention(d_model, nhead, dropout, freqs_cis)
    elif attn_type == "complex":
        return ComplexMultiHeadAttention(d_model, nhead, dropout, freqs_cis)
    elif attn_type == "real_valued":
        return RealValuedAttention(d_model, nhead, dropout, freqs_cis)
    elif attn_type == "phase_aware":
        return PhaseAwareAttention(d_model, nhead, dropout, freqs_cis=freqs_cis)
    else:
        raise ValueError(f"Unknown attn_type '{attn_type}'. Choose from: standard, complex, real_valued, phase_aware")


class FeedForward(nn.Module):
    """Two-layer MLP. Supports real and complex dtypes transparently."""
    def __init__(self, d_model, dim_feedforward, dropout, activation, is_complex=False):
        super().__init__()
        dtype = torch.cfloat if is_complex else None
        act = ComplexReLU() if (is_complex and activation == 'relu') else \
              ComplexGELU() if is_complex else \
              nn.ReLU() if activation == 'relu' else nn.GELU()
        drop = lambda: ComplexDropout(dropout) if is_complex else nn.Dropout(dropout)
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, dtype=dtype),
            act,
            drop(),
            nn.Linear(dim_feedforward, d_model, dtype=dtype),
            drop(),
        )

    def forward(self, x):
        return self.net(x)


def get_to_embedding(tokenizer_type, patch_height=None, patch_width=None, patch_dim=None, d_model=None,
                     image_height=None, image_width=None, numCh=None, is_complex=False):
    dtype = torch.cfloat if is_complex else None
    if tokenizer_type == "patch":
        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, d_model, dtype=dtype),
        )
    elif tokenizer_type == "kaleidoscope":
        return nn.Sequential(
            Rearrange('b c (k1 h) (k2 w) -> b (h w) (k1 k2 c)', k1=patch_height, k2=patch_width),
            nn.Linear(patch_dim, d_model, dtype=dtype),
        )
    elif tokenizer_type == "axial":
        return (
            nn.Sequential(
                Rearrange('b c h w -> b h (w c)'),
                nn.Linear(image_width * numCh, d_model, dtype=dtype),
            ),
            nn.Sequential(
                Rearrange('b c h w -> b w (h c)'),
                nn.Linear(image_height * numCh, d_model, dtype=dtype),
            ),
        )
    else:
        raise ValueError(f"Unknown tokenizer_type '{tokenizer_type}'. Choose from: patch, kaleidoscope, axial")


def get_from_embedding(tokenizer_type, patch_height=None, patch_width=None, grid_h=None, numCh=None,
                       image_height=None, image_width=None):
    if tokenizer_type == "patch":
        return Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         c=numCh, h=grid_h, p1=patch_height, p2=patch_width)
    elif tokenizer_type == "kaleidoscope":
        return Rearrange('b (h w) (k1 k2 c) -> b c (k1 h) (k2 w)',
                         k1=patch_height, k2=patch_width, h=grid_h, c=numCh)
    elif tokenizer_type == "axial":
        return (
            Rearrange('b h (w c) -> b c h w', c=numCh),
            Rearrange('b w (h c) -> b c h w', c=numCh),
        )
    else:
        raise ValueError(f"Unknown tokenizer_type '{tokenizer_type}'. Choose from: patch, kaleidoscope, axial")


def get_mlp_head(tokenizer_type, d_model, patch_dim=None, patch_height=None, patch_width=None,
                 grid_h=None, numCh=None, image_height=None, image_width=None, is_complex=False):
    norm = ComplexLayerNorm if is_complex else nn.LayerNorm
    dtype = torch.cfloat if is_complex else None
    if tokenizer_type in ("patch", "kaleidoscope"):
        from_emb = get_from_embedding(tokenizer_type, patch_height, patch_width, grid_h, numCh)
        return nn.Sequential(
            norm(d_model),
            nn.Linear(d_model, patch_dim, dtype=dtype),
            from_emb,
        )
    elif tokenizer_type == "axial":
        h_from, v_from = get_from_embedding("axial", numCh=numCh)
        return (
            nn.Sequential(norm(d_model), nn.Linear(d_model, image_width * numCh, dtype=dtype), h_from),
            nn.Sequential(norm(d_model), nn.Linear(d_model, image_height * numCh, dtype=dtype), v_from),
        )
    else:
        raise ValueError(f"Unknown tokenizer_type '{tokenizer_type}'. Choose from: patch, kaleidoscope, axial")
