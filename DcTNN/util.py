import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .complex_init import apply_trabelsi_


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
    """
    Trabelsi-style 2x2 covariance whitening for complex-valued layer norm.
    Computes the full 2x2 covariance matrix of (Re, Im) and applies its
    closed-form inverse square root, rather than the scalar |z|^2 variance
    used in circularly-symmetric normalization. This properly decorrelates
    real and imaginary parts without assuming equal variance or zero correlation.
    """
    def __init__(self, normalized_shape, eps=1e-8):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps

        inv_sqrt2 = 2 ** -0.5
        self.gamma_rr = nn.Parameter(torch.full(self.normalized_shape, inv_sqrt2))
        self.gamma_ii = nn.Parameter(torch.full(self.normalized_shape, inv_sqrt2))
        self.gamma_ri = nn.Parameter(torch.zeros(self.normalized_shape))
        self.beta = nn.Parameter(torch.zeros(self.normalized_shape, dtype=torch.cfloat))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        centered = x - mean

        r, i = centered.real, centered.imag
        V_rr = torch.mean(r * r, dim=-1, keepdim=True)
        V_ii = torch.mean(i * i, dim=-1, keepdim=True)
        V_ri = torch.mean(r * i, dim=-1, keepdim=True)

        # Add eps to diagonal for numerical stability before trace/det computation
        V_rr = V_rr + self.eps
        V_ii = V_ii + self.eps

        tau = V_rr + V_ii                                            # trace
        delta = (V_rr * V_ii - V_ri * V_ri).clamp(min=self.eps)     # det, clamped to avoid sqrt of negative
        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2.0 * s).clamp(min=self.eps)           # denominator of V^{-1/2}

        # Closed-form 2x2 symmetric matrix square root inverse: [[a, b], [b, d]]
        a = (V_ii + s) / (s * t)
        b = -V_ri       / (s * t)
        d = (V_rr + s) / (s * t)

        real_hat = a * r + b * i
        imag_hat = b * r + d * i

        real_out = self.gamma_rr * real_hat + self.gamma_ri * imag_hat + self.beta.real
        imag_out = self.gamma_ri * real_hat + self.gamma_ii * imag_hat + self.beta.imag

        return torch.complex(real_out, imag_out)


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
        if is_complex:
            # he: net[0] is immediately followed by a rectifying activation (CReLU/CGELU)
            apply_trabelsi_(self.net[0], criterion="he")
            # glorot: net[3] feeds dropout → residual add, no activation follows
            apply_trabelsi_(self.net[3], criterion="glorot")

    def forward(self, x):
        return self.net(x)


def get_to_embedding(tokenizer_type, patch_height=None, patch_width=None, patch_dim=None, d_model=None,
                     image_height=None, image_width=None, numCh=None, row_stride=1, is_complex=False):
    dtype = torch.cfloat if is_complex else None
    if tokenizer_type == "patch":
        seq = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, d_model, dtype=dtype),
        )
        if is_complex:
            # glorot: projection feeds pos-embed add → transformer stack, no immediate rectifier
            apply_trabelsi_(seq[1], criterion="glorot")
        return seq
    elif tokenizer_type == "kaleidoscope":
        seq = nn.Sequential(
            Rearrange('b c (k1 h) (k2 w) -> b (h w) (k1 k2 c)', k1=patch_height, k2=patch_width),
            nn.Linear(patch_dim, d_model, dtype=dtype),
        )
        if is_complex:
            apply_trabelsi_(seq[1], criterion="glorot")
        return seq
    elif tokenizer_type == "axial":
        h_seq = nn.Sequential(
            Rearrange('b c (h p) w -> b h (p w c)', p=row_stride),
            nn.Linear(row_stride * image_width * numCh, d_model, dtype=dtype),
        )
        v_seq = nn.Sequential(
            Rearrange('b c h w -> b w (h c)'),
            nn.Linear(image_height * numCh, d_model, dtype=dtype),
        )
        if is_complex:
            apply_trabelsi_(h_seq[1], criterion="glorot")
            apply_trabelsi_(v_seq[1], criterion="glorot")
        return h_seq, v_seq
    else:
        raise ValueError(f"Unknown tokenizer_type '{tokenizer_type}'. Choose from: patch, kaleidoscope, axial")


def get_from_embedding(tokenizer_type, patch_height=None, patch_width=None, grid_h=None, numCh=None,
                       image_height=None, image_width=None, row_stride=1):
    if tokenizer_type == "patch":
        return Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         c=numCh, h=grid_h, p1=patch_height, p2=patch_width)
    elif tokenizer_type == "kaleidoscope":
        return Rearrange('b (h w) (k1 k2 c) -> b c (k1 h) (k2 w)',
                         k1=patch_height, k2=patch_width, h=grid_h, c=numCh)
    elif tokenizer_type == "axial":
        return (
            Rearrange('b h (p w c) -> b c (h p) w', p=row_stride, w=image_width, c=numCh),
            Rearrange('b w (h c) -> b c h w', c=numCh),
        )
    else:
        raise ValueError(f"Unknown tokenizer_type '{tokenizer_type}'. Choose from: patch, kaleidoscope, axial")


def get_mlp_head(tokenizer_type, d_model, patch_dim=None, patch_height=None, patch_width=None,
                 grid_h=None, numCh=None, image_height=None, image_width=None, row_stride=1, is_complex=False):
    norm = ComplexLayerNorm if is_complex else nn.LayerNorm
    dtype = torch.cfloat if is_complex else None
    if tokenizer_type in ("patch", "kaleidoscope"):
        from_emb = get_from_embedding(tokenizer_type, patch_height, patch_width, grid_h, numCh)
        seq = nn.Sequential(
            norm(d_model),
            nn.Linear(d_model, patch_dim, dtype=dtype),
            from_emb,
        )
        if is_complex:
            # glorot: head projection feeds rearrange → output, no activation follows
            apply_trabelsi_(seq[1], criterion="glorot")
        return seq
    elif tokenizer_type == "axial":
        h_from, v_from = get_from_embedding("axial", numCh=numCh, image_width=image_width, row_stride=row_stride)
        h_seq = nn.Sequential(norm(d_model), nn.Linear(d_model, row_stride * image_width * numCh, dtype=dtype), h_from)
        v_seq = nn.Sequential(norm(d_model), nn.Linear(d_model, image_height * numCh, dtype=dtype), v_from)
        if is_complex:
            apply_trabelsi_(h_seq[1], criterion="glorot")
            apply_trabelsi_(v_seq[1], criterion="glorot")
        return h_seq, v_seq
    else:
        raise ValueError(f"Unknown tokenizer_type '{tokenizer_type}'. Choose from: patch, kaleidoscope, axial")
