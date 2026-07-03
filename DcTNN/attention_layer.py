import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope_vit import apply_rotary_emb, apply_rotary_emb_complex
from .util import FeedForward, ComplexLayerNorm, ComplexDropout, get_attention, _COMPLEX_ATTN_TYPES
from .complex_init import apply_trabelsi_
# ---------------------------------------------------------------------------
# Attention classes
# ---------------------------------------------------------------------------

class BaseAttention(nn.Module):
    """
    Shared base for all attention variants.
    Subclasses implement _attend(q, k, v) with their specific scoring strategy.
    Pass dtype=torch.cfloat for complex-valued subclasses.
    """
    def __init__(self, d_model, nhead, dropout=0.0, freqs_cis=None, dtype=None):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, dtype=dtype)
        self.proj = nn.Linear(d_model, d_model, dtype=dtype)
        self.attn_drop = ComplexDropout(dropout) if dtype == torch.cfloat else nn.Dropout(dropout)
        if dtype == torch.cfloat:
            # glorot: Q/K/V feed softmax (not a rectifier); proj feeds dropout→residual
            apply_trabelsi_(self.qkv, criterion="glorot")
            apply_trabelsi_(self.proj, criterion="glorot")
        self.use_rope = freqs_cis is not None
        if self.use_rope:
            self.register_buffer('freqs_cis', freqs_cis)

    def _apply_rope(self, q, k):
        freqs = self.freqs_cis.to(q.device)
        if q.is_complex():
            return apply_rotary_emb_complex(q, k, freqs_cis=freqs)
        return apply_rotary_emb(q, k, freqs_cis=freqs)

    def _attend(self, q, k, v, attn_mask=None):
        raise NotImplementedError

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rope:
            q, k = self._apply_rope(q, k)
        x = self._attend(q, k, v, attn_mask=attn_mask).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MultiHeadAttention(BaseAttention):
    def __init__(self, d_model, nhead, dropout=0.0, freqs_cis=None):
        super().__init__(d_model, nhead, dropout, freqs_cis, dtype=None)

    def _attend(self, q, k, v, attn_mask=None):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )


class ComplexMultiHeadAttention(BaseAttention):
    """
    Inspired by the complex attention implementation done in
    "Efficient Complex-Valued Vision Transformers for MRI Classification Directly
    from k-Space" and equation 11 from "Building Blocks for Complex-Valued Transformer Architectures".
    Hermitian inner product with complex softmax.
    """
    def __init__(self, d_model, nhead, dropout=0.0, freqs_cis=None):
        super().__init__(d_model, nhead, dropout, freqs_cis, dtype=torch.cfloat)

    def _attend(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1).conj()) * self.scale
        if attn_mask is not None:
            # Add mask directly to real/imag before softmax — avoids a [B,nhead,W,W] intermediate allocation
            attn_r = F.softmax(attn.real + attn_mask, dim=-1)
            attn_i = F.softmax(attn.imag + attn_mask, dim=-1)
        else:
            attn_r = F.softmax(attn.real, dim=-1)
            attn_i = F.softmax(attn.imag, dim=-1)
        attn = self.attn_drop(torch.complex(attn_r, attn_i))
        return torch.matmul(attn, v)


class RealValuedAttention(BaseAttention):
    """
    Similar to equation 8: Complex Valued Dot Product Attention in
    "Building Blocks for Complex-Valued Transformer Architectures".
    Real-valued scores from Hermitian inner product, complex output.
    """
    def __init__(self, d_model, nhead, dropout=0.0, freqs_cis=None):
        super().__init__(d_model, nhead, dropout, freqs_cis, dtype=torch.cfloat)

    def _attend(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1).conj()).real * self.scale
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        # matmul on real/imag separately avoids materialising a complex [B,nhead,W,d] intermediate
        return torch.complex(torch.matmul(attn, v.real), torch.matmul(attn, v.imag))


class PhaseAwareAttention(BaseAttention):
    """
    Inspired by "HOLOGRAPHIC TRANSFORMERS FOR COMPLEX-VALUED SIGNAL PROCESSING:
    INTEGRATING PHASE INTERFERENCE INTO SELF-ATTENTION".
    Phase-damped cosine similarity scores with coherent superposition output.
    """
    def __init__(self, d_model, nhead, dropout=0.0, alpha=1.0, eps=1e-8, freqs_cis=None):
        super().__init__(d_model, nhead, dropout, freqs_cis, dtype=torch.cfloat)
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def _attend(self, q, k, v, attn_mask=None):
        corr = torch.matmul(q, k.transpose(-2, -1).conj())
        q_norm = q.abs().pow(2).sum(-1).sqrt()
        k_norm = k.abs().pow(2).sum(-1).sqrt()
        denom = q_norm.unsqueeze(-1) * k_norm.unsqueeze(-2) + self.eps
        dphi = torch.angle(corr)
        scores = (corr.real / denom * self.scale) * torch.exp(-self.alpha * dphi.abs())
        if attn_mask is not None:
            scores = scores + attn_mask
        A = self.attn_drop(F.softmax(scores, dim=-1))
        return torch.matmul(A * torch.exp(1j * dphi), v)


# ---------------------------------------------------------------------------
# Encoder layer and stack
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                 freqs_cis=None, attn_type="standard"):
        super().__init__()
        is_complex = attn_type in _COMPLEX_ATTN_TYPES
        self.attn = get_attention(attn_type, d_model, nhead, dropout, freqs_cis)
        self.ff = FeedForward(d_model, dim_feedforward, dropout, activation, is_complex)
        if is_complex:
            self.norm1 = ComplexLayerNorm(d_model)
            self.norm2 = ComplexLayerNorm(d_model)
            self.drop = ComplexDropout(dropout)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, encoder_layer, num_layers):
        super().__init__(*[copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x, attn_mask=None):
        for layer in self:
            x = layer(x, attn_mask=attn_mask)
        return x
