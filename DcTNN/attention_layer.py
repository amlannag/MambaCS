import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope_vit import apply_rotary_emb, apply_rotary_emb_complex, reshape_for_broadcast
from .util import FeedForward, ComplexLayerNorm, ComplexDropout, get_attention, _COMPLEX_ATTN_TYPES
from .complex_init import apply_trabelsi_
# ---------------------------------------------------------------------------
# Attention classes
# ---------------------------------------------------------------------------


def _apply_rotary_emb_single(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a single Q or K tensor using preselected position frequencies."""
    if x.is_complex():
        freqs_cis = reshape_for_broadcast(freqs_cis, x)
        return (x * freqs_cis).to(x.device)

    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x).to(x.device)

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


class BaseCrossAttention(nn.Module):
    """
    Shared base for cross-attention variants with separate query and key/value inputs.
    """
    def __init__(self, d_model, nhead, dropout=0.0, freqs_cis=None, dtype=None):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model, dtype=dtype)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, dtype=dtype)
        self.proj = nn.Linear(d_model, d_model, dtype=dtype)
        self.attn_drop = ComplexDropout(dropout) if dtype == torch.cfloat else nn.Dropout(dropout)
        if dtype == torch.cfloat:
            apply_trabelsi_(self.q_proj, criterion="glorot")
            apply_trabelsi_(self.kv_proj, criterion="glorot")
            apply_trabelsi_(self.proj, criterion="glorot")
        self.use_rope = freqs_cis is not None
        if self.use_rope:
            self.register_buffer("freqs_cis", freqs_cis)

    def _select_freqs(self, positions: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        freqs = self.freqs_cis.to(device)
        if positions is None:
            if freqs.shape[0] != seq_len:
                raise ValueError(
                    f"RoPE positions required for cross-attention subset of length {seq_len}; "
                    f"stored frequencies have length {freqs.shape[0]}"
                )
            return freqs

        if positions.ndim != 1:
            raise ValueError(f"Expected 1D positions tensor, got shape {tuple(positions.shape)}")
        if positions.shape[0] != seq_len:
            raise ValueError(
                f"Positions length {positions.shape[0]} does not match sequence length {seq_len}"
            )
        return freqs.index_select(0, positions.to(device=device, dtype=torch.long))

    def _apply_rope(self, q, k, q_positions=None, kv_positions=None):
        q_freqs = self._select_freqs(q_positions, q.shape[-2], q.device)
        k_freqs = self._select_freqs(kv_positions, k.shape[-2], k.device)
        q = _apply_rotary_emb_single(q, q_freqs)
        k = _apply_rotary_emb_single(k, k_freqs)
        return q, k

    def _attend(self, q, k, v, attn_mask=None):
        raise NotImplementedError

    def forward(self, q_input, kv_input, attn_mask=None, q_positions=None, kv_positions=None):
        if q_input.shape[0] != kv_input.shape[0]:
            raise ValueError(
                f"Query batch {q_input.shape[0]} does not match key/value batch {kv_input.shape[0]}"
            )

        B, Nq, C = q_input.shape
        Nk = kv_input.shape[1]
        q = self.q_proj(q_input).reshape(B, Nq, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(kv_input).reshape(B, Nk, 2, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        if self.use_rope:
            q, k = self._apply_rope(q, k, q_positions=q_positions, kv_positions=kv_positions)
        x = self._attend(q, k, v, attn_mask=attn_mask).transpose(1, 2).reshape(B, Nq, C)
        return self.proj(x)


class MultiHeadCrossAttention(BaseCrossAttention):
    def __init__(self, d_model, nhead, dropout=0.0, freqs_cis=None):
        super().__init__(d_model, nhead, dropout, freqs_cis, dtype=None)

    def _attend(self, q, k, v, attn_mask=None):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )


class ComplexCrossAttention(BaseCrossAttention):
    """Complex-valued cross-attention using the same scoring rule as ComplexMultiHeadAttention."""
    def __init__(self, d_model, nhead, dropout=0.0, freqs_cis=None):
        super().__init__(d_model, nhead, dropout, freqs_cis, dtype=torch.cfloat)

    def _attend(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1).conj()) * self.scale
        if attn_mask is not None:
            attn_r = F.softmax(attn.real + attn_mask, dim=-1)
            attn_i = F.softmax(attn.imag + attn_mask, dim=-1)
        else:
            attn_r = F.softmax(attn.real, dim=-1)
            attn_i = F.softmax(attn.imag, dim=-1)
        attn = self.attn_drop(torch.complex(attn_r, attn_i))
        return torch.matmul(attn, v)


class RealValuedCrossAttention(BaseCrossAttention):
    def __init__(self, d_model, nhead, dropout=0.0, freqs_cis=None):
        super().__init__(d_model, nhead, dropout, freqs_cis, dtype=torch.cfloat)

    def _attend(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1).conj()).real * self.scale
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        return torch.complex(torch.matmul(attn, v.real), torch.matmul(attn, v.imag))


class PhaseAwareCrossAttention(BaseCrossAttention):
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
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        return torch.matmul(attn * torch.exp(1j * dphi), v)


def get_cross_attention(attn_type, d_model, nhead, dropout=0.0, freqs_cis=None):
    if attn_type == "standard":
        return MultiHeadCrossAttention(d_model, nhead, dropout, freqs_cis)
    elif attn_type == "complex":
        return ComplexCrossAttention(d_model, nhead, dropout, freqs_cis)
    elif attn_type == "real_valued":
        return RealValuedCrossAttention(d_model, nhead, dropout, freqs_cis)
    elif attn_type == "phase_aware":
        return PhaseAwareCrossAttention(d_model, nhead, dropout, freqs_cis=freqs_cis)
    else:
        raise ValueError(
            f"Unknown attn_type '{attn_type}'. Choose from: standard, complex, real_valued, phase_aware"
        )


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


class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                 freqs_cis=None, attn_type="standard"):
        super().__init__()
        is_complex = attn_type in _COMPLEX_ATTN_TYPES
        self.attn = get_cross_attention(attn_type, d_model, nhead, dropout, freqs_cis)
        self.ff = FeedForward(d_model, dim_feedforward, dropout, activation, is_complex)
        if is_complex:
            self.norm_q = ComplexLayerNorm(d_model)
            self.norm_kv = ComplexLayerNorm(d_model)
            self.norm2 = ComplexLayerNorm(d_model)
            self.drop = ComplexDropout(dropout)
        else:
            self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm_kv = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.drop = nn.Dropout(dropout)

    def forward(self, q, kv, attn_mask=None, q_positions=None, kv_positions=None):
        q = q + self.drop(
            self.attn(
                self.norm_q(q),
                self.norm_kv(kv),
                attn_mask=attn_mask,
                q_positions=q_positions,
                kv_positions=kv_positions,
            )
        )
        q = q + self.drop(self.ff(self.norm2(q)))
        return q


class CrossAttentionEncoder(nn.Sequential):
    def __init__(self, encoder_layer, num_layers):
        super().__init__(*[copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, q, kv, attn_mask=None, q_positions=None, kv_positions=None):
        for layer in self:
            q = layer(q, kv, attn_mask=attn_mask, q_positions=q_positions, kv_positions=kv_positions)
        return q
