"""
GRAPPA-style local cross-attention over multi-coil k-space.

Tokens are per (coil, kx-strip, ky-column): the column's `strip_len` consecutive kx samples of one coil, linearly
embedded to d_model, plus a coil-ID embedding, a sampled/unsampled flag embedding and (for unsampled columns) a learned
mask token. Every unsampled token attends to the tokens of the `n_neighbours` nearest acquired columns on each side of
its column, in all coils, in the kx strips s-kx_halo .. s+kx_halo. That is the footprint of a GRAPPA kernel
(source coil, target coil, dky, dkx) with content-dependent weights shared over k-space. Sampled columns are keys only
and pass through the stage unchanged; the data-consistency layer is untouched.

Keys and values are projected once per token and then gathered along ky with the neighbour index tables; the kx halo
is realised with shifted views of the strip axis instead of a second gather, so memory scales with the gathered key
set (B * heads * C * nS * Nu * 2n * head_dim), never with (number of tokens)^2. Because the key set of a query does not
depend on the query's coil, the per-query (Nq, Kn, d) layout of the spec is kept factored as (C, nS, Nu) x (C', 2h+1, 2n).

Positional information is axial RoPE driven by explicit coordinates: ky = the actual column index (so the score depends
on the signed column distance between query and key, not on the rank among acquired columns) and kx = the strip index
(relative kx offset in strip units).
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from .complex_init import apply_trabelsi_
from .encoders import crossAxialEncoder, _extract_shared_column_mask, pair
from .kspace_ops import build_kspace_stem
from .rope_vit import axial_cis_complex_from_coords
from .util import ComplexDropout, ComplexLayerNorm, FeedForward

NEIGHBOURHOOD_ATTN_TYPES = ("complex", "real_valued", "phase_aware")
NEIGHBOURHOOD_COMBINE_MODES = ("sum", "gate", "sequential", "local_only")


def build_neighbour_tables(candidates: torch.Tensor, n: int):
    """
    Per-column ky neighbour tables.

    candidates : bool [W], True where a column may serve as a key (the acquired columns).
    Returns (idx, valid, dist), each [W, 2n]: for every column w the n nearest candidate columns strictly left of w
    and the n nearest strictly right of w (by ky index). If one side has fewer than n candidates (k-space edge, long
    gap) the remaining slots are filled with the next-nearest candidates of the other side; if still short they are
    padded with idx 0 / dist 0 and marked invalid. dist = idx - w is the signed column distance.
    """
    if n < 1:
        raise ValueError(f"n_neighbours must be >= 1, got {n}")
    W = candidates.shape[0]
    device = candidates.device
    acq = candidates.nonzero(as_tuple=True)[0]
    A = acq.numel()
    if A == 0:
        raise ValueError("neighbourhood attention needs at least one candidate (acquired) column")

    w = torch.arange(W, device=device)
    pos = torch.searchsorted(acq, w)                        # candidates strictly left of w
    pos_right = pos + candidates.long()                     # first candidate strictly right of w
    r = torch.arange(2 * n, device=device)
    rank = torch.cat([pos[:, None] - 1 - r, pos_right[:, None] + r], dim=1)     # (W, 4n), nearest first per side
    in_range = (rank >= 0) & (rank < A)
    col = acq[rank.clamp(0, A - 1)]
    dist = col - w[:, None]
    # primary slots (side rank < n) always rank before fallback slots (side rank >= n); within a group by distance
    fallback = (r >= n).repeat(2)[None, :]
    key = dist.abs().float() + fallback.float() * (W + 1)
    key = key.masked_fill(~in_range, float("inf"))
    order = key.argsort(dim=1)[:, : 2 * n]
    valid = in_range.gather(1, order)
    idx = col.gather(1, order).masked_fill(~valid, 0)
    dist = dist.gather(1, order).masked_fill(~valid, 0)
    return idx, valid, dist


class NeighbourhoodCrossAttention(nn.Module):
    """
    Complex cross-attention from unsampled (coil, strip, column) tokens to their gathered GRAPPA neighbourhood.

    q_in  : (B, C, nS, Nu, d)  hidden state of the unsampled columns (queries)
    kv_in : (B, C, nS, W, d)   hidden state of every column (keys / values are gathered from it)
    nbr_idx, nbr_valid : (Nu, 2n) neighbour column tables for the query columns, q_cols : (Nu,) their column indices.

    attn_type selects the scoring rule, each reproducing the matching dense module of attention_layer.py:
      "complex"     Hermitian q.conj(k), separate softmax of Re and Im  (ComplexCrossAttention)
      "real_valued" Re(q.conj(k)) real softmax, convex combination of complex v  (RealValuedAttention)
      "phase_aware" phase-damped cosine scores, coherent e^{i dphi} aggregation  (PhaseAwareAttention)
    """
    def __init__(self, d_model, nhead, dropout=0.0, attn_type="complex", rope_theta=None, alpha=1.0, eps=1e-8):
        super().__init__()
        if attn_type not in NEIGHBOURHOOD_ATTN_TYPES:
            raise ValueError(f"Unknown neighbourhood attn_type '{attn_type}'. Choose from: {NEIGHBOURHOOD_ATTN_TYPES}")
        if d_model % nhead:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.attn_type = attn_type
        self.eps = eps
        self.q_proj = nn.Linear(d_model, d_model, dtype=torch.cfloat)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, dtype=torch.cfloat)
        self.proj = nn.Linear(d_model, d_model, dtype=torch.cfloat)
        self.attn_drop = ComplexDropout(dropout)
        apply_trabelsi_(self.q_proj, criterion="glorot")
        apply_trabelsi_(self.kv_proj, criterion="glorot")
        apply_trabelsi_(self.proj, criterion="glorot")
        if attn_type == "phase_aware":
            self._alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.use_rope = rope_theta is not None
        self.rope_theta = rope_theta
        if self.use_rope and self.head_dim % 2:
            raise ValueError(f"RoPE needs an even head_dim, got {self.head_dim}")

    @property
    def alpha(self):
        return F.softplus(self._alpha)

    def _rope(self, x, t_ky, t_kx):
        """Rotate the token axis (..., nS, N, head_dim) of x by the coordinates t_ky, t_kx of shape (nS, N)."""
        return x * axial_cis_complex_from_coords(t_ky, t_kx, self.head_dim, self.rope_theta)

    @staticmethod
    def _gather(t, nbr_idx, halo):
        """(B, nh, C, nS, W, d) -> (B, nh, C, nS + 2*halo, Nu, 2n, d): neighbour columns, strip axis zero-padded."""
        g = t[:, :, :, :, nbr_idx]
        if halo:
            pad = g.new_zeros(g.shape[:3] + (halo,) + g.shape[4:])
            g = torch.cat([pad, g, pad], dim=3)
        return g

    @staticmethod
    def _halo_corr(q, k_pad, nS, halo):
        """q (B,nh,C,nS,Nu,d) x k_pad (B,nh,C',nS+2h,Nu,2n,d) -> (B,nh,C,nS,Nu,C',2h+1,2n) Hermitian correlations."""
        return torch.stack(
            [torch.einsum("bhcsud,bhxsumd->bhcsuxm", q, k_pad[:, :, :, j:j + nS].conj()) for j in range(2 * halo + 1)],
            dim=-2,
        )

    @staticmethod
    def _halo_aggregate(A, v_pad, nS, halo):
        """A (B,nh,C,nS,Nu,C',2h+1,2n) complex x v_pad (B,nh,C',nS+2h,Nu,2n,d) -> (B,nh,C,nS,Nu,d)."""
        out = 0
        for j in range(2 * halo + 1):
            out = out + torch.einsum("bhcsuxm,bhxsumd->bhcsud", A[..., j, :], v_pad[:, :, :, j:j + nS])
        return out

    def _weights(self, corr, q, k_pad, add_mask, nS, halo):
        """Attention weights (complex) over the flattened key axis, following the selected scoring rule."""
        shape = corr.shape
        corr = corr.flatten(-3)
        if self.attn_type == "complex":
            attn = corr * self.scale
            A = torch.complex(F.softmax(attn.real + add_mask, dim=-1), F.softmax(attn.imag + add_mask, dim=-1))
        elif self.attn_type == "real_valued":
            A = F.softmax(corr.real * self.scale + add_mask, dim=-1).to(corr.dtype)
        else:
            q_norm = q.abs().pow(2).sum(-1).sqrt().unsqueeze(-1)                                   # (B,nh,C,nS,Nu,1)
            k_norm = k_pad.abs().pow(2).sum(-1).sqrt()                                             # (B,nh,C',nS+2h,Nu,2n)
            k_norm = torch.stack([k_norm[:, :, :, j:j + nS] for j in range(2 * halo + 1)], dim=-2)  # (B,nh,C',nS,Nu,2h+1,2n)
            k_norm = k_norm.permute(0, 1, 3, 4, 2, 5, 6).unsqueeze(2).flatten(-3)                  # (B,nh,1,nS,Nu,Kn)
            denom = q_norm * k_norm + self.eps
            dphi = torch.angle(corr)
            scores = (corr.real / denom * self.scale) * torch.exp(-self.alpha * dphi.abs()) + add_mask
            A = F.softmax(scores, dim=-1) * torch.exp(1j * dphi)
        return self.attn_drop(A).reshape(shape)

    def forward(self, q_in, kv_in, nbr_idx, nbr_valid, q_cols, halo):
        B, C, nS, Nu, D = q_in.shape
        W = kv_in.shape[3]
        n2 = nbr_idx.shape[1]
        nh, dh = self.nhead, self.head_dim

        q = self.q_proj(q_in).reshape(B, C, nS, Nu, nh, dh).permute(0, 4, 1, 2, 3, 5)              # (B,nh,C,nS,Nu,dh)
        kv = self.kv_proj(kv_in).reshape(B, C, nS, W, 2, nh, dh).permute(4, 0, 5, 1, 2, 3, 6)      # (2,B,nh,C,nS,W,dh)
        k, v = kv[0], kv[1]
        if self.use_rope:
            strips = torch.arange(nS, device=q.device)
            q = self._rope(q, q_cols[None, :].expand(nS, Nu), strips[:, None].expand(nS, Nu))
            cols = torch.arange(W, device=q.device)
            k = self._rope(k, cols[None, :].expand(nS, W), strips[:, None].expand(nS, W))

        k_pad = self._gather(k, nbr_idx, halo)                                                     # (B,nh,C',nS+2h,Nu,2n,dh)
        v_pad = self._gather(v, nbr_idx, halo)
        corr = self._halo_corr(q, k_pad, nS, halo)                                                 # (B,nh,C,nS,Nu,C',2h+1,2n)

        offsets = torch.arange(-halo, halo + 1, device=q.device)
        strip_ok = ((torch.arange(nS, device=q.device)[:, None] + offsets[None, :]) >= 0) & \
                   ((torch.arange(nS, device=q.device)[:, None] + offsets[None, :]) < nS)         # (nS, 2h+1)
        valid = strip_ok[:, None, None, :, None] & nbr_valid[None, :, None, None, :]              # (nS,Nu,1,2h+1,2n)
        valid = valid.expand(nS, Nu, C, 2 * halo + 1, n2).flatten(-3)                              # (nS,Nu,Kn)
        add_mask = torch.zeros(valid.shape, device=q.device, dtype=q.real.dtype).masked_fill(~valid, float("-inf"))

        A = self._weights(corr, q, k_pad, add_mask, nS, halo)
        out = self._halo_aggregate(A, v_pad, nS, halo)                                             # (B,nh,C,nS,Nu,dh)
        out = out.permute(0, 2, 3, 4, 1, 5).reshape(B, C, nS, Nu, D)
        return self.proj(out)


class _NeighbourhoodLayer(nn.Module):
    """Pre-LN block: unsampled tokens += attn(norm_q(unsampled), norm_kv(all)); unsampled tokens += FFN(norm2(.))."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                 attn_type="complex", rope_theta=None, ff=None):
        super().__init__()
        self.attn = NeighbourhoodCrossAttention(d_model, nhead, dropout, attn_type=attn_type, rope_theta=rope_theta)
        self.ff = ff if ff is not None else FeedForward(d_model, dim_feedforward, dropout, activation, True)
        self.norm_q = ComplexLayerNorm(d_model, eps=layer_norm_eps)
        self.norm_kv = ComplexLayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = ComplexLayerNorm(d_model, eps=layer_norm_eps)
        self.drop = ComplexDropout(dropout)

    def forward(self, h, unsampled_idx, nbr_idx, nbr_valid, halo):
        h_u = h.index_select(3, unsampled_idx)
        h_u = h_u + self.drop(self.attn(self.norm_q(h_u), self.norm_kv(h), nbr_idx, nbr_valid, unsampled_idx, halo))
        h_u = h_u + self.drop(self.ff(self.norm2(h_u)))
        h = h.clone()
        h[:, :, :, unsampled_idx] = h_u
        return h


class _NeighbourhoodStack(nn.Sequential):
    def __init__(self, layer, num_layers, tie_ffn=False):
        super().__init__(*[copy.deepcopy(layer) for _ in range(num_layers)])
        if tie_ffn:
            for l in self:
                l.ff = layer.ff

    def forward(self, h, unsampled_idx, nbr_idx, nbr_valid, halo):
        for layer in self:
            h = layer(h, unsampled_idx, nbr_idx, nbr_valid, halo)
        return h


class NeighbourhoodEncoder(nn.Module):
    """
    Stage: k-space (B, C, H, W) -> per-coil strip tokens -> num_layers of neighbourhood cross-attention on the
    unsampled columns -> linear head back to k-space. Returns a (B, C, H, W) residual that is zero at sampled columns.
    kx is zero-padded to a multiple of strip_len (the padded strip is a valid token; the padding is cropped on output).
    """
    def __init__(self, image_size, numCh=1, d_model=512, nhead=8, num_layers=1, dim_feedforward=None, dropout=0.1,
                 activation='relu', layer_norm_eps=1e-05, pos_emb_type="Rope-Axial", rope_theta=100.0,
                 attn_type="complex", strip_len=16, n_neighbours=3, kx_halo=1, keys_acquired_only=True,
                 ffn_sharing="none", shared_ffn=None, kspace_stem=None):
        super().__init__()
        if strip_len < 1 or n_neighbours < 1 or kx_halo < 0:
            raise ValueError("strip_len and n_neighbours must be >= 1 and kx_halo >= 0")
        H, W = pair(image_size)
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)
        self.strip_len = strip_len
        self.n_neighbours = n_neighbours
        self.kx_halo = kx_halo
        self.keys_acquired_only = keys_acquired_only
        self.num_strips = math.ceil(H / strip_len)
        self.pos_emb_type = pos_emb_type
        self.kspace_stem = build_kspace_stem(image_size, numCh, is_complex=True, **(kspace_stem or {}))

        self.to_token = nn.Linear(strip_len, d_model, dtype=torch.cfloat)
        self.coil_embedding = nn.Parameter(torch.randn(numCh, d_model, dtype=torch.cfloat))
        self.flag_embedding = nn.Parameter(torch.randn(2, d_model, dtype=torch.cfloat))
        self.mask_token = nn.Parameter(torch.randn(d_model, dtype=torch.cfloat))
        self.dropout = ComplexDropout(dropout)

        if pos_emb_type == "APE":
            self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.num_strips, W, d_model, dtype=torch.cfloat))
            rope = None
        elif pos_emb_type == "Rope-Axial":
            rope = rope_theta
        else:
            raise ValueError(f"NeighbourhoodEncoder supports pos_emb_type 'APE' or 'Rope-Axial', got '{pos_emb_type}'")

        if shared_ffn is None and ffn_sharing == "per_stage":
            shared_ffn = FeedForward(d_model, dim_feedforward, dropout, activation, True)
        layer = _NeighbourhoodLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                    attn_type=attn_type, rope_theta=rope, ff=shared_ffn)
        self.layers = _NeighbourhoodStack(layer, num_layers, tie_ffn=shared_ffn is not None)

        self.head = nn.Sequential(
            ComplexLayerNorm(d_model, eps=layer_norm_eps),
            nn.Linear(d_model, strip_len, dtype=torch.cfloat),
        )
        apply_trabelsi_(self.to_token, criterion="glorot")
        apply_trabelsi_(self.head[1], criterion="glorot")

    def neighbour_tables(self, sampled):
        candidates = sampled if self.keys_acquired_only else torch.ones_like(sampled)
        return build_neighbour_tables(candidates, self.n_neighbours)

    def forward(self, img, col_mask=None):
        if col_mask is None:
            raise ValueError("NeighbourhoodEncoder requires col_mask for sampled/unsampled routing")
        if self.kspace_stem is not None:
            img = self.kspace_stem(img)
        B, C, H, W = img.shape
        S, nS = self.strip_len, self.num_strips
        sampled = _extract_shared_column_mask(col_mask, W)
        unsampled_idx = (~sampled).nonzero(as_tuple=True)[0]
        if unsampled_idx.numel() == 0:
            return torch.zeros_like(img)
        nbr_idx, nbr_valid, _ = self.neighbour_tables(sampled)

        x = F.pad(img, (0, 0, 0, nS * S - H)) if nS * S != H else img
        x = x.reshape(B, C, nS, S, W).permute(0, 1, 2, 4, 3)                                     # (B,C,nS,W,S)
        # Unsampled columns start from the learned mask token only: their k-space content (zero-filled or a previous
        # stage's estimate) never enters this module, so the output depends on acquired data alone.
        h = torch.where(sampled[None, None, None, :, None], self.to_token(x), self.mask_token)
        h = h + self.coil_embedding[None, :, None, None, :]
        h = h + self.flag_embedding[sampled.long()][None, None, None, :, :]
        if self.pos_emb_type == "APE":
            h = h + self.pos_embedding
        h = self.dropout(h)

        h = self.layers(h, unsampled_idx, nbr_idx[unsampled_idx], nbr_valid[unsampled_idx], self.kx_halo)

        out = self.head(h).permute(0, 1, 2, 4, 3).reshape(B, C, nS * S, W)[:, :, :H]
        return out.masked_fill(sampled[None, None, None, :], 0)


class PerCoilCrossAxialEncoder(nn.Module):
    """
    crossAxialEncoder run on every coil independently with shared weights: coils are folded into the batch,
    so each coil's unsampled columns attend only to that coil's acquired columns (token = one coil's column,
    Linear(H -> d_model)). A learned coil-ID embedding is added to the column tokens so the shared weights know
    which coil they operate on. Returns (B, C, H, W) like crossAxialEncoder.
    """
    def __init__(self, image_size, numCh=1, d_model=512, nhead=8, num_layers=6, dim_feedforward=None,
                 dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=True,
                 device=None, dtype=None, norm=None,
                 pos_emb_type="Rope-Axial", rope_theta=100.0, attn_type="complex", row_stride=1,
                 ffn_sharing="none", shared_ffn=None, flattening_order="row_major", kspace_stem=None):
        super().__init__()
        if flattening_order != "row_major":
            raise ValueError("per-coil cross_axial requires flattening_order='row_major'")
        self.kspace_stem = build_kspace_stem(image_size, numCh, is_complex=True, **(kspace_stem or {}))
        self.inner = crossAxialEncoder(
            image_size, 1, d_model, nhead, num_layers, dim_feedforward, dropout, activation, layer_norm_eps,
            batch_first, device, dtype, norm, pos_emb_type=pos_emb_type, rope_theta=rope_theta,
            attn_type=attn_type, row_stride=row_stride, ffn_sharing=ffn_sharing, shared_ffn=shared_ffn,
            flattening_order=flattening_order, kspace_stem=None,
        )
        self.coil_embedding = nn.Parameter(torch.randn(numCh, 1, d_model, dtype=torch.cfloat))

    def forward(self, img, col_mask=None):
        if col_mask is None:
            raise ValueError("crossAxialEncoder requires col_mask for sampled/unsampled routing")
        if self.kspace_stem is not None:
            img = self.kspace_stem(img)
        B, C, H, W = img.shape
        enc = self.inner
        x = enc.to_vertical_embedding(img.reshape(B * C, 1, H, W))                       # (B*C, W, d)
        x = x + self.coil_embedding.repeat(B, 1, 1)
        if enc.pos_emb_type == "APE":
            x = x + enc.vertical_pos_embedding
        x = enc.dropout(x)
        sampled = _extract_shared_column_mask(col_mask, W)
        sampled_idx = sampled.nonzero(as_tuple=True)[0]
        unsampled_idx = (~sampled).nonzero(as_tuple=True)[0]
        x = enc.verticalEncoder(x, sampled_idx, unsampled_idx)
        return enc.vertical_mlp_head(x).reshape(B, C, H, W)


def build_cross_axial_encoder(coil_mode, image_size, numCh, *args, **kwargs):
    """'folded': the standard crossAxialEncoder (all coils in one column token); 'per_coil': PerCoilCrossAxialEncoder."""
    if coil_mode == "folded":
        return crossAxialEncoder(image_size, numCh, *args, **kwargs)
    if coil_mode == "per_coil":
        return PerCoilCrossAxialEncoder(image_size, numCh, *args, **kwargs)
    raise ValueError(f"Unknown cross_axial_coil_mode '{coil_mode}'. Choose from: folded, per_coil")


class CrossAxialNeighbourhoodEncoder(nn.Module):
    """
    Global sampled->unsampled column cross-attention (crossAxialEncoder) combined with the local per-coil
    neighbourhood attention on the same stage input. combine:
      "sum"         global(x) + local(x)
      "gate"        global(x) + sigmoid(gate) * local(x),  learned scalar gate initialised at 0
      "sequential"  g = global(x); g + local(x + g)
      "local_only"  local(x)  (the global branch is not built)
    global_coil_mode: "folded" (standard crossAxialEncoder) or "per_coil" (PerCoilCrossAxialEncoder).
    The optional k-space stem is built once here and applied before both branches.
    """
    def __init__(self, image_size, numCh=1, d_model=512, nhead=8, num_layers=6, dim_feedforward=None,
                 dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=True,
                 device=None, dtype=None, norm=None,
                 pos_emb_type="Rope-Axial", rope_theta=100.0, attn_type="complex", row_stride=1,
                 ffn_sharing="none", shared_ffn=None, flattening_order="row_major", kspace_stem=None,
                 strip_len=16, n_neighbours=3, kx_halo=1, keys_acquired_only=True,
                 local_attn_type="complex", combine="sum", local_layers=None, global_coil_mode="folded"):
        super().__init__()
        if combine not in NEIGHBOURHOOD_COMBINE_MODES:
            raise ValueError(f"Unknown neighbourhood combine mode '{combine}'. Choose from: {NEIGHBOURHOOD_COMBINE_MODES}")
        if flattening_order != "row_major":
            raise ValueError("cross_axial_neighbourhood requires flattening_order='row_major'")
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)
        self.combine = combine
        self.kspace_stem = build_kspace_stem(image_size, numCh, is_complex=True, **(kspace_stem or {}))
        if shared_ffn is None and ffn_sharing == "per_stage":
            shared_ffn = FeedForward(d_model, dim_feedforward, dropout, activation, True)
        self.global_branch = None if combine == "local_only" else build_cross_axial_encoder(
            global_coil_mode,
            image_size, numCh, d_model, nhead, num_layers, dim_feedforward, dropout, activation, layer_norm_eps,
            batch_first, device, dtype, norm, pos_emb_type=pos_emb_type, rope_theta=rope_theta, attn_type=attn_type,
            row_stride=row_stride, ffn_sharing=ffn_sharing, shared_ffn=shared_ffn,
            flattening_order=flattening_order, kspace_stem=None,
        )
        self.local_branch = NeighbourhoodEncoder(
            image_size, numCh, d_model, nhead, local_layers or num_layers, dim_feedforward, dropout, activation,
            layer_norm_eps, pos_emb_type=pos_emb_type, rope_theta=rope_theta, attn_type=local_attn_type,
            strip_len=strip_len, n_neighbours=n_neighbours, kx_halo=kx_halo, keys_acquired_only=keys_acquired_only,
            ffn_sharing=ffn_sharing, shared_ffn=shared_ffn, kspace_stem=None,
        )
        if combine == "gate":
            self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, img, col_mask=None):
        if self.kspace_stem is not None:
            img = self.kspace_stem(img)
        if self.combine == "local_only":
            return self.local_branch(img, col_mask=col_mask)
        g = self.global_branch(img, col_mask=col_mask)
        if self.combine == "sequential":
            return g + self.local_branch(img + g, col_mask=col_mask)
        local = self.local_branch(img, col_mask=col_mask)
        if self.combine == "gate":
            return g + torch.sigmoid(self.gate) * local
        return g + local
