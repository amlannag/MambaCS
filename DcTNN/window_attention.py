"""
Windowed local self-attention over axial column-segment tokens (Swin-style windows on k-space).

Tokens are the segmented axial tokens of `axialEncoder(col_segments=S, token_axis="vertical")`: every k-space column is cut
into S equal kx segments and token (s, w) = the H/S values (all coils) of segment s of column w, laid out as a row-major
(S, W) grid. Unlike the neighbourhood encoder nothing is routed by the sampling mask: sampled and unsampled tokens all
carry the current k-space estimate, all are queries and keys, and every token is updated.

Each encoder layer runs FULL self-attention inside non-overlapping windows of `window_height` segments x `window_width`
columns. With `window_shift=True` every odd layer cyclically shifts the grid by half a window in both directions before
partitioning (Swin), with the usual attention mask so tokens that wrapped around the k-space edges cannot attend across
the seam; information therefore crosses window borders over consecutive layers. RoPE is the 2-D (segment, column) RoPE of
the axial encoders applied with window-local coordinates: RoPE scores only depend on coordinate differences, so one
position table serves every window.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .attention_layer import TransformerEncoderLayer
from .encoders import pair
from .kspace_ops import build_kspace_stem
from .rope_vit import compute_axial_cis, compute_axial_cis_complex
from .util import ComplexDropout, FeedForward, get_mlp_head, get_to_embedding, _COMPLEX_ATTN_TYPES

WINDOW_ATTN_TYPES = ("complex", "real_valued", "phase_aware", "standard")


def window_partition_indices(grid, window, shift=(0, 0)):
    """
    (nWin, K) flat row-major token indices of every window of a (S, W) grid, optionally after a cyclic shift of the grid
    by `shift` (tokens are rolled UP/LEFT by shift so the windows start shift rows/cols later on the original grid).
    """
    S, W = grid
    wh, ww = window
    if S % wh or W % ww:
        raise ValueError(f"window {window} must tile the token grid {grid}")
    idx = torch.arange(S * W).view(S, W)
    if any(shift):
        idx = torch.roll(idx, shifts=(-shift[0], -shift[1]), dims=(0, 1))
    return idx.view(S // wh, wh, W // ww, ww).permute(0, 2, 1, 3).reshape(-1, wh * ww)


def shifted_window_mask(grid, window, shift):
    """
    Swin attention mask (nWin, K, K) for a shifted partition: 0 where both tokens come from the same contiguous region of
    the ORIGINAL grid, -inf where one of them wrapped around the k-space edge.
    """
    S, W = grid
    wh, ww = window
    sh, sw = shift
    region = torch.zeros(S, W, dtype=torch.long)
    row_bands = (slice(0, S - wh), slice(S - wh, S - sh), slice(S - sh, S))
    col_bands = (slice(0, W - ww), slice(W - ww, W - sw), slice(W - sw, W))
    for i, rows in enumerate(row_bands):
        for j, cols in enumerate(col_bands):
            region[rows, cols] = 3 * i + j
    windows = region.view(S // wh, wh, W // ww, ww).permute(0, 2, 1, 3).reshape(-1, wh * ww)
    different = windows.unsqueeze(2) != windows.unsqueeze(1)
    return torch.zeros(different.shape, dtype=torch.float32).masked_fill(different, float("-inf"))


class WindowAxialEncoder(nn.Module):
    def __init__(self, image_size, numCh=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=None, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-05, pos_emb_type="Rope-Axial", rope_theta=100.0,
                 attn_type="complex", col_segments=20, window_height=5, window_width=16, window_shift=True,
                 ffn_sharing="none", shared_ffn=None, kspace_stem=None):
        super().__init__()
        if attn_type not in WINDOW_ATTN_TYPES:
            raise ValueError(f"window_axial supports attn_type in {WINDOW_ATTN_TYPES}, got {attn_type!r} "
                             "(multi-scale Q/K needs the full token sequence)")
        if pos_emb_type not in ("Rope-Axial", "Rope-Mixed", "APE"):
            raise ValueError(f"window_axial supports pos_emb_type 'Rope-Axial' or 'APE', got {pos_emb_type!r}")
        H, W = pair(image_size)
        S, wh, ww = int(col_segments), int(window_height), int(window_width)
        if S < 1 or wh < 1 or ww < 1:
            raise ValueError("col_segments, window_height and window_width must be >= 1")
        if H % S:
            raise ValueError(f"image height {H} must be divisible by col_segments={S}")
        if S % wh or W % ww:
            raise ValueError(f"window ({wh} segments x {ww} columns) must tile the ({S} x {W}) token grid")
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)
        self.is_complex = attn_type in _COMPLEX_ATTN_TYPES
        self.grid, self.window = (S, W), (wh, ww)
        self.col_segments = S
        self.pos_emb_type = pos_emb_type
        self.d_model = d_model
        self.kspace_stem = build_kspace_stem(image_size, numCh, is_complex=self.is_complex, **(kspace_stem or {}))
        self.to_embedding = get_to_embedding("axial", image_height=H, image_width=W, numCh=numCh, d_model=d_model,
                                             is_complex=self.is_complex, col_segments=S)[1]
        self.head = get_mlp_head("axial", d_model, numCh=numCh, image_height=H, image_width=W, is_complex=self.is_complex,
                                 layer_norm_eps=layer_norm_eps, col_segments=S)[1]
        self.dropout = ComplexDropout(dropout) if self.is_complex else nn.Dropout(dropout)

        head_dim = d_model // nhead
        if pos_emb_type == "APE":
            freqs = None
            self.pos_embedding = nn.Parameter(torch.randn(1, S * W, d_model, dtype=torch.cfloat if self.is_complex else None))
        else:
            cis_fn = compute_axial_cis_complex if self.is_complex else compute_axial_cis
            freqs = cis_fn(dim=head_dim, end_x=W, end_y=S, theta=rope_theta)
        if shared_ffn is None and ffn_sharing == "per_stage":
            shared_ffn = FeedForward(d_model, dim_feedforward, dropout, activation, self.is_complex)
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                        freqs_cis=freqs, attn_type=attn_type, ff=shared_ffn)
        self.layers = nn.ModuleList([layer] + [_clone_layer(layer, shared_ffn) for _ in range(num_layers - 1)])

        # window-local RoPE coordinates: token (ls, lw) of any window uses the rotation of grid token (ls, lw)
        local = torch.arange(wh).view(wh, 1) * W + torch.arange(ww).view(1, ww)
        self.register_buffer("local_positions", local.reshape(-1), persistent=False)
        self.window_shift = bool(window_shift) and (wh > 1 or ww > 1) and (S > wh or W > ww)
        shift = (wh // 2, ww // 2)
        self.register_buffer("windows_plain", window_partition_indices(self.grid, self.window), persistent=False)
        self.register_buffer("inverse_plain", torch.argsort(self.windows_plain.reshape(-1)), persistent=False)
        if self.window_shift:
            shifted = window_partition_indices(self.grid, self.window, shift)
            self.register_buffer("windows_shifted", shifted, persistent=False)
            self.register_buffer("inverse_shifted", torch.argsort(shifted.reshape(-1)), persistent=False)
            self.register_buffer("mask_shifted", shifted_window_mask(self.grid, self.window, shift), persistent=False)

    def _layer_windows(self, layer_index):
        if self.window_shift and layer_index % 2 == 1:
            return self.windows_shifted, self.inverse_shifted, self.mask_shifted
        return self.windows_plain, self.inverse_plain, None

    def forward(self, img, col_mask=None):
        if self.kspace_stem is not None:
            img = self.kspace_stem(img)
        x = self.to_embedding(img)                                              # (B, S*W, d), token (s, w) at s*W + w
        if self.pos_emb_type == "APE":
            x = x + self.pos_embedding
        x = self.dropout(x)
        B, N, d = x.shape
        for i, layer in enumerate(self.layers):
            windows, inverse, mask = self._layer_windows(i)
            n_win, k = windows.shape
            xw = x[:, windows.reshape(-1)].reshape(B * n_win, k, d)
            attn_mask = None if mask is None else mask.to(x.real.dtype if x.is_complex() else x.dtype).repeat(B, 1, 1).unsqueeze(1)
            xw = layer(xw, attn_mask=attn_mask, positions=None if self.pos_emb_type == "APE" else self.local_positions)
            x = xw.reshape(B, n_win * k, d)[:, inverse]
        return self.head(x)


def _clone_layer(layer, shared_ffn):
    import copy
    clone = copy.deepcopy(layer)
    if shared_ffn is not None:
        clone.ff = layer.ff
    return clone
