"""
Windowed local self-attention over k-space tokens (Swin-style windows on k-space).

Tokens, two options:
  * column segments (default, `patch_size=None`): the segmented axial tokens of `axialEncoder(col_segments=S,
    token_axis="vertical")`: every k-space column is cut into S equal kx segments and token (s, w) = the H/S values (all
    coils) of segment s of column w, laid out as a row-major (S, W) grid. These tokens are 16 x 1 slivers: a vertical
    streak lies along one token, a horizontal or diagonal streak is one sample in each of many tokens.
  * isotropic patches (`patch_size=(ph, pw)`): token = a ph x pw k-space patch (all coils) on an (H/ph, W/pw) grid, so no
    direction is privileged inside a token.
Unlike the neighbourhood encoder nothing is routed by the sampling mask: sampled and unsampled tokens all carry the current
k-space estimate, all are queries and keys, and every token is updated.

Each encoder layer runs FULL self-attention inside non-overlapping windows of `window_height` x `window_width` tokens. With
`window_shift=True` every odd layer cyclically shifts the grid by half a (base) window in both directions before
partitioning (Swin), with an attention mask so tokens that wrapped around the k-space edges cannot attend across the seam;
information therefore crosses window borders over consecutive layers.

Radius-dependent windows (`radial_scales=(1, 2, 4, ...)`, `radial_bands`): the grid is tiled by the largest window
(base window x max scale) and every window is split quad-tree style into four windows of the next-finer scale while its
closest token lies inside the band of that finer scale (`radial_bands[k]` = outer radius of scale k as a fraction of half
the grid's shorter side, DC at the grid centre). Small windows therefore sit around DC where k-space energy and structure
are dense, and windows grow with |k| where the spectrum is sparse. Windows of different sizes run as separate batches
through the same layer.

RoPE is the 2-D (row, column) RoPE of the axial encoders applied with window-local coordinates: RoPE scores only depend on
coordinate differences, so one position table serves every window of a size class.
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


def seam_mask(windows, grid, shift):
    """
    Attention mask (nWin, K, K) for windows of ORIGINAL-grid token indices taken from a grid rolled by `shift`:
    -inf between a token that wrapped around the k-space edge (original row < shift[0] / col < shift[1]) and one that
    did not. Equivalent to `shifted_window_mask` for uniform windows and valid for windows of any size.
    """
    S, W = grid
    rows, cols = windows // W, windows % W
    label = 2 * (rows < shift[0]).long() + (cols < shift[1]).long()
    different = label.unsqueeze(2) != label.unsqueeze(1)
    return torch.zeros(different.shape, dtype=torch.float32).masked_fill(different, float("-inf"))


def build_window_groups(grid, window, shift=(0, 0), radial_scales=(1,), radial_bands=()):
    """
    Partition an (S, W) token grid into windows, optionally rolled by `shift` and refined towards DC.

    radial_scales : increasing window multipliers, each twice the previous (quad-tree split), e.g. (1, 2, 4).
    radial_bands  : outer radius of every scale but the largest, as a fraction of half the grid's shorter side
                    (DC at (S // 2, W // 2)); a window of scale radial_scales[k] is split into four windows of scale
                    radial_scales[k - 1] while its closest token has radius < radial_bands[k - 1].
    Returns a list of (window_size (wh, ww), indices (nWin, wh * ww)) per size class, largest windows first; the
    indices are original-grid token indices (row-major), every token appearing exactly once.
    """
    S, W = grid
    wh, ww = window
    scales = tuple(int(s) for s in radial_scales)
    if len(scales) < 1 or scales[0] != 1 or any(b != 2 * a for a, b in zip(scales, scales[1:])):
        raise ValueError(f"radial_scales must start at 1 and double at every step, got {radial_scales}")
    if len(radial_bands) != len(scales) - 1:
        raise ValueError(f"radial_bands needs one outer radius per scale but the largest: {len(scales) - 1} values")
    if any(b <= 0 for b in radial_bands) or any(b <= a for a, b in zip(radial_bands, radial_bands[1:])):
        raise ValueError(f"radial_bands must be positive and increasing, got {radial_bands}")
    top = scales[-1]
    if S % (wh * top) or W % (ww * top):
        raise ValueError(f"the largest window ({wh * top} x {ww * top}) must tile the token grid {grid}")

    idx = torch.arange(S * W).view(S, W)
    if any(shift):
        idx = torch.roll(idx, shifts=(-shift[0], -shift[1]), dims=(0, 1))
    radius = torch.sqrt(
        (torch.arange(S, dtype=torch.float64).view(S, 1) - S // 2) ** 2
        + (torch.arange(W, dtype=torch.float64).view(1, W) - W // 2) ** 2
    ) / (min(S, W) / 2)

    def split(block, level):
        """block: (h, w) index tile at scale scales[level]; yields (level, flat indices) leaves."""
        if level > 0 and radius.view(-1)[block.reshape(-1)].min().item() < radial_bands[level - 1]:
            h2, w2 = block.shape[0] // 2, block.shape[1] // 2
            for r in (0, h2):
                for c in (0, w2):
                    yield from split(block[r:r + h2, c:c + w2], level - 1)
        else:
            yield level, block.reshape(-1)

    th, tw = wh * top, ww * top
    leaves = {level: [] for level in range(len(scales))}
    for r in range(0, S, th):
        for c in range(0, W, tw):
            for level, flat in split(idx[r:r + th, c:c + tw], len(scales) - 1):
                leaves[level].append(flat)
    groups = []
    for level in reversed(range(len(scales))):
        if leaves[level]:
            s = scales[level]
            groups.append(((wh * s, ww * s), torch.stack(leaves[level])))
    return groups


class WindowAxialEncoder(nn.Module):
    def __init__(self, image_size, numCh=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=None, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-05, pos_emb_type="Rope-Axial", rope_theta=100.0,
                 attn_type="complex", col_segments=20, window_height=5, window_width=16, window_shift=True,
                 ffn_sharing="none", shared_ffn=None, kspace_stem=None, patch_size=None, radial_scales=(1,),
                 radial_bands=()):
        super().__init__()
        if attn_type not in WINDOW_ATTN_TYPES:
            raise ValueError(f"window_axial supports attn_type in {WINDOW_ATTN_TYPES}, got {attn_type!r} "
                             "(multi-scale Q/K needs the full token sequence)")
        if pos_emb_type not in ("Rope-Axial", "Rope-Mixed", "APE"):
            raise ValueError(f"window_axial supports pos_emb_type 'Rope-Axial' or 'APE', got {pos_emb_type!r}")
        H, W = pair(image_size)
        wh, ww = int(window_height), int(window_width)
        if wh < 1 or ww < 1:
            raise ValueError("window_height and window_width must be >= 1")
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)
        self.is_complex = attn_type in _COMPLEX_ATTN_TYPES
        self.pos_emb_type = pos_emb_type
        self.d_model = d_model
        self.kspace_stem = build_kspace_stem(image_size, numCh, is_complex=self.is_complex, **(kspace_stem or {}))

        if patch_size is None:
            S = int(col_segments)
            if S < 1:
                raise ValueError("col_segments must be >= 1")
            if H % S:
                raise ValueError(f"image height {H} must be divisible by col_segments={S}")
            self.grid = (S, W)
            self.to_embedding = get_to_embedding("axial", image_height=H, image_width=W, numCh=numCh, d_model=d_model,
                                                 is_complex=self.is_complex, col_segments=S)[1]
            self.head = get_mlp_head("axial", d_model, numCh=numCh, image_height=H, image_width=W,
                                     is_complex=self.is_complex, layer_norm_eps=layer_norm_eps, col_segments=S)[1]
        else:
            ph, pw = pair(patch_size)
            if ph < 1 or pw < 1 or H % ph or W % pw:
                raise ValueError(f"patch_size {patch_size} must be positive and tile the {H} x {W} k-space grid")
            self.grid = (H // ph, W // pw)
            patch_dim = ph * pw * numCh
            self.to_embedding = get_to_embedding("patch", patch_height=ph, patch_width=pw, patch_dim=patch_dim,
                                                 d_model=d_model, image_height=H, image_width=W, numCh=numCh,
                                                 is_complex=self.is_complex)
            self.head = get_mlp_head("patch", d_model, patch_dim=patch_dim, patch_height=ph, patch_width=pw,
                                     grid_h=self.grid[0], numCh=numCh, image_height=H, image_width=W,
                                     is_complex=self.is_complex, layer_norm_eps=layer_norm_eps)
        self.patch_size = None if patch_size is None else (ph, pw)
        self.col_segments = self.grid[0] if patch_size is None else None
        self.window = (wh, ww)
        self.radial_scales, self.radial_bands = tuple(radial_scales), tuple(radial_bands)
        S, W = self.grid
        if S % wh or W % ww:
            raise ValueError(f"window ({wh} x {ww} tokens) must tile the ({S} x {W}) token grid")
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

        self.window_shift = bool(window_shift) and (wh > 1 or ww > 1) and (S > wh or W > ww)
        shift = (wh // 2, ww // 2)
        # One group per window size class: (indices (nWin, K), window-local RoPE positions (K,), seam mask or None).
        # Window-local coordinates: token (ls, lw) of any window of a class uses the rotation of grid token (ls, lw).
        self._group_shapes = {}
        for name, sh in (("plain", (0, 0)), ("shifted", shift)):
            if name == "shifted" and not self.window_shift:
                continue
            groups = build_window_groups(self.grid, self.window, sh, self.radial_scales, self.radial_bands)
            self._group_shapes[name] = []
            for g, ((gh, gw), windows) in enumerate(groups):
                local = (torch.arange(gh).view(gh, 1) * W + torch.arange(gw).view(1, gw)).reshape(-1)
                self.register_buffer(f"windows_{name}_{g}", windows, persistent=False)
                self.register_buffer(f"positions_{name}_{g}", local, persistent=False)
                if any(sh):
                    self.register_buffer(f"mask_{name}_{g}", seam_mask(windows, self.grid, sh), persistent=False)
                self._group_shapes[name].append((gh, gw))
            flat = torch.cat([w.reshape(-1) for _, w in groups])
            self.register_buffer(f"inverse_{name}", torch.argsort(flat), persistent=False)

    @property
    def windows_plain(self):
        """Uniform-window partition (nWin, K); only defined without radial refinement (kept for tests/introspection)."""
        if len(self._group_shapes["plain"]) != 1:
            raise AttributeError("windows_plain is undefined with radial_scales (several window sizes)")
        return self.windows_plain_0

    def _layer_groups(self, layer_index):
        name = "shifted" if self.window_shift and layer_index % 2 == 1 else "plain"
        groups = [
            (getattr(self, f"windows_{name}_{g}"), getattr(self, f"positions_{name}_{g}"),
             getattr(self, f"mask_{name}_{g}", None))
            for g in range(len(self._group_shapes[name]))
        ]
        return groups, getattr(self, f"inverse_{name}")

    def forward(self, img, col_mask=None):
        if self.kspace_stem is not None:
            img = self.kspace_stem(img)
        x = self.to_embedding(img)                                              # (B, S*W, d), token (r, c) at r*W + c
        if self.pos_emb_type == "APE":
            x = x + self.pos_embedding
        x = self.dropout(x)
        B, N, d = x.shape
        real_dtype = x.real.dtype if x.is_complex() else x.dtype
        for i, layer in enumerate(self.layers):
            groups, inverse = self._layer_groups(i)
            outputs = []
            for windows, positions, mask in groups:
                n_win, k = windows.shape
                xw = x[:, windows.reshape(-1)].reshape(B * n_win, k, d)
                attn_mask = None if mask is None else mask.to(real_dtype).repeat(B, 1, 1).unsqueeze(1)
                xw = layer(xw, attn_mask=attn_mask, positions=None if self.pos_emb_type == "APE" else positions)
                outputs.append(xw.reshape(B, n_win * k, d))
            x = torch.cat(outputs, dim=1)[:, inverse]
        return self.head(x)


def _clone_layer(layer, shared_ffn):
    import copy
    clone = copy.deepcopy(layer)
    if shared_ffn is not None:
        clone.ff = layer.ff
    return clone
