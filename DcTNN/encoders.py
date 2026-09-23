import copy
import torch
from torch import nn
from einops.layers.torch import Rearrange
from .rope_vit import (compute_axial_cis, compute_mixed_cis,
                        compute_axial_cis_complex, compute_mixed_cis_complex)
from .attention_layer import (
    TransformerEncoderLayer,
    TransformerEncoder,
    CrossAttentionEncoderLayer,
)
from .util import (
    get_to_embedding,
    get_from_embedding,
    get_mlp_head,
    ComplexLayerNorm,
    ComplexDropout,
    FeedForward,
    PatchUnroller,
    _COMPLEX_ATTN_TYPES,
)
from .complex_init import apply_trabelsi_

def _build_vertical_attn_mask(sampled: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Build an additive [W, W] attention mask for vertical (column) attention.

    sampled : bool tensor [W] — True where column is sampled (mask=1).
    mode    : "lenient" — sampled queries attend only to sampled keys;
                          unsampled queries attend to all.
              "strict"  — all queries attend only to sampled keys.
    Returns float tensor [W, W] with 0 (allow) or -inf (block).
    """
    W = sampled.shape[0]
    unsampled = ~sampled
    if mode == "strict":
        mask = torch.zeros(W, W, device=sampled.device)
        mask[:, unsampled] = float("-inf")
    elif mode == "lenient":
        block = sampled.unsqueeze(1) & unsampled.unsqueeze(0)
        mask = torch.zeros(W, W, device=sampled.device).masked_fill(block, float("-inf"))
    else:
        mask = torch.zeros(W, W, device=sampled.device)
    return mask


def _extract_shared_column_mask(col_mask: torch.Tensor, width: int) -> torch.Tensor:
    """
    Collapse a broadcast column mask to [W].

    Current training uses a single broadcast mask shared across the batch.
    Reject per-example masks so the assumption stays explicit.
    """
    if col_mask.shape[-1] != width:
        raise ValueError(
            f"Column mask width {col_mask.shape[-1]} does not match token width {width}"
        )

    flat_mask = col_mask.reshape(-1, width)
    reference = flat_mask[0]
    if flat_mask.shape[0] > 1 and not torch.equal(flat_mask, reference.unsqueeze(0).expand_as(flat_mask)):
        raise ValueError(
            "Per-example column masks are not supported in axial attention. "
            "Expected one broadcast mask shared across the batch."
        )
    return reference.bool()


def pair(t):
    return tuple(t) if isinstance(t, (tuple, list)) else (t, t)


class _CrossAxialInnerLayer(nn.Module):
    """
    One vertical-only block:
    1. sampled -> unsampled cross-attention
    2. unsampled -> unsampled self-attention
    3. sampled -> unsampled cross-attention
    4. unsampled -> unsampled self-attention
    5. scatter updated unsampled tokens back into the full token tensor
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                 freqs_cis=None, attn_type="complex", ff=None):
        super().__init__()
        self.cross1 = CrossAttentionEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation,
            layer_norm_eps, freqs_cis=freqs_cis, attn_type=attn_type, ff=ff
        )
        self.self_attn1 = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation,
            layer_norm_eps, freqs_cis=freqs_cis, attn_type=attn_type, ff=ff
        )
        self.cross2 = CrossAttentionEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation,
            layer_norm_eps, freqs_cis=freqs_cis, attn_type=attn_type, ff=ff
        )
        self.self_attn2 = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation,
            layer_norm_eps, freqs_cis=freqs_cis, attn_type=attn_type, ff=ff
        )

    def forward(self, x, sampled_idx, unsampled_idx):
        if unsampled_idx.numel() == 0:
            return x
        if sampled_idx.numel() == 0:
            raise ValueError("cross_axial requires at least one sampled column")

        x_unsampled = x.index_select(1, unsampled_idx)
        x_sampled = x.index_select(1, sampled_idx)
        x_unsampled = self.cross1(
            x_unsampled,
            x_sampled,
            q_positions=unsampled_idx,
            kv_positions=sampled_idx,
        )
        x_unsampled = self.self_attn1(x_unsampled, positions=unsampled_idx)
        x_unsampled = self.cross2(
            x_unsampled,
            x_sampled,
            q_positions=unsampled_idx,
            kv_positions=sampled_idx,
        )
        x_unsampled = self.self_attn2(x_unsampled, positions=unsampled_idx)

        x = x.clone()
        x[:, unsampled_idx, :] = x_unsampled
        return x


class _CrossAxialEncoderStack(nn.Sequential):
    def __init__(self, encoder_layer, num_layers, tie_ffn=False):
        super().__init__(*[copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        if tie_ffn:
            # Re-point every deep-copied layer's FFN at the template's shared instance.
            for layer in self:
                layer.cross1.ff = encoder_layer.cross1.ff
                layer.self_attn1.ff = encoder_layer.self_attn1.ff
                layer.cross2.ff = encoder_layer.cross2.ff
                layer.self_attn2.ff = encoder_layer.self_attn2.ff

    def forward(self, x, sampled_idx, unsampled_idx):
        for layer in self:
            x = layer(x, sampled_idx, unsampled_idx)
        return x


# ---------------------------------------------------------------------------
# Base encoder for patch-like tokenisation schemes
# ---------------------------------------------------------------------------

class BaseTokenEncoder(nn.Module):
    """
    Shared base for TokenEncoder.

    Subclasses must set the following attributes in __init__ before calling
    _setup_pos_emb():
        self.pos_emb_type, self.d_model, self.nhead, self.is_complex
        self.to_embedding, self.mlp_head, self.dropout
    """

    def _setup_pos_emb(self, grid_h, grid_w, num_patches, num_layers,
                       d_model, nhead, dim_feedforward, dropout, activation,
                       layer_norm_eps, batch_first, device, dtype, norm,
                       rope_theta, rope_mixed_rotate, attn_type,
                       ffn_sharing="none", shared_ffn=None):

        if self.pos_emb_type == "APE":
            dtype = torch.cfloat if self.is_complex else None
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model, dtype=dtype))
            freqs_cis = None

        elif self.pos_emb_type == "Rope-Axial":
            head_dim = d_model // nhead
            cis_fn = compute_axial_cis_complex if self.is_complex else compute_axial_cis
            freqs_cis = cis_fn(dim=head_dim, end_x=grid_w, end_y=grid_h, theta=rope_theta)

        elif self.pos_emb_type == "Rope-Mixed":
            head_dim = d_model // nhead
            cis_fn = compute_mixed_cis_complex if self.is_complex else compute_mixed_cis
            freqs_cis = cis_fn(dim=head_dim, end_x=grid_w, end_y=grid_h, theta=rope_theta)


        if shared_ffn is None and ffn_sharing == "per_stage":
            shared_ffn = FeedForward(d_model, dim_feedforward, dropout, activation, self.is_complex)

        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                        layer_norm_eps, freqs_cis=freqs_cis, attn_type=attn_type,
                                        ff=shared_ffn)
        self.encoder = TransformerEncoder(layer, num_layers, tie_ffn=shared_ffn is not None)

    def forward(self, img):
        x = self.to_embedding(img)
        if self.pos_emb_type == "APE":
            x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.mlp_head(x)
        return x


# ---------------------------------------------------------------------------
# Encoder implementations
# ---------------------------------------------------------------------------

class TokenEncoder(BaseTokenEncoder):
    """
    Unified encoder supporting patch and kaleidoscope tokenisation strategies.
    tokenizer_type="patch"        — local N×N patch tokens
    tokenizer_type="kaleidoscope" — globally-spaced pixel tokens
    """
    def __init__(self, image_size, patch_size, numCh=1, tokenizer_type="patch", d_model=512, nhead=8,
                num_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05,
                batch_first=True, device=None, dtype=None, norm=None,
                pos_emb_type="APE", rope_theta=100.0, rope_mixed_rotate=True,
                attn_type="standard", ffn_sharing="none", shared_ffn=None, flattening_order="row_major"):
        super().__init__()

        self.pos_emb_type = pos_emb_type
        self.d_model = d_model
        self.nhead = nhead
        self.is_complex = attn_type in _COMPLEX_ATTN_TYPES

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        grid_h = image_height // patch_height
        grid_w = image_width // patch_width
        num_patches = grid_h * grid_w
        patch_dim = patch_height * patch_width * numCh

        self.to_embedding = get_to_embedding(tokenizer_type, patch_height, patch_width, patch_dim, d_model,
                                             image_height=image_height, image_width=image_width,
                                             is_complex=self.is_complex, flattening_order=flattening_order)
        self.mlp_head = get_mlp_head(tokenizer_type, d_model, patch_dim, patch_height, patch_width,
                                     grid_h, numCh, image_height=image_height, image_width=image_width,
                                     is_complex=self.is_complex, flattening_order=flattening_order,
                                     layer_norm_eps=layer_norm_eps)
        
        self.dropout = ComplexDropout(dropout) if self.is_complex else nn.Dropout(dropout)

        self._setup_pos_emb(grid_h, grid_w, num_patches, num_layers, d_model, nhead,
                            dim_feedforward, dropout, activation, layer_norm_eps,
                            batch_first, device, dtype, norm, rope_theta, rope_mixed_rotate,
                            attn_type, ffn_sharing=ffn_sharing, shared_ffn=shared_ffn)


class axialEncoder(nn.Module):
    """
    Standard Encoder that utilizes axial attention (separate row and column transformers).
    """
    def __init__(self, image_size, numCh=1, d_model=512, nhead=8, num_layers=6, dim_feedforward=None,
                    dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=True,
                    device=None, dtype=None, norm=None,
                    pos_emb_type="APE", rope_theta=100.0, attn_type="standard", row_stride=1,
                    mask_vertical_attn="none", ffn_sharing="none", shared_ffn=None,
                    flattening_order="row_major"):
        super().__init__()

        self.pos_emb_type = pos_emb_type
        self.d_model = d_model
        self.is_complex = attn_type in _COMPLEX_ATTN_TYPES
        self.mask_vertical_attn = mask_vertical_attn
        if mask_vertical_attn == "cross":
            raise ValueError(
                "mask_vertical_attn='cross' is no longer supported on axialEncoder. "
                "Use the 'cross_axial' encoder family instead."
            )

        image_height, image_width = pair(image_size)
        h_tokens = image_height // row_stride  # horizontal token count after row grouping

        self.to_horizontal_embedding, self.to_vertical_embedding = get_to_embedding(
            "axial", image_height=image_height, image_width=image_width, numCh=numCh, d_model=d_model,
            row_stride=row_stride, is_complex=self.is_complex, flattening_order=flattening_order)

        self.horizontal_mlp_head, self.vertical_mlp_head = get_mlp_head(
            "axial", d_model, numCh=numCh, image_height=image_height, image_width=image_width,
            row_stride=row_stride, is_complex=self.is_complex, flattening_order=flattening_order,
            layer_norm_eps=layer_norm_eps)

        self.dropout = ComplexDropout(dropout) if self.is_complex else nn.Dropout(dropout)

        numLayers = max(num_layers // 2, 1)
        head_dim = d_model // nhead
        cis_fn = compute_axial_cis_complex if self.is_complex else compute_axial_cis

        if pos_emb_type == "APE":
            freqs_h = None
            freqs_v = None
            ape_dtype = torch.cfloat if self.is_complex else None
            self.horizontal_pos_embedding = nn.Parameter(torch.randn(1, h_tokens,     d_model, dtype=ape_dtype))
            self.vertical_pos_embedding   = nn.Parameter(torch.randn(1, image_width,  d_model, dtype=ape_dtype))

        elif pos_emb_type in ("Rope-Axial", "Rope-Mixed"):
            freqs_h = cis_fn(dim=head_dim, end_x=h_tokens,    end_y=1, theta=rope_theta)
            freqs_v = cis_fn(dim=head_dim, end_x=image_width, end_y=1, theta=rope_theta)

        if shared_ffn is None and ffn_sharing == "per_stage":
            shared_ffn = FeedForward(d_model, dim_feedforward, dropout, activation, self.is_complex)

        h_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                          layer_norm_eps, freqs_cis=freqs_h, attn_type=attn_type,
                                          ff=shared_ffn)
        v_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                          layer_norm_eps, freqs_cis=freqs_v, attn_type=attn_type,
                                          ff=shared_ffn)
        self.horizontalEncoder = TransformerEncoder(h_layer, numLayers, tie_ffn=shared_ffn is not None)
        self.verticalEncoder = TransformerEncoder(v_layer, numLayers, tie_ffn=shared_ffn is not None)

    def forward(self, img, col_mask=None):
        x = self.to_horizontal_embedding(img)
        if self.pos_emb_type == "APE":
            x = x + self.horizontal_pos_embedding
        x = self.dropout(x)
        x = self.horizontalEncoder(x)
        x = self.horizontal_mlp_head(x)

        x = self.to_vertical_embedding(x)
        if self.pos_emb_type == "APE":
            x = x + self.vertical_pos_embedding
        x = self.dropout(x)
        if col_mask is not None and self.mask_vertical_attn != "none":
            W = x.shape[1]
            sampled = _extract_shared_column_mask(col_mask, W)
            attn_mask = _build_vertical_attn_mask(sampled, self.mask_vertical_attn)
            x = self.verticalEncoder(x, attn_mask=attn_mask)
        else:
            x = self.verticalEncoder(x)

        x = self.vertical_mlp_head(x)

        return x


class crossAxialEncoder(nn.Module):
    """
    Vertical-only encoder:
    each inner layer does sampled->unsampled cross-attention followed by
    unsampled-only self-attention, then scatters unsampled updates back into
    the full vertical token tensor before the shared vertical output head.
    """
    def __init__(self, image_size, numCh=1, d_model=512, nhead=8, num_layers=6, dim_feedforward=None,
                    dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=True,
                    device=None, dtype=None, norm=None,
                    pos_emb_type="APE", rope_theta=100.0, attn_type="complex", row_stride=1,
                    ffn_sharing="none", shared_ffn=None, flattening_order="row_major"):
        super().__init__()
        if row_stride != 1:
            raise ValueError("crossAxialEncoder supports vertical tokens only and requires row_stride=1")
        if attn_type != "complex":
            raise ValueError(
                "crossAxialEncoder only supports attn_type='complex'. "
                f"Received '{attn_type}'."
            )

        self.pos_emb_type = pos_emb_type
        self.d_model = d_model
        self.is_complex = attn_type in _COMPLEX_ATTN_TYPES

        image_height, image_width = pair(image_size)
        head_dim = d_model // nhead
        cis_fn = compute_axial_cis_complex if self.is_complex else compute_axial_cis
        dtype = torch.cfloat if self.is_complex else None
        norm = ComplexLayerNorm if self.is_complex else nn.LayerNorm
        v_from = get_from_embedding("axial", numCh=numCh, image_height=image_height,
                                    image_width=image_width, flattening_order=flattening_order)[1]

        self.to_vertical_embedding = nn.Sequential(
            (PatchUnroller((image_height, image_width), (image_height, 1), flattening_order)
             if flattening_order == 'dc_radial' else Rearrange('b c h w -> b w (h c)')),
            nn.Linear(image_height * numCh, d_model, dtype=dtype),
        )
        self.vertical_mlp_head = nn.Sequential(
            norm(d_model, eps=layer_norm_eps),
            nn.Linear(d_model, image_height * numCh, dtype=dtype),
            v_from,
        )
        if self.is_complex:
            apply_trabelsi_(self.to_vertical_embedding[1], criterion="glorot")
            apply_trabelsi_(self.vertical_mlp_head[1], criterion="glorot")
        self.dropout = ComplexDropout(dropout) if self.is_complex else nn.Dropout(dropout)

        if pos_emb_type == "APE":
            freqs_v = None
            ape_dtype = torch.cfloat if self.is_complex else None
            self.vertical_pos_embedding = nn.Parameter(torch.randn(1, image_width, d_model, dtype=ape_dtype))
        elif pos_emb_type in ("Rope-Axial", "Rope-Mixed"):
            freqs_v = cis_fn(dim=head_dim, end_x=image_width, end_y=1, theta=rope_theta)
        else:
            freqs_v = None

        if shared_ffn is None and ffn_sharing == "per_stage":
            shared_ffn = FeedForward(d_model, dim_feedforward, dropout, activation, self.is_complex)

        inner_layer = _CrossAxialInnerLayer(
            d_model, nhead, dim_feedforward, dropout, activation,
            layer_norm_eps, freqs_cis=freqs_v, attn_type=attn_type, ff=shared_ffn
        )
        self.verticalEncoder = _CrossAxialEncoderStack(inner_layer, num_layers,
                                                       tie_ffn=shared_ffn is not None)

    def forward(self, img, col_mask=None):
        if col_mask is None:
            raise ValueError("crossAxialEncoder requires col_mask for sampled/unsampled routing")

        x = self.to_vertical_embedding(img)
        if self.pos_emb_type == "APE":
            x = x + self.vertical_pos_embedding
        x = self.dropout(x)

        W = x.shape[1]
        sampled = _extract_shared_column_mask(col_mask, W)
        sampled_idx = sampled.nonzero(as_tuple=True)[0]
        unsampled_idx = (~sampled).nonzero(as_tuple=True)[0]

        x = self.verticalEncoder(x, sampled_idx, unsampled_idx)
        x = self.vertical_mlp_head(x)
        return x


# ---------------------------------------------------------------------------
# FNet: parameter-free Fourier token mixing (Lee-Thorp et al. 2021) on vertical column tokens
# ---------------------------------------------------------------------------

class FourierMixing(nn.Module):
    """
    FNet token mixing: 2-D FFT over the (sequence, hidden) axes of [B, seq, hidden].
    complex tokens -> the complex FFT output is kept (no information discarded);
    real tokens    -> the real part is taken, as in the original FNet.
    fft_norm="ortho" keeps |mix(x)| ~ |x| so the residual stream is not swamped.
    """
    def __init__(self, fft_norm="ortho"):
        super().__init__()
        if fft_norm not in {"ortho", "backward", "forward"}:
            raise ValueError("fft_norm must be 'ortho', 'backward' or 'forward'")
        self.fft_norm = fft_norm

    def forward(self, x):
        y = torch.fft.fft2(x, dim=(-2, -1), norm=self.fft_norm)
        return y if x.is_complex() else y.real


class _FNetLayer(nn.Module):
    """x = LN(x + mix(x)); x = LN(x + FFN(x))  (pre-LN variant matching TransformerEncoderLayer style)."""
    def __init__(self, d_model, dim_feedforward, dropout, activation, layer_norm_eps,
                 is_complex=True, ff=None, fft_norm="ortho"):
        super().__init__()
        norm = ComplexLayerNorm if is_complex else nn.LayerNorm
        self.mix = FourierMixing(fft_norm)
        self.ff = ff if ff is not None else FeedForward(d_model, dim_feedforward, dropout, activation, is_complex)
        self.norm1 = norm(d_model, eps=layer_norm_eps)
        self.norm2 = norm(d_model, eps=layer_norm_eps)
        self.drop = ComplexDropout(dropout) if is_complex else nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.mix(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class _FNetStack(nn.Sequential):
    def __init__(self, layer, num_layers, tie_ffn=False):
        super().__init__(*[copy.deepcopy(layer) for _ in range(num_layers)])
        if tie_ffn:
            for l in self:
                l.ff = layer.ff

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


_FNET_TOKEN_AXES = ("vertical", "horizontal", "both")


class fnetEncoder(nn.Module):
    """
    FNet encoder on axial (row / column) tokens.

    token_axis:
      "vertical"   — each k-space column is one token (same tokenisation as the axial vertical branch
                     and crossAxialEncoder); mixing runs over (columns, hidden).
      "horizontal" — each k-space row (group of `row_stride` rows) is one token; mixing over (rows, hidden).
      "both"       — horizontal branch followed by the vertical branch, exactly like axialEncoder, each
                     with its own embedding / FNet stack / head.
    with_embedding:
      True  — tokens go through the same Rearrange -> Linear(token_dim, d_model) embedding as the other
              encoders and come back through LayerNorm -> Linear(d_model, token_dim) -> Rearrange.
      False — raw k-space rows / columns are the tokens (hidden dim = token_dim: H*numCh for columns,
              row_stride*W*numCh for rows); no linear projection in or out, only the Rearrange.
    All learning is in the per-token FFN; the column mask is accepted for interface compatibility
    but not used (the Fourier mixing is global).
    """
    def __init__(self, image_size, numCh=1, d_model=512, nhead=8, num_layers=2, dim_feedforward=None,
                    dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=True,
                    device=None, dtype=None, norm=None,
                    pos_emb_type="APE", rope_theta=100.0, attn_type="complex", row_stride=1,
                    ffn_sharing="none", shared_ffn=None, flattening_order="row_major", fft_norm="ortho",
                    token_axis="vertical", with_embedding=True):
        super().__init__()
        if token_axis not in _FNET_TOKEN_AXES:
            raise ValueError(f"token_axis must be one of {_FNET_TOKEN_AXES}, got '{token_axis}'")
        if token_axis == "vertical" and row_stride != 1:
            raise ValueError("row_stride only applies to horizontal tokens; use row_stride=1 with token_axis='vertical'")

        self.token_axis = token_axis
        self.pos_emb_type = pos_emb_type
        self.d_model = d_model
        self.with_embedding = with_embedding
        self.is_complex = attn_type in _COMPLEX_ATTN_TYPES

        image_height, image_width = pair(image_size)
        h_tokens = image_height // row_stride
        ape_dtype = torch.cfloat if self.is_complex else None

        h_emb, v_emb = get_to_embedding(
            "axial", image_height=image_height, image_width=image_width, numCh=numCh, d_model=d_model,
            row_stride=row_stride, is_complex=self.is_complex, flattening_order=flattening_order)
        h_head, v_head = get_mlp_head(
            "axial", d_model, numCh=numCh, image_height=image_height, image_width=image_width,
            row_stride=row_stride, is_complex=self.is_complex, flattening_order=flattening_order,
            layer_norm_eps=layer_norm_eps)
        # Hidden width per branch: d_model when embedded, otherwise the raw token length.
        h_dim = d_model if with_embedding else row_stride * image_width * numCh
        v_dim = d_model if with_embedding else image_height * numCh
        if not with_embedding:
            h_emb, v_emb = h_emb[0], v_emb[0]            # keep only the Rearrange (token unroll)
            h_head, v_head = h_head[-1], v_head[-1]      # keep only the Rearrange back to the image
        self.dropout = ComplexDropout(dropout) if self.is_complex else nn.Dropout(dropout)

        if shared_ffn is None and ffn_sharing == "per_stage" and with_embedding:
            shared_ffn = FeedForward(d_model, dim_feedforward, dropout, activation, self.is_complex)
        if shared_ffn is not None and not with_embedding:
            for name, dim in (("horizontal", h_dim), ("vertical", v_dim)):
                if name in (token_axis, "both") and shared_ffn.net[0].in_features != dim:
                    raise ValueError(
                        f"Shared FFN expects d_model={shared_ffn.net[0].in_features} but the un-embedded FNet "
                        f"{name} tokens have width {dim}; use with_embedding=True or ffn_sharing='none'"
                    )

        def make_stack(dim):
            ff = shared_ffn
            if ff is None and ffn_sharing == "per_stage":      # un-embedded branches may differ in width
                ff = FeedForward(dim, dim_feedforward, dropout, activation, self.is_complex)
            layer = _FNetLayer(dim, dim_feedforward, dropout, activation, layer_norm_eps,
                               is_complex=self.is_complex, ff=ff, fft_norm=fft_norm)
            return _FNetStack(layer, num_layers, tie_ffn=ff is not None)

        # RoPE has no q/k to rotate; the DFT is already position-dependent. Only APE adds a learned embedding.
        if token_axis in ("horizontal", "both"):
            self.to_horizontal_embedding, self.horizontal_mlp_head = h_emb, h_head
            self.horizontalEncoder = make_stack(h_dim)
            if pos_emb_type == "APE":
                self.horizontal_pos_embedding = nn.Parameter(torch.randn(1, h_tokens, h_dim, dtype=ape_dtype))
        if token_axis in ("vertical", "both"):
            self.to_vertical_embedding, self.vertical_mlp_head = v_emb, v_head
            self.verticalEncoder = make_stack(v_dim)
            if pos_emb_type == "APE":
                self.vertical_pos_embedding = nn.Parameter(torch.randn(1, image_width, v_dim, dtype=ape_dtype))

    def _run_branch(self, x, embed, pos_emb, encoder, head):
        x = embed(x)
        if pos_emb is not None:
            x = x + pos_emb
        x = self.dropout(x)
        return head(encoder(x))

    def forward(self, img, col_mask=None):
        x = img
        if self.token_axis in ("horizontal", "both"):
            x = self._run_branch(x, self.to_horizontal_embedding, getattr(self, "horizontal_pos_embedding", None),
                                 self.horizontalEncoder, self.horizontal_mlp_head)
        if self.token_axis in ("vertical", "both"):
            x = self._run_branch(x, self.to_vertical_embedding, getattr(self, "vertical_pos_embedding", None),
                                 self.verticalEncoder, self.vertical_mlp_head)
        return x


# ---------------------------------------------------------------------------
# PCA-channel encoder: volume-wise PCA bins -> per-bin branches -> k-space merge -> single-channel encoder
# ---------------------------------------------------------------------------

PCA_BRANCH_TOKENIZERS = ("axial", "fixed_apt")
PCA_SCOPES = ("volume", "batch")


def make_pca_branch(tokenizer, image_size, num_layers, d_model=None, nhead=8, dim_feedforward=None,
                    dropout=0.1, activation="relu", layer_norm_eps=1e-5, pos_emb_type="APE", rope_theta=100.0,
                    attn_type="complex", row_stride=1, flattening_order="row_major",
                    ffn_sharing="none", shared_ffn=None, apt_layout=None, apt_embed_dim=256,
                    apt_rope_ref_grid=None, apt_use_abs_pos_emb=False, **tokenizer_kwargs):
    """
    Build one single-channel encoder module ([B,1,H,W] -> [B,1,H,W]) for a PCA channel.
    `tokenizer` selects the tokenisation; every branch owns its embedding, transformer layers and head.
        "axial"     : axialEncoder; one "layer" = one horizontal + one vertical attention layer
                      (axialEncoder takes num_layers as the total, split in two).
        "fixed_apt" : FixedAPTVIT with a single inner encoder of `num_layers` transformer layers on the
                      fixed adaptive-patch layout (apt_layout / apt_embed_dim / apt_rope_ref_grid / apt_use_abs_pos_emb).
    """
    if tokenizer == "axial":
        _, image_width = pair(image_size)
        d_model = d_model or image_width
        dim_feedforward = dim_feedforward or int(d_model * 4)
        return axialEncoder(image_size, numCh=1, d_model=d_model, nhead=nhead, num_layers=2 * num_layers,
                            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                            layer_norm_eps=layer_norm_eps, pos_emb_type=pos_emb_type, rope_theta=rope_theta,
                            attn_type=attn_type, row_stride=row_stride, ffn_sharing=ffn_sharing,
                            shared_ffn=shared_ffn, flattening_order=flattening_order)
    if tokenizer == "fixed_apt":
        from .fixed_apt import FixedAPTVIT, resolve_fixed_apt_layout
        d_model = d_model or apt_embed_dim
        return FixedAPTVIT(image_size, layerNo=1, numCh=1, d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                           layer_norm_eps=layer_norm_eps, attn_type=attn_type, ffn_sharing=ffn_sharing,
                           shared_ffn=shared_ffn, layout=resolve_fixed_apt_layout(apt_layout), rope_theta=rope_theta,
                           rope_ref_grid=apt_rope_ref_grid, use_abs_pos_emb=apt_use_abs_pos_emb)
    raise ValueError(f"Unknown PCA branch tokenizer '{tokenizer}'. Choose from {PCA_BRANCH_TOKENIZERS}")


class pcaEncoder(nn.Module):
    """
    Stage that splits its input into principal-component channels and processes them separately.

        x [B,1,H,W] --volume PCA--> channels c_b [B,1,H,W] (b = 1..n_bins), volume mean m
        c_b' = c_b + branch_b(c_b)                    (separate weights per bin; spatial attention within a bin only)
        x~   = m + sum_b c_b'                          (k-space merge; == x + sum_b branch_b(c_b))
        out  = x~ + post(x~) - x                       (single-channel encoder after the merge; returned as a residual
                                                        so cascadeNet's x + stage(x) gives x~ + post(x~))

    scope="volume": the PCA is computed per volume on the stage input (slices sharing `volume_id`);
    scope="batch" : the whole batch is treated as one set of slices (volume_id ignored).
    The basis is detached by default (see DcTNN.pca). Tokenisation of the branches and of the post-merge
    encoder is selected with `tokenizer` ("axial" | "fixed_apt", see make_pca_branch).
    """
    def __init__(self, image_size, numCh=1, d_model=None, nhead=8, num_layers=2, dim_feedforward=None,
                 dropout=0.1, activation="relu", layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None,
                 norm=None, pos_emb_type="APE", rope_theta=100.0, attn_type="complex", row_stride=1,
                 flattening_order="row_major", ffn_sharing="none", shared_ffn=None,
                 tokenizer="axial", n_bins=3, bin_rule="equal_variance", detach_basis=True, center=True,
                 layers_per_bin=1, layers_after_merge=1, scope="volume",
                 apt_layout=None, apt_embed_dim=256, apt_rope_ref_grid=None, apt_use_abs_pos_emb=False):
        super().__init__()
        if scope not in PCA_SCOPES:
            raise ValueError(f"scope must be one of {PCA_SCOPES}, got '{scope}'")
        self.scope = scope
        if numCh != 1:
            raise ValueError("pcaEncoder expects a single-channel complex k-space input (numCh=1)")
        if attn_type not in _COMPLEX_ATTN_TYPES:
            raise ValueError("pcaEncoder operates on complex k-space; use a complex attn_type")
        self.image_size = pair(image_size)
        self.tokenizer, self.n_bins, self.bin_rule = tokenizer, int(n_bins), bin_rule
        self.detach_basis, self.center = bool(detach_basis), bool(center)
        self.last_pca_info = None
        common = dict(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                      activation=activation, layer_norm_eps=layer_norm_eps, pos_emb_type=pos_emb_type,
                      rope_theta=rope_theta, attn_type=attn_type, row_stride=row_stride,
                      flattening_order=flattening_order, ffn_sharing=ffn_sharing, shared_ffn=shared_ffn,
                      apt_layout=apt_layout, apt_embed_dim=apt_embed_dim, apt_rope_ref_grid=apt_rope_ref_grid,
                      apt_use_abs_pos_emb=apt_use_abs_pos_emb)
        self.branches = nn.ModuleList([
            make_pca_branch(tokenizer, self.image_size, layers_per_bin, **common) for _ in range(self.n_bins)])
        self.post = (make_pca_branch(tokenizer, self.image_size, layers_after_merge, **common)
                     if layers_after_merge > 0 else None)

    def forward(self, img, col_mask=None, volume_id=None):
        from .pca import volume_pca_channels
        if self.scope == "batch":
            volume_id = None
        channels, mean, info = volume_pca_channels(img, volume_id, n_bins=self.n_bins, rule=self.bin_rule,
                                                   detach_basis=self.detach_basis, center=self.center,
                                                   return_info=True)
        self.last_pca_info = info
        merged = mean
        for b, branch in enumerate(self.branches):
            c = channels[:, b:b + 1]
            merged = merged + c + branch(c, col_mask=col_mask)
        out = merged + self.post(merged, col_mask=col_mask) if self.post is not None else merged
        return out - img
