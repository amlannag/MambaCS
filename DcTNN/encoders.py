import torch
from torch import nn
from .rope_vit import (compute_axial_cis, compute_mixed_cis,
                        compute_axial_cis_complex, compute_mixed_cis_complex)
from .attention_layer import (
    TransformerEncoderLayer,
    TransformerEncoder,
    CrossAttentionEncoderLayer,
    CrossAttentionEncoder,
)
from .util import get_to_embedding, get_mlp_head, ComplexDropout, _COMPLEX_ATTN_TYPES


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
        # All queries blocked from unsampled keys: column mask on keys only
        mask = torch.zeros(W, W, device=sampled.device)
        mask[:, unsampled] = float("-inf")
    elif mode == "lenient":
        # Sampled queries blocked from unsampled keys; unsampled queries attend freely
        # block[i,j] = True when query i is sampled AND key j is unsampled
        block = sampled.unsqueeze(1) & unsampled.unsqueeze(0)   # [W, W] bool, no large intermediates
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
                       rope_theta, rope_mixed_rotate, attn_type):

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


        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                        layer_norm_eps, freqs_cis=freqs_cis, attn_type=attn_type)
        self.encoder = TransformerEncoder(layer, num_layers)

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
                attn_type="standard"):
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
                                             is_complex=self.is_complex)
        self.mlp_head = get_mlp_head(tokenizer_type, d_model, patch_dim, patch_height, patch_width,
                                     grid_h, numCh, is_complex=self.is_complex)
        
        self.dropout = ComplexDropout(dropout) if self.is_complex else nn.Dropout(dropout)

        self._setup_pos_emb(grid_h, grid_w, num_patches, num_layers, d_model, nhead,
                            dim_feedforward, dropout, activation, layer_norm_eps,
                            batch_first, device, dtype, norm, rope_theta, rope_mixed_rotate,
                            attn_type)


class axialEncoder(nn.Module):
    """
    Standard Encoder that utilizes axial attention (separate row and column transformers).
    """
    def __init__(self, image_size, numCh=1, d_model=512, nhead=8, num_layers=6, dim_feedforward=None,
                    dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=True,
                    device=None, dtype=None, norm=None,
                    pos_emb_type="APE", rope_theta=100.0, attn_type="standard", row_stride=1,
                    mask_vertical_attn="none"):
        super().__init__()

        self.pos_emb_type = pos_emb_type
        self.d_model = d_model
        self.is_complex = attn_type in _COMPLEX_ATTN_TYPES
        self.mask_vertical_attn = mask_vertical_attn

        image_height, image_width = pair(image_size)
        h_tokens = image_height // row_stride  # horizontal token count after row grouping

        self.to_horizontal_embedding, self.to_vertical_embedding = get_to_embedding(
            "axial", image_height=image_height, image_width=image_width, numCh=numCh, d_model=d_model,
            row_stride=row_stride, is_complex=self.is_complex)

        self.horizontal_mlp_head, self.vertical_mlp_head = get_mlp_head(
            "axial", d_model, numCh=numCh, image_height=image_height, image_width=image_width,
            row_stride=row_stride, is_complex=self.is_complex)

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

        h_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                          layer_norm_eps, freqs_cis=freqs_h, attn_type=attn_type)
        v_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                          layer_norm_eps, freqs_cis=freqs_v, attn_type=attn_type)
        v_cross_layer = CrossAttentionEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation,
            layer_norm_eps, freqs_cis=freqs_v, attn_type=attn_type
        )
        self.horizontalEncoder = TransformerEncoder(h_layer, numLayers)
        self.verticalEncoder = TransformerEncoder(v_layer, numLayers)
        self.verticalCrossEncoder = CrossAttentionEncoder(v_cross_layer, numLayers)

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

            if self.mask_vertical_attn == "cross":
                sampled_idx = sampled.nonzero(as_tuple=True)[0]
                unsampled_idx = (~sampled).nonzero(as_tuple=True)[0]
                if sampled_idx.numel() == 0:
                    raise ValueError("Cross attention requires at least one sampled column")
                if unsampled_idx.numel() > 0:
                    x_unsampled = x.index_select(1, unsampled_idx)
                    x_sampled = x.index_select(1, sampled_idx)
                    x_updated = self.verticalCrossEncoder(
                        x_unsampled,
                        x_sampled,
                        q_positions=unsampled_idx,
                        kv_positions=sampled_idx,
                    )
                    x = x.clone()
                    x[:, unsampled_idx, :] = x_updated
            else:
                attn_mask = _build_vertical_attn_mask(sampled, self.mask_vertical_attn)
                x = self.verticalEncoder(x, attn_mask=attn_mask)
        else:
            x = self.verticalEncoder(x)

        x = self.vertical_mlp_head(x)

        return x
