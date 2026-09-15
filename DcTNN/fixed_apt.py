import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .attention_layer import TransformerEncoder, TransformerEncoderLayer
from .complex_init import apply_trabelsi_, trabelsi_init_
from .util import ComplexDropout, ComplexLayerNorm, FeedForward, _COMPLEX_ATTN_TYPES


PATCH_SIZES = (8, 16, 32, 64)


def _image_pair(size):
    size = (size, size) if isinstance(size, int) else tuple(size)
    if len(size) != 2 or any(type(d) is not int or d <= 0 or d % 8 for d in size):
        raise ValueError("fixed_apt image dimensions must be positive integer multiples of 8")
    return size


def resolve_fixed_apt_layout(layout=None):
    if layout is None:
        with Path(__file__).with_name("fixed_apt_layout.json").open() as handle:
            layout = json.load(handle)
    if not isinstance(layout, dict):
        raise ValueError("apt_layout must be a layout dictionary, not a path")
    layout = copy.deepcopy(layout)
    if layout.get("version") != 1 or layout.get("base_patch_size") != 8:
        raise ValueError("fixed_apt requires layout version 1 and base_patch_size=8")
    image_size = _image_pair(layout.get("image_size", ()))
    leaves = layout.get("leaves")
    if not isinstance(leaves, (list, tuple)) or not leaves:
        raise ValueError("fixed_apt layout must contain leaf boxes")
    coverage = torch.zeros(image_size[0] // 8, image_size[1] // 8, dtype=torch.int32)
    boxes = []
    for box in leaves:
        if not isinstance(box, (list, tuple)) or len(box) != 3 or any(type(v) is not int for v in box):
            raise ValueError("Each fixed_apt leaf must be integer [y, x, size]")
        y, x, size = box
        if size not in PATCH_SIZES or y < 0 or x < 0 or y % size or x % size:
            raise ValueError("fixed_apt leaves must be aligned 8/16/32/64 pixel squares")
        if y + size > image_size[0] or x + size > image_size[1]:
            raise ValueError("fixed_apt leaf lies outside the image")
        coverage[y // 8:(y + size) // 8, x // 8:(x + size) // 8] += 1
        boxes.append([y, x, size])
    if not torch.all(coverage == 1):
        raise ValueError("fixed_apt leaves must cover the image exactly once without overlaps or gaps")
    layout.update(image_size=list(image_size), leaves=sorted(boxes, key=lambda box: (box[2], box[0], box[1])))
    return layout


def layout_signature(layout):
    geometry = {key: layout[key] for key in ("version", "image_size", "base_patch_size", "leaves")}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def patch_center_positions(boxes, image_size, reference_grid=None):
    height, width = _image_pair(image_size)
    reference_grid = (height / 8, width / 8) if reference_grid is None else tuple(reference_grid)
    if len(reference_grid) != 2 or any(not math.isfinite(v) or v <= 0 for v in reference_grid):
        raise ValueError("apt_rope_ref_grid must contain two positive finite dimensions")
    boxes = boxes.to(dtype=torch.float32)
    y, x, size = boxes.unbind(-1)
    tx = (x + (size - 1) / 2 - width // 2) * (reference_grid[1] / width)
    ty = (y + (size - 1) / 2 - height // 2) * (reference_grid[0] / height)
    return torch.stack((tx, ty), dim=-1)


def fixed_apt_rope(positions, head_dim, theta=100.0):
    if type(head_dim) is not int or head_dim <= 0 or head_dim % 2:
        raise ValueError("fixed_apt complex axial RoPE requires a positive even head dimension")
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("rope_theta must be positive and finite")
    if positions.ndim != 2 or positions.shape[1] != 2 or not torch.isfinite(positions).all():
        raise ValueError("RoPE positions must be finite [tokens, 2] coordinates")
    freq = theta ** (-torch.arange(0, head_dim, 2, device=positions.device, dtype=positions.dtype) / head_dim)
    angles = torch.cat((positions[:, :1] * freq, positions[:, 1:] * freq), dim=-1)
    return torch.polar(torch.ones_like(angles), angles)


def resize_complex(x, size, mode="bilinear"):
    options = dict(size=size, mode=mode, align_corners=False, antialias=mode == "bicubic")
    return torch.complex(F.interpolate(x.real, **options), F.interpolate(x.imag, **options))


def _init_complex_conv(layer):
    fan_in = layer.in_channels * math.prod(layer.kernel_size)
    trabelsi_init_(layer.weight, fan_in=fan_in, fan_out=layer.out_channels, criterion="glorot")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class FixedAPTGeometry(nn.Module):
    def __init__(self, image_size, layout=None, rope_ref_grid=None):
        super().__init__()
        self.layout = resolve_fixed_apt_layout(layout)
        self.image_size = _image_pair(image_size)
        if tuple(self.layout["image_size"]) != self.image_size:
            raise ValueError("fixed_apt image size does not match the frozen layout")
        self.base_patch_size = 8
        self.rope_ref_grid = tuple(rope_ref_grid) if rope_ref_grid is not None else tuple(d / 8 for d in self.image_size)
        boxes = torch.tensor(self.layout["leaves"], dtype=torch.long)
        self.register_buffer("boxes", boxes)
        self.register_buffer("positions", patch_center_positions(boxes, self.image_size, self.rope_ref_grid), persistent=False)
        self.num_tokens = len(boxes)
        self.patch_sizes = tuple(size for size in PATCH_SIZES if (boxes[:, 2] == size).any())
        for size in self.patch_sizes:
            indices = (boxes[:, 2] == size).nonzero().flatten()
            origins = boxes[indices, :2]
            self.register_buffer(f"token_indices_{size}", indices, persistent=False)
            self.register_buffer(f"grid_indices_{size}", origins[:, 0] // size * (self.image_size[1] // size) + origins[:, 1] // size, persistent=False)
            for unit, name in ((1, "pixel"), (8, "feature")):
                side = size // unit
                offsets = torch.arange(side)
                rows = origins[:, 0, None, None] // unit + offsets[None, :, None]
                cols = origins[:, 1, None, None] // unit + offsets[None, None, :]
                spatial_indices = (rows * (self.image_size[1] // unit) + cols).reshape(-1)
                self.register_buffer(f"{name}_indices_{size}", spatial_indices, persistent=False)

    def indices(self, size):
        return getattr(self, f"token_indices_{size}")

    def _extract(self, x, size, unit, name):
        if x.ndim != 4 or tuple(x.shape[-2:]) != tuple(d // unit for d in self.image_size):
            raise ValueError("Input spatial dimensions do not match the fixed_apt geometry")
        indices = getattr(self, f"{name}_indices_{size}")
        batch, channels = x.shape[:2]
        count, side = self.indices(size).numel(), size // unit
        return x.flatten(2).index_select(2, indices).reshape(batch, channels, count, side, side).permute(0, 2, 1, 3, 4)

    def extract(self, x, size):
        return self._extract(x, size, 1, "pixel")

    def extract_features(self, x, size):
        return self._extract(x, size, 8, "feature")

    def scatter(self, groups):
        if set(groups) != set(self.patch_sizes):
            raise ValueError("Decoded groups must match the fixed_apt patch sizes")
        first = groups[self.patch_sizes[0]]
        batch, channels = first.shape[0], first.shape[2]
        output = first.new_zeros(batch, channels, math.prod(self.image_size))
        for size in self.patch_sizes:
            patches = groups[size]
            if tuple(patches.shape) != (batch, self.indices(size).numel(), channels, size, size):
                raise ValueError("Decoded patch shape does not match the fixed_apt layout")
            values = patches.permute(0, 2, 1, 3, 4).reshape(batch, channels, -1)
            output = output.index_copy(2, getattr(self, f"pixel_indices_{size}"), values)
        return output.reshape(batch, channels, *self.image_size)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "boxes")
        if saved is not None and not torch.equal(saved.cpu(), self.boxes.cpu()):
            error_msgs.append(f"{prefix}fixed_apt checkpoint layout does not match the configured geometry")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class FixedAPTPatchEmbedding(nn.Module):
    def __init__(self, geometry, num_channels=1, embed_dim=256, use_abs_pos_emb=False):
        super().__init__()
        self.geometry = geometry
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(num_channels, embed_dim, kernel_size=8, stride=8, dtype=torch.cfloat)
        self.patch_attn = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, dtype=torch.cfloat)
        self.zero_conv = nn.Linear(embed_dim, embed_dim, dtype=torch.cfloat)
        _init_complex_conv(self.proj)
        _init_complex_conv(self.patch_attn)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        self.pos_embedding = (nn.Parameter(torch.randn(1, embed_dim, *(d // 8 for d in geometry.image_size), dtype=torch.cfloat) * 0.02)
                              if use_abs_pos_emb else None)

    def forward(self, x):
        if x.ndim != 4 or not x.is_complex() or x.shape[1] != self.num_channels or tuple(x.shape[-2:]) != self.geometry.image_size:
            raise ValueError("fixed_apt expects complex [B, configured channels, H, W] input")
        batch = x.shape[0]
        fine = self.proj(x)
        output = x.new_zeros(batch, self.geometry.num_tokens, self.embed_dim)
        for size in self.geometry.patch_sizes:
            count = self.geometry.indices(size).numel()
            features = self.geometry.extract_features(fine, size)
            if size == 8:
                tokens = features[..., 0, 0]
            else:
                patches = self.geometry.extract(x, size).reshape(batch * count, self.num_channels, size, size)
                resized = resize_complex(patches, (8, 8))
                coarse = self.proj(resized).flatten(1).reshape(batch, count, self.embed_dim)
                reduced = features.reshape(batch * count, self.embed_dim, size // 8, size // 8)
                for _ in range(int(math.log2(size // 8))):
                    reduced = self.patch_attn(reduced)
                residual = self.zero_conv(reduced.flatten(1)).reshape(batch, count, self.embed_dim)
                tokens = coarse + residual
            if self.pos_embedding is not None:
                positions = resize_complex(self.pos_embedding, tuple(d // size for d in self.geometry.image_size), mode="bicubic")
                positions = positions.flatten(2).index_select(2, getattr(self.geometry, f"grid_indices_{size}")).transpose(1, 2)
                tokens = tokens + positions
            output = output.index_copy(1, self.geometry.indices(size), tokens)
        return output


class FixedAPTReconstructionHead(nn.Module):
    def __init__(self, geometry, num_channels=1, embed_dim=256, layer_norm_eps=1e-5):
        super().__init__()
        self.geometry = geometry
        self.num_channels = num_channels
        self.norm = ComplexLayerNorm(embed_dim, eps=layer_norm_eps)
        self.projections = nn.ModuleDict({str(size): nn.Linear(embed_dim, num_channels * size * size, dtype=torch.cfloat)
                                         for size in geometry.patch_sizes})
        for projection in self.projections.values():
            apply_trabelsi_(projection, criterion="glorot")

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] != self.geometry.num_tokens or not tokens.is_complex():
            raise ValueError("Reconstruction requires complex tokens matching the frozen layout")
        tokens = self.norm(tokens)
        groups = {}
        for size in self.geometry.patch_sizes:
            selected = tokens.index_select(1, self.geometry.indices(size))
            groups[size] = self.projections[str(size)](selected).reshape(tokens.shape[0], selected.shape[1], self.num_channels, size, size)
        return self.geometry.scatter(groups)


class FixedAPTEncoder(nn.Module):
    def __init__(self, N, layout=None, numCh=1, d_model=256, nhead=8, num_encoder_layers=2,
                 dim_feedforward=None, dropout=0.1, activation="relu", layer_norm_eps=1e-5,
                 rope_theta=100.0, rope_ref_grid=None, use_abs_pos_emb=False, attn_type="complex", shared_ffn=None):
        super().__init__()
        if attn_type not in _COMPLEX_ATTN_TYPES:
            raise ValueError("fixed_apt requires a complex-valued attention type")
        if type(d_model) is not int or type(nhead) is not int or nhead <= 0 or d_model <= 0 or d_model % nhead:
            raise ValueError("fixed_apt embedding width must be positive and divisible by nhead")
        if type(numCh) is not int or numCh <= 0 or type(num_encoder_layers) is not int or num_encoder_layers <= 0:
            raise ValueError("fixed_apt channels and encoder depth must be positive integers")
        geometry = FixedAPTGeometry(N, layout, rope_ref_grid)
        freqs = fixed_apt_rope(geometry.positions, d_model // nhead, rope_theta)
        self.d_model, self.nhead, self.is_complex = d_model, nhead, True
        self.to_embedding = FixedAPTPatchEmbedding(geometry, numCh, d_model, use_abs_pos_emb)
        self.mlp_head = FixedAPTReconstructionHead(geometry, numCh, d_model, layer_norm_eps=layer_norm_eps)
        self.dropout = ComplexDropout(dropout)
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * d_model
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                        freqs_cis=freqs, attn_type=attn_type, ff=shared_ffn)
        self.encoder = TransformerEncoder(layer, num_encoder_layers, tie_ffn=shared_ffn is not None)
        spec = dict(layout=layout_signature(geometry.layout), channels=numCh, embed_dim=d_model, nhead=nhead,
                    rope_theta=float(rope_theta), rope_ref_grid=list(map(float, geometry.rope_ref_grid)),
                    use_abs_pos_emb=bool(use_abs_pos_emb), attention=attn_type, origin="physical_dc_sample")
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).digest()
        self.register_buffer("configuration_fingerprint", torch.tensor(list(digest), dtype=torch.uint8))

    @property
    def geometry(self):
        return self.to_embedding.geometry

    def forward(self, x):
        return self.mlp_head(self.encoder(self.dropout(self.to_embedding(x))))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "configuration_fingerprint")
        if saved is not None and not torch.equal(saved.cpu(), self.configuration_fingerprint.cpu()):
            error_msgs.append(f"{prefix}fixed_apt checkpoint layout or positional configuration mismatch")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class FixedAPTVIT(nn.Module):
    def __init__(self, N, layerNo=1, numCh=1, d_model=256, nhead=8, num_encoder_layers=2,
                 dim_feedforward=None, dropout=0.1, activation="relu", layer_norm_eps=1e-5,
                 attn_type="complex", ffn_sharing="none", shared_ffn=None, flattening_order="row_major",
                 layout=None, rope_theta=100.0, rope_ref_grid=None, use_abs_pos_emb=False):
        super().__init__()
        if flattening_order != "row_major":
            raise ValueError("fixed_apt only supports row_major features inside patches")
        if ffn_sharing not in {"none", "per_stage", "global"}:
            raise ValueError("Invalid fixed_apt ffn_sharing mode")
        if type(layerNo) is not int or layerNo <= 0:
            raise ValueError("fixed_apt layerNo must be a positive integer")
        self.N, self.layerNo, self.numCh = N, layerNo, numCh
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * d_model
        if shared_ffn is None and ffn_sharing != "none":
            shared_ffn = FeedForward(d_model, dim_feedforward, dropout, activation, True)
        self.transformers = nn.ModuleList([
            FixedAPTEncoder(N, layout, numCh, d_model, nhead, num_encoder_layers, dim_feedforward,
                            dropout, activation, layer_norm_eps, rope_theta, rope_ref_grid,
                            use_abs_pos_emb, attn_type, shared_ffn)
            for _ in range(layerNo)
        ])

    def forward(self, xPrev, col_mask=None):
        for transformer in self.transformers:
            xPrev = transformer(xPrev)
        return xPrev
