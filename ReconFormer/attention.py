"""Recurrent multi-scale window attention used by ReconFormer."""

from collections.abc import Iterable

import torch
from torch import nn
from torch.utils import checkpoint


def to_2tuple(value):
    """Small local equivalent of timm's ``to_2tuple`` helper."""
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return (value, value)


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (stochastic depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.3f}"


class PatchEmbed(nn.Module):
    """Flatten an image feature map into patch tokens."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            int(img_size[0] // patch_size[0]),
            int(img_size[1] // patch_size[1]),
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        height, width = self.img_size
        return height * width * self.embed_dim if self.norm is not None else 0


class PatchUnEmbed(nn.Module):
    """Restore patch tokens to an image feature map."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            int(img_size[0] // patch_size[0]),
            int(img_size[1] // patch_size[1]),
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        batch, _, _ = x.shape
        return x.transpose(1, 2).reshape(batch, self.embed_dim, x_size[0], x_size[1])

    def flops(self):
        return 0


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


def window_partition(x, window_size):
    """Partition a ``[B,H,W,C]`` tensor into non-overlapping windows."""
    batch, height, width, channels = x.shape
    x = x.reshape(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(
        -1, window_size, window_size, channels
    )


def window_reverse(windows, window_size, height, width):
    """Reverse :func:`window_partition`."""
    batch = windows.shape[0] // ((height // window_size) * (width // window_size))
    x = windows.reshape(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch, height, width, -1)


class WindowAttention(nn.Module):
    """Multi-scale window attention with optional recurrent attention logits."""

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        scale=(1, 3, 5),
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shift_size=0,
        rec_att=False,
    ):
        super().__init__()
        if num_heads % len(scale) != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by the number of attention scales ({len(scale)})"
            )
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.per_head_dim = dim // num_heads
        if self.per_head_dim == 0:
            raise ValueError(f"dim ({dim}) must be at least num_heads ({num_heads})")
        # Released configurations divide exactly. Using the projected dimension here also
        # keeps the source's nominal defaults usable when dim is not divisible by heads.
        self.attention_dim = self.per_head_dim * num_heads
        self.scale = qk_scale or self.per_head_dim**-0.5
        self.shift_size = shift_size
        self.rec_att = rec_att

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                num_heads,
            )
        )
        if rec_att:
            self.lambda_att = nn.Parameter(torch.tensor([0.25]))

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.v = nn.Conv2d(dim, self.attention_dim, 1, 1, bias=qkv_bias)
        self.heads_per_scale = num_heads // len(scale)
        branch_dim = self.per_head_dim * self.heads_per_scale * 2
        qk = []
        for kernel_size in scale:
            if kernel_size == 1:
                qk.append(nn.Conv2d(dim, branch_dim, 1, stride=1, padding=0))
            elif kernel_size == 3:
                hidden_dim = branch_dim // 4
                qk.append(
                    nn.Sequential(
                        nn.Conv2d(dim, hidden_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(hidden_dim, branch_dim, 3, 1, 1),
                    )
                )
            elif kernel_size == 5:
                hidden_dim = branch_dim // 8
                qk.append(
                    nn.Sequential(
                        nn.Conv2d(dim, hidden_dim, 5, 1, 2),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(hidden_dim, branch_dim, 5, 1, 2),
                    )
                )
            else:
                raise ValueError(f"Unsupported attention scale: {kernel_size}")
        self.qk = nn.ModuleList(qk)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        previous_att = None
        if self.rec_att:
            x, previous_att = x

        batch, height, width, _ = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        q_parts = []
        k_parts = []
        branch_features = self.per_head_dim * self.heads_per_scale
        for conv in self.qk:
            qk = (
                conv(x)
                .reshape(batch, 2, branch_features, height, width)
                .permute(1, 0, 3, 4, 2)
                .contiguous()
            )
            q_parts.append(qk[0])
            k_parts.append(qk[1])
        q = torch.cat(q_parts, dim=-1)
        k = torch.cat(k_parts, dim=-1)
        v = self.v(x).permute(0, 2, 3, 1).contiguous()
        qkv = torch.cat((q, k, v), dim=-1)

        if self.shift_size > 0:
            qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        window = self.window_size[0]
        qkv_windows = window_partition(qkv, window)
        qkv = qkv_windows.reshape(-1, window * window, 3, self.attention_dim).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        batch_windows, tokens, _ = q.shape
        q = q.reshape(batch_windows, tokens, self.num_heads, self.per_head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_windows, tokens, self.num_heads, self.per_head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_windows, tokens, self.num_heads, self.per_head_dim).permute(0, 2, 1, 3)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(window * window, window * window, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.reshape(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                tokens,
                tokens,
            )
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.num_heads, tokens, tokens)

        attn_before_softmax = attn
        if not self.rec_att or previous_att is None:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        if self.rec_att and previous_att is not None:
            attn_before_softmax = (
                previous_att * self.lambda_att
                + attn_before_softmax * (1.0 - self.lambda_att)
            )
            attn = self.softmax(attn_before_softmax)

        x = (attn @ v).transpose(1, 2).reshape(batch_windows, tokens, self.attention_dim)
        x = self.proj_drop(self.proj(x))
        return (x, attn_before_softmax) if self.rec_att else x

    def extra_repr(self):
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, tokens):
        flops = tokens * self.dim * 3 * self.dim
        flops += self.num_heads * tokens * self.per_head_dim * tokens
        flops += self.num_heads * tokens * tokens * self.per_head_dim
        flops += tokens * self.dim * self.dim
        return flops


class SwinTransformerBlock_MS(nn.Module):
    """ReconFormer's multi-scale Swin transformer block."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rec_att=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = tuple(int(v) for v in input_resolution)
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError("shift_size must be in [0, window_size)")
        if any(size % self.window_size for size in self.input_resolution):
            raise ValueError(
                f"input resolution {self.input_resolution} must be divisible by window size {self.window_size}"
            )

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            shift_size=self.shift_size,
            rec_att=rec_att,
        )
        self.rec_att = rec_att
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        attn_mask = self.calculate_mask(self.input_resolution) if self.shift_size > 0 else None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        height, width = x_size
        img_mask = torch.zeros((1, height, width, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x):
        height, width = self.input_resolution
        previous_att = None
        if self.rec_att:
            x, previous_att = x
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError(
                f"token length {length} does not match configured resolution {height}x{width}"
            )

        shortcut = x
        x = self.norm1(x).reshape(batch, height, width, channels)
        attn_input = (x, previous_att) if self.rec_att else x
        attn_windows = self.attn(attn_input, mask=self.attn_mask)
        if self.rec_att:
            attn_windows, previous_att = attn_windows

        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, channels)
        shifted_x = window_reverse(attn_windows, self.window_size, height, width)
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x
        x = x.reshape(batch, height * width, channels)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return (x, previous_att) if self.rec_att else x

    def extra_repr(self):
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, window_size={self.window_size}, "
            f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        height, width = self.input_resolution
        flops = self.dim * height * width
        num_windows = height * width / self.window_size / self.window_size
        flops += num_windows * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * height * width * self.dim * self.dim * self.mlp_ratio
        return flops + self.dim * height * width


class BasicLayer(nn.Module):
    """A sequence of recurrent multi-scale Swin blocks."""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        rec_att=False,
        shift=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.rec_att = rec_att

        if depth == 1:
            shifts = [window_size // 2 if shift else 0]
        else:
            shifts = [0 if index % 2 == 0 else window_size // 2 for index in range(depth)]
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock_MS(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shifts[index],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    rec_att=rec_att,
                )
                for index in range(depth)
            ]
        )
        self.downsample = (
            downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            if downsample is not None
            else None
        )

    def forward(self, x):
        if self.rec_att and self.depth > 1:
            previous = x[1]
            previous_list = [None] * self.depth if previous is None else list(previous)
            features = x[0]
            for index, block in enumerate(self.blocks):
                block_input = (features, previous_list[index])
                if self.use_checkpoint:
                    block_output = checkpoint.checkpoint(
                        block, block_input, use_reentrant=False
                    )
                else:
                    block_output = block(block_input)
                features, previous_list[index] = block_output
            x = (features, previous_list)
        else:
            for block in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = sum(block.flops() for block in self.blocks)
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RPTL(nn.Module):
    """Recurrent Pyramid Transformer Layer."""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
        rec_att=False,
        shift=False,
    ):
        super().__init__()
        self.rec_att = rec_att
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            rec_att=rec_att,
            shift=shift,
        )
        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unknown residual connection: {resi_connection!r}")
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(self, x, x_size):
        if self.rec_att:
            residual = x[0]
            x, previous_att = self.residual_group(x)
            x = self.patch_embed(self.conv(self.patch_unembed(x, x_size))) + residual
            return x, previous_att
        residual = x
        x = self.residual_group(x)
        return self.patch_embed(self.conv(self.patch_unembed(x, x_size))) + residual

    def flops(self):
        flops = self.residual_group.flops()
        height, width = self.input_resolution
        flops += height * width * self.dim * self.dim * 9
        return flops + self.patch_embed.flops() + self.patch_unembed.flops()


__all__ = [
    "BasicLayer",
    "DropPath",
    "Mlp",
    "PatchEmbed",
    "PatchUnEmbed",
    "RPTL",
    "SwinTransformerBlock_MS",
    "WindowAttention",
    "drop_path",
    "to_2tuple",
    "window_partition",
    "window_reverse",
]
