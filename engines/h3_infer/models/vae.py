# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image VAE, following the DiffSynth image and video VAE ports.

The encoder uses one causal video frame. The decoder attends over a 2D latent
grid. Both operate in ImageNet-normalized RGB space. Latent normalization
and image quantization are the caller's responsibility.
"""

from __future__ import annotations

import gc
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..config import MiniMaxH3VAEConfig
from ..ops.attention import AttentionBackend, apply_rotary, attention

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DiagonalGaussianDistribution:
    """A diagonal posterior over the encoded image latents."""

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.zeros_like(self.mean) if deterministic else (0.5 * self.logvar).exp()
        self.var = torch.zeros_like(self.mean) if deterministic else self.logvar.exp()

    def mode(self):
        return self.mean

    def sample(self, generator=None):
        device = self.mean.device if generator is None else generator.device
        noise = torch.randn(self.mean.shape, generator=generator, device=device, dtype=self.mean.dtype)
        return self.mean + self.std * noise.to(self.mean.device)

    def kl(self, other=None):
        if self.deterministic:
            return self.mean.new_zeros(1)
        if other is None:
            terms = self.mean.square() + self.var - 1.0 - self.logvar
        else:
            terms = (self.mean - other.mean).square() / other.var
            terms = terms + self.var / other.var - 1.0 - self.logvar + other.logvar
        return 0.5 * terms.sum(dim=(1, 2, 3))


class _CausalConv3d(nn.Conv3d):
    def __init__(self, *args, padding_mode="reflect", **kwargs):
        super().__init__(*args, padding_mode=padding_mode, **kwargs)
        self.pad_mode = "constant" if padding_mode == "zeros" else padding_mode

    def forward(self, x):
        x = x.to(self.weight.dtype)
        pt, ph, pw = self.padding
        if ph or pw:
            x = F.pad(x, (pw, pw, ph, ph, 0, 0), mode=self.pad_mode)
        if pt:
            if x.shape[2] == 1:
                zeros = torch.zeros_like(x).expand(-1, -1, self.kernel_size[0] - 1, -1, -1)
                x = torch.cat((zeros, x), dim=2)
            else:
                x = F.pad(x, (0, 0, 0, 0, 2 * pt, 0))
        return F.conv3d(x, self.weight, self.bias, stride=self.stride, dilation=self.dilation)


class _FrameGroupNorm(nn.GroupNorm):
    def forward(self, x):
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, 1, h, w).contiguous()
        frames = super().forward(frames)
        return frames.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


class _ResnetBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super().__init__()
        self.norm1 = _FrameGroupNorm(config.norm_num_groups, in_channels, eps=config.norm_eps)
        self.norm2 = _FrameGroupNorm(config.norm_num_groups, out_channels, eps=config.norm_eps)
        mode = config.spatial_padding_mode
        self.conv1 = _CausalConv3d(in_channels, out_channels, 3, padding=1, padding_mode=mode)
        self.conv2 = _CausalConv3d(out_channels, out_channels, 3, padding=1, padding_mode=mode)
        if in_channels != out_channels:
            self.nin_shortcut = _CausalConv3d(in_channels, out_channels, 1, padding_mode=mode)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if hasattr(self, "nin_shortcut"):
            x = self.nin_shortcut(x)
        return x + h


class _Downsample3D(nn.Module):
    def __init__(self, channels, time_stride, space_stride, padding_mode):
        super().__init__()
        self.space_stride = space_stride
        self.conv = _CausalConv3d(channels, channels, 3, padding=(1, 0, 0),
                                  stride=(time_stride, space_stride, space_stride), padding_mode=padding_mode)

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode=self.conv.pad_mode)
        return self.conv(x)


class MiniMaxH3ImageEncoder3d(nn.Module):
    """Causal encoder using the released encoder.down.N.block.M keys."""

    def __init__(self, config: MiniMaxH3VAEConfig | None = None):
        super().__init__()
        config = config if config is not None else MiniMaxH3VAEConfig()
        widths = config.block_out_channels
        mode = config.spatial_padding_mode
        self.conv_in = _CausalConv3d(config.in_channels, widths[0], 3, padding=1, padding_mode=mode)
        self.down = nn.ModuleList()
        strides = zip(widths, config.temporal_downsample_factors, config.spatial_downsample_factors, strict=True)
        for in_ch, (out_ch, stride_t, stride_xy) in zip((widths[0], *widths[:-1]), strides, strict=True):
            level = nn.Module()
            level.block = nn.ModuleList(
                _ResnetBlock3d(in_ch if i == 0 else out_ch, out_ch, config)
                for i in range(config.layers_per_block)
            )
            if stride_t * stride_xy > 1:
                level.downsample = _Downsample3D(out_ch, stride_t, stride_xy, mode)
            self.down.append(level)

        self.norm_out = _FrameGroupNorm(config.norm_num_groups, widths[-1], eps=config.norm_eps)
        self.conv_out = _CausalConv3d(widths[-1], 2 * config.latent_channels, 3, padding=1, padding_mode=mode)

    def forward(self, x):
        x = self.conv_in(x)
        for level in self.down:
            for block in level.block:
                x = block(x)
            if hasattr(level, "downsample"):
                x = level.downsample(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class _DecoderAttention(nn.Module):
    def __init__(self, heads, dim_head, qk_norm_type, qk_norm_affine, eps, backend):
        super().__init__()
        self.dim_head = dim_head
        self.backend = backend
        dim = heads * dim_head
        norm = {"rms_norm": nn.RMSNorm, "layer_norm": nn.LayerNorm, None: None}[qk_norm_type]
        self.norm_q = norm(dim_head, eps=eps, elementwise_affine=qk_norm_affine) if norm else nn.Identity()
        self.norm_k = norm(dim_head, eps=eps, elementwise_affine=qk_norm_affine) if norm else nn.Identity()
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, rotary):
        b, seq, _ = x.shape
        qkv = self.to_qkv(x.to(self.to_qkv.weight.dtype)).view(b, seq, -1, 3 * self.dim_head)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.norm_q(q).to(v.dtype), self.norm_k(k).to(v.dtype)
        if rotary is not None:
            q, k = apply_rotary(q, *rotary), apply_rotary(k, *rotary)
        out = attention(q, k, v, backend=self.backend).reshape(b, seq, -1)
        return self.to_out(out.to(self.to_out.weight.dtype))


class _FeedForward(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        inner = int(dim * mult)
        self.w1 = nn.Linear(dim, 2 * inner)
        self.w2 = nn.Linear(inner, dim)

    def forward(self, x):
        gate, value = self.w1(x.to(self.w1.weight.dtype)).chunk(2, dim=-1)
        return self.w2((F.silu(gate) * value).to(self.w2.weight.dtype))


class _DecoderBlock(nn.Module):
    def __init__(self, heads, dim_head, ffn_mult, norm_affine, qk_norm_type, qk_norm_affine, eps, backend):
        super().__init__()
        dim = heads * dim_head
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=norm_affine)
        self.attn = _DecoderAttention(heads, dim_head, qk_norm_type, qk_norm_affine, eps, backend)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=norm_affine)
        self.ff = _FeedForward(dim, ffn_mult)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x, rotary):
        x = x + self.attn(self.norm1(x), rotary) * self.scale1
        return x + self.ff(self.norm2(x)) * self.scale2


class ViT2DDecoder(nn.Module):
    """One token per latent pixel, with rotary positions and learned register tokens."""

    def __init__(self, patch_size=16, in_channels=24, out_channels=3, num_layers=24, heads=16,
                 dim_head=64, ffn_mult=4, rope_theta=100.0, rope_dim_ratio=0.5, norm_eps=1e-5,
                 num_register_tokens=4, norm_affine=True, qk_norm_type="rms_norm", qk_norm_affine=False,
                 attention_backend: AttentionBackend = "auto"):
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.rotary_dim = int(dim_head * rope_dim_ratio)
        self.rope_theta = rope_theta
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = (
            nn.Parameter(torch.randn(1, num_register_tokens, dim) * 0.02) if num_register_tokens else None
        )
        self.transformer_blocks = nn.ModuleList(
            _DecoderBlock(heads, dim_head, ffn_mult, norm_affine, qk_norm_type,
                          qk_norm_affine, norm_eps, attention_backend)
            for _ in range(num_layers)
        )
        self.norm_out = nn.LayerNorm(dim, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size ** 2)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _rotary(self, h, w, batch, device):
        if not self.rotary_dim:
            return None
        axes = [2.0 * (torch.arange(0.5, dim, device=device, dtype=torch.float32) / dim) - 1.0
                for dim in (h, w)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, 1)
        suffix = grid.new_zeros((1 + self.num_register_tokens, 2))
        positions = torch.cat((grid, suffix)).expand(batch, -1, -1)
        inv_freq = 1 / self.rope_theta ** torch.arange(
            0, 1, 4 / self.rotary_dim, dtype=torch.float32, device=device,
        )
        angles = ((2.0 * math.pi) * positions[..., None] * inv_freq).flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()

    def forward(self, x):
        batch, channels, h, w = x.shape
        patches = x.permute(0, 2, 3, 1).reshape(batch, h * w, channels)
        patches = self.x_embedder(patches.to(self.x_embedder.weight.dtype))
        tokens = [patches]
        if self.register_tokens is not None:
            tokens.append(self.register_tokens.expand(batch, -1, -1).to(patches.dtype))
        tokens.append(torch.zeros_like(patches[:, :1]))
        hidden = torch.cat(tokens, dim=1)
        rotary = self._rotary(h, w, batch, x.device)
        for block in self.transformer_blocks:
            hidden = block(hidden, rotary)
        hidden = self.norm_out(hidden)
        hidden = self.proj_out(hidden.to(self.proj_out.weight.dtype))[:, :h * w]
        size = self.patch_size
        pixels = hidden.reshape(batch, h, w, self.out_channels, size, size)
        return pixels.permute(0, 3, 1, 4, 2, 5).reshape(batch, self.out_channels, h * size, w * size)


class MiniMaxH3ImageVAE(nn.Module):
    """Encode [B, 3, H, W] pixels and decode [B, C, h, w] latents.

    Tiled execution uses the release's 256-pixel tiles and at least 64 pixels
    of overlap. The latent statistics are stored in the config, outside the
    checkpoint state dict.
    """

    def __init__(self, config: MiniMaxH3VAEConfig | None = None, *,
                 attention_backend: AttentionBackend = "auto",
                 stats_device: str | torch.device | None = None):
        super().__init__()
        config = config if config is not None else MiniMaxH3VAEConfig()
        options = config.vit_decoder_kwargs
        if (not config.vit_decoder_2d or options["norm_type"] != "rms_norm"
                or options["ffn_activation_fn"] != "silu" or not options["ffn_use_gated"]):
            raise ValueError("Expected the MiniMax-H3-Image 2D gated SiLU ViT decoder")

        self.config = config
        self.spatial_compression_ratio = math.prod(config.spatial_downsample_factors)
        self.vae_ratio = self.spatial_compression_ratio
        self.tile_size = config.vae_tile_size
        self.tile_overlap_min = config.vae_tile_overlap_min
        ratio = self.spatial_compression_ratio
        if not 0 <= self.tile_overlap_min < self.tile_size or self.tile_size % ratio or self.tile_overlap_min % ratio:
            raise ValueError("Tile size and overlap must be aligned to the VAE compression ratio")

        channels = config.latent_channels
        self.encoder = MiniMaxH3ImageEncoder3d(config)
        self.quant_conv = nn.Conv3d(2 * channels, 2 * channels, 1)
        self.post_quant_conv = nn.Conv2d(channels, channels, 1)
        self.decoder = ViT2DDecoder(
            patch_size=ratio, in_channels=channels, out_channels=config.out_channels,
            num_layers=options["num_layers"], heads=options["heads"], dim_head=options["dim_head"],
            ffn_mult=config.decoder_ffn_mult, rope_theta=options["rope_theta"],
            rope_dim_ratio=options["rope_dim_ratio"], norm_eps=config.decoder_norm_eps,
            num_register_tokens=config.decoder_num_register_tokens, norm_affine=options["norm_affine"],
            qk_norm_type=options["qk_norm_type"], qk_norm_affine=options["qk_norm_affine"],
            attention_backend=attention_backend,
        )
        for name in ("latents_mean", "latents_std"):
            stats = torch.tensor(getattr(config, name), device=stats_device, dtype=torch.float32)
            self.register_buffer(name, stats.view(1, -1, 1, 1), persistent=False)

    def _split_tiles(self, length, tile_size, min_overlap):
        """Spread the tile slack over adjacent overlaps in whole latent steps."""
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil((length - min_overlap) / (tile_size - min_overlap))
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        ratio = self.spatial_compression_ratio
        for index in range(remaining // ratio):
            overlaps[index % len(overlaps)] += ratio
        starts = [0]
        for overlap in overlaps:
            starts.append(starts[-1] + tile_size - overlap)
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(a, b, blend_extent, dim):
        extent = min(a.shape[dim], b.shape[dim], blend_extent)
        if not extent:
            return b
        shape = [1] * b.ndim
        shape[dim] = extent
        positions = torch.arange(extent, device=b.device, dtype=b.dtype)
        weight_a = (1 - positions / extent).view(shape)
        weight_b = (positions / extent).view(shape)
        tail = a.narrow(dim, a.shape[dim] - extent, extent)
        head = b.narrow(dim, 0, extent)
        blended = tail * weight_a + head * weight_b
        return torch.cat((blended, b.narrow(dim, extent, b.shape[dim] - extent)), dim=dim)

    def _stitch_tiles(self, tiles, height_overlaps, width_overlaps):
        """Blend adjacent raw tiles, then trim the overlap from preceding tiles."""
        output = []
        for i, row in enumerate(tiles):
            combined = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1 and height_overlaps[i]:
                    tile = tile[..., :-height_overlaps[i], :]
                if j < len(row) - 1 and width_overlaps[j]:
                    tile = tile[..., :-width_overlaps[j]]
                combined.append(tile)
            output.append(torch.cat(combined, dim=-1))
        return torch.cat(output, dim=-2)

    def _tiled(self, x, operation, *, input_scale, output_scale, tile_batch_size: int | None = None):
        if tile_batch_size is None:
            tile_batch_size = (64 if output_scale == 1 else 8) if x.device.type == "cuda" else 1
        if tile_batch_size <= 0:
            raise ValueError("tile_batch_size must be positive")

        h, w = (size * input_scale for size in x.shape[-2:])
        ys, heights, y_overlaps = self._split_tiles(h, self.tile_size, self.tile_overlap_min)
        xs, widths, x_overlaps = self._split_tiles(w, self.tile_size, self.tile_overlap_min)
        tile_inputs = [
            [x[..., y // input_scale:(y + th) // input_scale,
                 z // input_scale:(z + tw) // input_scale]
             for z, tw in zip(xs, widths)]
            for y, th in zip(ys, heights)
        ]
        num_tiles = len(heights) * len(widths)
        if tile_batch_size <= 1 or num_tiles <= 1 or len(set(heights)) != 1 or len(set(widths)) != 1:
            tiles = [[operation(tile) for tile in row] for row in tile_inputs]
        else:
            flat_tiles = [tile for row in tile_inputs for tile in row]
            flat_outputs = []
            pos = 0
            chunk_tiles = min(tile_batch_size, num_tiles)
            while pos < num_tiles:
                chunk_tiles = min(chunk_tiles, num_tiles - pos)
                chunk = flat_tiles[pos:pos + chunk_tiles]
                try:
                    batch_out = operation(torch.cat(chunk, dim=0) if chunk_tiles > 1 else chunk[0])
                except torch.cuda.OutOfMemoryError:
                    if chunk_tiles == 1:
                        raise
                else:
                    flat_outputs.extend(batch_out.split(x.shape[0], dim=0))
                    pos += chunk_tiles
                    continue

                # Release the failed operation's traceback before reclaiming memory.
                gc.collect()
                if x.device.type == "cuda":
                    with torch.cuda.device(x.device):
                        torch.cuda.empty_cache()
                chunk_tiles = max(1, chunk_tiles // 2)

            cols = len(widths)
            tiles = [flat_outputs[pos:pos + cols] for pos in range(0, num_tiles, cols)]
        return self._stitch_tiles(tiles, [size // output_scale for size in y_overlaps],
                                 [size // output_scale for size in x_overlaps])

    def _encode_tile(self, x):
        frames = self.encoder(x[:, :, None])
        return self.quant_conv(frames.to(self.quant_conv.weight.dtype))[:, :, -1]

    def _decode_tile(self, z):
        return self.decoder(self.post_quant_conv(z.to(self.post_quant_conv.weight.dtype)))

    def encode(self, x, tiling: bool | None = None, sample=False, generator=None, *,
               tile_batch_size: int | None = None):
        """Encode pixels using the configured tiling unless explicitly overridden."""
        if tiling is None:
            tiling = self.config.vae_encoder_tiling
        moments = (
            self._tiled(x, self._encode_tile, input_scale=1, output_scale=self.vae_ratio,
                        tile_batch_size=tile_batch_size)
            if tiling else self._encode_tile(x)
        )
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.sample(generator) if sample else posterior

    def decode(self, z, tiling: bool | None = None, *, tile_batch_size: int | None = None):
        """Decode latents using the configured tiling unless explicitly overridden."""
        if tiling is None:
            tiling = self.config.vae_decoder_tiling
        return (
            self._tiled(z, self._decode_tile, input_scale=self.vae_ratio, output_scale=1,
                        tile_batch_size=tile_batch_size)
            if tiling else self._decode_tile(z)
        )

    def forward(self, x, tiling=None, sample=False, generator=None):
        posterior = self.encode(x, tiling=tiling)
        z = posterior.sample(generator) if sample else posterior.mode()
        return self.decode(z, tiling=tiling)
