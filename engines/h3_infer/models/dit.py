"""MiniMax-H3-Image transformer and its separate distilled output heads.

The attention, timestep and packing equations follow the DiffSynth image port.
The base output head and RoPE parameter retain their release checkpoint names.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Mapping
from threading import RLock
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..config import MiniMaxH3DiTConfig, MiniMaxH3PDDConfig
from ..ops.attention import AttentionBackend, apply_rotary, attention
from ..ops.quant import Int8ConvRotLinear
from ..runtime.compile import CompileMode, compile_dit, run_block
from ..runtime.offload import AsyncBlockOffloader
from ..runtime.scheduler import MiniMaxH3Scheduler

PATCH_SIZE = (1, 2, 2)
REFERENCE_TIMESTEP = 0.999
_MODALITIES = 3
_ADALN_TABLE_ROWS = 7


def patchify_video(latent: torch.Tensor, patch_size=PATCH_SIZE) -> torch.Tensor:
    """Pack [B, C, T, H, W] into rows in batch, time, height, width order."""
    b, c, ft, fh, fw = latent.shape
    pt, ph, pw = patch_size
    if ft % pt or fh % ph or fw % pw:
        raise ValueError("Latent dimensions must be divisible by the patch size")
    groups = latent.reshape(b, c, ft // pt, pt, fh // ph, ph, fw // pw, pw)
    return groups.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, c * pt * ph * pw)


def unpatchify_video(rows, ft, fh, fw, channel=24, patch_size=PATCH_SIZE):
    """Restore patch rows to [B, C, T, H, W]."""
    pt, ph, pw = patch_size
    if ft % pt or fh % ph or fw % pw:
        raise ValueError("Latent dimensions must be divisible by the patch size")
    groups = rows.reshape(-1, ft // pt, fh // ph, fw // pw, channel, pt, ph, pw)
    return groups.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(-1, channel, ft, fh, fw)


def _spatial_axis(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * 32
    return torch.from_numpy(grid)


def build_packed_sequence(text_token_tags, reference_grids, latent_height, latent_width,
                          *, patch_size=PATCH_SIZE):
    """Lay out [text | references | generated image] and return the DiffSynth indices.

    The result is (position_ids [S, 3], token_tags, image_indices, text_indices,
    num_reference_rows). The aspect-normalized rotary grid is built in float64.
    """
    _, ph, pw = patch_size
    text_token_tags = torch.as_tensor(text_token_tags, dtype=torch.long, device="cpu").reshape(-1)
    num_text = text_token_tags.numel()
    grids = [*reference_grids, (latent_height, latent_width)]
    counts = [(h // ph) * (w // pw) for h, w in grids]
    seq_len = num_text + sum(counts)
    position_ids = torch.zeros(seq_len, 3, dtype=torch.float64)
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)
    cursor = num_text
    for index, ((height, width), count) in enumerate(zip(grids, counts)):
        if height <= 0 or width <= 0 or height % ph or width % pw:
            raise ValueError("Image grids must contain whole latent patches")
        sqrt_area = np.sqrt(height * width)
        axes = (_spatial_axis(height, ph, sqrt_area), _spatial_axis(width, pw, sqrt_area))
        spatial = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 2)
        position_ids[cursor:cursor + count, 0] = num_text + index
        position_ids[cursor:cursor + count, 1:] = spatial
        cursor += count

    tags = torch.zeros(seq_len, dtype=torch.long)
    tags[:num_text] = text_token_tags
    return position_ids, tags, torch.arange(num_text, seq_len), torch.arange(num_text), sum(counts[:-1])


def _linear(layer, x):
    return layer(_linear_input(layer, x))


def _linear_input(layer, x):
    return x.to(layer.weight.dtype) if layer.weight.is_floating_point() else x


class _Rope(nn.Module):
    def __init__(self, inv_freq_len):
        super().__init__()
        self.inv_freq_len = inv_freq_len
        self.inv_freq = nn.Parameter(self._build())

    def _build(self, device=None):
        steps = torch.arange(self.inv_freq_len, device=device, dtype=torch.float32)
        return 1.0 / (10000.0 ** (steps / self.inv_freq_len))

    def forward(self, position_ids):
        # The release computes the full-precision frequencies from scratch.
        pos = position_ids[0].to(torch.float32)
        freqs = pos.unsqueeze(-1) * self._build(pos.device).view(1, 1, -1)
        half = freqs.flatten(1)
        angles = torch.cat((half, half), dim=-1).unsqueeze(1)
        return angles.cos(), angles.sin()


class _TimeEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj_in = nn.Linear(config.timestep_input_dim, config.time_embed_hidden_size)
        self.proj_out = nn.Linear(config.time_embed_hidden_size, config.time_embed_dim)

    def forward(self, t):
        half = self.proj_in.in_features // 2
        freqs = torch.exp(-math.log(10000.0) *
                          torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return _linear(self.proj_out, F.silu(_linear(self.proj_in, t_freq)))


class _Attention(nn.Module):
    def __init__(self, config, backend, qkv_layout: Literal["interleaved", "split"] = "interleaved"):
        super().__init__()
        if qkv_layout not in ("interleaved", "split"):
            raise ValueError(f"Unknown QKV layout {qkv_layout!r}")
        self.heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.backend = backend
        self.qkv_layout = qkv_layout
        inner = self.heads * self.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.qk_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.qk_norm_eps)
        self.out_proj = nn.Linear(inner, config.hidden_size, bias=False)

    def forward(self, x, cu_seqlens, rope=None):
        qkv = _linear(self.qkv_proj, x)
        if self.qkv_layout == "split":
            q, k, v = qkv.view(x.shape[0], 3, self.heads, self.head_dim).unbind(1)
        else:
            q, k, v = qkv.view(x.shape[0], self.heads, 3, self.head_dim).unbind(2)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        if rope is not None:
            q, k = apply_rotary(q, *rope), apply_rotary(k, *rope)
        out = attention(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=x.shape[0],
                        softmax_scale=self.head_dim ** -0.5, backend=self.backend)
        return _linear(self.out_proj, out.flatten(1))


class _MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_hidden_size * 2, bias=False)
        self.fc2 = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate, up = _linear(self.fc1, x).chunk(2, dim=-1)
        return _linear(self.fc2, F.silu(gate) * up)


def _scale_shift(x, shift, scale, indices):
    return (x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(x.dtype)


def _gate(x, gate, update, indices):
    return (x + gate.index_select(0, indices) * update).to(x.dtype)


class _AdaLN(nn.Module):
    def __init__(self, config, out_features, chunks, modalities, *, in_features=None, apply_silu=True):
        super().__init__()
        self.chunks = chunks
        self.hidden_size = config.hidden_size
        self.modalities = modalities
        if out_features != config.hidden_size * chunks * modalities:
            raise ValueError("AdaLN width must equal hidden_size × chunks × modalities")
        self.apply_silu = apply_silu
        in_dim = config.time_embed_dim if in_features is None else in_features
        self.linear = nn.Linear(in_dim, out_features)

    def forward(self, t_emb):
        # The released time embedder is fp32. Round only after SiLU.
        inp = F.silu(t_emb) if self.apply_silu else t_emb
        weight, bias = self.linear.weight, self.linear.bias
        if weight.is_floating_point():
            result = F.linear(inp.to(weight.dtype), weight, None if bias is None else bias.to(weight.dtype))
        else:
            result = self.linear(inp)
        return result.reshape(-1, self.chunks * self.hidden_size).chunk(self.chunks, dim=-1)


class MiniMaxH3AdalnTable(nn.Module):
    """Distilled AdaLN outputs in step, modality, chunk, hidden order."""

    def __init__(self, hidden_size, modality_num, expand_ratio,
                 num_timesteps=_ADALN_TABLE_ROWS, dtype=torch.bfloat16):
        super().__init__()
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = hidden_size
        self.table = nn.Parameter(
            torch.empty(num_timesteps, modality_num, expand_ratio, hidden_size, dtype=dtype),
            requires_grad=False,
        )

    def forward(self, timestep_rows):
        return self.table.index_select(0, timestep_rows).reshape(
            -1, self.expand_ratio * self.hidden_size,
        ).chunk(self.expand_ratio, dim=-1)


class _RefinerBlock(nn.Module):
    def __init__(self, config, backend, qkv_layout: Literal["interleaved", "split"] = "interleaved"):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = _Attention(config, backend, qkv_layout=qkv_layout)
        self.mlp = _MLP(config)

    def forward(self, x, cu_seqlens):
        x = x + self.attn(self.norm1(x), cu_seqlens)
        return x + self.mlp(self.norm2(x))


class _TokenRefiner(nn.Module):
    def __init__(self, config, backend, qkv_layout: Literal["interleaved", "split"] = "interleaved"):
        super().__init__()
        self.blocks = nn.ModuleList(
            _RefinerBlock(config, backend, qkv_layout=qkv_layout) for _ in range(config.token_refiner_num_layers)
        )
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)

    def forward(self, x, cu_seqlens):
        for block in self.blocks:
            x = block(x, cu_seqlens)
        return self.final_norm(x)


class _DiTBlock(_RefinerBlock):
    def __init__(self, config, backend, pruned_adaln=False, adaln_curve_dim=None,
                 qkv_layout: Literal["interleaved", "split"] = "interleaved"):
        super().__init__(config, backend, qkv_layout=qkv_layout)
        if pruned_adaln:
            self.adaln_proj = MiniMaxH3AdalnTable(config.hidden_size, _MODALITIES, 6)
        else:
            self.adaln_proj = _AdaLN(config, config.adaln_out_features, 6, _MODALITIES,
                                    in_features=adaln_curve_dim, apply_silu=adaln_curve_dim is None)

    def forward(self, x, t_emb, combined_indices, rope, cu_seqlens):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _scale_shift(self.norm1(x), shift_attn, scale_attn, combined_indices)
        x = _gate(x, gate_attn, self.attn(h, cu_seqlens, rope), combined_indices)
        h = _scale_shift(self.norm2(x), shift_mlp, scale_mlp, combined_indices)
        return _gate(x, gate_mlp, self.mlp(h), combined_indices)


class _FinalLayer(nn.Module):
    def __init__(self, config, pruned_adaln=False, adaln_curve_dim=None):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)
        if pruned_adaln:
            self.adaln_proj = MiniMaxH3AdalnTable(config.hidden_size, 1, 2)
        else:
            self.adaln_proj = _AdaLN(config, config.final_adaln_out_features, 2, 1,
                                    in_features=adaln_curve_dim, apply_silu=adaln_curve_dim is None)
            self.video_out = nn.Linear(config.hidden_size, config.patch_dim)

    def forward(self, x, t_emb, timestep_indices):
        shift, scale = self.adaln_proj(t_emb)
        return _scale_shift(self.norm(x), shift, scale, timestep_indices)


class _HeadParameters(nn.Module):
    def __init__(self, num_heads, hidden_size, patch_dim):
        super().__init__()
        weight = torch.empty(num_heads, patch_dim, hidden_size, dtype=torch.float32)
        self.weight = nn.Parameter(nn.init.normal_(weight, std=hidden_size ** -0.5))
        self.bias = nn.Parameter(torch.zeros(num_heads, patch_dim, dtype=torch.float32))


class MiniMaxH3PDDHeads(nn.Module):
    """The six output projections stacked as video_out.weight and video_out.bias."""

    def __init__(self, num_heads: int = 6, hidden_size: int = 5376, patch_dim: int = 96):
        super().__init__()
        self.num_heads = num_heads
        self.video_out = _HeadParameters(num_heads, hidden_size, patch_dim)

    @property
    def dtype(self):
        return self.video_out.weight.dtype

    def forward(self, hidden_states, step_index):
        return F.linear(hidden_states.to(self.dtype),
                        self.video_out.weight[step_index], self.video_out.bias[step_index])


class MiniMaxH3DiT(nn.Module):
    """Predict data-ward velocity for packed image rows, including reference rows.

    Inputs follow DiffSynth's MiniMaxH3ImageDiT contract. The sampler passes
    the separate PDD heads, then uses only the generated rows of the output.
    With offload=True, CPU-backed blocks can use async CUDA transfers.
    """

    def __init__(self, config: MiniMaxH3DiTConfig | None = None, *,
                 attention_backend: AttentionBackend = "auto", offload: bool = False,
                 pruned_adaln: bool = False, adaln_curve_grid: int | None = None,
                 adaln_curve_dim: int | None = None, quant_map: Mapping[str, Mapping] | None = None,
                 qkv_layout: Literal["interleaved", "split"] = "interleaved"):
        super().__init__()
        if qkv_layout not in ("interleaved", "split"):
            raise ValueError(f"Unknown QKV layout {qkv_layout!r}")
        config = config if config is not None else MiniMaxH3DiTConfig()
        if config.audio_enabled:
            raise ValueError("MiniMaxH3DiT supports the image-only checkpoint")
        if 6 * config.rope_inv_freq_len > config.attention_head_dim:
            raise ValueError("The rotary dimensions exceed the DiT attention head dimension")
        if adaln_curve_grid is not None:
            if pruned_adaln or adaln_curve_grid < 2 or adaln_curve_dim is None or adaln_curve_dim < 1:
                raise ValueError("AdaLN curves require an unpruned projection, at least two rows and positive width")
            self.register_buffer("adaln_t_table", torch.empty(adaln_curve_grid, adaln_curve_dim, dtype=torch.float32))
        else:
            if adaln_curve_dim is not None:
                raise ValueError("AdaLN curve width requires a grid size")
            self.adaln_t_table = None

        self.config = config
        self.hidden_size = config.hidden_size
        self.offload = offload
        self.pruned_adaln = pruned_adaln
        self._offload_lock = RLock() if offload else None
        self._offloader = None
        self._execution_device = None
        self._block_runner = run_block
        self.video_patch_proj = nn.Linear(config.patch_dim, config.hidden_size)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size)
        self.time_embedder = None if pruned_adaln or adaln_curve_grid is not None else _TimeEmbedder(config)
        self.rope = _Rope(config.rope_inv_freq_len)
        self.token_refiner = _TokenRefiner(config, attention_backend, qkv_layout=qkv_layout)
        self.blocks = nn.ModuleList(
            _DiTBlock(config, attention_backend, pruned_adaln, adaln_curve_dim, qkv_layout=qkv_layout)
            for _ in range(config.num_layers)
        )
        self.final_layer = _FinalLayer(config, pruned_adaln, adaln_curve_dim)
        for name, meta in (quant_map or {}).items():
            if meta.get("format") != "int8_tensorwise":
                raise ValueError(f"{name}: unsupported quantization format {meta.get('format')!r}")
            try:
                layer = self.get_submodule(name)
            except AttributeError as error:
                raise ValueError(f"{name}: quantization metadata must target a linear layer") from error
            if not isinstance(layer, nn.Linear):
                raise ValueError(f"{name}: quantization metadata must target a linear layer")
            self.set_submodule(name, Int8ConvRotLinear(
                layer.in_features, layer.out_features, bias=layer.bias is not None,
                convrot=meta.get("convrot", False), convrot_groupsize=meta.get("convrot_groupsize", 256),
            ))

    def _eval_adaln_curve(self, t_vals):
        table = self.adaln_t_table.to(device=t_vals.device, dtype=torch.float32)
        pos = t_vals.to(torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

    @torch.no_grad()
    def prune_adaln_tables(self, sigmas=None, reference_timestep=REFERENCE_TIMESTEP,
                           device=None) -> MiniMaxH3DiT:
        """Replace the timestep embedder and AdaLN projections with fixed PDD tables."""
        if self.pruned_adaln:
            return self
        offloader = self._offloader
        if offloader is not None and offloader.prepared and offloader.device.type == "cuda":
            raise ValueError("Release the CUDA offloader before pruning AdaLN tables")
        sigmas = tuple(MiniMaxH3PDDConfig().sigmas[:-1] if sigmas is None else sigmas)
        if self.adaln_t_table is None and len(sigmas) != _ADALN_TABLE_ROWS - 1:
            raise ValueError("AdaLN pruning requires six PDD sigmas")
        # Match the sampler's float32 sigma -> timestep -> model timestep order.
        num_rows = len(sigmas) + 1
        scheduler = MiniMaxH3Scheduler(MiniMaxH3PDDConfig(
            num_inference_steps=len(sigmas), sigmas=(*map(float, sigmas), 0.0),
        ))
        times = torch.cat((scheduler.model_timesteps, scheduler.model_timesteps.new_tensor([reference_timestep])))
        if self.adaln_t_table is not None:
            compute_device = device if device is not None else self.adaln_t_table.device
            t_emb = self._eval_adaln_curve(times.to(compute_device))
        else:
            compute_device = device if device is not None else self.time_embedder.proj_in.weight.device
            time_weights = {
                name: param.to(device=compute_device, dtype=torch.float32) if param.is_floating_point()
                else param.to(compute_device)
                for name, param in self.time_embedder.named_parameters()
            }
            t_emb = torch.func.functional_call(self.time_embedder, time_weights, (times.to(compute_device),))
            del time_weights

        # Build the small tables first so a failed conversion leaves the model usable.
        # Only one AdaLN linear is copied to the working device at a time.
        tables = []
        for layer in (*self.blocks, self.final_layer):
            proj = layer.adaln_proj
            weights = {name: param.to(compute_device) for name, param in proj.named_parameters()}
            values = torch.cat(torch.func.functional_call(proj, weights, (t_emb,)), dim=-1)
            shape = (num_rows, proj.modalities, proj.chunks, proj.hidden_size)
            with torch.device("meta"):
                table = MiniMaxH3AdalnTable(
                    proj.hidden_size, proj.modalities, proj.chunks, num_timesteps=num_rows, dtype=values.dtype,
                )
            table.table = nn.Parameter(values.reshape(shape).to(proj.linear.weight.device), requires_grad=False)
            tables.append((layer, table))
        for layer, table in tables:
            layer.adaln_proj = table
        if hasattr(self.final_layer, "video_out"):
            del self.final_layer.video_out
        self.time_embedder = None
        self.adaln_t_table = None
        self.pruned_adaln = True
        tables.clear()
        del proj, weights, values, t_emb
        if offloader is not None:
            self.enable_async_offload(offloader.device, offloader.resident_blocks, offloader.pin_staging)
        gc.collect()
        if torch.device(compute_device).type == "cuda":
            with torch.cuda.device(compute_device):
                torch.cuda.empty_cache()
        return self

    def enable_async_offload(self, device, resident_blocks="auto", pin_staging=True):
        if self._offload_lock is None:
            self._offload_lock = RLock()
        with self._offload_lock:
            if self._offloader is not None and self._offloader.prepared:
                self._offloader.release_gpu()
            self._offloader = AsyncBlockOffloader(self.blocks, device, resident_blocks, pin_staging)
            self.offload = True
        return self

    def prepare_gpu(self, device=None):
        if device is None:
            device = self._execution_device
            if self._offloader is not None:
                device = self._offloader.device
            if device is None:
                device = self.video_patch_proj.weight.device
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self.offload:
            with self._offload_lock:
                if device.type == "cuda" and (self._offloader is None or self._offloader.device != device):
                    settings = self._offloader
                    self.enable_async_offload(
                        device, settings.resident_blocks if settings else "auto",
                        settings.pin_staging if settings else True,
                    )
                if self._offloader is not None and self._offloader.device == device:
                    self._offloader.prepare_gpu()
        elif self.video_patch_proj.weight.device != device:
            self.to(device)
        self._execution_device = device

    def release_gpu(self):
        if self._offload_lock is not None:
            with self._offload_lock:
                if self._offloader is not None and self._offloader.prepared:
                    self._offloader.release_gpu()
        if self.video_patch_proj.weight.is_cuda:
            with torch.cuda.device(self.video_patch_proj.weight.device):
                self.to("cpu")
                torch.cuda.empty_cache()

    def compile(self, mode: CompileMode = "regional", **kwargs):
        """Return the DiT with a shared block runner, or a full compiled wrapper."""
        return compile_dit(self, mode, **kwargs)

    def _run(self, module, *args, runner=run_block):
        if self._offload_lock is None:
            return runner(module, *args)
        # functional_call temporarily swaps in the copied parameters.
        with self._offload_lock:
            device = args[0].device
            if next(module.parameters(), args[0]).device == device:
                return runner(module, *args)
            weights = {name: parameter.to(device) for name, parameter in module.named_parameters()}
            weights.update((name, buffer.to(device)) for name, buffer in module.named_buffers())
            return torch.func.functional_call(module, weights, args, strict=True)

    def _run_blocks(self, hidden, *args):
        if self.offload:
            with self._offload_lock:
                device = hidden.device
                if device.type == "cuda":
                    offloader = self._offloader
                    if offloader is None:
                        needs_prepare = self.video_patch_proj.weight.device != device
                    else:
                        needs_prepare = offloader.device != device or not offloader.prepared
                    if needs_prepare:
                        self.prepare_gpu(device)
                if self._offloader is not None and self._offloader.device == device:
                    return self._offloader.run_blocks(self._block_runner, hidden, *args)
        for block in self.blocks:
            hidden = self._run(block, hidden, *args, runner=self._block_runner)
        return hidden

    def forward(self, x, prompt_embeds, position_ids, unique_timesteps, timestep_indices,
                token_tags, img_pos, text_pos, pdd_heads: MiniMaxH3PDDHeads, pdd_step_index: int = 0,
                precomputed_rope: tuple[torch.Tensor, torch.Tensor] | None = None):
        if x.ndim != 2 or position_ids.ndim != 3 or position_ids.shape[0] != 1 or position_ids.shape[-1] != 3:
            raise ValueError("Expected image rows [N, patch_dim] and position_ids [1, S, 3]")
        device = x.device
        seq_len = position_ids.shape[1]
        timestep_indices, token_tags, img_pos, text_pos = (
            values.reshape(-1).to(device=device, dtype=torch.long)
            for values in (timestep_indices, token_tags, img_pos, text_pos)
        )
        if timestep_indices.numel() != seq_len or token_tags.numel() != seq_len:
            raise ValueError("Timestep indices and token tags must have one value per packed row")

        image = self._run(self.video_patch_proj, _linear_input(self.video_patch_proj, x))
        text = self._run(self.condition_proj, _linear_input(self.condition_proj, prompt_embeds.to(device)))
        text_bounds = torch.tensor([0, text.shape[0]], device=device, dtype=torch.int32)
        text = self._run(self.token_refiner, text, text_bounds)
        hidden = text.new_zeros((seq_len, self.hidden_size))
        hidden.index_copy_(0, text_pos, text)
        hidden.index_copy_(0, img_pos, image.to(hidden.dtype))

        if self.pruned_adaln:
            timestep = torch.full((unique_timesteps.numel(),), int(pdd_step_index),
                                  dtype=torch.long, device=device)
            timestep[1:] = self.final_layer.adaln_proj.table.shape[0] - 1
        elif self.adaln_t_table is not None:
            timestep = self._eval_adaln_curve(unique_timesteps.reshape(-1).to(device))
        else:
            timestep = self._run(self.time_embedder, unique_timesteps.reshape(-1).to(device))
        combined = timestep_indices * _MODALITIES + token_tags.clamp(min=0)
        rope = self.rope(position_ids.to(device)) if precomputed_rope is None else precomputed_rope
        bounds = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
        hidden = self._run_blocks(hidden, timestep, combined, rope, bounds)
        hidden = self._run(self.final_layer, hidden, timestep, timestep_indices)
        return pdd_heads(hidden, pdd_step_index).index_select(0, img_pos)
