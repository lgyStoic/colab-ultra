"""Six-step image generation or editing from precomputed text embeddings."""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors import safe_open

from .config import MiniMaxH3DiTConfig, MiniMaxH3PDDConfig, MiniMaxH3VAEConfig
from .models.dit import (
    REFERENCE_TIMESTEP,
    MiniMaxH3DiT,
    MiniMaxH3PDDHeads,
    build_packed_sequence,
    patchify_video,
    unpatchify_video,
)
from .models.vae import IMAGENET_MEAN, IMAGENET_STD, MiniMaxH3ImageVAE
from .ops.attention import AttentionBackend
from .runtime.compile import CompileMode
from .runtime.loader import inspect_comfy_checkpoint, load_safetensors
from .runtime.scheduler import MiniMaxH3Scheduler

MIN_GENERATION_AREA = 2048 * 2048
_CANVAS_MULTIPLE = 32
_UNUSED_ATTENTION_SUFFIX = (".comfy_attention.config",)
_COMFY_DIT_FILENAME = "minimax_h3_image_pruned_int8_convrot.safetensors"
_EMBEDDED_PDD_KEYS = ("final_layer.video_out.weight", "final_layer.video_out.bias")
_DIT_RESIDENT_VAE_ERROR = (
    "Cannot move VAE to GPU while keep_dit_resident() is active; "
    "decode latents after exiting or use generate_batch()."
)


@dataclass(frozen=True)
class InferenceStats:
    step_ms: tuple[float, ...]
    peak_vram_bytes: int | None


def get_canvas_size(height, width, min_generation_area=MIN_GENERATION_AREA, multiple=_CANVAS_MULTIPLE):
    """Scale to the minimum image area and round each side to the nearest multiple."""
    scale = max(1.0, math.sqrt(min_generation_area / (height * width)))
    return tuple(max(multiple, round(side * scale / multiple) * multiple) for side in (height, width))


def _resize_reference(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    short = min(width, height)
    if short < _CANVAS_MULTIPLE:
        raise ValueError(f"Reference image sides must be at least {_CANVAS_MULTIPLE} pixels, got {image.size}")
    if 1024 <= short <= 2048:
        return image
    scale = (1024 if short < 1024 else 2048) / short
    size = tuple(max(32, round(side * scale / 32) * 32) for side in (width, height))
    return image.resize(size, Image.Resampling.LANCZOS)


def _load_embedded_pdd_heads(heads, path, device, dtype_override):
    state = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key, expected in heads.state_dict().items():
            source = f"final_layer.{key}"
            if source not in keys:
                raise ValueError(f"{path}: missing embedded PDD tensor {source}")
            packed = handle.get_tensor(source)
            flat_shape = (expected.shape[0] * expected.shape[1], *expected.shape[2:])
            if not packed.is_floating_point() or packed.shape != flat_shape:
                raise ValueError(f"{path}: {source} must contain {expected.shape[0]} flattened PDD heads")
            rows = packed.to(device).reshape(expected.shape)
            unfolded = torch.cat((rows[:1], rows[:1] + rows[1:]), dim=0)
            state[key] = unfolded.to(dtype_override) if dtype_override is not None else unfolded
    heads.load_state_dict(state, assign=True)


class MiniMaxH3ImagePipeline:
    """Generate or edit images with the distilled MiniMax-H3-Image model."""

    def __init__(self, dit: MiniMaxH3DiT, pdd_heads: MiniMaxH3PDDHeads, vae: MiniMaxH3ImageVAE,
                 scheduler: MiniMaxH3Scheduler | None = None, *, device: str | torch.device | None = None,
                 staged_vram: bool | None = None):
        self.dit = dit.eval()
        self.pdd_heads = pdd_heads.eval()
        self.vae = vae.eval()
        self.scheduler = scheduler if scheduler is not None else MiniMaxH3Scheduler()
        self.device = torch.device(device if device is not None else pdd_heads.video_out.weight.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.staged_vram = self.device.type == "cuda" and staged_vram is not False
        self._dit_resident_depth = 0
        self._vae_device_depth = 0
        if dit.config.patch_size[0] != 1 or dit.config.latents_dim != vae.config.latent_channels:
            raise ValueError("The image pipeline requires one-frame patches and matching VAE/DiT latent channels")
        if pdd_heads.num_heads != self.scheduler.num_inference_steps:
            raise ValueError("The number of output heads must match the PDD schedule")
        if self.staged_vram:
            self.dit.release_gpu()
            if self.vae_device.type != "cpu":
                self.vae.to("cpu")
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

    @property
    def vae_device(self):
        return self.vae.post_quant_conv.weight.device

    @classmethod
    def from_pretrained(cls, root: str | Path, *, device: str | torch.device = "cpu",
                        attention_backend: AttentionBackend = "auto",
                        offload_dit: bool | None = None, pdd_heads_path: str | Path | None = None,
                        dit_checkpoint: str | Path | None = None,
                        dtype_override: torch.dtype | None = None, staged_vram: bool | None = None,
                        prune_adaln: bool | Literal["auto"] = "auto", resident_blocks: int | str = "auto",
                        pin_staging: bool = True, compile_mode: CompileMode = "none"):
        """Load a local release directory or Comfy-Org DiT with a companion VAE.

        With offload_dit=None, use file-backed CPU weights and CUDA layer
        transfers if there is insufficient free GPU memory for the DiT.
        """
        root = Path(root)
        dit_path = Path(dit_checkpoint) if dit_checkpoint is not None else None
        if root.suffix == ".safetensors" and not root.is_dir():
            if dit_path is None:
                dit_path = root
            root = root.parent.parent if root.parent.name == "transformer" else root.parent
        device = torch.device(device)
        staged_vram = device.type == "cuda" and staged_vram is not False
        vae_device = torch.device("cpu") if staged_vram else device
        index = root / "transformer/model.safetensors.index.json"
        if dit_path is None and not index.is_file():
            candidates = (root / _COMFY_DIT_FILENAME, root / "transformer" / _COMFY_DIT_FILENAME)
            dit_path = next((path for path in candidates if path.is_file()), None)

        dit_config_path = root / "transformer/config.json"
        pdd_config_path = root / "pdd_config.json"
        vae_config_path = root / "vae/config.json"
        vae_config = (MiniMaxH3VAEConfig.from_json(vae_config_path) if vae_config_path.is_file()
                      else MiniMaxH3VAEConfig())
        if dit_path is not None:
            if dit_path.suffix != ".safetensors":
                raise ValueError("dit_checkpoint must be a .safetensors file")
            info = inspect_comfy_checkpoint(dit_path)
            dit_config = (MiniMaxH3DiTConfig.from_json(dit_config_path) if dit_config_path.is_file()
                          else MiniMaxH3DiTConfig())
            pdd_config = (MiniMaxH3PDDConfig.from_json(pdd_config_path) if pdd_config_path.is_file()
                          else info.pdd_config or MiniMaxH3PDDConfig())
            model_bytes = dit_path.stat().st_size if offload_dit is None else None
            pruned_adaln = info.pruned_adaln
            curve_grid, curve_dim = info.adaln_curve_shape or (None, None)
            quant_map = info.quant_map
            embedded_heads = pdd_heads_path is None and info.video_out_shape == (
                pdd_config.num_inference_steps * dit_config.patch_dim, dit_config.hidden_size,
            )
            qkv_layout = ("split" if info.adaln_curve_shape is not None or info.quant_map or embedded_heads
                          else "interleaved")
        else:
            with index.open(encoding="utf-8") as handle:
                index_data = json.load(handle)
            weight_map = index_data["weight_map"]
            pruned_adaln = ("blocks.0.adaln_proj.table" in weight_map
                            or "final_layer.adaln_proj.table" in weight_map)
            dit_config = MiniMaxH3DiTConfig.from_json(dit_config_path)
            pdd_config = MiniMaxH3PDDConfig.from_json(pdd_config_path)
            model_bytes = index_data.get("metadata", {}).get("total_size")
            if model_bytes is not None and dtype_override is not None:
                element_bytes = torch.empty((), device="meta", dtype=dtype_override).element_size()
                model_bytes = model_bytes * max(2, element_bytes) // 2
            curve_grid = curve_dim = quant_map = None
            embedded_heads = False
            qkv_layout = "interleaved"
        if offload_dit is None:
            offload_dit = False
            if device.type == "cuda":
                free_bytes, _ = torch.cuda.mem_get_info(device)
                offload_dit = model_bytes is None or free_bytes < model_bytes + 6 * 1024 ** 3
        if compile_mode == "full" and offload_dit:
            raise ValueError("Full compilation requires offload=False")
        if pdd_heads_path is None and not embedded_heads:
            pdd_heads_path = root / "pdd_fused_heads.safetensors"
            if not pdd_heads_path.is_file():
                pdd_heads_path = root / "transformer/pdd_fused_heads.safetensors"

        with torch.device("meta"):
            dit = MiniMaxH3DiT(dit_config, attention_backend=attention_backend, offload=offload_dit,
                              pruned_adaln=pruned_adaln, adaln_curve_grid=curve_grid,
                              adaln_curve_dim=curve_dim, quant_map=quant_map, qkv_layout=qkv_layout)
            pdd_heads = MiniMaxH3PDDHeads(pdd_config.num_inference_steps, dit_config.hidden_size, dit_config.patch_dim)
            vae = MiniMaxH3ImageVAE(vae_config, attention_backend=attention_backend, stats_device=vae_device)

        dit_device = "cpu" if offload_dit or staged_vram else device
        if dit_path is not None:
            if hasattr(dit.final_layer, "video_out"):
                del dit.final_layer.video_out
            load_safetensors(dit, dit_path, device=dit_device, dtype_override=dtype_override,
                             ignore_keys=_EMBEDDED_PDD_KEYS,
                             ignore_suffixes=(*_UNUSED_ATTENTION_SUFFIX, ".comfy_quant"))
        else:
            load_safetensors(dit, index, device=dit_device, dtype_override=dtype_override,
                             ignore_suffixes=_UNUSED_ATTENTION_SUFFIX)
        if not pruned_adaln and (prune_adaln is True or (
                prune_adaln == "auto" and offload_dit and (dit_path is not None or device.type == "cuda"))):
            dit.prune_adaln_tables(pdd_config.sigmas[:-1], device=device)
        if embedded_heads:
            _load_embedded_pdd_heads(pdd_heads, dit_path, device, dtype_override)
        else:
            load_safetensors(pdd_heads, pdd_heads_path, device=device, dtype_override=dtype_override,
                             ignore_suffixes=_UNUSED_ATTENTION_SUFFIX)
        load_safetensors(vae, root / "vae/diffusion_pytorch_model.safetensors", device=vae_device,
                         dtype_override=dtype_override, ignore_keys=("decoder.mask_token",),
                         ignore_suffixes=_UNUSED_ATTENTION_SUFFIX)
        if offload_dit:
            dit.enable_async_offload(device, resident_blocks, pin_staging)
        dit = dit.eval().compile(compile_mode)
        return cls(dit, pdd_heads, vae, MiniMaxH3Scheduler(pdd_config), device=device, staged_vram=staged_vram)

    @contextmanager
    def keep_dit_resident(self):
        """Keep the staged DiT prepared across latent-only calls."""
        if self._dit_resident_depth == 0 and self.staged_vram:
            self.dit.prepare_gpu(self.device)
        self._dit_resident_depth += 1
        try:
            yield
        finally:
            self._dit_resident_depth -= 1
            if self._dit_resident_depth == 0 and self.staged_vram:
                self.dit.release_gpu()

    @contextmanager
    def _vae_on_device(self):
        if self.staged_vram and self._dit_resident_depth > 0:
            raise RuntimeError(_DIT_RESIDENT_VAE_ERROR)
        stage_vae = self._vae_device_depth == 0 and self.staged_vram
        try:
            if stage_vae:
                self.vae.to(self.device)
            self._vae_device_depth += 1
            try:
                yield
            finally:
                self._vae_device_depth -= 1
        finally:
            if stage_vae:
                self.vae.to("cpu")
                gc.collect()
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()

    def _generation_sizes(self, height, width, min_generation_area):
        if height <= 0 or width <= 0:
            raise ValueError("Output height and width must be positive")
        multiple = _CANVAS_MULTIPLE
        output_h, output_w = ((side + multiple - 1) // multiple * multiple for side in (height, width))
        ratio = self.vae.spatial_compression_ratio
        config = self.dit.config
        _, ph, pw = config.patch_size
        canvas_h, canvas_w = get_canvas_size(
            output_h, output_w, MIN_GENERATION_AREA if min_generation_area is None else min_generation_area,
            math.lcm(multiple, ratio * ph, ratio * pw),
        )
        return (output_w, output_h), (1, config.latents_dim, 1, canvas_h // ratio, canvas_w // ratio)

    @staticmethod
    def _initial_latents(shape, seed, rand_device, latents):
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        if latents is None:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=torch.float32)
        return generator, latents

    def _latent_stats(self, device):
        return tuple(stats.to(device=device, dtype=torch.float32)
                     for stats in (self.vae.latents_mean, self.vae.latents_std))

    @staticmethod
    def _pixel_stats(device):
        return tuple(torch.tensor(stat, dtype=torch.float32, device=device).view(1, 3, 1, 1)
                     for stat in (IMAGENET_MEAN, IMAGENET_STD))

    def _encode_references(self, images, generator, rand_device, tiling, reference_noise):
        if not isinstance(images, (list, tuple)):
            images = [images]
        mean, std = self._latent_stats(self.vae_device)
        pixel_mean, pixel_std = self._pixel_stats(self.vae_device)
        ratio = self.vae.spatial_compression_ratio
        _, ph, pw = self.dit.config.patch_size
        t = torch.tensor(REFERENCE_TIMESTEP, dtype=torch.float32, device=self.vae_device)
        output = []
        for i, image in enumerate(images):
            if isinstance(image, (str, Path)):
                with Image.open(image) as opened:
                    image = _resize_reference(opened)
            else:
                image = _resize_reference(image)
            pixels = torch.from_numpy(np.array(image)).to(self.vae_device).permute(2, 0, 1)[None]
            h, w = pixels.shape[-2:]
            pixels = pixels[..., :h // ratio * ratio, :w // ratio * ratio]
            pixels = (pixels.float().div(255.0) - pixel_mean) / pixel_std
            posterior = self.vae.encode(pixels, tiling=tiling)
            latent = posterior.sample(generator).half().float()
            h, w = latent.shape[-2:]
            latent = (latent[..., :h // ph * ph, :w // pw * pw] - mean) / std
            if reference_noise is None:
                noise = torch.randn(latent.shape, generator=generator, device=rand_device, dtype=torch.float32)
            else:
                noise = reference_noise[i].reshape(latent.shape)
            noise = noise.to(self.vae_device, dtype=torch.float32)
            output.append((t * latent + (1.0 - t) * noise)[:, :, None].to(self.device))
        return output

    @torch.inference_mode()
    def decode_latents(self, latents, *, tiling=None, output_size=None):
        """Denormalize, decode, truncate to uint8, then LANCZOS resize to (width, height)."""
        if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[2] != 1:
            raise ValueError("Expected one image's latents [1, C, 1, H, W]")
        with self._vae_on_device():
            mean, std = self._latent_stats(self.vae_device)
            pixels = self.vae.decode(latents[:, :, 0].to(self.vae_device, torch.float32) * std + mean, tiling=tiling)
            pixel_mean, pixel_std = self._pixel_stats(self.vae_device)
            pixels = (pixels.float() * pixel_std + pixel_mean).clamp(0, 1)
            pixels = (pixels[0].permute(1, 2, 0) * 255).to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(pixels.numpy())
        if output_size is not None and image.size != output_size:
            image = image.resize(output_size, Image.Resampling.LANCZOS)
        return image

    def _decode_output(self, latents, tiling, output_size, output_type):
        image = self.decode_latents(latents, tiling=tiling, output_size=output_size)
        if output_type == "tensor":
            return torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].contiguous()
        return image

    @torch.inference_mode()
    def __call__(
        self, prompt_embeds, *, text_token_tags=None, edit_images=None, ref_latents=None,
        height=2048, width=2048, min_generation_area=MIN_GENERATION_AREA,
        seed=0, rand_device="cpu", latents=None, reference_noise=None, tiling=None,
        output_type="pil", return_stats=False, step_callback=None,
    ):
        """Generate from [tokens, text_dim] embeddings or a .pt file.

        The file can contain the embeddings tensor or a dict with prompt_embeds
        (or text_embeds) and optional text_token_tags. For edits, pass PIL
        images/paths or prepacked conditioning latents (normalized, already
        noised to t=0.999, shaped [1, C, 1, h, w] or [1, C, h, w] per reference).
        The output size rounds up to a multiple of 32, as in DiffSynth.
        Returns a PIL image, a matching CPU uint8 BCHW tensor (output_type="tensor"),
        or the final normalized latents (output_type="latent"). With return_stats,
        returns (output, stats). step_callback sees (zero-based step, latents,
        data-ward velocity) after each update.
        """
        if (edit_images is not None or output_type != "latent") and self.staged_vram and self._dit_resident_depth > 0:
            raise RuntimeError(_DIT_RESIDENT_VAE_ERROR)
        output_size, shape = self._generation_sizes(height, width, min_generation_area)
        if edit_images is not None and ref_latents is not None:
            raise ValueError("Supply either edit_images or ref_latents")
        if output_type not in ("pil", "tensor", "latent"):
            raise ValueError(f"Unknown output_type {output_type!r}")
        if isinstance(prompt_embeds, (str, Path)):
            prompt_embeds = torch.load(prompt_embeds, map_location="cpu", weights_only=True)
        if isinstance(prompt_embeds, Mapping):
            if text_token_tags is None:
                text_token_tags = prompt_embeds.get("text_token_tags")
            prompt_embeds = prompt_embeds.get("prompt_embeds", prompt_embeds.get("text_embeds"))
        if isinstance(prompt_embeds, torch.Tensor) and prompt_embeds.ndim == 3 and prompt_embeds.shape[0] == 1:
            prompt_embeds = prompt_embeds[0]
        config = self.dit.config
        if (not isinstance(prompt_embeds, torch.Tensor) or prompt_embeds.ndim != 2
                or prompt_embeds.shape[0] == 0 or prompt_embeds.shape[-1] != config.text_dim):
            raise ValueError(f"Expected prompt_embeds [tokens, {config.text_dim}]")
        prompt_embeds = prompt_embeds.to(self.device)
        if text_token_tags is None:
            text_token_tags = torch.ones(prompt_embeds.shape[0], dtype=torch.long, device="cpu")
        else:
            text_token_tags = torch.as_tensor(text_token_tags, device="cpu").reshape(-1)
            if text_token_tags.numel() != prompt_embeds.shape[0]:
                raise ValueError("Supply one text_token_tag per embedding row")

        h, w = shape[-2:]
        track_cuda = return_stats and self.device.type == "cuda"
        if track_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        generator, latents = self._initial_latents(shape, seed, rand_device, latents)
        latents = latents.to(self.device, torch.float32).reshape(shape)
        if edit_images is not None:
            with self._vae_on_device():
                ref_latents = self._encode_references(edit_images, generator, rand_device, tiling, reference_noise)
        elif isinstance(ref_latents, torch.Tensor):
            ref_latents = [ref_latents]

        grids, ref_rows = [], []
        for reference in ref_latents or ():
            if reference.ndim == 4:
                reference = reference[:, :, None]
            if reference.ndim != 5 or tuple(reference.shape[:3]) != shape[:3]:
                raise ValueError("Each reference must be [1, C, 1, h, w] or [1, C, h, w]")
            grids.append(tuple(reference.shape[-2:]))
            ref_rows.append(patchify_video(reference.to(self.device, torch.float32), config.patch_size))
        refs = torch.cat(ref_rows) if ref_rows else None
        positions, tags, image_indices, text_indices, ref_count = build_packed_sequence(
            text_token_tags, grids, h, w, patch_size=config.patch_size,
        )
        positions = positions[None].to(self.device)
        tags, image_indices, text_indices = (t.to(self.device) for t in (tags, image_indices, text_indices))
        timestep_indices = torch.zeros_like(tags)
        times = self.scheduler.model_timesteps.unsqueeze(1)
        if ref_count:
            timestep_indices[image_indices[:ref_count]] = 1
            times = torch.cat((times, torch.full_like(times, REFERENCE_TIMESTEP)), dim=1)
        times = times.to(self.device)

        step_ms, events = [], []
        if track_cuda:
            stream = torch.cuda.current_stream(self.device)
        manage_dit = self.staged_vram and self._dit_resident_depth == 0
        try:
            if manage_dit:
                self.dit.prepare_gpu(self.device)
            rope = self.dit.rope(positions)
            for step, timestep in enumerate(times):
                if track_cuda:
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record(stream)
                elif return_stats:
                    wall_start = perf_counter()
                rows = patchify_video(latents, config.patch_size)
                if refs is not None:
                    rows = torch.cat((refs, rows))
                velocity = self.dit(
                    rows, prompt_embeds, positions, timestep, timestep_indices, tags,
                    image_indices, text_indices, self.pdd_heads, step, precomputed_rope=rope,
                )
                velocity = unpatchify_video(velocity[ref_count:], 1, h, w, config.latents_dim, config.patch_size)
                latents = self.scheduler.step(velocity, step, latents)
                if track_cuda:
                    end.record(stream)
                    events.append((start, end))
                elif return_stats:
                    step_ms.append((perf_counter() - wall_start) * 1000)
                if step_callback is not None:
                    step_callback(step, latents, velocity)
        finally:
            if manage_dit:
                self.dit.release_gpu()

        if output_type == "latent":
            result = latents
        else:
            result = self._decode_output(latents, tiling, output_size, output_type)
        if not return_stats:
            return result
        peak = None
        if track_cuda:
            events[-1][1].synchronize()
            step_ms = [start.elapsed_time(end) for start, end in events]
            peak = torch.cuda.max_memory_allocated(self.device)
        return result, InferenceStats(tuple(step_ms), peak)

    @torch.inference_mode()
    def generate_batch(
        self, items: Sequence, *, height=2048, width=2048, min_generation_area=MIN_GENERATION_AREA,
        seed=None, rand_device="cpu", tiling=None, output_type="pil", return_stats=False,
        step_callback=None,
    ):
        """Encode references, denoise each item, then decode with one VAE residency.

        Each item is an embedding tensor, a .pt path, or a mapping with embeddings
        and per-item generation settings. Missing seeds use zero or batch seed plus
        item index. step_callback(item_index, step, latents, velocity) uses zero-based
        indices. With return_stats, each result includes its diffusion stats.
        """
        if output_type not in ("pil", "tensor", "latent"):
            raise ValueError(f"Unknown output_type {output_type!r}")
        specs = []
        for idx, item in enumerate(items):
            spec = dict(item) if isinstance(item, Mapping) else {"prompt_embeds": item}
            embeds = spec.pop("prompt_embeds", spec.pop("text_embeds", None))
            options = {
                "height": height, "width": width, "min_generation_area": min_generation_area,
                "seed": 0 if seed is None else seed + idx, "rand_device": rand_device,
                "text_token_tags": None, "edit_images": None, "ref_latents": None,
                "latents": None, "reference_noise": None,
            }
            unknown = spec.keys() - options.keys()
            if unknown:
                raise ValueError(f"Unknown batch item options: {', '.join(sorted(unknown))}")
            options.update(spec)
            if options["edit_images"] is not None and options["ref_latents"] is not None:
                raise ValueError("Supply either edit_images or ref_latents")
            size, shape = self._generation_sizes(
                options["height"], options["width"], options["min_generation_area"],
            )
            specs.append((embeds, options, size, shape))
        if not specs:
            return []

        has_edits = any(options["edit_images"] is not None for _, options, _, _ in specs)
        if (has_edits or output_type != "latent") and self.staged_vram and self._dit_resident_depth > 0:
            raise RuntimeError(_DIT_RESIDENT_VAE_ERROR)
        if has_edits:
            with self._vae_on_device():
                # Draw initial latents in item order to preserve the RNG sequence for seed=None.
                for _, options, _, shape in specs:
                    generator, options["latents"] = self._initial_latents(
                        shape, options["seed"], options["rand_device"], options["latents"],
                    )
                    if options["edit_images"] is not None:
                        options["ref_latents"] = self._encode_references(
                            options["edit_images"], generator, options["rand_device"], tiling,
                            options["reference_noise"],
                        )
                        options["edit_images"] = None

        results = []
        with self.keep_dit_resident():
            for idx, (embeds, options, _, _) in enumerate(specs):
                callback = None if step_callback is None else partial(step_callback, idx)
                results.append(self(
                    embeds, **options, tiling=tiling, output_type="latent",
                    return_stats=return_stats, step_callback=callback,
                ))
        if output_type == "latent":
            return results

        with self._vae_on_device():
            for idx, (_, _, size, _) in enumerate(specs):
                latents, stats = results[idx] if return_stats else (results[idx], None)
                image = self._decode_output(latents, tiling, size, output_type)
                results[idx] = (image, stats) if return_stats else image
        return results
