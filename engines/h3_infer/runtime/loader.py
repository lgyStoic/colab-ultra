"""Strict one-shard-at-a-time loading of local safetensors checkpoints."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from torch import nn

from ..config import MiniMaxH3PDDConfig


@dataclass(frozen=True)
class ComfyCheckpoint:
    quant_map: dict[str, dict]
    adaln_curve_shape: tuple[int, int] | None
    video_out_shape: tuple[int, ...] | None
    pdd_config: MiniMaxH3PDDConfig | None
    pruned_adaln: bool


def inspect_comfy_checkpoint(path: Path) -> ComfyCheckpoint:
    """Read Comfy-Org model shapes, schedule and quantization metadata."""
    quant_map = {}
    quant_suffix = ".comfy_quant"
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in sorted(keys):
            if not key.endswith(quant_suffix):
                continue
            data = handle.get_tensor(key)
            if data.dtype != torch.uint8 or data.ndim != 1:
                raise ValueError(f"{path}: {key} must be a one-dimensional uint8 JSON tensor")
            try:
                spec = json.loads(bytes(data.tolist()).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}: {key} contains invalid UTF-8 JSON") from error
            if not isinstance(spec, dict):
                raise ValueError(f"{path}: {key} must contain a JSON object")
            quant_map[key.removesuffix(quant_suffix)] = spec

        curve = None
        if "adaln_t_table" in keys:
            shape = handle.get_slice("adaln_t_table").get_shape()
            if len(shape) != 2:
                raise ValueError(f"{path}: adaln_t_table must have shape [grid, dim]")
            curve = (shape[0], shape[1])
        video_out = "final_layer.video_out.weight"
        video_out_shape = tuple(handle.get_slice(video_out).get_shape()) if video_out in keys else None
        transformer_config = json.loads((handle.metadata() or {}).get("config", "{}")).get("transformer", {})
        pdd_config = None
        if "pdd_sigmas" in transformer_config:
            sigmas = transformer_config["pdd_sigmas"]
            pdd_config = MiniMaxH3PDDConfig(transformer_config.get("pdd_steps", len(sigmas) - 1), sigmas)
        pruned_adaln = bool(keys & {"blocks.0.adaln_proj.table", "final_layer.adaln_proj.table"})
    return ComfyCheckpoint(quant_map, curve, video_out_shape, pdd_config, pruned_adaln)


def _fixed_dtype(key, target):
    if target.dtype == torch.int8:
        return "I8"
    if key == "adaln_t_table" or key == "weight_scale" or key.endswith(".weight_scale"):
        return "F32"
    return None


def _check_keys(expected, actual, location):
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"{location}: missing_keys={missing}, unexpected_keys={unexpected}")


def load_safetensors(
    module: nn.Module,
    checkpoint: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype_override: torch.dtype | None = None,
    ignore_keys: Iterable[str] = (),
    ignore_suffixes: Iterable[str] = (),
) -> nn.Module:
    """Load matching keys and shapes from a single file or safetensors index.

    Validate the index and every shard header before assigning tensors. Preserve
    each tensor's checkpoint dtype unless dtype_override is set for floating
    weights. Quantization scales and AdaLN curves retain float32 precision.
    Parameters on meta materialize on CPU by default. CPU tensors can
    retain safetensors' file-backed storage for layer streaming. Call before
    creating an optimizer, since assigning weights replaces the parameters.
    """
    path = Path(checkpoint)
    keys_to_ignore = set(ignore_keys)
    suffixes = tuple(ignore_suffixes)

    def kept(key):
        return key not in keys_to_ignore and not key.endswith(suffixes)

    targets = module.state_dict()
    expected = {key for key in targets if kept(key)}
    fixed_dtypes = {key: _fixed_dtype(key, targets[key]) for key in expected}
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        _check_keys(expected, {key for key in weight_map if kept(key)}, path)
        shards = defaultdict(set)
        for key, filename in weight_map.items():
            assigned = shards[path.parent / filename]
            if kept(key):
                assigned.add(key)
    else:
        shards = {path: expected}

    for shard, assigned in shards.items():
        with safe_open(shard, framework="pt", device="cpu") as handle:
            observed = {key for key in handle.keys() if kept(key)}
            _check_keys(assigned, observed, shard)
            for key in sorted(observed):
                view = handle.get_slice(key)
                shape = tuple(view.get_shape())
                if shape != targets[key].shape:
                    raise ValueError(f"{shard}: {key} has shape {shape}, expected {tuple(targets[key].shape)}")
                dtype = view.get_dtype()
                expected_dtype = fixed_dtypes[key]
                if expected_dtype is not None and dtype != expected_dtype:
                    raise ValueError(f"{shard}: {key} has dtype {dtype}, expected {expected_dtype}")
                if dtype == "I8" and expected_dtype is None:
                    raise ValueError(f"{shard}: {key} has int8 data without an int8 target")

    for shard, assigned in shards.items():
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensors = {}
            for key in sorted(assigned):
                tensor = handle.get_tensor(key)
                target = targets[key]
                placement = device if device is not None else ("cpu" if target.is_meta else target.device)
                dtype = tensor.dtype
                if dtype_override is not None and tensor.is_floating_point() and fixed_dtypes[key] is None:
                    dtype = dtype_override
                tensors[key] = tensor.to(device=placement, dtype=dtype)
            module.load_state_dict(tensors, strict=False, assign=True)
            del tensors
    return module
