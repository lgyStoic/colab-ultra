#!/usr/bin/env python3
"""Reusable GPU VRAM + System RAM & Resolution-Aware Offload Policy for Colab Ultra.

Automatically selects the fastest safe placement strategy for DiffSynth-Studio / PyTorch
pipelines from live hardware telemetry:
1. Active GPU total VRAM & SM compute capability (`torch.cuda.mem_get_info`).
2. Available host system RAM (`/proc/meminfo` MemAvailable) — automatically degrades
   oversized offloaded components to `disk` offload when RAM cannot hold them.
3. Target output resolution (`1K / 2K / 3K / 4K+`) — automatically scales activation
   `headroom_gb` and decides `vae_tiling`.

Usage on Colab:
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")
    from cu_remote.vram_policy import build_vram_plan, format_vram_plan

    plan = build_vram_plan(
        {"dit": 7.7, "text_encoder": 15.0, "pdd_heads": 0.3, "vae": 0.8},
        resolution=(2048, 2048),
    )
    print(format_vram_plan(plan))
"""

from __future__ import annotations

import os
from typing import Any


def get_available_ram_gib() -> float:
    """Return available host RAM in GiB (MemAvailable on Linux, total physical RAM fallback)."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 2)
    except OSError:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return round((pages * page_size) / (1024 ** 3), 2)
    except (ValueError, OSError, AttributeError):
        pass
    return 16.0


def get_gpu_profile() -> dict[str, Any]:
    """Inspect active CUDA GPU VRAM, compute capability, and host available RAM."""
    ram_gib = get_available_ram_gib()
    try:
        import torch
    except ImportError:
        return {
            "name": "CPU",
            "total_vram_gib": 0.0,
            "avail_ram_gib": ram_gib,
            "sm_major": 0,
            "sm_minor": 0,
            "supports_bf16": False,
            "supports_int8_linear": False,
            "supports_fp8": False,
        }

    if not torch.cuda.is_available():
        return {
            "name": "CPU",
            "total_vram_gib": 0.0,
            "avail_ram_gib": ram_gib,
            "sm_major": 0,
            "sm_minor": 0,
            "supports_bf16": False,
            "supports_int8_linear": False,
            "supports_fp8": False,
        }

    total_vram_gib = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_vram_gib": round(total_vram_gib, 2),
        "avail_ram_gib": ram_gib,
        "sm_major": major,
        "sm_minor": minor,
        "supports_bf16": major >= 8,
        "supports_int8_linear": major >= 8,
        "supports_fp8": (major > 8) or (major == 8 and minor >= 9),
    }


def estimate_headroom_gb(resolution: tuple[int, int] | None) -> tuple[float, float]:
    """Return (activation_headroom_gb, vae_tiling_vram_threshold_gib) for target (width, height)."""
    if not resolution:
        return 3.5, 18.0
    w, h = int(resolution[0]), int(resolution[1])
    pixels = max(1, w * h)
    if pixels <= 1024 * 1024:
        return 2.5, 6.0
    if pixels <= 2048 * 2048:
        return 3.5, 8.0
    if pixels <= 3072 * 3072:
        return 5.0, 14.0
    return 6.5, 22.0


def _make_cfg(offload_device: str, onload_device: str, dtype) -> dict[str, Any]:
    return {
        "offload_dtype": dtype,
        "offload_device": offload_device,
        "onload_dtype": dtype,
        "onload_device": onload_device,
        "preparing_dtype": dtype,
        "preparing_device": "cuda",
        "computation_dtype": dtype,
        "computation_device": "cuda",
    }


def build_vram_plan(
    component_sizes_gb: dict[str, float],
    headroom_gb: float | None = None,
    resolution: tuple[int, int] | None = (2048, 2048),
    dtype=None,
    vram_reserve_gb: float = 1.5,
    ram_reserve_gb: float = 6.0,
    total_vram_gib_override: float | None = None,
    avail_ram_gib_override: float | None = None,
) -> dict[str, Any]:
    """Compute per-component DiffSynth `vram_config` and execution flags from GPU VRAM, host RAM & resolution.

    Args:
        component_sizes_gb: Mapping of component name -> weight footprint in GB.
        headroom_gb: Explicit GPU activation headroom in GB (auto-derived from `resolution` if None).
        resolution: Target `(width, height)` for automatic headroom & `vae_tiling` sizing.
        dtype: Torch dtype (defaults to `torch.bfloat16`).
        vram_reserve_gb: Safety margin subtracted from total VRAM for DiffSynth `vram_limit`.
        ram_reserve_gb: Host RAM safety margin before spilling largest offloaded components to `disk`.
        total_vram_gib_override: Optional VRAM override for dry-run / unit tests.
        avail_ram_gib_override: Optional available RAM override for dry-run / unit tests.
    """
    if dtype is None:
        try:
            import torch
            dtype = torch.bfloat16
        except ImportError:
            dtype = "bfloat16"

    auto_headroom, tiling_threshold_gib = estimate_headroom_gb(resolution)
    eff_headroom = auto_headroom if headroom_gb is None else float(headroom_gb)

    if total_vram_gib_override is not None:
        gpu = {
            "name": f"Simulated-{total_vram_gib_override}GiB",
            "total_vram_gib": float(total_vram_gib_override),
            "avail_ram_gib": float(avail_ram_gib_override) if avail_ram_gib_override is not None else get_available_ram_gib(),
            "sm_major": 8,
            "sm_minor": 9,
            "supports_bf16": True,
            "supports_int8_linear": True,
            "supports_fp8": True,
        }
    else:
        gpu = get_gpu_profile()
        if avail_ram_gib_override is not None:
            gpu["avail_ram_gib"] = float(avail_ram_gib_override)

    total_vram = float(gpu["total_vram_gib"])
    avail_ram = float(gpu["avail_ram_gib"])
    total_weights = sum(float(v) for v in component_sizes_gb.values())
    configs: dict[str, dict[str, Any]] = {}

    if total_vram >= total_weights + eff_headroom:
        mode = "full_resident"
        for name in component_sizes_gb:
            configs[name] = _make_cfg("cuda", "cuda", dtype)
        vram_limit = None
        staged_batch = False
        vae_tiling = False
    else:
        # Step 1: Decide GPU-resident-when-active (`cpu->cuda`) vs layer-wise offload (`cpu->cpu`)
        for name, size_gb in component_sizes_gb.items():
            if total_vram >= float(size_gb) + eff_headroom:
                configs[name] = _make_cfg("cpu", "cuda", dtype)
            else:
                configs[name] = _make_cfg("cpu", "cpu", dtype)

        # Step 2: Check host RAM budget — spill largest offloaded components to `disk` if RAM is insufficient
        ram_budget = max(0.0, avail_ram - ram_reserve_gb)
        cpu_total = sum(float(sz) for sz in component_sizes_gb.values())
        spilled_to_disk = False
        if cpu_total > ram_budget:
            for name, size_gb in sorted(component_sizes_gb.items(), key=lambda kv: kv[1], reverse=True):
                if cpu_total <= ram_budget:
                    break
                configs[name] = _make_cfg("disk", "disk", dtype)
                cpu_total -= float(size_gb)
                spilled_to_disk = True

        if spilled_to_disk:
            mode = "disk_offload"
        elif total_vram < tiling_threshold_gib:
            mode = "tight_offload"
        else:
            mode = "selective_offload"

        vram_limit = round(max(1.0, total_vram - vram_reserve_gb), 2)
        staged_batch = True
        # Default to tiling=False for maximum decode speed; pipelines with OOM fallback will
        # automatically enable tiling=True only if full-frame decode actually hits CUDA OOM.
        vae_tiling = False

    return {
        "gpu": gpu,
        "mode": mode,
        "configs": configs,
        "vram_limit": vram_limit,
        "vae_tiling": vae_tiling,
        "staged_batch": staged_batch,
        "headroom_gb": eff_headroom,
        "resolution": resolution,
        "total_weights_gb": round(total_weights, 2),
    }


def format_vram_plan(plan: dict[str, Any]) -> str:
    """Return a concise single-line summary suitable for logs."""
    gpu = plan["gpu"]
    placements = ", ".join(
        f"{k}={v['offload_device']}->{v['onload_device']}"
        for k, v in plan["configs"].items()
    )
    res = plan.get("resolution")
    res_str = f"{res[0]}x{res[1]}" if res else "auto"
    return (
        f"[VRAM Policy] gpu={gpu['name']} ({gpu['total_vram_gib']} GiB VRAM, {gpu['avail_ram_gib']} GiB RAM, "
        f"sm_{gpu['sm_major']}{gpu['sm_minor']}) | res={res_str} (headroom={plan['headroom_gb']}G) | "
        f"mode={plan['mode']} | vram_limit={plan['vram_limit']} | vae_tiling={plan['vae_tiling']} | "
        f"staged_batch={plan['staged_batch']} | {placements}"
    )
