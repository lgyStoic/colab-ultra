#!/usr/bin/env python3
"""Colab Ultra — Modern Diffusion Inference Template (NVMe Staging + Auto VRAM Policy).

Demonstrates the canonical Colab Ultra inference pattern:
1. Never read multi-GB weights over slow Drive FUSE; stage onto local NVMe `/content/models`
   via `fast_stage_nvme.stage_dual_channel()` (HF GCP CDN + Drive v3 API) overlapped with `uv` setup.
2. Never hard-code offload flags per GPU; call `vram_policy.build_vram_plan()` to auto-select
   `full_resident` (G4/H100), `selective_offload` (L4), or `tight_offload` (T4).
3. Run via `colab-ultra exec -s <session> -p <project> -f <script.py> && colab-ultra down -s <session> --json`
   (heartbeats, live streaming, token refresh, and exact exit codes are handled automatically).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")
from fast_stage_nvme import stage_dual_channel
from vram_policy import build_vram_plan, format_vram_plan


def setup_env(pkgs: list[str]) -> None:
    uv_bin = shutil.which("uv")
    cmd = [uv_bin, "pip", "install", "--system", "-q", *pkgs] if uv_bin else [sys.executable, "-m", "pip", "install", "-q", *pkgs]
    subprocess.check_call(cmd)


def main() -> None:
    t0 = time.time()
    nvme_root = Path("/content/models")
    out_dir = Path("/content/drive/MyDrive/ColabUltra/outputs/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Example: Tri-Parallel Overlap (background dual-channel NVMe staging || main-thread uv install)
    stage_err: list[Exception] = []

    def _bg_stage() -> None:
        try:
            stage_dual_channel(
                dest_root=nvme_root,
                drive_files=[
                    # ("subdir", "ColabUltra/models/<model>/<variant>/lora.safetensors"),
                ],
                hf_urls=[
                    # ("subdir", "model.safetensors", "https://huggingface.co/<org>/<repo>/resolve/main/<file>"),
                ],
            )
        except Exception as exc:
            stage_err.append(exc)

    t_stage = threading.Thread(target=_bg_stage, daemon=True)
    t_stage.start()

    setup_env(["safetensors", "pillow"])

    t_stage.join()
    if stage_err:
        raise stage_err[0]

    # Auto VRAM placement based on live GPU memory (replace sizes with real staged file sizes)
    plan = build_vram_plan({"dit": 7.7, "text_encoder": 15.0, "vae": 0.8})
    print(format_vram_plan(plan), flush=True)
    print(f"[Template] Ready in {time.time() - t0:.2f}s | outputs -> {out_dir}", flush=True)
    os.sync()


if __name__ == "__main__":
    main()
