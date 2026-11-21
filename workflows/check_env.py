#!/usr/bin/env python3
"""Colab Ultra — Runtime Environment, GPU Capability & Local NVMe Check."""

import os
import shutil
import sys

sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")

print("=" * 60)
print("🚀 Colab Ultra Runtime Environment Check")
print(f"Python : {sys.version.split()[0]}")

try:
    import torch
    print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            from vram_policy import get_gpu_profile
            prof = get_gpu_profile()
            print(
                f"GPU    : {prof['name']} | VRAM: {prof['total_vram_gib']} GiB | "
                f"SM {prof['sm_major']}.{prof['sm_minor']} | "
                f"BF16={prof['supports_bf16']} INT8={prof['supports_int8_linear']} FP8={prof['supports_fp8']}"
            )
        except Exception:
            vram = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
            print(f"GPU    : {torch.cuda.get_device_name(0)} | VRAM: {vram:.2f} GiB")
except ImportError:
    print("PyTorch: not installed")

for path in ("/content", "/tmp", "/content/drive/MyDrive"):
    if os.path.exists(path):
        usage = shutil.disk_usage(path)
        print(f"Disk   : {path:<24} free={usage.free / (1024**3):.1f} GiB / total={usage.total / (1024**3):.1f} GiB")

print(f"Tools  : uv={shutil.which('uv')} | aria2c={shutil.which('aria2c')}")
print("=" * 60)
