"""Dense Tensor Core specifications for Colab and Kaggle hardware."""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Literal


_METRIC_FIELDS = {
    "fp16": "fp16_tflops",
    "bf16": "bf16_tflops",
    "int8": "int8_tops",
    "fp8": "fp8_tflops",
    "nvfp4": "nvfp4_tflops",
}


def _metric_field(precision: str) -> str:
    try:
        return _METRIC_FIELDS[precision.lower()]
    except KeyError:
        raise ValueError(f"Unknown precision: {precision!r}") from None


@dataclass(frozen=True)
class HardwareSpec:
    key: str
    name: str
    provider: Literal["kaggle", "colab"]
    kind: Literal["gpu", "cpu", "tpu"]
    arch: str
    sm: str | None
    gpu_count: int
    vram_gb: float
    bandwidth_gb_s: float
    fp16_tflops: float | None
    bf16_tflops: float | None
    int8_tops: float | None
    fp8_tflops: float | None
    nvfp4_tflops: float | None
    tdp_w: float | None
    idle_watts: float | None
    cu_per_hour: float
    cpu_cores: int
    ram_gb: float
    notes: str

    @property
    def supports_fp16(self) -> bool:
        return self.fp16_tflops is not None

    @property
    def supports_bf16(self) -> bool:
        return self.bf16_tflops is not None

    @property
    def supports_int8(self) -> bool:
        return self.int8_tops is not None

    @property
    def supports_fp8(self) -> bool:
        return self.fp8_tflops is not None

    @property
    def supports_nvfp4(self) -> bool:
        return self.nvfp4_tflops is not None

    def metric_for(self, precision: str) -> float | None:
        """Return dense TFLOPS, or TOPS for INT8."""
        return getattr(self, _metric_field(precision))

    def tflops_per_cu(self, precision: str) -> float | None:
        """Return peak throughput per CU/h for a supported, paid precision."""
        value = self.metric_for(precision)
        if value is not None and self.cu_per_hour > 0:
            return round(value / self.cu_per_hour, 2)
        return None


SPECS = {
    "T4X2": HardwareSpec(
        key="T4X2", name="Kaggle Free Dual-T4", provider="kaggle", kind="gpu",
        arch="Turing", sm="sm_75", gpu_count=2, vram_gb=32.0, bandwidth_gb_s=640.0,
        fp16_tflops=130.0, bf16_tflops=None, int8_tops=260.0, fp8_tflops=None, nvfp4_tflops=None,
        tdp_w=140.0, idle_watts=28.0, cu_per_hour=0.0, cpu_cores=4, ram_gb=29.0,
        notes="2x16 GB GDDR6 (~30 GiB usable); 320 GB/s/card; 30h/week free",
    ),
    "T4": HardwareSpec(
        key="T4", name="Colab T4", provider="colab", kind="gpu",
        arch="Turing", sm="sm_75", gpu_count=1, vram_gb=16.0, bandwidth_gb_s=320.0,
        fp16_tflops=65.0, bf16_tflops=None, int8_tops=130.0, fp8_tflops=None, nvfp4_tflops=None,
        tdp_w=70.0, idle_watts=28.0, cu_per_hour=1.19, cpu_cores=2, ram_gb=12.7,
        notes="Fallback when Kaggle quota unavailable; INT8/FP16 inference",
    ),
    "L4": HardwareSpec(
        key="L4", name="Colab L4", provider="colab", kind="gpu",
        arch="Ada Lovelace", sm="sm_89", gpu_count=1, vram_gb=24.0, bandwidth_gb_s=300.0,
        fp16_tflops=121.0, bf16_tflops=121.0, int8_tops=242.5, fp8_tflops=242.5, nvfp4_tflops=None,
        tdp_w=72.0, idle_watts=28.0, cu_per_hour=1.71, cpu_cores=12, ram_gb=53.0,
        notes="Best CU efficiency for FP8/INT8 image generation (Sparse 2:4 2x)",
    ),
    "A100": HardwareSpec(
        key="A100", name="Colab A100 80GB", provider="colab", kind="gpu",
        arch="Ampere", sm="sm_80", gpu_count=1, vram_gb=80.0, bandwidth_gb_s=2039.0,
        fp16_tflops=312.0, bf16_tflops=312.0, int8_tops=624.0, fp8_tflops=None, nvfp4_tflops=None,
        tdp_w=400.0, idle_watts=75.0, cu_per_hour=8.71, cpu_cores=12, ram_gb=83.5,
        notes="SXM4 2039 GB/s (PCIe 1935 GB/s); high-bandwidth BF16/INT8",
    ),
    "G4": HardwareSpec(
        key="G4", name="Colab G4 (RTX PRO 6000 Blackwell Server Edition)", provider="colab", kind="gpu",
        arch="Blackwell", sm="sm_120", gpu_count=1, vram_gb=96.0, bandwidth_gb_s=1597.0,
        fp16_tflops=504.0, bf16_tflops=504.0, int8_tops=1008.0, fp8_tflops=1008.0, nvfp4_tflops=2015.0,
        tdp_w=600.0, idle_watts=75.0, cu_per_hour=10.05, cpu_cores=12, ram_gb=96.0,
        notes="96 GB GDDR7 ECC; native NVFP4/FP8/INT8 (Sparse 2:4 2x); 400-600W TDP",
    ),
    "H100": HardwareSpec(
        key="H100", name="Colab H100 80GB", provider="colab", kind="gpu",
        arch="Hopper", sm="sm_90", gpu_count=1, vram_gb=80.0, bandwidth_gb_s=3350.0,
        fp16_tflops=989.0, bf16_tflops=989.0, int8_tops=1979.0, fp8_tflops=1979.0, nvfp4_tflops=None,
        tdp_w=700.0, idle_watts=75.0, cu_per_hour=18.05, cpu_cores=26, ram_gb=235.0,
        notes="SXM5 3350 GB/s (PCIe 2000 GB/s, 756 BF16 / 1513 FP8/INT8)",
    ),
    "CPU-STD": HardwareSpec(
        key="CPU-STD", name="Colab Standard CPU", provider="colab", kind="cpu",
        arch="EPYC", sm=None, gpu_count=0, vram_gb=0.0, bandwidth_gb_s=0.0,
        fp16_tflops=None, bf16_tflops=None, int8_tops=None, fp8_tflops=None, nvfp4_tflops=None,
        tdp_w=None, idle_watts=None, cu_per_hour=0.08, cpu_cores=2, ram_gb=12.0,
        notes="Standard CPU (--cpu --std-mem); lightweight prep/pull",
    ),
    "CPU-HIGH": HardwareSpec(
        key="CPU-HIGH", name="Colab High-RAM CPU", provider="colab", kind="cpu",
        arch="EPYC", sm=None, gpu_count=0, vram_gb=0.0, bandwidth_gb_s=0.0,
        fp16_tflops=None, bf16_tflops=None, int8_tops=None, fp8_tflops=None, nvfp4_tflops=None,
        tdp_w=None, idle_watts=None, cu_per_hour=0.26, cpu_cores=8, ram_gb=50.0,
        notes="Default --cpu high-RAM worker; 64x staging/ETL",
    ),
    "TPU-V5E1": HardwareSpec(
        key="TPU-V5E1", name="Colab TPU v5e1", provider="colab", kind="tpu",
        arch="TPU v5e", sm=None, gpu_count=0, vram_gb=0.0, bandwidth_gb_s=0.0,
        fp16_tflops=None, bf16_tflops=None, int8_tops=None, fp8_tflops=None, nvfp4_tflops=None,
        tdp_w=None, idle_watts=None, cu_per_hour=3.09, cpu_cores=24, ram_gb=47.0,
        notes="24-core EPYC host (--tpu v5e1)",
    ),
    "TPU-V6E1": HardwareSpec(
        key="TPU-V6E1", name="Colab TPU v6e1", provider="colab", kind="tpu",
        arch="TPU v6e", sm=None, gpu_count=0, vram_gb=0.0, bandwidth_gb_s=0.0,
        fp16_tflops=None, bf16_tflops=None, int8_tops=None, fp8_tflops=None, nvfp4_tflops=None,
        tdp_w=None, idle_watts=None, cu_per_hour=5.09, cpu_cores=44, ram_gb=172.0,
        notes="44-core Zen4 EPYC host (--tpu v6e1)",
    ),
}

_ALIASES = {
    "2XT4": "T4X2",
    "DUALT4": "T4X2",
    "KAGGLET4": "T4X2",
    "CPU": "CPU-STD",
    "CPU_STD": "CPU-STD",
    "CPU-HIGHRAM": "CPU-HIGH",
    "HIGH_RAM": "CPU-HIGH",
    "V5E1": "TPU-V5E1",
    "V6E1": "TPU-V6E1",
    "RTX PRO 6000": "G4",
}


def get_spec(name: str) -> HardwareSpec:
    """Look up a hardware key or alias."""
    key = name.strip().upper()
    return SPECS[_ALIASES.get(key, key)]


def list_specs(
    *, precision: str | None = None, min_vram_gb: float = 0.0, kind: str | None = None,
) -> list[HardwareSpec]:
    """Filter by native precision, total VRAM, and runtime kind."""
    field = _metric_field(precision) if precision is not None else None
    return [
        spec for spec in SPECS.values()
        if spec.vram_gb >= min_vram_gb
        and (kind is None or spec.kind == kind)
        and (field is None or getattr(spec, field) is not None)
    ]


def as_dict(spec: HardwareSpec) -> dict:
    """Serialize the hardware fields and native precision flags."""
    return {
        **asdict(spec),
        **{f"supports_{precision}": getattr(spec, f"supports_{precision}") for precision in _METRIC_FIELDS},
    }


def current() -> HardwareSpec | None:
    """Identify the active NVIDIA GPU using CUDA, then fall back to nvidia-smi."""
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and torch.cuda.is_available():
        count = torch.cuda.device_count()
        if not count:
            return None
        name = torch.cuda.get_device_name(0)
    else:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        names = [line.strip() for line in output.splitlines() if line.strip()]
        count = len(names)
        if not count:
            return None
        name = names[0]

    match = re.search(r"\b(T4|RTX PRO 6000|H100|A100|L4)\b", name.upper())
    if match is None:
        return None
    model = match[0]
    return get_spec("T4X2" if model == "T4" and count >= 2 else model)
