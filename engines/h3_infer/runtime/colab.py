"""CUDA capability and scratch/cache locations for Colab inference."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ..ops.attention import recommended_attention_backend


@dataclass(frozen=True)
class GPUCapabilities:
    device_name: str
    compute_capability: str | None
    vram_gb: float
    supports_bf16: bool
    supports_fp8: bool
    attention_backend: str


def detect_gpu(device: int | str | torch.device | None = None) -> GPUCapabilities:
    """Report native BF16 and FP8 hardware support and VRAM in decimal GB."""
    if isinstance(device, str):
        device = torch.device(device)
    if (isinstance(device, torch.device) and device.type != "cuda") or not torch.cuda.is_available():
        return GPUCapabilities("CPU", None, 0.0, False, False, "sdpa")
    props = torch.cuda.get_device_properties(device)
    capability = (props.major, props.minor)
    return GPUCapabilities(
        props.name, f"sm_{props.major}{props.minor}", props.total_memory / 1e9,
        capability >= (8, 0), capability >= (8, 9), recommended_attention_backend(props.major),
    )


@dataclass(frozen=True)
class ColabPaths:
    models: Path
    triton_cache: Path
    inductor_cache: Path

    def cache_environment(self) -> dict[str, str]:
        """Environment variables for directing Triton and Inductor caches."""
        return {
            "TRITON_CACHE_DIR": str(self.triton_cache),
            "TORCHINDUCTOR_CACHE_DIR": str(self.inductor_cache),
        }


def staging_paths(content: str | Path = "/content", drive: str | Path | None = None) -> ColabPaths:
    """Return local scratch for weights and use Google Drive for caches when mounted."""
    content = Path(content)
    models = (content if content.is_dir() else Path("/tmp")) / "models"
    drive_root = Path(drive) if drive is not None else content / "drive" / "MyDrive"
    cache = (drive_root / ".cache" if drive is not None or drive_root.is_dir() else models / ".cache") / "h3_infer"
    return ColabPaths(models, cache / "triton", cache / "inductor")
