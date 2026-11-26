"""Configuration for the official MiniMax-H3-Image BF16 release."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TypeVar

_Config = TypeVar("_Config", bound="_JsonConfig")


class _JsonConfig:
    @classmethod
    def from_json(cls: type[_Config], path: str | Path) -> _Config:
        """Read model fields from a release JSON, ignoring its metadata."""
        with Path(path).open(encoding="utf-8") as handle:
            values = json.load(handle)
        return cls(**{item.name: values[item.name] for item in fields(cls) if item.name in values})


@dataclass(frozen=True)
class MiniMaxH3DiTConfig(_JsonConfig):
    """Transformer dimensions, including the release's disabled audio settings."""

    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 96768
    final_adaln_out_features: int = 10752
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-05
    qk_norm_eps: float = 1e-05
    final_norm_eps: float = 1e-05
    audio_enabled: bool = False

    def __post_init__(self):
        object.__setattr__(self, "patch_size", tuple(self.patch_size))

    @property
    def patch_dim(self) -> int:
        pt, ph, pw = self.patch_size
        return self.latents_dim * pt * ph * pw


_IMAGE_LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328,
    -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
    -0.2553254961967468, -0.26907554268836975, -0.5376840829849243,
    -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
    0.25928452610969543, -0.30133944749832153, 0.211341992020607,
    -1.1206848621368408, 0.3581933379173279, -0.04225143790245056,
    0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
_IMAGE_LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887,
    1.7549455165863037, 1.5636216402053833, 2.194143533706665,
    0.9653137922286987, 1.0569885969161987, 0.841948926448822,
    0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
    0.6936293244361877, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180145264, 3.276226282119751,
    3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)

_VIT_DECODER_DEFAULTS = {
    "dim_head": 64,
    "ffn_activation_fn": 'silu',
    "ffn_use_gated": True,
    "heads": 16,
    "norm_affine": True,
    "norm_type": 'rms_norm',
    "num_layers": 24,
    "qk_norm_affine": False,
    "qk_norm_type": 'rms_norm',
    "rope_dim_ratio": 0.5,
    "rope_theta": 100.0,
}


@dataclass(frozen=True)
class MiniMaxH3VAEConfig(_JsonConfig):
    """VAE release JSON fields and encoder defaults from the reference model."""

    latent_channels: int = 24
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None
    vae_chunk_dim: int = -1
    vae_clip_length: int = 17
    vae_decoder_tiling: int = 1
    vae_encoder_tiling: int = 1
    vae_parallel_tiling: int = 0
    vae_tile_overlap_min: int = 64
    vae_tile_size: int = 256
    vae_token_drop: int = 0
    vit_decoder_2d: bool = True
    vit_decoder_kwargs: dict = field(default_factory=_VIT_DECODER_DEFAULTS.copy)

    in_channels: int = 3
    out_channels: int = 3
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024)
    layers_per_block: int = 2
    spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    spatial_padding_mode: str = "reflect"
    decoder_num_register_tokens: int = 4
    decoder_ffn_mult: float = 4
    decoder_norm_eps: float = 1e-5

    def __post_init__(self):
        for name, released, fill in (
            ("latents_mean", _IMAGE_LATENTS_MEAN, 0.0),
            ("latents_std", _IMAGE_LATENTS_STD, 1.0),
        ):
            values = getattr(self, name)
            if values is None:
                values = released if self.latent_channels == len(released) else (fill,) * self.latent_channels
            if len(values) != self.latent_channels:
                raise ValueError(f"{name} must have {self.latent_channels} values")
            object.__setattr__(self, name, tuple(values))
        for name in ("block_out_channels", "spatial_downsample_factors", "temporal_downsample_factors"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "vit_decoder_kwargs", _VIT_DECODER_DEFAULTS | self.vit_decoder_kwargs)


@dataclass(frozen=True)
class MiniMaxH3PDDConfig(_JsonConfig):
    """Six distilled denoising steps plus the terminal zero sigma."""

    num_inference_steps: int = 6
    sigmas: tuple[float, ...] = (
        1.0, 0.9675236940383911, 0.8875485062599182,
        0.75, 0.5327739119529724, 0.23200887441635132,
        0.0,
    )

    def __post_init__(self):
        sigmas = tuple(self.sigmas)
        if (self.num_inference_steps <= 0 or len(sigmas) != self.num_inference_steps + 1
                or sigmas[0] != 1.0 or sigmas[-1] != 0.0
                or not all(a > b for a, b in zip(sigmas, sigmas[1:]))):
            raise ValueError("PDD sigmas must decrease from 1 to 0 and include one terminal value")
        object.__setattr__(self, "sigmas", sigmas)
