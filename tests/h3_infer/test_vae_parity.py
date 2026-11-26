"""Compare the pure-PyTorch image VAE with DiffSynth on small tiled grids."""

from types import ModuleType
import sys

import pytest
import torch
from torch.nn.functional import cosine_similarity

from h3_infer.config import MiniMaxH3VAEConfig
from h3_infer.models.vae import MiniMaxH3ImageVAE


class _Normalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, value):
        mean = value.new_tensor(self.mean).view(-1, 1, 1)
        std = value.new_tensor(self.std).view(-1, 1, 1)
        return (value - mean) / std


@pytest.fixture
def vae_pair(monkeypatch, diffsynth_reference):
    # Only the reference imports this package. The real encoder and decoder
    # implementations are used, along with DiffSynth's own CPU attention.
    torchvision = ModuleType("torchvision")
    transforms = ModuleType("torchvision.transforms")
    transforms.Normalize = _Normalize
    torchvision.transforms = transforms
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms)
    _, load = diffsynth_reference
    reference = load("models.minimax_h3_image_vae")

    torch.manual_seed(456)
    config = MiniMaxH3VAEConfig(
        latent_channels=4, block_out_channels=(8, 16), layers_per_block=1,
        spatial_downsample_factors=(2, 2), temporal_downsample_factors=(1, 2),
        norm_num_groups=4, vae_tile_size=24, vae_tile_overlap_min=8, decoder_ffn_mult=2,
        vit_decoder_kwargs={"num_layers": 2, "heads": 2, "dim_head": 16},
    )
    ours = MiniMaxH3ImageVAE(config, attention_backend="sdpa").eval()
    ref = reference.MiniMaxH3ImageVAE(
        latent_channels=4, block_out_channels=(8, 16), layers_per_block=1,
        spatial_downsample_factors=(2, 2), temporal_downsample_factors=(1, 2),
        norm_num_groups=4, decoder_num_layers=2, decoder_num_attention_heads=2,
        decoder_attention_head_dim=16, decoder_ffn_mult=2, tile_size=24,
        tile_overlap_min=8, latents_mean=config.latents_mean, latents_std=config.latents_std,
    ).eval()
    with torch.no_grad():
        # Nonzero learned residual scales make attention and FFN affect parity.
        for block in ref.decoder.transformer_blocks:
            block.scale1.weight.uniform_(0.1, 0.6)
            block.scale2.weight.uniform_(0.1, 0.6)
    release = {}
    for key, value in ref.state_dict().items():
        if key.endswith(("register_tokens.weight", "scale1.weight", "scale2.weight")):
            key = key.removesuffix(".weight")
        release[key] = value
    assert {k: v.shape for k, v in release.items()} == {k: v.shape for k, v in ours.state_dict().items()}
    assert set(dict(ours.named_buffers())) >= {"latents_mean", "latents_std"}
    assert "latents_mean" not in release and "latents_std" not in release
    ours.load_state_dict(release, strict=True)
    return ours, ref


def _close(actual, expected):
    assert actual.shape == expected.shape
    assert (actual - expected).abs().max().item() < 1e-4
    assert cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item() > 0.99999


@pytest.mark.parametrize("tiling", [False, True], ids=["untiled", "tiled"])
@torch.no_grad()
def test_vae_parity(vae_pair, tiling):
    ours, ref = vae_pair
    pixels = torch.randn(2, 3, 36, 44)
    assert ours._split_tiles(44, 24, 8) == ref._split_tiles(44, 24, 8)
    assert len(ours._split_tiles(44, 24, 8)[0]) == 3
    actual = ours.encode(pixels, tiling=tiling)
    expected = ref.encode(pixels, tiling=tiling)
    _close(actual.mode(), expected.mode())
    _close(actual.logvar, expected.logvar)
    _close(actual.kl(), expected.kl())
    _close(actual.sample(torch.Generator().manual_seed(1)), expected.sample(torch.Generator().manual_seed(1)))

    latent = torch.randn(2, 4, 9, 11)
    _close(ours.decode(latent, tiling=tiling), ref.decode(latent, tiling=tiling))
