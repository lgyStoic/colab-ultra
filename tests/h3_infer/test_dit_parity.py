"""Checkpoint layout and numeric parity of the image DiT and PDD heads."""

import ast
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn.functional import cosine_similarity

from h3_infer.config import MiniMaxH3DiTConfig
from h3_infer.models.dit import (
    MiniMaxH3DiT,
    MiniMaxH3PDDHeads,
    build_packed_sequence,
    patchify_video,
    unpatchify_video,
)

_RELEASE_INDEX = Path(
    "/Users/garry/ColabUltra/models/minimax-h3-image/bf16/transformer/model.safetensors.index.json"
)


@pytest.fixture
def reference_builder(diffsynth_reference):
    # Execute the actual builder from the pipeline source. Its base class is
    # irrelevant to packing, and importing the full pipeline loads text models.
    root, _ = diffsynth_reference
    source_path = root / "pipelines/minimax_h3_image.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "MINIMAX_H3_IMAGE_PATCH_SIZE", "MINIMAX_H3_IMAGE_ROPE_SPATIAL_SCALE",
        "MINIMAX_H3_IMAGE_IMAGE_TAG",
    }
    body = [
        node for node in tree.body
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names)
        or isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3ImageUnit_PackedSequenceBuilder"
    ]
    namespace = {"torch": torch, "np": np, "PipelineUnit": object, "MiniMaxH3ImagePipeline": object}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"), namespace)
    builder = namespace["MiniMaxH3ImageUnit_PackedSequenceBuilder"]
    return builder.__new__(builder)


@pytest.mark.parametrize("mode", ["t2i", "edit"])
@pytest.mark.parametrize("mixed", [False, True], ids=["fp32", "bf16-mixed"])
@torch.no_grad()
def test_dit_parity(diffsynth_reference, reference_builder, mode, mixed):
    torch.manual_seed(123)
    _, load = diffsynth_reference
    ref_dit = load("models.minimax_h3_image_dit")
    ref_patch = load("models.minimax_h3_dit")
    config = MiniMaxH3DiTConfig(
        num_layers=2, token_refiner_num_layers=1, hidden_size=128,
        num_attention_heads=4, attention_head_dim=32, ffn_hidden_size=256,
        adaln_out_features=2304, final_adaln_out_features=256,
        latents_dim=4, text_dim=64, timestep_input_dim=32,
        time_embed_hidden_size=128, time_embed_dim=64, rope_inv_freq_len=4,
    )
    ref = ref_dit.MiniMaxH3ImageDiT(**asdict(config)).eval()
    ref_heads = ref_dit.MiniMaxH3ImagePDDHeads(6, 128, config.patch_dim).eval()
    model = MiniMaxH3DiT(config, attention_backend="sdpa").eval()
    heads = MiniMaxH3PDDHeads(6, 128, config.patch_dim).eval()
    for expected, actual in ((ref, model), (ref_heads, heads)):
        assert {k: v.shape for k, v in expected.state_dict().items()} == {
            k: v.shape for k, v in actual.state_dict().items()
        }
    if mixed:
        for key, child in ref.named_children():
            if key not in ("video_patch_proj", "time_embedder"):
                child.bfloat16()
    model.load_state_dict(ref.state_dict(), strict=True, assign=True)
    heads.load_state_dict(ref_heads.state_dict(), strict=True)
    if mixed:
        assert model.time_embedder.proj_out.weight.dtype == torch.float32
        assert model.blocks[0].adaln_proj.linear.weight.dtype == torch.bfloat16

    grids = [(4, 8), (6, 4)] if mode == "edit" else []
    tags = torch.tensor([1, 0, 0, 1, 1]) if grids else torch.ones(5, dtype=torch.long)
    height, width = 6, 10
    packed = build_packed_sequence(tags, grids, height, width)
    ref_packed = reference_builder.build_packed_sequence(tags, grids, height, width)
    for got, expected in zip(packed[:4], ref_packed[:4]):
        assert torch.equal(got, expected)
    assert packed[0].dtype == torch.float64
    assert packed[-1] == ref_packed[-1]
    positions, tags, image_pos, text_pos, num_ref = packed
    latents = [torch.randn(1, config.latents_dim, 1, h, w) for h, w in [*grids, (height, width)]]
    rows = torch.cat([patchify_video(latent) for latent in latents])
    ref_rows = torch.cat([ref_patch.patchify_video(latent) for latent in latents])
    assert torch.equal(rows, ref_rows)
    assert torch.equal(unpatchify_video(rows[num_ref:], 1, height, width, config.latents_dim), latents[-1])
    times = torch.tensor([0.11245149, 0.999] if grids else [0.11245149])
    indices = torch.zeros(tags.shape, dtype=torch.long)
    indices[image_pos[:num_ref]] = 1
    args = (rows, torch.randn(5, config.text_dim), positions[None], times, indices, tags, image_pos, text_pos)
    actual = model(*args, pdd_heads=heads, pdd_step_index=2)
    expected = ref(*args, pdd_heads=ref_heads, pdd_step_index=2)
    cosine = cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    if mixed:
        assert cosine > 0.9999
        # The same fp32-SiLU-then-bf16-cast path is exact on CPU SDPA.
        assert torch.equal(actual, expected)
    else:
        assert (actual - expected).abs().max().item() < 1e-5
        assert cosine > 0.999999


@pytest.mark.skipif(not _RELEASE_INDEX.is_file(), reason="local release index unavailable")
def test_official_release_index_keys(diffsynth_reference):
    _, load = diffsynth_reference
    ref_dit = load("models.minimax_h3_image_dit")
    with torch.device("meta"):
        actual = MiniMaxH3DiT()
        expected = ref_dit.MiniMaxH3ImageDiT()
    first = {k: tuple(v.shape) for k, v in actual.state_dict().items()}
    second = {k: tuple(v.shape) for k, v in expected.state_dict().items()}
    mapping = json.loads(_RELEASE_INDEX.read_text(encoding="utf-8"))["weight_map"]
    assert set(mapping) == set(first) == set(second)
    assert len(set(mapping.values())) == 12
    assert first == second
