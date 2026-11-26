"""Checkpoint parity, offload lifecycle, staging, compilation, and batch inference."""

import json
import os
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from h3_infer.config import MiniMaxH3PDDConfig
from h3_infer.models.dit import (
    REFERENCE_TIMESTEP,
    MiniMaxH3AdalnTable,
    MiniMaxH3DiT,
    build_packed_sequence,
)
from h3_infer.ops import attention as attention_ops
from h3_infer.ops.attention import _optional_kernel, attention, recommended_attention_backend
from h3_infer.pipeline import MiniMaxH3ImagePipeline
from h3_infer.runtime.loader import load_safetensors
from h3_infer.runtime.offload import AsyncBlockOffloader
from h3_infer.runtime.scheduler import MiniMaxH3Scheduler

from test_scheduler_and_loader import tiny_checkpoint


def _packed_args(dit, heads, edit, device="cpu"):
    config = dit.config
    grids = [(4, 6), (4, 4)] if edit else []
    positions, tags, img_pos, text_pos, num_ref = build_packed_sequence([1, 0, 1], grids, 6, 8)
    indices = torch.zeros_like(tags)
    indices[img_pos[:num_ref]] = 1
    times = MiniMaxH3Scheduler().model_timesteps[:, None]
    if edit:
        times = torch.cat((times, torch.full_like(times, REFERENCE_TIMESTEP)), dim=1)
    rows = torch.randn(len(img_pos), config.patch_dim)
    embeds = torch.randn(len(text_pos), config.text_dim)
    positions = positions[None]

    def for_step(step):
        return (*[value.to(device) for value in
                  (rows, embeds, positions, times[step], indices, tags, img_pos, text_pos)], heads, step)

    return positions.to(device), for_step


@pytest.mark.parametrize("edit", [False, True], ids=["t2i", "edit"])
@torch.inference_mode()
def test_pruned_adaln_parity(tiny_checkpoint, diffsynth_reference, edit):
    _, dit, heads, _ = tiny_checkpoint
    _, load = diffsynth_reference
    reference = load("models.minimax_h3_image_dit")
    with torch.device("meta"):
        diff_pruned = reference.MiniMaxH3ImageDiTPruned(**asdict(dit.config))
    cfg = MiniMaxH3PDDConfig()
    scheduler = MiniMaxH3Scheduler(cfg)
    times = torch.tensor([1.0 - sigma for sigma in cfg.sigmas[:-1]] + [REFERENCE_TIMESTEP])
    assert torch.equal(times[:-1], scheduler.model_timesteps)
    assert torch.equal(times[-1], torch.tensor(0.999, dtype=torch.float32))
    assert dit.time_embedder.proj_in.weight.dtype == torch.float32
    assert dit.blocks[0].adaln_proj.linear.weight.dtype == torch.bfloat16

    model = deepcopy(dit).eval()
    assert model.prune_adaln_tables() is model
    assert model.prune_adaln_tables() is model
    assert model.time_embedder is None
    assert not hasattr(model.final_layer, "video_out")
    assert isinstance(model.blocks[0].adaln_proj, MiniMaxH3AdalnTable)
    assert not model.blocks[0].adaln_proj.table.requires_grad
    assert {key: value.shape for key, value in model.state_dict().items()} == {
        key: value.shape for key, value in diff_pruned.state_dict().items()
    }
    diff_pruned.load_state_dict(model.state_dict(), assign=True)
    _, args = _packed_args(dit, heads, edit)
    for step in range(6):
        expected = dit(*args(step))
        result = model(*args(step))
        assert torch.equal(result, expected), step
        assert torch.equal(result, diff_pruned(*args(step))), step


@torch.inference_mode()
def test_load_pruned_checkpoint(tiny_checkpoint):
    root, dit, heads, _ = tiny_checkpoint
    converted = MiniMaxH3ImagePipeline.from_pretrained(root, prune_adaln=True, attention_backend="sdpa")
    assert converted.dit.pruned_adaln
    weights = converted.dit.state_dict()
    transformer = root / "transformer"
    checkpoint = transformer / "pruned.safetensors"
    save_file(weights, checkpoint)
    (transformer / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": dict.fromkeys(weights, checkpoint.name),
    }), encoding="utf-8")
    with torch.device("meta"):
        direct = MiniMaxH3DiT(dit.config, attention_backend="sdpa", pruned_adaln=True)
    load_safetensors(direct, checkpoint)
    pipe = MiniMaxH3ImagePipeline.from_pretrained(root, attention_backend="sdpa", prune_adaln=False)
    assert pipe.dit.pruned_adaln
    for model in (direct, pipe.dit):
        assert model.state_dict().keys() == weights.keys()
        for key, value in model.state_dict().items():
            assert value.device.type == "cpu"
            assert torch.equal(value, weights[key])
    _, args = _packed_args(dit, heads, True)
    assert torch.equal(direct(*args(5)), pipe.dit(*args(5)))
    assert torch.equal(direct(*args(5)), dit(*args(5)))


@pytest.mark.parametrize("pruned", [False, True])
@torch.inference_mode()
def test_hoisted_rope(tiny_checkpoint, pruned):
    _, dit, heads, _ = tiny_checkpoint
    if pruned:
        dit.prune_adaln_tables()
    positions, args = _packed_args(dit, heads, True)
    rope = dit.rope(positions)
    expected = dit(*args(4))
    with patch.object(dit.rope, "forward", side_effect=AssertionError("RoPE recomputed")):
        assert torch.equal(dit(*args(4), precomputed_rope=rope), expected)


@pytest.mark.parametrize("resident", [0, 1, 2])
@pytest.mark.parametrize("pin_staging", [False, True])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable")),
])
@torch.inference_mode()
def test_async_offload(tiny_checkpoint, resident, pin_staging, device):
    _, tiny, heads, _ = tiny_checkpoint
    dit = MiniMaxH3DiT(replace(tiny.config, num_layers=4), attention_backend="sdpa").eval()
    dit.prune_adaln_tables()
    direct = deepcopy(dit).to(device)
    heads.to(device)
    _, args = _packed_args(dit, heads, True, device)
    backing = [block.attn.qkv_proj.weight for block in dit.blocks]

    dit.enable_async_offload(device, resident, pin_staging)
    dit.prepare_gpu()
    offloader = dit._offloader
    assert isinstance(offloader, AsyncBlockOffloader)
    pointers = [slot.attn.qkv_proj.weight.data_ptr() for slot in offloader.slot_modules]
    if device == "cpu":
        assert offloader.num_resident == len(dit.blocks)
        assert not pointers and offloader.copy_stream is None
    else:
        assert offloader.num_resident == resident and len(set(pointers)) == 2
        assert len(offloader.staging) == 2 * pin_staging
        assert all(value.is_pinned() for slot in offloader.staging for value in slot.values())
    for step in (0, 5):
        assert torch.equal(dit(*args(step)), direct(*args(step)))
        assert [slot.attn.qkv_proj.weight.data_ptr() for slot in offloader.slot_modules] == pointers
    dit.release_gpu()
    assert not offloader.prepared and not offloader.slot_modules and not offloader.staging
    assert all(block.attn.qkv_proj.weight is weight for block, weight in zip(dit.blocks, backing))
    dit.prepare_gpu()
    try:
        assert torch.equal(dit(*args(3)), direct(*args(3)))
    finally:
        dit.release_gpu()


@pytest.mark.parametrize("prepared", [False, True])
def test_cuda_block_run_checks_offloader_residency(tiny_checkpoint, prepared):
    _, dit, _, _ = tiny_checkpoint
    dit.enable_async_offload("cpu")
    device = torch.device("cuda:0")
    hidden = SimpleNamespace(device=device)
    dit._offloader = SimpleNamespace(device=device, prepared=prepared, run_blocks=Mock())
    with patch.object(dit, "prepare_gpu") as prepare:
        assert dit._run_blocks(hidden) is dit._offloader.run_blocks.return_value
    if prepared:
        prepare.assert_not_called()
    else:
        prepare.assert_called_once_with(device)
    dit._offloader.run_blocks.assert_called_once_with(dit._block_runner, hidden)


@pytest.mark.parametrize("edit", [False, True], ids=["t2i", "edit"])
def test_staged_pipeline(staged_cpu_pipeline, edit):
    pipe, calls = staged_cpu_pipeline
    assert pipe.vae_device.type == "cpu"
    hook = pipe.dit.rope.register_forward_hook(lambda *_: calls.append("rope"))
    try:
        result = pipe(
            torch.randn(3, pipe.dit.config.text_dim),
            edit_images=Image.new("RGB", (32, 32)) if edit else None,
            height=32, width=32, min_generation_area=1024,
        )
    finally:
        hook.remove()
    assert result.size == (32, 32)
    prefix = ["vae enter", "vae leave"] if edit else []
    assert calls == prefix + ["dit enter", "rope", "dit leave", "vae enter", "vae leave"]
    assert pipe.vae_device.type == "cpu"


@pytest.fixture
def staged_cpu_pipeline(tiny_checkpoint, monkeypatch):
    root, _, _, _ = tiny_checkpoint
    pipe = MiniMaxH3ImagePipeline.from_pretrained(root, prune_adaln=True)
    # Exercise the staged call order with real CPU inference and mocked transfers.
    pipe.staged_vram = True
    calls = []
    monkeypatch.setattr(pipe.dit, "prepare_gpu", lambda device: calls.append("dit enter"))
    monkeypatch.setattr(pipe.dit, "release_gpu", lambda: calls.append("dit leave"))
    monkeypatch.setattr(pipe.vae, "to", lambda device: calls.append(
        "vae enter" if isinstance(device, torch.device) else "vae leave"
    ))
    monkeypatch.setattr("h3_infer.pipeline._resize_reference", lambda image: image)
    return pipe, calls


def test_keep_dit_resident(staged_cpu_pipeline):
    pipe, calls = staged_cpu_pipeline
    embeds = torch.randn(3, pipe.dit.config.text_dim)
    options = dict(height=32, width=32, min_generation_area=1024)
    message = r"Cannot move VAE to GPU while keep_dit_resident\(\) is active"
    with pipe.keep_dit_resident():
        with pipe.keep_dit_resident():
            latents = pipe(embeds, output_type="latent", **options)
            assert calls == ["dit enter"]
            with pytest.raises(RuntimeError, match=message), pipe._vae_on_device():
                pytest.fail("VAE context was entered with DiT resident")
            with pytest.raises(RuntimeError, match=message):
                pipe.decode_latents(latents)
            with patch.object(pipe.dit, "forward", side_effect=AssertionError("DiT ran")):
                with pytest.raises(RuntimeError, match=message):
                    pipe(embeds, output_type="pil", **options)
                with pytest.raises(RuntimeError, match=message):
                    pipe(embeds, edit_images=Image.new("RGB", (32, 32)), output_type="latent", **options)
        assert torch.equal(pipe(embeds, output_type="latent", **options), latents)
        assert calls == ["dit enter"]
    assert calls == ["dit enter", "dit leave"]
    assert pipe.decode_latents(latents).size == (32, 32)
    assert calls[-2:] == ["vae enter", "vae leave"]


def test_keep_dit_resident_unwinds(staged_cpu_pipeline, monkeypatch):
    pipe, calls = staged_cpu_pipeline
    with pytest.raises(ValueError, match="inference failed"), pipe.keep_dit_resident():
        raise ValueError("inference failed")
    assert calls == ["dit enter", "dit leave"]
    assert pipe._dit_resident_depth == 0

    def fail_prepare(device):
        raise ValueError("prepare failed")

    monkeypatch.setattr(pipe.dit, "prepare_gpu", fail_prepare)
    with pytest.raises(ValueError, match="prepare failed"), pipe.keep_dit_resident():
        pytest.fail("DiT context was entered after failed prepare")
    assert calls == ["dit enter", "dit leave"]
    assert pipe._dit_resident_depth == 0


@pytest.mark.parametrize("edit", [False, True], ids=["t2i", "edit"])
def test_generate_batch_matches_single_calls(staged_cpu_pipeline, tmp_path, edit):
    pipe, calls = staged_cpu_pipeline
    embeds = torch.randn(3, pipe.dit.config.text_dim)
    tags = torch.tensor([1, 0, 1])
    raw_embeds = {"text_embeds": embeds * 0.5, "text_token_tags": tags}
    path = tmp_path / "embeds.pt"
    torch.save(raw_embeds, path)
    options = dict(height=32, width=32, min_generation_area=1024)
    second_options = {"width": 37}
    first_options = {}
    if edit:
        first_options["edit_images"] = Image.new("RGB", (32, 32), "red")
        second_options.update(
            edit_images=[Image.new("RGB", (32, 32), "blue")], seed=103,
            latents=torch.randn(1, pipe.dit.config.latents_dim, 1, 4, 8),
            reference_noise=torch.randn(1, pipe.dit.config.latents_dim, 4, 4),
        )
    first = {"prompt_embeds": path, **first_options} if edit else path
    second = MappingProxyType({**raw_embeds, **second_options})
    seed = 101 if edit else None
    expected = [
        pipe(path, **options, **first_options, seed=101 if edit else 0, output_type="tensor"),
        pipe(raw_embeds, **(options | {"seed": 102 if edit else 0} | second_options), output_type="tensor"),
    ]
    calls.clear()
    steps = []

    def progress(idx, step, latents, velocity):
        steps.append((idx, step))
        if step == 0:
            calls.append(f"denoise {idx}")

    original_decode = pipe.vae.decode

    def record_decode(*args, **kwargs):
        calls.append("decode")
        return original_decode(*args, **kwargs)

    with patch.object(pipe.vae, "decode", side_effect=record_decode) as decode:
        results = pipe.generate_batch(
            [first, second], **options, seed=seed, output_type="tensor", return_stats=True,
            step_callback=progress,
        )
    for (result, stats), individual in zip(results, expected, strict=True):
        assert result.dtype == torch.uint8
        assert torch.equal(result, individual)
        assert len(stats.step_ms) == 6
    assert steps == [(idx, step) for idx in range(2) for step in range(6)]
    assert [call.kwargs["tiling"] for call in decode.call_args_list] == [None, None]
    prefix = ["vae enter", "vae leave"] if edit else []
    assert calls == prefix + [
        "dit enter", "denoise 0", "denoise 1", "dit leave", "vae enter", "decode", "decode", "vae leave",
    ]


def test_generate_batch_latents(staged_cpu_pipeline):
    pipe, calls = staged_cpu_pipeline
    embeds = torch.randn(3, pipe.dit.config.text_dim)
    options = dict(height=32, width=32, min_generation_area=1024, output_type="latent")
    assert pipe.generate_batch([], **options) == []
    with pytest.raises(ValueError, match="Unknown output_type"):
        pipe.generate_batch([], output_type="bad")
    assert calls == []
    expected = [pipe(embeds, seed=seed, **options) for seed in (41, 42)]
    calls.clear()
    results = pipe.generate_batch([embeds, embeds], seed=41, **options)
    assert all(torch.equal(result, item) for result, item in zip(results, expected, strict=True))
    assert calls == ["dit enter", "dit leave"]


@pytest.mark.parametrize("operation", ["encode", "decode"])
@torch.inference_mode()
def test_vae_batched_tiling_parity(tiny_checkpoint, monkeypatch, operation):
    _, _, _, vae = tiny_checkpoint
    for block in vae.decoder.transformer_blocks:
        block.scale1.fill_(0.2)
        block.scale2.fill_(0.3)

    # Six overlapping tiles exercise stitching, a short final chunk, and inputs with B > 1.
    ratio = vae.spatial_compression_ratio
    shape = (2, 3, 56, 80) if operation == "encode" else (2, 4, 56 // ratio, 80 // ratio)
    x = torch.randn(shape, generator=torch.Generator().manual_seed(77))
    run = getattr(vae, operation)
    tile_method = f"_{operation}_tile"
    tile = getattr(vae, tile_method)

    def values(result):
        return result.parameters if operation == "encode" else result

    expected = values(run(x, tiling=True, tile_batch_size=1))
    for batch_size, batch_lengths in ((None, [2] * 6), (2, [4, 4, 4]), (4, [8, 4])):
        with patch.object(vae, tile_method, wraps=tile) as tracked:
            actual = values(run(x, tiling=None if batch_size is None else True, tile_batch_size=batch_size))
        assert [call.args[0].shape[0] for call in tracked.call_args_list] == batch_lengths
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    expected = values(run(x[:1], tiling=True, tile_batch_size=1))
    calls = []

    def limited_tile(inputs):
        calls.append(inputs.shape[0])
        if inputs.shape[0] > 1:
            raise torch.cuda.OutOfMemoryError("mock OOM")
        return tile(inputs)

    monkeypatch.setattr(vae, tile_method, limited_tile)
    with (
        patch("h3_infer.models.vae.gc.collect", side_effect=lambda: calls.append("gc")),
        patch.object(torch.cuda, "empty_cache", side_effect=AssertionError("CPU input called empty_cache")),
    ):
        actual = values(run(x[:1], tiling=True, tile_batch_size=4))
    assert calls == [4, "gc", 2, "gc"] + [1] * 6
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    for batch_size in (0, -1):
        with pytest.raises(ValueError, match="tile_batch_size must be positive"):
            run(x, tiling=True, tile_batch_size=batch_size)


@pytest.fixture
def dense_sageattention(monkeypatch):
    def sageattn(q, k, v, *, tensor_layout, is_causal, sm_scale):
        assert tensor_layout == "NHD" and not is_causal
        assert all(x.ndim == 4 and x.is_contiguous() for x in (q, k, v))
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=sm_scale,
        ).transpose(1, 2)

    sage = Mock(side_effect=sageattn)
    flash = Mock(side_effect=AssertionError("FlashAttention selected instead of SageAttention"))
    monkeypatch.setitem(sys.modules, "sageattention", SimpleNamespace(sageattn=sage))
    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace(flash_attn_varlen_func=flash))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    _optional_kernel.cache_clear()
    try:
        yield sage
    finally:
        _optional_kernel.cache_clear()


def test_recommended_attention_backend(dense_sageattention):
    assert recommended_attention_backend(8) == "sageattention"
    assert recommended_attention_backend(9) == "sageattention"
    assert recommended_attention_backend(7) == "sdpa"


@pytest.mark.parametrize("bounds, expected_lengths", [
    (None, [10]), ([0, 10], [10]), ([0, 4, 4, 10], [4, 6]),
], ids=["dense", "packed-single", "packed-multiple"])
def test_auto_attention_with_dense_sageattention(dense_sageattention, bounds, expected_lengths):
    shape = (2, 2, 10, 64) if bounds is None else (2, 10, 64)
    qkv = [torch.randn(shape, dtype=torch.bfloat16).transpose(-3, -2) for _ in range(3)]
    offsets = None if bounds is None else torch.tensor(bounds, dtype=torch.int32)
    options = dict(cu_seqlens=offsets, softmax_scale=0.25)
    expected = attention(*qkv, backend="sdpa", **options)
    output = attention(*qkv, backend="auto", **options)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert [call.args[0].shape[1] for call in dense_sageattention.call_args_list] == expected_lengths
    for call in dense_sageattention.call_args_list:
        assert call.kwargs == dict(tensor_layout="NHD", is_causal=False, sm_scale=0.25)


@pytest.mark.parametrize("scale", [None, 0.25])
def test_compile_auto_attention_with_dense_sageattention(
    dense_sageattention, torch_compile_discovery, monkeypatch, scale,
):
    select_kernel = attention_ops._cuda_kernel

    def select_for_cuda(backend, major, dtype, head_dim, packed):
        # Dynamo reads is_cuda from actual tensor metadata during tracing.
        return select_kernel(backend, 8, dtype, head_dim, packed)

    monkeypatch.setattr(attention_ops, "_cuda_kernel", select_for_cuda)

    def run(q, k, v, offsets):
        return attention(q, k, v, cu_seqlens=offsets, softmax_scale=scale, backend="auto")

    qkv = [torch.randn(10, 2, 64, dtype=torch.bfloat16) for _ in range(3)]
    offsets = torch.tensor([0, 10], dtype=torch.int32)
    expected = attention(*qkv, cu_seqlens=offsets, softmax_scale=scale, backend="sdpa")
    compiled = torch.compile(run, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(*qkv, offsets), expected, atol=0, rtol=0)
    dense_sageattention.assert_called_once()
    assert dense_sageattention.call_args.kwargs["sm_scale"] == scale


def test_compile_dynamic_packed_sageattention_varlen(torch_compile_discovery, monkeypatch):
    calls = []

    def varlen(q, k, v, cu_q, cu_k, max_q, max_k, *, is_causal, sm_scale):
        calls.append((q.shape[0], max_q, max_k, is_causal, sm_scale))
        return torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2),
            k.unsqueeze(0).transpose(1, 2),
            v.unsqueeze(0).transpose(1, 2),
            scale=sm_scale,
        ).transpose(1, 2).squeeze(0)

    monkeypatch.setitem(sys.modules, "sageattention", SimpleNamespace(sageattn_varlen=varlen))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    _optional_kernel.cache_clear()
    select_kernel = attention_ops._cuda_kernel
    monkeypatch.setattr(
        attention_ops,
        "_cuda_kernel",
        lambda backend, major, dtype, head_dim, packed: select_kernel(backend, 8, dtype, head_dim, packed),
    )

    def run(q, k, v, offsets):
        return attention(q, k, v, cu_seqlens=offsets, max_seqlen=q.shape[0], softmax_scale=0.25, backend="auto")

    try:
        compiled = torch.compile(run, backend="eager", fullgraph=True, dynamic=True)
        for length in (10, 17):
            qkv = [torch.randn(length, 2, 64, dtype=torch.bfloat16) for _ in range(3)]
            offsets = torch.tensor([0, length], dtype=torch.int32)
            expected = attention(*qkv, cu_seqlens=offsets, max_seqlen=length, softmax_scale=0.25, backend="sdpa")
            torch.testing.assert_close(compiled(*qkv, offsets), expected, atol=0, rtol=0)
        assert calls == [(10, 10, 10, False, 0.25), (17, 17, 17, False, 0.25)]
    finally:
        _optional_kernel.cache_clear()


def test_compile_auto_attention_with_closure_function(torch_compile_discovery, monkeypatch):
    calls = []

    def make_closure(tag):
        def sageattn(q, k, v, *, tensor_layout, is_causal, sm_scale):
            calls.append((tag, tensor_layout, is_causal, sm_scale))
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=sm_scale,
            ).transpose(1, 2)

        return sageattn

    monkeypatch.setitem(sys.modules, "sageattention", SimpleNamespace(sageattn=make_closure("closure")))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    _optional_kernel.cache_clear()
    select_kernel = attention_ops._cuda_kernel
    monkeypatch.setattr(
        attention_ops,
        "_cuda_kernel",
        lambda backend, major, dtype, head_dim, packed: select_kernel(backend, 8, dtype, head_dim, packed),
    )

    def run(q, k, v, offsets):
        return attention(q, k, v, cu_seqlens=offsets, softmax_scale=0.25, backend="auto")

    try:
        qkv = [torch.randn(10, 2, 64, dtype=torch.bfloat16) for _ in range(3)]
        offsets = torch.tensor([0, 10], dtype=torch.int32)
        expected = attention(*qkv, cu_seqlens=offsets, softmax_scale=0.25, backend="sdpa")
        compiled = torch.compile(run, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(*qkv, offsets), expected, atol=0, rtol=0)
        assert calls == [("closure", "NHD", False, 0.25)]
    finally:
        _optional_kernel.cache_clear()


@pytest.fixture
def torch_compile_discovery(monkeypatch):
    # The NAS creates macOS AppleDouble files that 2.15 tries to import as code.
    # Filter them only during compiler discovery without modifying the environment.
    listdir = os.listdir
    torch_root = Path(torch.__file__).parent
    discovery_dirs = {torch_root / "_dynamo/backends", torch_root / "_inductor/kernel"}

    def list_modules(path):
        entries = listdir(path)
        if isinstance(path, (str, Path)) and Path(path) in discovery_dirs:
            return [entry for entry in entries if not entry.startswith("._")]
        return entries

    monkeypatch.setattr(os, "listdir", list_modules)


@torch.inference_mode()
def test_regional_compile_one_graph(tiny_checkpoint, torch_compile_discovery):
    _, tiny, heads, _ = tiny_checkpoint
    dit = MiniMaxH3DiT(replace(tiny.config, num_layers=50), attention_backend="sdpa", offload=True).eval()
    dit.prune_adaln_tables()
    dit.enable_async_offload("cpu")
    _, args = _packed_args(dit, heads, False)
    expected = dit(*args(0))
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    with patch.object(torch, "compile", wraps=torch.compile) as compile_fn:
        assert dit.compile("regional", backend=backend, fullgraph=True) is dit
        torch.testing.assert_close(dit(*args(0)), expected, atol=0, rtol=0)
        assert compile_fn.call_count == 1
    assert len(graphs) == 1


@torch.inference_mode()
def test_regional_and_full_compile_modes(tiny_checkpoint, torch_compile_discovery):
    root, dit, heads, _ = tiny_checkpoint
    with pytest.raises(ValueError, match="offload=False"):
        MiniMaxH3ImagePipeline.from_pretrained(root, offload_dit=True, compile_mode="full")
    with pytest.raises(ValueError, match="offload=False"):
        MiniMaxH3DiT(dit.config, offload=True).compile("full")
    _, args = _packed_args(dit, heads, False)
    expected = dit(*args(0))
    with patch.object(torch, "compile", wraps=torch.compile) as compile_fn:
        compiled = dit.compile("full", backend="eager", fullgraph=True)
        assert compile_fn.call_count == 1 and compile_fn.call_args.args == (dit,)
        assert torch.equal(compiled(*args(0)), expected)
    with patch.object(torch, "compile", side_effect=lambda fn, **kwargs: fn) as compile_fn:
        pipe = MiniMaxH3ImagePipeline.from_pretrained(root, compile_mode="regional")
        assert compile_fn.call_count == 1
        assert torch.equal(pipe.dit(*args(0)), expected)


@torch.inference_mode()
def test_regional_inductor_parity(tiny_checkpoint, torch_compile_discovery):
    _, dit, heads, _ = tiny_checkpoint
    _, args = _packed_args(dit, heads, False)
    expected = dit(*args(0))
    dit.compile("regional", fullgraph=True)
    try:
        compiled = dit(*args(0))
    except torch._dynamo.exc.BackendCompilerFailed as error:
        if type(error.inner_exception).__name__ not in (
                "CppCompileError", "InvalidCxxCompiler", "MissingCompiler", "ModuleNotFoundError"):
            raise
        pytest.skip(f"Native Inductor unavailable: {type(error.inner_exception).__name__}")
    torch.testing.assert_close(compiled, expected, atol=0.02, rtol=0.02)
