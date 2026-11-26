"""Native INT8 operators and Comfy-Org checkpoint integration."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from h3_infer.config import MiniMaxH3DiTConfig, MiniMaxH3PDDConfig, MiniMaxH3VAEConfig
from h3_infer.models.dit import REFERENCE_TIMESTEP, MiniMaxH3DiT, MiniMaxH3PDDHeads, build_packed_sequence
from h3_infer.models.vae import MiniMaxH3ImageVAE
from h3_infer.ops import (
    Int8ConvRotLinear,
    build_regular_hadamard,
    int8_convrot_linear,
    quantize_int8_rowwise,
    rotate_activation,
    rotate_weight,
)
from h3_infer.ops import quant as quant_ops
from h3_infer.pipeline import MiniMaxH3ImagePipeline
from h3_infer.runtime.compile import run_block
from h3_infer.runtime.loader import inspect_comfy_checkpoint, load_safetensors
from h3_infer.runtime.offload import AsyncBlockOffloader

from test_optimizations import torch_compile_discovery


_DEVICES = (
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
)
_AUTO_CHECKPOINT = "minimax_h3_image_pruned_int8_convrot.safetensors"


@pytest.mark.parametrize("size", [4, 16, 64, 256])
def test_regular_hadamard(size):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]) / 2
    expected = h4
    while expected.shape[0] < size:
        expected = torch.kron(expected, h4)
    h = build_regular_hadamard(size)
    assert torch.equal(h, expected)
    assert torch.equal(h, h.T)
    assert torch.equal(h @ h.T, torch.eye(size))
    assert torch.equal(h @ h, torch.eye(size))


@pytest.mark.parametrize("size", [0, 2, 8, 15, 128])
def test_invalid_hadamard_size(size):
    with pytest.raises(ValueError, match="power of 4"):
        build_regular_hadamard(size)


def test_hadamard_cache_shares_one_build_across_threads(monkeypatch):
    monkeypatch.setattr(quant_ops, "_hadamard_cache", {})
    device = torch.device("cpu")

    def get_hadamard(_):
        return quant_ops._get_cached_hadamard(256, device, torch.float32)

    with patch.object(quant_ops, "build_regular_hadamard", wraps=build_regular_hadamard) as build:
        with ThreadPoolExecutor(max_workers=4) as executor:
            matrices = list(executor.map(get_hadamard, range(8)))
    build.assert_called_once_with(256, device, torch.float32)
    assert all(matrix is matrices[0] for matrix in matrices)


def test_group_divisibility_and_zero_row_quantization():
    x = torch.zeros(2, 9)
    with pytest.raises(ValueError, match="divisible"):
        rotate_activation(x, 4)
    with pytest.raises(ValueError, match="divisible"):
        rotate_weight(x, 4)
    quantized, scales = quantize_int8_rowwise(x)
    assert torch.equal(quantized, x.to(torch.int8))
    torch.testing.assert_close(scales, torch.full((2, 1), 1e-12 / 127), atol=0, rtol=0)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("convrot", [False, True])
@torch.no_grad()
def test_int8_linear_float_parity(device, convrot):
    generator = torch.Generator().manual_seed(41)
    weight = (torch.randn(32, 512, generator=generator) / 512**0.5).to(device)
    bias = torch.randn(32, generator=generator).to(device)
    h = build_regular_hadamard(256, device=device)
    rotated = rotate_weight(weight, hadamard=h) if convrot else weight
    weight_i8, scales = quantize_int8_rowwise(rotated)
    linear = Int8ConvRotLinear(512, 32, convrot=convrot, bias=True).to(device)
    linear.weight.copy_(weight_i8)
    linear.weight_scale.copy_(scales)
    linear.bias.copy_(bias)

    for m in (1, 5, 16, 17, 64):
        x = torch.randn(1, 512, m, generator=generator).to(device).transpose(-1, -2)
        if convrot:
            torch.testing.assert_close(rotate_activation(rotate_activation(x, hadamard=h), hadamard=h), x)
        expected = F.linear(x, weight, bias)
        torch.testing.assert_close(linear(x), expected, atol=0.047, rtol=0.015)


@pytest.mark.parametrize("device", _DEVICES)
@torch.inference_mode()
def test_int8_linear_dynamic_fullgraph(device, torch_compile_discovery):
    linear = Int8ConvRotLinear(256, 37, bias=True).to(device)
    linear.weight.random_(-127, 128)
    linear.weight_scale.fill_(0.01)
    linear.bias.normal_()
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(linear, backend=backend, fullgraph=True, dynamic=True)
    for m in (17, 31):
        x = torch.randn(1, m, 256, device=device)
        torch.testing.assert_close(compiled(x), linear(x), atol=0, rtol=0)
    assert len(graphs) == 1
    op = torch.ops.h3_infer.int8_convrot_linear
    assert any(node.op == "call_function" and node.target in (op, op.default) for node in graphs[0].graph.nodes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [256, 512])
@pytest.mark.parametrize("n", [32, 37, 256])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("convrot", [False, True])
@torch.inference_mode()
def test_triton_int8_linear_matches_eager(dtype, k, n, has_bias, convrot):
    pytest.importorskip("triton")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 unavailable")
    generator = torch.Generator(device="cuda").manual_seed(73)
    weight = torch.randn(n, k, device="cuda", generator=generator) / k**0.5
    bias = torch.randn(n, device="cuda", generator=generator) if has_bias else None
    h = build_regular_hadamard(256, device="cuda", dtype=dtype)
    weight_i8, scales = quantize_int8_rowwise(rotate_weight(weight, hadamard=h) if convrot else weight)
    weight_i8 = weight_i8.T.contiguous().T

    with (
        patch.object(
            quant_ops, "_cublas_int8_matmul_triton_dequant", wraps=quant_ops._cublas_int8_matmul_triton_dequant,
        ) as cublas_dequant,
        patch.object(quant_ops, "_triton_int8_matmul_dequant", wraps=quant_ops._triton_int8_matmul_dequant) as fused,
    ):
        for m in (1, 5, 16, 17, 64, 129):
            x = torch.randn(1, k, m, device="cuda", generator=generator).to(dtype).transpose(-1, -2)
            expected = int8_convrot_linear(x, weight_i8, scales, bias, convrot=convrot, hadamard=h)
            actual = int8_convrot_linear(x, weight_i8, scales, bias, convrot=convrot)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        if k % 8 == 0 and n % 8 == 0:
            assert cublas_dequant.call_count == 6
            assert fused.call_count == 0
        else:
            assert fused.call_count == 6
            assert cublas_dequant.call_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
def test_triton_dequant_int32_matches_eager(dtype, has_bias):
    pytest.importorskip("triton")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 unavailable")
    m, n = 17, 1029
    generator = torch.Generator(device="cuda").manual_seed(73)
    out_i32 = torch.randint(-2**20, 2**20, (24, n), device="cuda", dtype=torch.int32, generator=generator)
    a_scale = torch.rand(m, device="cuda", generator=generator) / 128
    b_scale = (torch.rand(n, device="cuda", generator=generator) / 128).to(dtype)
    bias = torch.randn(n, device="cuda", generator=generator).to(dtype) if has_bias else None

    actual = quant_ops._triton_dequant_int32(out_i32, a_scale, b_scale, bias, m=m, out_dtype=dtype)
    expected = out_i32[:m].float() * a_scale[:, None] * b_scale.float()
    if bias is not None:
        expected = expected + bias.float()
    torch.testing.assert_close(actual, expected.to(dtype), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_rowwise_quantization_across_tiles():
    pytest.importorskip("triton")
    x = torch.zeros(3, 1029, device="cuda")
    x[0, -5:] = torch.tensor([127, 2.5, 3.5, -2.5, -3.5], device=x.device)
    x[1, -1] = 2.6e-12
    quantized, scale = quant_ops._triton_quantize_rowwise(x)
    expected = torch.zeros_like(x, dtype=torch.int8)
    expected[0, -5:] = torch.tensor([127, 2, 4, -2, -4], device=x.device)
    expected[1, -1] = 3
    torch.testing.assert_close(quantized, expected, atol=0, rtol=0)
    torch.testing.assert_close(scale, x.new_tensor([1.0, 1e-12, 1e-12]), atol=0, rtol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_int8_unaligned_accumulation(device):
    weight = torch.arange(5, dtype=torch.int8, device=device).expand(2051, -1)
    scale = torch.full((weight.shape[0], 1), 0.5, device=device)
    bias = torch.ones(weight.shape[0], device=device)
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0]], dtype=torch.bfloat16, device=device).expand(2053, -1)
    result = int8_convrot_linear(x, weight, scale, bias, convrot=False)
    assert result.dtype == x.dtype
    torch.testing.assert_close(result, x.new_full(result.shape, 2.0), rtol=0, atol=0)


@pytest.mark.parametrize(("key", "bad_tensor", "message"), [
    ("weight", torch.zeros(8, 8), "weight has dtype F32, expected I8"),
    ("weight_scale", torch.zeros(8), "weight_scale has shape"),
    ("weight_scale", torch.zeros(8, 1, dtype=torch.float16), "weight_scale has dtype F16, expected F32"),
])
def test_strict_quantized_tensor_validation(tmp_path, key, bad_tensor, message):
    module = Int8ConvRotLinear(8, 8, convrot=False)
    with torch.no_grad():
        module.weight.zero_()
        module.weight_scale.fill_(1)
    path = tmp_path / "invalid.safetensors"
    save_file({**module.state_dict(), key: bad_tensor}, path)
    with pytest.raises(ValueError, match=message):
        load_safetensors(module, path)
    assert torch.equal(module.weight, torch.zeros_like(module.weight))
    assert torch.equal(module.weight_scale, torch.ones_like(module.weight_scale))


@pytest.mark.parametrize("quantized", [False, True], ids=["float", "int8-convrot"])
@torch.inference_mode()
def test_qkv_layout_parity(quantized):
    torch.manual_seed(42)
    config = MiniMaxH3DiTConfig(
        hidden_size=32, num_layers=1, token_refiner_num_layers=1, num_attention_heads=2,
        attention_head_dim=16, ffn_hidden_size=64, latents_dim=4, text_dim=16,
        timestep_input_dim=16, time_embed_hidden_size=32, time_embed_dim=16,
        adaln_out_features=576, final_adaln_out_features=64, rope_inv_freq_len=2,
    )
    projections = ("blocks.0.attn.qkv_proj", "token_refiner.blocks.0.attn.qkv_proj")
    interleaved = MiniMaxH3DiT(config, attention_backend="sdpa", qkv_layout="interleaved")
    state = interleaved.state_dict()
    quant_map = None
    if quantized:
        meta = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}
        quant_map = dict.fromkeys(projections, meta)
        for name in projections:
            rotated = rotate_weight(state[f"{name}.weight"], group_size=16)
            state[f"{name}.weight"], state[f"{name}.weight_scale"] = quantize_int8_rowwise(rotated)
        interleaved = MiniMaxH3DiT(config, attention_backend="sdpa", qkv_layout="interleaved",
                                  quant_map=quant_map)
        interleaved.load_state_dict(state)

    split = MiniMaxH3DiT(config, attention_backend="sdpa", qkv_layout="split", quant_map=quant_map)
    qkv_shape = (config.num_attention_heads, 3, config.attention_head_dim)
    for name, tensor in state.items():
        if name.rsplit(".", 1)[0] in projections:
            state[name] = tensor.unflatten(0, qkv_shape).transpose(0, 1).flatten(0, 2)
    split.load_state_dict(state)

    positions, tags, img_pos, text_pos, _ = build_packed_sequence([0, 1, 1], [], 4, 4)
    rows = torch.randn(len(img_pos), config.patch_dim)
    embeds = torch.randn(len(text_pos), config.text_dim)
    heads = MiniMaxH3PDDHeads(1, config.hidden_size, config.patch_dim)
    args = (rows, embeds, positions[None], torch.tensor([0.5]), torch.zeros_like(tags),
            tags, img_pos, text_pos, heads)
    torch.testing.assert_close(split(*args), interleaved(*args), atol=0, rtol=0)


@pytest.fixture
def comfy_checkpoint(tmp_path):
    torch.manual_seed(71)
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    dit_config = MiniMaxH3DiTConfig(
        hidden_size=256, num_layers=3, token_refiner_num_layers=1, num_attention_heads=4,
        attention_head_dim=64, ffn_hidden_size=256, latents_dim=4, text_dim=16,
        time_embed_dim=16, adaln_out_features=4608, final_adaln_out_features=512, rope_inv_freq_len=2,
    )
    vae_config = MiniMaxH3VAEConfig(
        latent_channels=4, block_out_channels=(8, 8, 8), layers_per_block=1,
        spatial_downsample_factors=(2, 2, 2), temporal_downsample_factors=(1, 1, 1),
        norm_num_groups=2, vae_tile_size=32, vae_tile_overlap_min=8,
        vit_decoder_kwargs={"num_layers": 1, "heads": 2, "dim_head": 16},
    )
    (transformer / "config.json").write_text(json.dumps(asdict(dit_config)), encoding="utf-8")
    (vae_dir / "config.json").write_text(json.dumps(asdict(vae_config)), encoding="utf-8")
    save_file(MiniMaxH3ImageVAE(vae_config).state_dict(), vae_dir / "diffusion_pytorch_model.safetensors")

    dit = MiniMaxH3DiT(dit_config, adaln_curve_grid=3, adaln_curve_dim=2, qkv_layout="split").bfloat16()
    state = dit.state_dict()
    state["adaln_t_table"] = torch.tensor([[0.25, -0.5], [1.0, 0.5], [-0.75, 2.0]])
    h = build_regular_hadamard(256)
    for index in range(dit_config.num_layers):
        prefix = f"blocks.{index}"
        state[f"{prefix}.adaln_proj.linear.weight"] = state[f"{prefix}.adaln_proj.linear.weight"].half()
        state[f"{prefix}.attn.comfy_attention.config"] = torch.tensor([1], dtype=torch.uint8)
        for proj in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            layer = f"{prefix}.{proj}"
            convrot = proj != "mlp.fc2"
            weight = state[f"{layer}.weight"].float()
            if convrot:
                weight = rotate_weight(weight, hadamard=h)
            state[f"{layer}.weight"], state[f"{layer}.weight_scale"] = quantize_int8_rowwise(weight)
            meta = {"format": "int8_tensorwise", "convrot": convrot, "convrot_groupsize": 256}
            state[f"{layer}.comfy_quant"] = torch.tensor(list(json.dumps(meta).encode("utf-8")), dtype=torch.uint8)
    state["final_layer.adaln_proj.linear.weight"] = state["final_layer.adaln_proj.linear.weight"].half()

    heads = MiniMaxH3PDDHeads(6, dit_config.hidden_size, dit_config.patch_dim)
    with torch.no_grad():
        for name, param in heads.named_parameters():
            packed = torch.randint(-16, 16, param.shape, dtype=torch.float32) / 256
            param.copy_(packed)
            param[1:].add_(param[0])
            state[f"final_layer.{name}"] = packed.flatten(0, 1)
    sigmas = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0)
    path = tmp_path / "pruned_int8_convrot.safetensors"
    save_file(state, path, metadata={"config": json.dumps({"transformer": {"pdd_steps": 6, "pdd_sigmas": sigmas}})})
    return path, heads, sigmas


@torch.inference_mode()
def test_comfy_curve_and_embedded_heads(comfy_checkpoint):
    path, standalone_heads, sigmas = comfy_checkpoint
    inspection = inspect_comfy_checkpoint(path)
    assert inspection.adaln_curve_shape == (3, 2)
    assert inspection.video_out_shape == (96, 256)
    assert len(inspection.quant_map) == 12
    assert inspection.quant_map["blocks.0.attn.qkv_proj"]["convrot_groupsize"] == 256
    assert inspection.pdd_config.sigmas == sigmas
    pipe = MiniMaxH3ImagePipeline.from_pretrained(path.parent, dit_checkpoint=path, attention_backend="sdpa")
    dit = pipe.dit
    assert pipe.scheduler.config.sigmas == sigmas
    assert dit.time_embedder is None
    assert isinstance(dit.blocks[0].attn.qkv_proj, Int8ConvRotLinear)
    assert dit.blocks[0].attn.qkv_layout == "split"
    assert dit.token_refiner.blocks[0].attn.qkv_layout == "split"
    assert dit.token_refiner.blocks[0].mlp.fc1.weight.dtype == torch.bfloat16
    assert not any(tensor.is_meta for tensor in dit.state_dict().values())

    times = torch.tensor([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    first, middle, last = dit.adaln_t_table
    expected = torch.stack((first, first, (first + middle) / 2, middle, (middle + last) / 2, last, last))
    curve = dit._eval_adaln_curve(times)
    assert torch.equal(curve, expected)
    for proj in (dit.blocks[0].adaln_proj, dit.final_layer.adaln_proj):
        assert not proj.apply_silu
        weight, bias = proj.linear.weight, proj.linear.bias
        assert (weight.dtype, bias.dtype) == (torch.float16, torch.bfloat16)
        projected = F.linear(curve.half(), weight, bias.half()).reshape(-1, proj.hidden_size * proj.chunks)
        assert torch.equal(torch.cat(proj(curve), dim=-1), projected)

    hidden = torch.randn(5, dit.hidden_size)
    for step in range(6):
        assert torch.equal(pipe.pdd_heads(hidden, step), standalone_heads(hidden, step))
    latents = pipe(torch.randn(3, dit.config.text_dim), height=32, width=32,
                   min_generation_area=0, output_type="latent")
    assert latents.shape == (1, 4, 1, 4, 4) and torch.isfinite(latents).all()


def _packed_args(pipe):
    device = pipe.device
    positions, tags, img_pos, text_pos, ref_count = build_packed_sequence([0, 1, 1], [(2, 2)], 4, 4)
    indices = torch.zeros_like(tags)
    indices[img_pos[:ref_count]] = 1
    config = pipe.dit.config
    rows = torch.randn(len(img_pos), config.patch_dim)
    embeds = torch.randn(len(text_pos), config.text_dim)
    model_times = pipe.scheduler.model_timesteps

    def at_step(step):
        times = torch.stack((model_times[step], model_times.new_tensor(REFERENCE_TIMESTEP)))
        return (*(t.to(device) for t in (rows, embeds, positions[None], times, indices, tags, img_pos, text_pos)),
                pipe.pdd_heads, step)

    return at_step


@pytest.mark.parametrize("device", _DEVICES)
@torch.inference_mode()
def test_comfy_forward_prune_offload_and_compile(comfy_checkpoint, torch_compile_discovery, device):
    path, _, sigmas = comfy_checkpoint
    pipe = MiniMaxH3ImagePipeline.from_pretrained(path.parent, dit_checkpoint=path, device=device,
                                                 attention_backend="sdpa", staged_vram=False)
    dit = pipe.dit
    args = _packed_args(pipe)
    expected = [dit(*args(step)) for step in range(6)]
    assert all(torch.isfinite(out).all() for out in expected)

    compiled = torch.compile(dit, fullgraph=True, backend="eager")
    torch.testing.assert_close(compiled(*args(0)), expected[0], atol=0, rtol=0)
    dit.prune_adaln_tables(sigmas[:-1])
    assert dit.pruned_adaln and dit.adaln_t_table is None
    assert "adaln_t_table" not in dit.state_dict()
    for step, out in enumerate(expected):
        torch.testing.assert_close(dit(*args(step)), out, atol=0, rtol=0)

    streamed = MiniMaxH3ImagePipeline.from_pretrained(
        path.parent, dit_checkpoint=path, device=device, attention_backend="sdpa", staged_vram=True,
        offload_dit=True, resident_blocks=1,
    )
    assert streamed.dit.pruned_adaln
    offloader = streamed.dit._offloader
    assert isinstance(offloader, AsyncBlockOffloader) and offloader.resident_blocks == 1
    streamed.dit.compile("regional", fullgraph=True, backend="eager")
    try:
        for step in (0, 5):
            torch.testing.assert_close(streamed.dit(*args(step)), expected[step], atol=0, rtol=0)
        assert offloader.prepared
        if device == "cuda":
            assert offloader.num_resident == 1 and len(offloader.slot_modules) == 2
    finally:
        streamed.dit.release_gpu()


@pytest.mark.parametrize("device", _DEVICES)
@torch.inference_mode()
def test_offloader_preserves_each_blocks_rotation(device):
    blocks = nn.ModuleList([nn.Identity(), Int8ConvRotLinear(256, 256, convrot=False),
                            Int8ConvRotLinear(256, 256, convrot=True)])
    for block in blocks[1:]:
        block.weight.copy_(torch.eye(256, dtype=torch.int8))
        block.weight_scale.fill_(1.0)
    x = torch.linspace(-1, 1, 256, device=device).reshape(1, -1)
    expected = x
    for block in deepcopy(blocks).to(device):
        expected = block(expected)

    offloader = AsyncBlockOffloader(blocks, device, resident_blocks=1)
    try:
        torch.testing.assert_close(offloader.run_blocks(run_block, x), expected, atol=0, rtol=0)
        if device == "cuda":
            assert offloader.num_resident == 1 and not offloader.slot_modules
    finally:
        offloader.release_gpu()


@pytest.mark.parametrize("location", ["file", "root", "transformer"])
def test_comfy_checkpoint_discovery(comfy_checkpoint, location):
    path, _, sigmas = comfy_checkpoint
    root = path.parent
    if location == "file":
        source = path
    elif location == "root":
        path.rename(root / _AUTO_CHECKPOINT)
        source = root
    else:
        path.rename(root / "transformer" / _AUTO_CHECKPOINT)
        source = root
    pipe = MiniMaxH3ImagePipeline.from_pretrained(source, dtype_override=torch.float64, prune_adaln=True)
    assert pipe.dit.pruned_adaln and pipe.scheduler.config.sigmas == sigmas
    assert pipe.dit.blocks[0].attn.qkv_proj.weight.dtype == torch.int8
    assert pipe.dit.blocks[0].attn.qkv_proj.weight_scale.dtype == torch.float32
    assert pipe.pdd_heads.dtype == torch.float64


def test_comfy_explicit_pdd_files_take_precedence(comfy_checkpoint):
    path, heads, _ = comfy_checkpoint
    root = path.parent
    config = MiniMaxH3PDDConfig()
    (root / "pdd_config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    weights = root / "pdd_fused_heads.safetensors"
    save_file({name: -tensor for name, tensor in heads.state_dict().items()}, weights)
    pipe = MiniMaxH3ImagePipeline.from_pretrained(root, dit_checkpoint=path, pdd_heads_path=weights)
    assert pipe.scheduler.config == config
    assert torch.equal(pipe.pdd_heads.video_out.weight, -heads.video_out.weight)
