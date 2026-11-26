"""Scheduler, strict checkpoint loading, attention fallback, and CLI smoke tests."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from h3_infer.config import MiniMaxH3DiTConfig, MiniMaxH3PDDConfig, MiniMaxH3VAEConfig
from h3_infer.models.dit import MiniMaxH3DiT, MiniMaxH3PDDHeads
from h3_infer.models.vae import MiniMaxH3ImageVAE
from h3_infer.ops.attention import attention
from h3_infer.pipeline import MiniMaxH3ImagePipeline, get_canvas_size
from h3_infer.runtime.colab import detect_gpu, staging_paths
from h3_infer.runtime.loader import load_safetensors
from h3_infer.runtime.scheduler import MiniMaxH3Scheduler


def test_euler_parity(diffsynth_reference):
    _, load = diffsynth_reference
    reference = load("diffusion.flow_match").FlowMatchScheduler("MiniMax-H3-Image")
    reference.set_timesteps(6, distilled=True)
    scheduler = MiniMaxH3Scheduler()
    assert scheduler.sigmas[-1].item() == 0
    assert torch.equal(scheduler.sigmas[:-1], reference.sigmas)
    assert torch.equal(scheduler.timesteps, reference.timesteps)
    current = expected = torch.randn(1, 4, 1, 8, 6, generator=torch.Generator().manual_seed(4))
    for i, timestep in enumerate(reference.timesteps):
        velocity = current.sin() + 0.1 * i
        expected = reference.step(-velocity, timestep, expected)
        current = scheduler.step(velocity, i, current)
        assert torch.equal(current, expected)
    torch.testing.assert_close(scheduler.predict_x0(velocity, 0, current), current + velocity)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def tiny_checkpoint(tmp_path):
    root = tmp_path / "model"
    transformer = root / "transformer"
    vae_dir = root / "vae"
    transformer.mkdir(parents=True)
    vae_dir.mkdir()
    dit_config = MiniMaxH3DiTConfig(
        hidden_size=32, num_layers=1, token_refiner_num_layers=1, num_attention_heads=2,
        attention_head_dim=16, ffn_hidden_size=64, latents_dim=4, text_dim=16,
        timestep_input_dim=16, time_embed_hidden_size=32, time_embed_dim=16,
        adaln_out_features=576, final_adaln_out_features=64, rope_inv_freq_len=2,
    )
    vae_config = MiniMaxH3VAEConfig(
        latent_channels=4, block_out_channels=(8, 8, 8), layers_per_block=1,
        spatial_downsample_factors=(2, 2, 2), temporal_downsample_factors=(1, 1, 1),
        norm_num_groups=2, vae_tile_size=32, vae_tile_overlap_min=8,
        vit_decoder_kwargs={"num_layers": 1, "heads": 2, "dim_head": 16},
    )
    _write_json(transformer / "config.json", {"_class_name": "ignored", **asdict(dit_config)})
    _write_json(vae_dir / "config.json", asdict(vae_config))
    _write_json(root / "pdd_config.json", asdict(MiniMaxH3PDDConfig()))

    dit = MiniMaxH3DiT(dit_config)
    for key, child in dit.named_children():
        if key not in ("video_patch_proj", "time_embedder"):
            child.bfloat16()
    heads = MiniMaxH3PDDHeads(6, dit_config.hidden_size, dit_config.patch_dim)
    vae = MiniMaxH3ImageVAE(vae_config)
    state = dit.state_dict()
    names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    weights = {key: names[i % len(names)] for i, key in enumerate(state)}
    extra = "blocks.0.attn.comfy_attention.config"
    weights[extra] = names[0]
    for name in names:
        shard = {key: tensor for key, tensor in state.items() if weights[key] == name}
        if name == names[0]:
            shard[extra] = torch.tensor([1])
        save_file(shard, transformer / name)
    _write_json(transformer / "model.safetensors.index.json", {"weight_map": weights})
    save_file(heads.state_dict(), root / "pdd_fused_heads.safetensors")
    save_file({**vae.state_dict(), "decoder.mask_token": torch.zeros(1, 1, 32)},
              vae_dir / "diffusion_pytorch_model.safetensors")
    return root, dit, heads, vae


def test_strict_sharded_loading(tiny_checkpoint):
    root, dit, heads, vae = tiny_checkpoint
    pipe = MiniMaxH3ImagePipeline.from_pretrained(root, attention_backend="sdpa")
    for loaded, original in ((pipe.dit, dit), (pipe.pdd_heads, heads), (pipe.vae, vae)):
        left, right = loaded.state_dict(), original.state_dict()
        assert set(left) == set(right)
        for key in right:
            assert left[key].dtype == right[key].dtype
            assert left[key].device.type == "cpu"
            assert torch.equal(left[key], right[key])
    assert pipe.vae.latents_std.device.type == "cpu"

    index = root / "transformer/model.safetensors.index.json"
    with pytest.raises(ValueError, match=r"missing_keys=\[\], unexpected_keys=.*comfy_attention.config"):
        load_safetensors(dit, index)
    with pytest.raises(ValueError, match="decoder.mask_token"):
        load_safetensors(vae, root / "vae/diffusion_pytorch_model.safetensors")

    with torch.device("meta"):
        cast = MiniMaxH3PDDHeads(6, dit.config.hidden_size, dit.config.patch_dim)
    load_safetensors(cast, root / "pdd_fused_heads.safetensors", dtype_override=torch.float64, device="cpu")
    assert cast.video_out.weight.dtype == torch.float64
    assert torch.equal(cast.video_out.weight, heads.video_out.weight.double())


def test_loader_rejects_wrong_shard_or_shape_before_loading(tmp_path):
    module = torch.nn.Linear(3, 2)
    originals = {k: v.clone() for k, v in module.state_dict().items()}
    save_file({"weight": torch.ones(2, 3)}, tmp_path / "one.safetensors")
    save_file({"bias": torch.ones(1)}, tmp_path / "two.safetensors")
    index = tmp_path / "model.safetensors.index.json"
    _write_json(index, {"weight_map": {"weight": "one.safetensors", "bias": "two.safetensors"}})
    with pytest.raises(ValueError, match="bias has shape"):
        load_safetensors(module, index)
    _write_json(index, {"weight_map": {"bias": "one.safetensors", "weight": "two.safetensors"}})
    with pytest.raises(ValueError, match=r"missing_keys=\['bias'\], unexpected_keys=\['weight'\]"):
        load_safetensors(module, index)
    for key, value in module.state_dict().items():
        assert torch.equal(value, originals[key])


@pytest.mark.parametrize("backend", ["auto", "flash_attn", "sageattention", "sdpa"])
def test_dense_and_packed_cpu_attention(backend):
    q, k, v = torch.randn(3, 10, 2, 8)
    scale = 0.25
    expected = []
    for start, stop in ((0, 4), (4, 10)):
        parts = [value[start:stop].transpose(0, 1)[None] for value in (q, k, v)]
        ref = torch.nn.functional.scaled_dot_product_attention(*parts, scale=scale)[0].transpose(0, 1)
        expected.append(ref)
        dense = attention(q[None, start:stop], k[None, start:stop], v[None, start:stop],
                          softmax_scale=scale, backend=backend)
        assert torch.equal(dense[0], ref)
    packed = attention(q, k, v, cu_seqlens=torch.tensor([0, 4, 4, 10], dtype=torch.int32),
                       max_seqlen=6, softmax_scale=scale, backend=backend)
    assert torch.equal(packed, torch.cat(expected))


def test_pipeline_and_parity_cli(tiny_checkpoint, tmp_path):
    root, dit, _, vae = tiny_checkpoint
    pipe = MiniMaxH3ImagePipeline.from_pretrained(root, attention_backend="sdpa")
    assert get_canvas_size(1024, 1024) == (2048, 2048)
    assert get_canvas_size(640, 960) == (1664, 2496)
    embeds = torch.randn(1, 3, dit.config.text_dim)
    path = tmp_path / "text_embeds.pt"
    tags = torch.tensor([1, 0, 1])
    torch.save({"prompt_embeds": embeds, "text_token_tags": tags}, path)
    noise = torch.randn(1, 4, 1, 12, 12)
    ref_latents = torch.randn(1, 4, 1, 4, 6)
    ref_path = tmp_path / "edit.pt"
    torch.save(ref_latents, ref_path)
    observed = {"initial_latents": noise}

    def collect(step, latents, velocity):
        if step == 0:
            observed["step_1_latents"] = latents
            observed["step_1_velocity"] = -velocity

    size = {"height": 47, "width": 53, "min_generation_area": 9216, "latents": noise}
    final, stats = pipe(embeds, text_token_tags=tags, ref_latents=ref_latents,
                        step_callback=collect, return_stats=True, output_type="latent", **size)
    t2i = pipe(path, output_type="latent", **size)
    assert not torch.equal(t2i, final)
    assert len(stats.step_ms) == 6 and stats.peak_vram_bytes is None
    observed["final_latents"] = final
    reference_path = tmp_path / "reference.pt"
    torch.save(observed, reference_path)
    image_path = tmp_path / "result.png"
    result = subprocess.run(
        [sys.executable, "scripts/colab_bf16_parity.py", "--model-dir", str(root),
         "--prompt-embeds", str(path), "--reference", str(reference_path), "--ref-latents", str(ref_path),
         "--output-image", str(image_path), "--height", "47", "--width", "53",
         "--min-generation-area", "9216", "--device", "cpu", "--attention", "sdpa"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    report = json.loads(result.stdout)
    assert len(report["per_step_latency_ms"]) == 6
    for key in ("single_step", "six_step", "single_step_velocity"):
        assert report[key]["max_abs_diff"] < 1e-5
        assert report[key]["cosine_similarity"] > 0.999999
    with torch.no_grad():
        latents = final[:, :, 0] * vae.latents_std + vae.latents_mean
        decoded = vae.decode(latents)[0].permute(1, 2, 0)
    assert decoded.shape == (96, 96, 3)
    pixels = (decoded * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406]))
    quantized = (pixels.clamp(0, 1) * 255).to(dtype=torch.uint8).numpy()
    expected = Image.fromarray(quantized).resize((64, 64), Image.Resampling.LANCZOS)
    with Image.open(image_path) as image:
        np.testing.assert_array_equal(np.array(image), np.array(expected))
    as_tensor = pipe(path, ref_latents=ref_latents, output_type="tensor", **size)
    assert as_tensor.dtype == torch.uint8 and as_tensor.shape == (1, 3, 64, 64)
    np.testing.assert_array_equal(as_tensor[0].permute(1, 2, 0).numpy(), np.array(expected))


def test_colab_paths_and_cpu_detection(tmp_path):
    content = tmp_path / "content"
    assert staging_paths(content).models == Path("/tmp/models")
    drive = content / "drive/MyDrive"
    drive.mkdir(parents=True)
    paths = staging_paths(content)
    assert paths.models == content / "models"
    assert paths.triton_cache.is_relative_to(drive)
    assert paths.inductor_cache.is_relative_to(drive)
    cpu = detect_gpu("cpu")
    assert (cpu.compute_capability, cpu.vram_gb, cpu.supports_bf16, cpu.supports_fp8, cpu.attention_backend) == (
        None, 0, False, False, "sdpa",
    )


@pytest.mark.parametrize(("major", "minor", "bf16", "fp8"), [
    (7, 5, False, False), (8, 0, True, False), (8, 9, True, True), (9, 0, True, True),
])
def test_colab_cuda_capabilities(monkeypatch, major, minor, bf16, fp8):
    properties = SimpleNamespace(name="Test GPU", major=major, minor=minor, total_memory=24_000_000_000)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    monkeypatch.setattr("h3_infer.ops.attention._optional_kernel", lambda backend, packed: object())
    result = detect_gpu("cuda:0")
    assert result.compute_capability == f"sm_{major}{minor}"
    assert result.vram_gb == 24
    assert result.supports_bf16 is bf16 and result.supports_fp8 is fp8
    assert result.attention_backend == ("sageattention" if bf16 else "sdpa")


def test_official_config_defaults():
    root = Path("/Users/garry/ColabUltra/models/minimax-h3-image/bf16")
    if not (root / "scheduler/scheduler_config.json").is_file():
        pytest.skip("release configs unavailable")
    for name, config in (
        ("transformer/config.json", MiniMaxH3DiTConfig),
        ("vae/config.json", MiniMaxH3VAEConfig),
        ("pdd_config.json", MiniMaxH3PDDConfig),
    ):
        path = root / name
        assert config.from_json(path) == config()
        actual = json.loads(json.dumps(asdict(config())))
        released = json.loads(path.read_text(encoding="utf-8"))
        assert all(actual[key] == value for key, value in released.items() if not key.startswith("_"))
    released_scheduler = json.loads((root / "scheduler/scheduler_config.json").read_text(encoding="utf-8"))
    assert released_scheduler["euler_update_form"] == "velocity"
    assert released_scheduler["shift_provided_sigmas"] is False
