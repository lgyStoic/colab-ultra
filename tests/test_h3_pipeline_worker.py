"""Exercise the H3 worker with offline queues and mocked model computation."""

import json
import os
import pickle
import subprocess
import sys
import tarfile
from contextlib import contextmanager, nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from colab_ultra.remote import models, queue, recipe
from recipes import h3_pipeline_worker as worker
from test_queue import FakeRedis


@pytest.fixture(autouse=True)
def worker_environment(monkeypatch, tmp_path):
    for name in ("COLAB_ULTRA_WORKER_ROLE", "COLAB_ULTRA_WORKER_ID", "COLAB_ULTRA_SESSION",
                 "COLAB_ULTRA_REDIS_URL", "COLAB_ULTRA_VAE_DEVICE", "COLAB_ULTRA_DRIVE_TOKEN", "DRIVE_ACCESS_TOKEN",
                 "COLAB_ULTRA_DIT_VARIANT", "COLAB_ULTRA_STAGED_MODELS", "COLAB_ULTRA_KAGGLE", "CU_DRY_RUN",
                 "COLAB_ULTRA_TE_OFFLOAD_LAYERS", "COLAB_ULTRA_VAE_TILE_BATCH_SIZE",
                 "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(worker, "WORKER_ENV_FILE", tmp_path / ".cu_worker_env.json")
    monkeypatch.setattr(worker, "REDIS_URL_FILE", tmp_path / ".cu_redis_url")


@pytest.mark.parametrize(("own", "upstreams", "expected"), [
    ({"unclaimed": 0, "pending": 2}, [], True),
    ({}, [{"all_done": True}], True),
    ({"unclaimed": 1}, [{"all_done": True}], False),
    ({"unclaimed": 0}, [{"all_done": True}, {"all_done": False}], False),
    ({"unclaimed": 0}, [{}], False),
])
def test_should_exit_worker(own, upstreams, expected):
    assert worker.should_exit_worker(own, upstreams) is expected


def test_loop_requires_two_consecutive_eligible_empty_polls(monkeypatch):
    settings = {**worker.DEFAULTS, "poll_interval_sec": 0.25, "reclaim_idle_ms": 1234}
    sleep = Mock()
    monkeypatch.setattr(worker.time, "sleep", sleep)
    own, embed, dit = Mock(), Mock(), Mock()
    polls = [
        (False, True, 0),
        (True, True, 0),
        (True, False, 0),
        (True, True, 0),
        (True, True, 1),
        (True, True, 0),
        (True, True, 0),
    ]
    own.claim_batch.side_effect = [[] for _ in polls]
    embed.status.side_effect = [{"all_done": e} for e, _, _ in polls]
    dit.status.side_effect = [{"all_done": d} for _, d, _ in polls]
    own.status.side_effect = [{"unclaimed": n} for _, _, n in polls]
    process = Mock()

    assert worker._run_loop("vae", "cpu", {"embed": embed, "dit": dit, "vae": own},
                            settings, process, str) == 0
    assert own.claim_batch.call_args_list == [call("cpu", count=1, reclaim_idle_ms=1234, block_ms=250)] * len(polls)
    assert embed.status.call_count == dit.status.call_count == own.status.call_count == len(polls)
    assert sleep.call_args_list == [call(0.25)] * (len(polls) - 1)
    own.set_worker_status.assert_called_once_with("cpu", "stopped")
    process.assert_not_called()


def test_resolve_redis_url_precedence(monkeypatch, tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / ".redis_url").write_text(" redis://drive.example\n", encoding="utf-8")
    worker.REDIS_URL_FILE.write_text(" https://temp-token@temp.example\n", encoding="utf-8")
    monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", " rediss://env.example\n")

    assert worker.resolve_redis_url(code) == "rediss://env.example"
    monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", " \n")
    assert worker.resolve_redis_url(code) == "https://temp-token@temp.example"
    worker.REDIS_URL_FILE.unlink()
    assert worker.resolve_redis_url(code) == "redis://drive.example"
    (code / ".redis_url").unlink()
    with pytest.raises(ValueError, match="COLAB_ULTRA_REDIS_URL"):
        worker.resolve_redis_url(code)


def test_resolve_worker_precedence(monkeypatch):
    worker.WORKER_ENV_FILE.write_text(json.dumps({"role": "vae", "worker_id": "temp"}), encoding="utf-8")
    monkeypatch.setenv("COLAB_ULTRA_WORKER_ROLE", "dit")
    monkeypatch.setenv("COLAB_ULTRA_WORKER_ID", "env")
    monkeypatch.setenv("COLAB_ULTRA_SESSION", "session")
    monkeypatch.setattr(worker.socket, "gethostname", lambda: "host")

    assert worker.resolve_worker("embed", "cli") == ("embed", "cli")
    assert worker.resolve_worker() == ("dit", "env")
    monkeypatch.delenv("COLAB_ULTRA_WORKER_ROLE")
    assert worker.resolve_worker() == ("vae", "env")
    monkeypatch.delenv("COLAB_ULTRA_WORKER_ID")
    assert worker.resolve_worker() == ("vae", "temp")
    worker.WORKER_ENV_FILE.unlink()
    assert worker.resolve_worker("embed") == ("embed", "session")
    monkeypatch.delenv("COLAB_ULTRA_SESSION")
    assert worker.resolve_worker("dit") == ("dit", "host")
    with pytest.raises(ValueError, match="worker role"):
        worker.resolve_worker()


def test_settings_embed_directory_and_dit_batch_size(tmp_path):
    config = tmp_path / "pipeline_worker.json"
    settings = worker._settings(tmp_path)
    assert settings["embeds_dir"] == "runs/h3_embeds_cache"
    assert settings["dit_batch_size"] == 4

    config.write_text(json.dumps({"embeds_dir": "runs/custom/", "dit_batch_size": "2"}), encoding="utf-8")
    settings = worker._settings(tmp_path)
    assert settings["embeds_dir"] == "runs/custom/"
    assert settings["dit_batch_size"] == 2

    config.write_text('{"dit_batch_size": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="dit_batch_size"):
        worker._settings(tmp_path)


def test_dit_variant_setting_precedence(monkeypatch, tmp_path):
    config = tmp_path / "pipeline_worker.json"
    assert worker._settings(tmp_path)["dit_variant"] == "bf16"
    config.write_text('{"dit_variant": "bf16"}', encoding="utf-8")
    assert worker._settings(tmp_path)["dit_variant"] == "bf16"
    config.write_text('{"dit_variant": "int8_convrot"}', encoding="utf-8")
    assert worker._settings(tmp_path)["dit_variant"] == "int8_convrot"
    monkeypatch.setenv("COLAB_ULTRA_DIT_VARIANT", " bf16\n")
    assert worker._settings(tmp_path)["dit_variant"] == "bf16"
    monkeypatch.setenv("COLAB_ULTRA_DIT_VARIANT", " \n")
    assert worker._settings(tmp_path)["dit_variant"] == "int8_convrot"


@pytest.mark.parametrize("source", ["config", "env"])
def test_settings_reject_invalid_dit_variant(monkeypatch, tmp_path, source):
    if source == "config":
        (tmp_path / "pipeline_worker.json").write_text('{"dit_variant": "int8"}', encoding="utf-8")
    else:
        monkeypatch.setenv("COLAB_ULTRA_DIT_VARIANT", "int8")
    with pytest.raises(ValueError, match="^dit_variant must be bf16 or int8_convrot$"):
        worker._settings(tmp_path)


@pytest.mark.parametrize("path", ["../cache", "runs/../cache", "/runs/cache", "runs/cache\n"])
def test_settings_rejects_unsafe_embed_directory(tmp_path, path):
    (tmp_path / "pipeline_worker.json").write_text(json.dumps({"embeds_dir": path}), encoding="utf-8")
    with pytest.raises(ValueError, match="embeds_dir"):
        worker._settings(tmp_path)


def test_vae_device_setting_precedence(monkeypatch, tmp_path):
    config = tmp_path / "pipeline_worker.json"
    assert worker._settings(tmp_path)["vae_device"] == "auto"
    config.write_text('{"vae_device": "cuda:2"}', encoding="utf-8")
    assert worker._settings(tmp_path)["vae_device"] == "cuda:2"
    worker.WORKER_ENV_FILE.write_text('{"vae_device": "cpu"}', encoding="utf-8")
    assert worker._settings(tmp_path)["vae_device"] == "cpu"
    monkeypatch.setenv("COLAB_ULTRA_VAE_DEVICE", "cuda")
    assert worker._settings(tmp_path)["vae_device"] == "cuda"


def test_offload_and_tile_settings_precedence(monkeypatch, tmp_path):
    config = tmp_path / "pipeline_worker.json"

    def configured_values():
        settings = worker._settings(tmp_path)
        return settings["te_offload_layers"], settings["vae_tile_batch_size"]

    assert configured_values() == (0, None)
    config.write_text('{"te_offload_layers": "9", "vae_tile_batch_size": "4"}', encoding="utf-8")
    assert configured_values() == (9, 4)
    monkeypatch.setenv("COLAB_ULTRA_TE_OFFLOAD_LAYERS", " ")
    monkeypatch.setenv("COLAB_ULTRA_VAE_TILE_BATCH_SIZE", " ")
    assert configured_values() == (9, 4)
    monkeypatch.setenv("COLAB_ULTRA_TE_OFFLOAD_LAYERS", " 50 ")
    monkeypatch.setenv("COLAB_ULTRA_VAE_TILE_BATCH_SIZE", " 1 ")
    assert configured_values() == (50, 1)


@pytest.mark.parametrize(("key", "value", "message"), [
    ("te_offload_layers", -1, "te_offload_layers must be between 0 and 50"),
    ("te_offload_layers", 51, "te_offload_layers must be between 0 and 50"),
    ("vae_tile_batch_size", 0, "vae_tile_batch_size must be positive"),
    ("vae_tile_batch_size", -1, "vae_tile_batch_size must be positive"),
])
@pytest.mark.parametrize("from_env", [False, True])
def test_settings_reject_invalid_offload_and_tile_values(monkeypatch, tmp_path, key, value, message, from_env):
    if from_env:
        monkeypatch.setenv(f"COLAB_ULTRA_{key.upper()}", str(value))
    else:
        (tmp_path / "pipeline_worker.json").write_text(json.dumps({key: value}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        worker._settings(tmp_path)


@pytest.mark.parametrize("device", ["gpu", "cuda:-1", "cuda:1.5", "mps", None])
def test_settings_reject_invalid_vae_device(tmp_path, device):
    (tmp_path / "pipeline_worker.json").write_text(json.dumps({"vae_device": device}), encoding="utf-8")
    with pytest.raises(ValueError, match="vae_device"):
        worker._settings(tmp_path)


@pytest.mark.parametrize(("requested", "gpu", "capability", "memory", "override", "device", "dtype", "tiles"), [
    ("auto", True, (8, 9), 22, None, "cuda", "bfloat16", 64),
    ("cuda:1", True, (7, 5), 15, None, "cuda:1", "float32", 16),
    ("auto", False, None, None, None, "cpu", "float32", 4),
    ("cpu", True, None, None, None, "cpu", "float32", 4),
    ("cuda", True, (8, 9), 22, 4, "cuda", "bfloat16", 4),
])
def test_load_vae_device_dtype_and_tiles(
    monkeypatch, tmp_path, capsys, requested, gpu, capability, memory, override, device, dtype, tiles,
):
    class Device(str):
        @property
        def type(self):
            return self.split(":")[0]

    cuda = SimpleNamespace(is_available=Mock(return_value=gpu),
                           get_device_capability=Mock(return_value=capability),
                           get_device_properties=Mock(return_value=SimpleNamespace(total_memory=(memory or 0) * 1024**3)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        device=lambda name: nullcontext() if name == "meta" else Device(name),
        cuda=cuda, bfloat16="torch.bfloat16", float32="torch.float32",
    ))
    config = {"vae_device": requested}
    if override is not None:
        config["vae_tile_batch_size"] = override
    (tmp_path / "pipeline_worker.json").write_text(json.dumps(config), encoding="utf-8")
    config_from_json, initialize, load = Mock(), Mock(), Mock()
    monkeypatch.setitem(sys.modules, "h3_infer", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "h3_infer.config", SimpleNamespace(
        MiniMaxH3VAEConfig=SimpleNamespace(from_json=config_from_json),
    ))
    monkeypatch.setitem(sys.modules, "h3_infer.models.vae", SimpleNamespace(MiniMaxH3ImageVAE=initialize))
    monkeypatch.setitem(sys.modules, "h3_infer.runtime.loader", SimpleNamespace(load_safetensors=load))

    vae, tile_batch_size = worker._load_vae(tmp_path, worker._settings(tmp_path))
    assert vae is initialize.return_value.eval.return_value and tile_batch_size == tiles
    config_from_json.assert_called_once_with(tmp_path / "vae/config.json")
    initialize.assert_called_once_with(config_from_json.return_value, attention_backend="sdpa", stats_device=device)
    load.assert_called_once_with(vae, tmp_path / "vae/diffusion_pytorch_model.safetensors",
                                 device=device, dtype_override=f"torch.{dtype}", ignore_keys=("decoder.mask_token",))
    if device.startswith("cuda"):
        cuda.get_device_capability.assert_called_once_with(device)
    else:
        cuda.get_device_capability.assert_not_called()
    if device.startswith("cuda") and override is None:
        cuda.get_device_properties.assert_called_once_with(device)
    else:
        cuda.get_device_properties.assert_not_called()
    assert capsys.readouterr().out == f"[VAE] device={device} dtype={dtype} tile_batch_size={tiles}\n"


@pytest.mark.parametrize(("versions", "missing"), [
    (("", "0.4.2", ""), ("transformers>=4.57.0", "safetensors>=0.4.3", "pillow")),
    (("4.57.0rc1", "0.4.3", "11.3.0"), ("transformers>=4.57.0",)),
    (("5.20.2", "0.4.3", "11.3.0"), ()),
])
def test_embed_dependency_versions(monkeypatch, versions, missing):
    installed = dict(zip(("transformers", "safetensors", "pillow"), versions, strict=True))
    monkeypatch.setattr(worker, "_package_version", installed.__getitem__)
    install = Mock()
    monkeypatch.setattr(worker, "_pip_install", install)
    worker._install_embed_dependencies()
    assert install.call_args_list == ([call(*missing)] if missing else [])


class EmbedModule(SimpleNamespace):
    def get_submodule(self, path):
        module = self
        for name in path.split(".") if path else ():
            module = getattr(module, name)
        return module


def _mock_embed_loader(monkeypatch, gpu_count, vram, legacy_rotary):
    device_stack = []
    optional_backends = ("accelerate", "bitsandbytes")
    for name in optional_backends:
        monkeypatch.delitem(sys.modules, name, raising=False)

    class Device(str):
        def __enter__(self):
            device_stack.append(self)

        def __exit__(self, *_exc):
            device_stack.pop()

    class ConfigRotary:
        def __init__(self, config):
            self.config = config
            self.device = device_stack[-1]

    class LegacyVisionRotary:
        def __init__(self, dim):
            assert dim == 36
            self.device = device_stack[-1]

    def new_config(**kwargs):
        for name in ("text_config", "vision_config"):
            kwargs[name] = SimpleNamespace(**kwargs[name])
        return SimpleNamespace(**kwargs)

    projections = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
                   "mlp": ("gate_proj", "up_proj", "down_proj")}

    def new_model(config):
        assert device_stack == ["meta"]
        assert all(sys.modules[name] is None for name in optional_backends)
        layers = []
        for _ in range(config.text_config.num_hidden_layers):
            layer = EmbedModule(input_layernorm=EmbedModule(), post_attention_layernorm=EmbedModule())
            for name, children in projections.items():
                setattr(layer, name, EmbedModule(**{
                    child: SimpleNamespace(in_features=256, out_features=3) for child in children
                }))
            layer.self_attn.q_norm = EmbedModule()
            layer.self_attn.k_norm = EmbedModule()
            layers.append(layer)
        text = EmbedModule(layers=layers, embed_tokens=EmbedModule(), rotary_emb=ConfigRotary(config.text_config))
        vision = EmbedModule(patch=EmbedModule(weight=SimpleNamespace(is_meta=True)))
        vision.named_parameters = lambda: [("patch.weight", vision.patch.weight)]
        vision.rotary_pos_emb = (LegacyVisionRotary(36) if legacy_rotary else ConfigRotary(config.vision_config))
        model = EmbedModule(
            config=config, language_model=text, visual=vision,
            named_parameters=lambda: [("embed", text.embed_tokens.weight), *vision.named_parameters()],
            named_buffers=lambda: [
                ("rotary", SimpleNamespace(is_meta=rotary.device == "meta"))
                for rotary in (text.rotary_emb, vision.rotary_pos_emb)
            ],
        )
        model.eval = Mock(return_value=model)
        return model

    def new_projection(in_features, out_features, **kwargs):
        assert device_stack == ["meta"]
        return EmbedModule(in_features=in_features, out_features=out_features, **kwargs)

    quant_key = "model.layers.22.self_attn.q_proj.comfy_quant"

    def read_tensor(key):
        if key == quant_key:
            return SimpleNamespace(tolist=lambda: list(b'{"convrot": false, "convrot_groupsize": 128}'))

        def materialize(*, device, dtype):
            data = Mock(device=device, dtype=dtype, is_meta=False)
            data.numel.return_value = 1 if key.startswith(("model.layers.0.", "model.layers.36.")) else 3
            data.reshape.return_value = data
            data.expand.return_value = data
            data.contiguous.return_value = data
            data.pin_memory.return_value = data
            return data

        return SimpleNamespace(to=materialize)

    def parameter(data, *, requires_grad):
        assert requires_grad is False
        return data

    reader = SimpleNamespace(keys=lambda: [quant_key], get_tensor=Mock(side_effect=read_tensor))
    safe_open = Mock(return_value=nullcontext(reader))
    processor = Mock()
    replace_linear = Mock(side_effect=new_projection)
    cuda = SimpleNamespace(
        is_available=lambda: gpu_count > 0, device_count=lambda: gpu_count,
        get_device_properties=Mock(return_value=SimpleNamespace(total_memory=vram * 1024**3)),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        device=Device, cuda=cuda, nn=SimpleNamespace(Identity=Mock(), Parameter=parameter),
        float16="float16", float32="float32", int8="int8",
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3VLConfig=new_config, Qwen3VLModel=new_model, AutoProcessor=SimpleNamespace(from_pretrained=processor),
    ))
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=safe_open))
    monkeypatch.setitem(sys.modules, "h3_infer.ops", SimpleNamespace(Int8ConvRotLinear=replace_linear))
    for name in ("_ensure_h3_import_path", "_install_embed_dependencies", "_install_embed_device_hooks"):
        monkeypatch.setattr(worker, name, Mock())
    return processor, safe_open, replace_linear, projections


@pytest.mark.parametrize(("gpu_count", "vram", "embed_device", "legacy_rotary", "requested_offload"), [
    (2, 15, "cpu", False, 0),
    (2, 15, "cpu", False, 14),
    (1, 31, "cpu", True, 0),
    (1, 32, "cuda:0", False, 0),
    (0, 0, "cpu", True, 14),
])
def test_load_embed_streams_int8_checkpoint(
    monkeypatch, tmp_path, capsys, gpu_count, vram, embed_device, legacy_rotary, requested_offload,
):
    processor, safe_open, replace_linear, projections = _mock_embed_loader(monkeypatch, gpu_count, vram, legacy_rotary)
    local_dir = tmp_path / "cache"
    monkeypatch.setattr(worker, "H3_CACHE_DIR", local_dir)
    monkeypatch.setattr(worker, "_ACTIVE_COMPILE_CACHE", (None, 0))
    monkeypatch.setattr(worker, "_compile_cache_archive", Mock(return_value=None))
    monkeypatch.setenv("TRITON_CACHE_DIR", "previous")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "previous")

    settings = {**worker.DEFAULTS, "te_offload_layers": requested_offload}
    model, loaded_processor = worker._load_embed(settings, tmp_path / "bf16", tmp_path / "te")
    assert all(name not in sys.modules for name in ("accelerate", "bitsandbytes"))
    assert local_dir.is_dir() == (gpu_count > 0)
    assert loaded_processor is processor.return_value
    processor.assert_called_once_with(str(tmp_path / "bf16/processor"))
    safe_open.assert_called_once_with(str(tmp_path / "te/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
                                      framework="pt", device="cpu")
    worker._ensure_h3_import_path.assert_called_once_with()
    worker._install_embed_dependencies.assert_called_once_with()
    model.eval.assert_called_once_with()
    assert model.language_model.norm is sys.modules["torch"].nn.Identity.return_value
    assert (model.config.image_token_id, model.config.video_token_id,
            model.config.vision_start_token_id, model.config.vision_end_token_id) == (151655, 151656, 151652, 151653)
    assert model.config.text_config.rope_theta == 5000000
    assert model.config.text_config.rope_scaling["mrope_section"] == [24, 20, 20]
    assert model.config.text_config.rope_scaling["mrope_interleaved"] is True
    assert model.config.vision_config.deepstack_visual_indexes == [8, 16, 24]

    dev0 = "cuda:0" if gpu_count else "cpu"
    dev1 = "cuda:1" if gpu_count >= 2 else None
    dtype = "float16" if gpu_count else "float32"
    text = model.language_model
    assert (text.embed_tokens.weight.device, text.embed_tokens.weight.dtype) == (embed_device, dtype)
    assert (model.visual.patch.weight.device, model.visual.patch.weight.dtype) == (dev0, dtype)
    assert text.rotary_emb.device == model.visual.rotary_pos_emb.device == dev0
    offload_layers = requested_offload if gpu_count else 0
    worker._install_embed_device_hooks.assert_called_once_with(text, embed_device, dev0, dev1,
                                                              offload_layers=offload_layers)
    assert capsys.readouterr().out == f"[Embed] split=22 offload_layers={offload_layers}\n"
    assert len(text.layers) == 50 and replace_linear.call_count == 350
    for idx, layer in enumerate(text.layers):
        offloaded = idx >= 50 - offload_layers
        target = "cpu" if offloaded else dev1 if dev1 and idx >= 22 else dev0
        for path in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            weight = layer.get_submodule(path).weight
            assert (weight.device, weight.dtype) == (target, dtype)
            assert weight.pin_memory.call_count == int(offloaded)
        for parent, children in projections.items():
            for child in children:
                linear = layer.get_submodule(f"{parent}.{child}")
                assert (linear.weight.device, linear.weight.dtype) == (target, "int8")
                assert (linear.weight_scale.device, linear.weight_scale.dtype) == (target, "float32")
                scalar_scale = idx in (0, 36)
                linear.weight_scale.reshape.assert_called_once_with(1 if scalar_scale else 3, 1)
                assert linear.weight_scale.expand.call_args_list == ([call(3, 1)] if scalar_scale else [])
                linear.weight_scale.contiguous.assert_called_once_with()
                assert linear.weight.pin_memory.call_count == int(offloaded)
                assert linear.weight_scale.pin_memory.call_count == int(offloaded)
    assert (text.layers[22].self_attn.q_proj.convrot, text.layers[22].self_attn.q_proj.convrot_groupsize) == (False, 128)
    assert (text.layers[21].self_attn.q_proj.convrot, text.layers[21].self_attn.q_proj.convrot_groupsize) == (True, 256)


def test_pin_cpu_tensor_falls_back_if_pinning_fails():
    tensor = Mock()
    tensor.pin_memory.side_effect = RuntimeError
    assert worker._pin_cpu_tensor(tensor, True) is tensor
    tensor.pin_memory.assert_called_once_with()


@pytest.mark.parametrize(("layer_count", "dev1", "targets"), [
    (50, "cuda:1", ("cuda:1", "cuda:1")),
    (23, "cuda:1", ("cuda:0", "cuda:1")),
    (50, None, ("cuda:0", "cuda:0")),
])
def test_embed_device_hooks_restore_offloaded_parameters(monkeypatch, layer_count, dev1, targets):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    layers = [Mock() for _ in range(layer_count)]
    cpu_tensors = [Mock(), Mock()]
    params = [SimpleNamespace(data=tensor) for tensor in cpu_tensors]
    for layer, param in zip(layers[-2:], params, strict=True):
        layer.named_parameters.return_value = [("weight", param)]
    text = Mock(layers=layers)
    worker._install_embed_device_hooks(text, "cpu", "cuda:0", dev1, offload_layers=2)

    for layer in layers[:-2]:
        layer.register_forward_hook.assert_not_called()
    for layer, param, cpu_tensor, target in zip(layers[-2:], params, cpu_tensors, targets, strict=True):
        load = next(entry.args[0] for entry in layer.register_forward_pre_hook.call_args_list if not entry.kwargs)
        release = layer.register_forward_hook.call_args.args[0]
        layer.register_forward_hook.assert_called_once_with(release, always_call=True)
        with pytest.raises(RuntimeError, match="forward failed"):
            try:
                load(layer, ())
                cpu_tensor.to.assert_called_once_with(target, non_blocking=True)
                assert param.data is cpu_tensor.to.return_value
                raise RuntimeError("forward failed")
            finally:
                release(layer, (), None)
        assert param.data is cpu_tensor


def test_embed_device_hooks_move_shared_inputs_once_per_forward(monkeypatch):
    class Tensor:
        def __init__(self, device):
            self.device = device
            self.to = Mock(side_effect=lambda target: self if target == device else Tensor(target))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Tensor=Tensor))
    text = Mock(layers=[Mock() for _ in range(50)])
    worker._install_embed_device_hooks(text, "cpu", "cuda:0", "cuda:1")
    lookup = text.embed_tokens.register_forward_pre_hook.call_args.args[0]
    back_to_cuda = text.embed_tokens.register_forward_hook.call_args.args[0]
    (cpu_ids,) = lookup(None, (Tensor("cuda:0"),))
    assert cpu_ids.device == "cpu"
    assert back_to_cuda(None, (), Tensor("cpu")).device == "cuda:0"
    for layer in text.layers[:22]:
        layer.register_forward_pre_hook.assert_not_called()

    rotary = (Tensor("cuda:0"), Tensor("cuda:0"))
    mask = Tensor("cuda:0")
    positions = Tensor("cuda:0")
    hidden = Tensor("cuda:0")
    kwargs = {"position_embeddings": rotary, "attention_mask": mask, "position_ids": positions,
              "past_key_values": None}

    def forward():
        previous_kwargs = None
        hidden_states = hidden
        for layer in text.layers[22:]:
            args, options = layer.register_forward_pre_hook.call_args
            assert options == {"with_kwargs": True}
            (hidden_states,), moved = args[0](layer, (hidden_states,), kwargs)
            assert hidden_states.device == "cuda:1"
            assert moved["past_key_values"] is None
            for value in (*moved["position_embeddings"], moved["attention_mask"], moved["position_ids"]):
                assert value.device == "cuda:1"
            if previous_kwargs is not None:
                assert moved == previous_kwargs
            previous_kwargs = moved
        return previous_kwargs

    first = forward()
    hidden.to.assert_called_once_with("cuda:1")
    for original in (*rotary, mask, positions):
        original.to.assert_called_once_with("cuda:1")
    clear, options = text.register_forward_hook.call_args
    assert options == {"always_call": True}
    clear[0](text, (), None)
    second = forward()
    assert second["attention_mask"] is not first["attention_mask"]
    for original in (*rotary, mask, positions):
        assert original.to.call_args_list == [call("cuda:1"), call("cuda:1")]


@pytest.fixture
def embed_prompt_model(monkeypatch):
    class Tensor:
        def __init__(self, values, *, device="cpu", dtype=None):
            self.values = values
            self.device = device
            self.dtype = dtype

        def to(self, device=None, dtype=None):
            return Tensor(self.values, device=device or self.device, dtype=dtype or self.dtype)

        def __eq__(self, token):
            return Tensor([[int(item == token) for item in row] for row in self.values], device=self.device)

    def ones_like(input_ids):
        return Tensor([[1] * len(row) for row in input_ids.values], device=input_ids.device, dtype=input_ids.dtype)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        tensor=Tensor, ones_like=ones_like, inference_mode=nullcontext, long="long", int32="int32", bfloat16="bfloat16",
    ))
    hidden = Tensor([[0.25, 1.5]], device="cuda:1", dtype="float16")
    model = Mock(return_value=SimpleNamespace(last_hidden_state=[hidden]))
    model.visual.parameters.return_value = iter([Tensor([], device="cuda:0", dtype="float16")])
    model.config.image_token_id = 151655
    return model, Mock()


def test_encode_prompt_text_applies_fallback_and_token_limit(embed_prompt_model):
    model, processor = embed_prompt_model
    processor.tokenizer.return_value = {"input_ids": list(range(4100))}
    embeds, tags = worker._encode_prompt((model, processor), "", [])
    processor.tokenizer.assert_called_once_with(" ", add_special_tokens=False)
    processor.image_processor.assert_not_called()
    kwargs = model.call_args.kwargs
    assert kwargs["input_ids"].values == [list(range(4096))]
    assert kwargs["input_ids"].device == kwargs["attention_mask"].device == "cuda:0"
    assert kwargs["attention_mask"].values == [[1] * 4096]
    for name in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
        assert kwargs[name] is None
    assert kwargs["output_hidden_states"] is kwargs["use_cache"] is False
    assert kwargs["return_dict"] is True
    assert (embeds.values, embeds.device, embeds.dtype) == ([[0.25, 1.5]], "cpu", "bfloat16")
    assert (tags.values, tags.device, tags.dtype) == ([1] * 4096, "cpu", "long")


def test_encode_prompt_edit_matches_h3_image_and_token_preprocessing(monkeypatch, tmp_path, embed_prompt_model):
    model, processor = embed_prompt_model
    paths = [tmp_path / "first.png", tmp_path / "second.png"]
    image_open, exif_transpose, lanczos = Mock(), Mock(), object()
    images = [Mock(), Mock()]
    exif_transpose.side_effect = images
    for image, size in zip(images, [(539, 300), (1024, 1537)], strict=True):
        image.convert.return_value.size = size
    image_open.side_effect = [nullcontext(path) for path in paths]
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(
        Image=SimpleNamespace(open=image_open, Resampling=SimpleNamespace(LANCZOS=lanczos)),
        ImageOps=SimpleNamespace(exif_transpose=exif_transpose),
    ))
    vision_tokens = {"<|vision_start|>": 151652, "<|image_pad|>": 151655, "<|vision_end|>": 151653}
    processor.tokenizer.convert_tokens_to_ids.side_effect = vision_tokens.__getitem__
    processor.tokenizer.side_effect = [{"input_ids": ids} for ids in ([11, 12], [13], [21, 22])]
    pixels, grid = Mock(), MagicMock()
    grid.__iter__.return_value = iter([Mock(prod=lambda: 8), Mock(prod=lambda: 12)])
    processor.image_processor.merge_size = 2
    processor.image_processor.return_value = {"pixel_values": pixels, "image_grid_thw": grid}

    embeds, tags = worker._encode_prompt((model, processor), "Describe both", paths)
    assert image_open.call_args_list == exif_transpose.call_args_list == [call(path) for path in paths]
    for image, size in zip(images, [(1824, 1024), (1024, 1536)], strict=True):
        image.convert.assert_called_once_with("RGB")
        image.convert.return_value.resize.assert_called_once_with(size, lanczos)
    processor.image_processor.assert_called_once_with(
        images=[image.convert.return_value.resize.return_value for image in images], return_tensors="pt",
    )
    assert processor.tokenizer.call_args_list == [
        call("Picture 1: ", add_special_tokens=False), call("Picture 2: ", add_special_tokens=False),
        call("Describe both", add_special_tokens=False),
    ]
    kwargs = model.call_args.kwargs
    assert kwargs["input_ids"].values == [[11, 12, 151652, 151655, 151655, 151653,
                                           13, 151652, 151655, 151655, 151655, 151653, 21, 22]]
    assert kwargs["attention_mask"].values == [[1] * 14]
    assert kwargs["mm_token_type_ids"].values == [[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]]
    assert (kwargs["mm_token_type_ids"].dtype, kwargs["mm_token_type_ids"].device) == ("int32", "cuda:0")
    assert kwargs["pixel_values"] is pixels.to.return_value
    assert kwargs["image_grid_thw"] is grid.to.return_value
    pixels.to.assert_called_once_with(device="cuda:0", dtype="float16")
    grid.to.assert_called_once_with("cuda:0")
    assert (tags.values, tags.device, tags.dtype) == ([1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1], "cpu", "long")
    assert (embeds.values, embeds.device, embeds.dtype) == ([[0.25, 1.5]], "cpu", "bfloat16")


@pytest.mark.parametrize(("variant", "checkpoint"), [
    ("bf16", None),
    ("int8_convrot", "minimax-h3-image/int8-convrot/minimax_h3_image_pruned_int8_convrot.safetensors"),
])
def test_load_dit_passes_variant_checkpoint(monkeypatch, tmp_path, variant, checkpoint):
    local_dir = tmp_path / "cache"
    archive = tmp_path / "compile.tar"
    monkeypatch.setattr(worker, "H3_CACHE_DIR", local_dir)
    monkeypatch.setattr(worker, "_ACTIVE_COMPILE_CACHE", (None, 0))
    monkeypatch.setattr(worker, "_compile_cache_archive", Mock(return_value=archive))
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setattr(worker, "_ensure_h3_import_path", Mock())
    cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda _: (7, 5))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    def restore_cache(cache_archive, cache_dir):
        assert (cache_archive, cache_dir) == (archive, local_dir)
        for name, var in (("triton", "TRITON_CACHE_DIR"), ("inductor", "TORCHINDUCTOR_CACHE_DIR")):
            assert (local_dir / name).is_dir()
            assert os.environ[var] == str(local_dir / name)
        return 2

    def load_pipeline(*args, **kwargs):
        assert worker._ACTIVE_COMPILE_CACHE == (archive, 2)
        return pipe

    pipe = object()
    monkeypatch.setattr(worker, "_restore_compile_cache", restore_cache)
    from_pretrained = Mock(side_effect=load_pipeline)
    monkeypatch.setitem(sys.modules, "h3_infer", SimpleNamespace(
        MiniMaxH3ImagePipeline=SimpleNamespace(from_pretrained=from_pretrained),
    ))
    monkeypatch.setitem(sys.modules, "h3_infer.ops.attention", SimpleNamespace(
        recommended_attention_backend=lambda _: "sdpa",
    ))
    bf16_dir = tmp_path / "minimax-h3-image" / "bf16"
    settings = {**worker.DEFAULTS, "dit_variant": variant}

    assert worker._load_dit(settings, bf16_dir) == (pipe, "sdpa")
    from_pretrained.assert_called_once()
    assert from_pretrained.call_args.args == (bf16_dir,)
    expected_checkpoint = None if checkpoint is None else tmp_path / checkpoint
    assert from_pretrained.call_args.kwargs["dit_checkpoint"] == expected_checkpoint


@pytest.mark.parametrize(("torch_attrs", "triton_ver", "filename"), [
    ({"__version__": "2.8.0+cu128"}, "3.4.0 /build-1", "sm89_pt2.8.0_cu128_tr3.4.0_build-1.tar"),
    ({}, "", "sm89_ptunknown_trunknown.tar"),
])
def test_compile_cache_archive_key(monkeypatch, tmp_path, torch_attrs, triton_ver, filename):
    drive_project = tmp_path / "ColabUltra"
    drive_project.mkdir()
    cache_dir = drive_project / ".cache" / "h3_infer"
    monkeypatch.setattr(worker, "DRIVE_COMPILE_CACHE_DIR", cache_dir)
    package_version = Mock(return_value=triton_ver)
    monkeypatch.setattr(worker, "_package_version", package_version)
    cuda = SimpleNamespace(get_device_capability=Mock(return_value=(8, 9)))
    torch = SimpleNamespace(cuda=cuda, **torch_attrs)

    assert worker._compile_cache_archive(torch) == cache_dir / filename
    cuda.get_device_capability.assert_called_once_with(0)
    package_version.assert_called_once_with("triton")


@pytest.mark.parametrize("kaggle", [False, True], ids=["unmounted-drive", "kaggle"])
def test_compile_cache_archive_skips_unavailable_drive(monkeypatch, tmp_path, kaggle):
    drive_project = tmp_path / "ColabUltra"
    cache_dir = drive_project / ".cache" / "h3_infer"
    monkeypatch.setattr(worker, "DRIVE_COMPILE_CACHE_DIR", cache_dir)
    token = Mock(return_value=None)
    monkeypatch.setattr(worker, "_drive_token", token)
    if kaggle:
        monkeypatch.setenv("COLAB_ULTRA_KAGGLE", "1")

    assert worker._compile_cache_archive(None) is None
    token.return_value = "drive-token"
    if kaggle:
        monkeypatch.setattr(worker, "_package_version", Mock(return_value="3.4.0"))
        cuda = SimpleNamespace(get_device_capability=Mock(return_value=(7, 5)))
        torch = SimpleNamespace(cuda=cuda, __version__="2.8.0+cu128")
        assert worker._compile_cache_archive(torch) == cache_dir / "sm75_pt2.8.0_cu128_tr3.4.0.tar"
        cuda.get_device_capability.assert_called_once_with(0)
    else:
        assert worker._compile_cache_archive(None) is None


@pytest.mark.parametrize("unsafe", [False, True], ids=["valid", "unsafe"])
def test_kaggle_restore_compile_cache_downloads_and_validates_tar(monkeypatch, tmp_path, capsys, unsafe):
    monkeypatch.setenv("COLAB_ULTRA_KAGGLE", "1")
    monkeypatch.setenv("COLAB_ULTRA_DRIVE_TOKEN", "drive-token")
    archive = tmp_path / "drive" / "sm75_pt2.8.0_tr3.4.0.tar"
    local_dir = tmp_path / "local"
    source = tmp_path / "source.tar"
    files = {"triton/hash/kernel": b"kernel", "inductor/module.so": b"module"}
    with tarfile.open(source, "w") as cache_tar:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            cache_tar.addfile(member, BytesIO(content))
        if unsafe:
            cache_tar.addfile(tarfile.TarInfo("../escaped"))
    entry = {"id": "archive-id", "size": str(source.stat().st_size)}

    def download(_entry, dest):
        assert dest.name == archive.name
        assert dest.parent.parent.resolve() == Path("/tmp").resolve()
        dest.write_bytes(source.read_bytes())

    drive = Mock()
    drive.folder.return_value = "folder-id"
    drive.find.return_value = entry
    drive.download.side_effect = download
    drive_factory = Mock(return_value=drive)
    monkeypatch.setattr(worker, "_Drive", drive_factory)

    assert worker._restore_compile_cache(archive, local_dir) == (0 if unsafe else len(files))
    drive_factory.assert_called_once_with("drive-token")
    drive.folder.assert_called_once_with(("ColabUltra", ".cache", "h3_infer"))
    drive.find.assert_called_once_with("folder-id", archive.name)
    dest = drive.download.call_args.args[1]
    drive.download.assert_called_once_with(entry, dest)
    assert not dest.parent.exists()
    assert not archive.exists()
    output = capsys.readouterr().out
    if unsafe:
        assert not local_dir.exists()
        assert not (tmp_path / "escaped").exists()
        assert "[Warn] compile cache restore failed (ExtractError)" in output
    else:
        for name, content in files.items():
            assert (local_dir / name).read_bytes() == content
        assert f"[Cache] restored 2 compile cache files from {archive.name}" in output


@pytest.mark.parametrize(("folder", "entry"), [
    (None, None), ("folder-id", None), ("folder-id", {"id": "archive-id", "size": "0"}),
], ids=["missing-folder", "missing-file", "empty-file"])
def test_kaggle_restore_compile_cache_skips_missing_archive(monkeypatch, tmp_path, capsys, folder, entry):
    monkeypatch.setenv("COLAB_ULTRA_KAGGLE", "1")
    monkeypatch.setenv("COLAB_ULTRA_DRIVE_TOKEN", "drive-token")
    drive = Mock()
    drive.folder.return_value = folder
    drive.find.return_value = entry
    monkeypatch.setattr(worker, "_Drive", Mock(return_value=drive))

    assert worker._restore_compile_cache(tmp_path / "cache.tar", tmp_path / "local") == 0
    drive.download.assert_not_called()
    assert capsys.readouterr().out == ""


def test_kaggle_save_compile_cache_uploads_only_when_grown(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("COLAB_ULTRA_KAGGLE", "1")
    monkeypatch.setenv("COLAB_ULTRA_DRIVE_TOKEN", "drive-token")
    archive = tmp_path / "drive" / "sm75_pt2.8.0_tr3.4.0.tar"
    local_dir = tmp_path / "local"
    files = {"triton/hash/kernel": b"kernel", "inductor/module.so": b"module"}
    for name, content in files.items():
        target = local_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (local_dir / "ignored").write_bytes(b"not compile cache data")

    def upload(path, _parts):
        assert path.name == archive.name
        assert path.parent.parent.resolve() == Path("/tmp").resolve()
        with tarfile.open(path) as cache_tar:
            assert {member.name: cache_tar.extractfile(member).read()
                    for member in cache_tar.getmembers() if member.isfile()} == files

    drive = Mock()
    drive.upload.side_effect = upload
    drive_factory = Mock(return_value=drive)
    monkeypatch.setattr(worker, "_Drive", drive_factory)

    assert not worker._save_compile_cache(archive, local_dir, initial_files=2)
    assert not worker._save_compile_cache(archive, local_dir, initial_files=3)
    drive_factory.assert_not_called()
    assert worker._save_compile_cache(archive, local_dir, initial_files=1)
    drive_factory.assert_called_once_with("drive-token")
    staged = drive.upload.call_args.args[0]
    drive.upload.assert_called_once_with(staged, ("ColabUltra", ".cache", "h3_infer", archive.name))
    assert not staged.parent.exists()
    assert not archive.exists()
    assert f"[Cache] saved 2 compile cache files to {archive.name}" in capsys.readouterr().out


def test_kaggle_compile_cache_drive_errors_are_nonfatal(monkeypatch, tmp_path, capsys):
    from requests.exceptions import HTTPError

    monkeypatch.setenv("COLAB_ULTRA_KAGGLE", "1")
    monkeypatch.setenv("COLAB_ULTRA_DRIVE_TOKEN", "drive-token")
    archive = tmp_path / "drive" / "cache.tar"
    local_dir = tmp_path / "local"
    kernel = local_dir / "triton" / "kernel"
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"kernel")
    drive = Mock()
    drive.folder.side_effect = HTTPError("lookup failed")
    drive.upload.side_effect = HTTPError("upload failed")
    monkeypatch.setattr(worker, "_Drive", Mock(return_value=drive))

    assert worker._restore_compile_cache(archive, local_dir) == 0
    assert not worker._save_compile_cache(archive, local_dir, initial_files=0)
    output = capsys.readouterr().out
    assert "[Warn] compile cache restore failed (HTTPError)" in output
    assert "[Warn] compile cache save failed (HTTPError)" in output


def test_compile_cache_round_trip_and_save_only_when_grown(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(worker.os, "sync", Mock())
    archive = tmp_path / "drive" / "sm89_pt2.8.0_tr3.4.0.tar"
    local_dir = tmp_path / "local"
    triton = local_dir / "triton" / "hash" / "kernel"
    inductor = local_dir / "inductor" / "module.so"
    triton.parent.mkdir(parents=True)
    inductor.parent.mkdir()
    triton.write_bytes(b"kernel")
    triton.chmod(0o700)
    inductor.write_bytes(b"module")
    (local_dir / "ignore").write_bytes(b"not cache data")
    triton.with_name("symlink").symlink_to(triton)

    assert worker._save_compile_cache(archive, local_dir, initial_files=2)
    restored = tmp_path / "restored"
    initial_files = worker._restore_compile_cache(archive, restored)
    assert initial_files == 2
    assert (restored / "triton" / "hash" / "kernel").read_bytes() == b"kernel"
    assert (restored / "triton" / "hash" / "kernel").stat().st_mode & 0o100
    assert (restored / "inductor" / "module.so").read_bytes() == b"module"
    assert set(restored.iterdir()) == {restored / "triton", restored / "inductor"}
    assert not (restored / "triton" / "hash" / "symlink").is_symlink()
    output = capsys.readouterr().out
    assert f"[Cache] saved 2 compile cache files to {archive.name}" in output
    assert f"[Cache] restored 2 compile cache files from {archive.name}" in output

    original_archive = archive.read_bytes()
    assert not worker._save_compile_cache(archive, restored, initial_files)
    (restored / "inductor" / "module.so").unlink()
    assert not worker._save_compile_cache(archive, restored, initial_files)
    assert archive.read_bytes() == original_archive

    (restored / "inductor" / "module.so").write_bytes(b"module")
    (restored / "inductor" / "new.so").write_bytes(b"new module")
    assert worker._save_compile_cache(archive, restored, initial_files)
    fresh = tmp_path / "fresh"
    assert worker._restore_compile_cache(archive, fresh) == 3
    assert (fresh / "inductor" / "new.so").read_bytes() == b"new module"


@pytest.mark.parametrize(("name", "member_type", "link"), [
    ("../evil", tarfile.REGTYPE, ""),
    ("/abs/path", tarfile.REGTYPE, ""),
    ("other/file", tarfile.REGTYPE, ""),
    ("triton/link", tarfile.SYMTYPE, "../../evil"),
    ("inductor/link", tarfile.LNKTYPE, "triton/valid"),
    ("triton/fifo", tarfile.FIFOTYPE, ""),
])
def test_restore_compile_cache_rejects_unsafe_members(tmp_path, capsys, name, member_type, link):
    archive = tmp_path / "unsafe.tar"
    local_dir = tmp_path / "local"
    with tarfile.open(archive, "w") as cache_tar:
        cache_tar.addfile(tarfile.TarInfo("triton/valid"))
        unsafe = tarfile.TarInfo(name)
        unsafe.type = member_type
        unsafe.linkname = link
        cache_tar.addfile(unsafe)

    assert worker._restore_compile_cache(archive, local_dir) == 0
    assert not local_dir.exists()
    assert not (tmp_path / "evil").exists()
    assert "[Warn] compile cache restore failed" in capsys.readouterr().out


def test_restore_compile_cache_rejects_existing_symlink(tmp_path, capsys):
    archive = tmp_path / "cache.tar"
    local_dir = tmp_path / "local"
    outside = tmp_path / "outside"
    local_dir.mkdir()
    outside.mkdir()
    (local_dir / "triton").symlink_to(outside, target_is_directory=True)
    with tarfile.open(archive, "w") as cache_tar:
        cache_tar.addfile(tarfile.TarInfo("triton/kernel"))

    assert worker._restore_compile_cache(archive, local_dir) == 0
    assert not list(outside.iterdir())
    assert "[Warn] compile cache restore failed" in capsys.readouterr().out


@pytest.mark.parametrize("bad_tar", [b"not a tar archive", b"\x1f\x8b\x08\x00"], ids=["plain", "gzip"])
def test_restore_compile_cache_handles_corrupt_archive(tmp_path, capsys, bad_tar):
    archive = tmp_path / "corrupt.tar"
    archive.write_bytes(bad_tar)
    local_dir = tmp_path / "local"

    assert worker._restore_compile_cache(archive, local_dir) == 0
    assert not local_dir.exists()
    assert "[Warn] compile cache restore failed" in capsys.readouterr().out


def test_embed_restores_before_model_load_and_saves_on_worker_error(monkeypatch, tmp_path):
    processor, safe_open, replace_linear, _ = _mock_embed_loader(monkeypatch, gpu_count=2, vram=15, legacy_rotary=False)
    monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", "https://token@example.test")
    monkeypatch.setenv("COLAB_ULTRA_STAGED_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("TRITON_CACHE_DIR", "previous")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "previous")
    monkeypatch.setattr(worker, "_import_remote", Mock(return_value=SimpleNamespace(TaskQueue=Mock())))
    local_dir = tmp_path / "local"
    archive = tmp_path / "cache.tar"
    monkeypatch.setattr(worker, "H3_CACHE_DIR", local_dir)
    monkeypatch.setattr(worker, "_ACTIVE_COMPILE_CACHE", (None, 0))
    monkeypatch.setattr(worker, "_compile_cache_archive", Mock(return_value=archive))
    operations = Mock()
    operations.attach_mock(worker._ensure_h3_import_path, "import_h3")

    def restore_cache(cache_archive, cache_dir):
        assert (cache_archive, cache_dir) == (archive, local_dir)
        for name, var in (("triton", "TRITON_CACHE_DIR"), ("inductor", "TORCHINDUCTOR_CACHE_DIR")):
            assert (local_dir / name).is_dir()
            assert os.environ[var] == str(local_dir / name)
        processor.assert_not_called()
        safe_open.assert_not_called()
        replace_linear.assert_not_called()
        return 2

    operations.restore.side_effect = restore_cache
    operations.run.side_effect = RuntimeError("worker stopped")
    monkeypatch.setattr(worker, "_restore_compile_cache", operations.restore)
    monkeypatch.setattr(worker, "_run_loop", operations.run)
    monkeypatch.setattr(worker, "_save_compile_cache", operations.save)

    with pytest.raises(SystemExit, match="H3 embed worker failed .*worker stopped"):
        worker.main(["--role", "embed", "--worker-id", "test"])

    replace_linear.assert_called()
    assert [entry[0] for entry in operations.mock_calls] == ["restore", "import_h3", "run", "save"]
    operations.save.assert_called_once_with(archive, local_dir, 2)
    assert worker._ACTIVE_COMPILE_CACHE == (None, 0)


def test_dit_saves_compile_cache_after_closing_on_worker_error(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", "https://token@example.test")
    monkeypatch.setenv("COLAB_ULTRA_STAGED_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(worker, "_import_remote", Mock(return_value=SimpleNamespace(TaskQueue=Mock())))
    archive = tmp_path / "cache.tar"
    monkeypatch.setattr(worker, "_ACTIVE_COMPILE_CACHE", (archive, 2))
    monkeypatch.setattr(worker, "_load_dit", Mock(return_value=(object(), "sdpa")))
    operations = Mock()
    monkeypatch.setattr(worker, "_DiTRunner", Mock(return_value=SimpleNamespace(close=operations.close)))
    monkeypatch.setattr(worker, "_run_loop", operations.run)
    monkeypatch.setattr(worker, "_save_compile_cache", operations.save)
    operations.run.side_effect = RuntimeError("worker stopped")

    with pytest.raises(SystemExit, match="worker stopped"):
        worker.main(["--role", "dit", "--worker-id", "test"])

    assert [entry[0] for entry in operations.mock_calls] == ["run", "close", "save"]
    operations.save.assert_called_once_with(archive, worker.H3_CACHE_DIR, 2)
    assert worker._ACTIVE_COMPILE_CACHE == (None, 0)


def test_wait_for_drive_file_refreshes_missing_parent_and_empty_file(monkeypatch, tmp_path):
    target = tmp_path / "input" / "embeds.pt"
    listdir = Mock(wraps=os.listdir)
    monkeypatch.setattr(worker.os, "listdir", listdir)
    elapsed = 0.0

    def sleep(delay):
        nonlocal elapsed
        assert delay == 0.5
        elapsed += delay
        if elapsed == 0.5:
            target.parent.mkdir()
            target.touch()
        else:
            target.write_bytes(b"published")

    monkeypatch.setattr(worker.time, "sleep", sleep)
    monkeypatch.setattr(worker.time, "monotonic", lambda: elapsed)

    assert worker.wait_for_drive_file(target, timeout_sec=4, poll_sec=0.5) == target
    assert elapsed == 1.0
    assert listdir.call_args_list == [call(target.parent)] * 3


def test_wait_for_drive_file_times_out(monkeypatch, tmp_path):
    clock = iter([0.0, 0.0, 0.25, 0.4])
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(clock))
    sleep = Mock()
    monkeypatch.setattr(worker.time, "sleep", sleep)
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        worker.wait_for_drive_file(tmp_path / "missing.pt", timeout_sec=0.4, poll_sec=0.25)
    assert sleep.call_args_list == [call(0.25), call(pytest.approx(0.15))]


def test_wait_for_artifact_retries_drive_and_publishes_complete_download(monkeypatch, tmp_path, capsys):
    project = tmp_path / "images' project"
    relative = "data/latents/image.pt"
    target = project / relative
    monkeypatch.setattr(worker, "_cu_drive_tok", "private-drive-token", raising=False)
    monkeypatch.setattr(worker, "_import_remote", {"models": models}.__getitem__)
    monkeypatch.setattr(worker.os, "sync", Mock())
    elapsed = 0.0
    file_lookups = 0

    def sleep(delay):
        nonlocal elapsed
        elapsed += delay

    monkeypatch.setattr(worker.time, "sleep", sleep)
    monkeypatch.setattr(worker.time, "monotonic", lambda: elapsed)

    def request(method, url, *, headers, params, timeout, stream=False):
        nonlocal file_lookups
        assert method == "GET" and headers == {"Authorization": "Bearer private-drive-token"}
        assert 0 < timeout <= 4 - elapsed
        assert not target.exists()
        response = MagicMock()
        if stream:
            assert url.endswith("/image-id") and params == {"alt": "media"}
            content = [b"partial"] if file_lookups == 2 else [b"complete ", b"latents"]
            response.iter_content.return_value = content
            response.__enter__.return_value = response
            return response
        assert url == worker.DRIVE_FILES_URL
        for parent, name, entry in (
            ("root", "ColabUltra", "hub"), ("hub", "images\\' project", "project"),
            ("project", "data", "data-id"), ("data-id", "latents", "latents-id"),
        ):
            if f"'{parent}' in parents and name = '{name}'" in params["q"]:
                assert f"mimeType = '{worker.DRIVE_FOLDER_MIME}'" in params["q"]
                response.json.return_value = {"files": [{"id": entry}]}
                return response
        assert "'latents-id' in parents and name = 'image.pt'" in params["q"]
        file_lookups += 1
        response.json.return_value = {"files": [] if file_lookups == 1 else [{"id": "image-id", "size": "16"}]}
        return response

    http = Mock(side_effect=request)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=http))
    assert worker._wait_for_artifact(project, relative, 4) == target
    assert target.read_bytes() == b"complete latents" and elapsed == 2
    assert list(target.parent.iterdir()) == [target]
    worker.os.sync.assert_called_once()
    http.reset_mock()
    assert worker._wait_for_artifact(project, relative, 4) == target
    http.assert_not_called()
    assert "private-drive-token" not in capsys.readouterr().out


def test_drive_api_upload_creates_folder_and_replaces_existing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_ULTRA_DRIVE_TOKEN", "private-drive-token")
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    responses = iter((
        {"files": [{"id": "hub"}]}, {"files": [{"id": "project"}]}, {"files": [{"id": "runs"}]},
        {"files": []}, {"id": "new-run"}, {"files": [{"id": "existing-image"}]}, {}, {},
    ))

    def respond(method, url, **kwargs):
        if method == "PUT":
            assert kwargs["data"].read() == b"image"
        return Mock(headers={"Location": "https://upload.example/session"}, **{"json.return_value": next(responses)})

    http = Mock(side_effect=respond)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=http))
    worker._publish_artifact(image, tmp_path, "runs/new-run/image.png")
    assert [entry.args[0] for entry in http.call_args_list] == ["GET"] * 4 + ["POST", "GET", "PATCH", "PUT"]
    create_folder, patch_file, upload = [http.call_args_list[index] for index in (4, 6, 7)]
    assert create_folder.kwargs["json"] == {"name": "new-run", "mimeType": worker.DRIVE_FOLDER_MIME,
                                           "parents": ["runs"]}
    assert patch_file.args[1].endswith("/existing-image")
    assert patch_file.kwargs["params"] == {"uploadType": "resumable"}
    assert patch_file.kwargs["headers"] == {"Authorization": "Bearer private-drive-token",
                                         "X-Upload-Content-Length": "5"}
    assert upload.args[1] == "https://upload.example/session" and upload.kwargs["headers"] == {"Content-Length": "5"}


@pytest.mark.parametrize(("kaggle", "on_drive", "uploads"), [
    (False, True, False),
    (True, True, True),
    (False, False, True),
])
def test_publish_artifact_uploads_only_when_needed(monkeypatch, tmp_path, kaggle, on_drive, uploads):
    monkeypatch.setenv("DRIVE_ACCESS_TOKEN", "tok")
    if kaggle:
        monkeypatch.setenv("COLAB_ULTRA_KAGGLE", "1")
    project = Path("/content/drive/MyDrive/ColabUltra/demo") if on_drive else tmp_path / "demo"
    relative = "runs/run/image.png"
    path = project / relative
    drive = Mock()
    monkeypatch.setattr(worker, "_Drive", drive)

    worker._publish_artifact(path, project, relative)

    if uploads:
        drive.assert_called_once_with("tok")
        drive.return_value.upload.assert_called_once_with(path, ("ColabUltra", "demo", "runs", "run", "image.png"))
    else:
        drive.assert_not_called()


def test_worker_discovers_bundled_h3_library(monkeypatch):
    engines_root = str(Path(worker.__file__).resolve().parents[1] / "engines")
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path != engines_root])
    monkeypatch.setitem(sys.modules, "h3_infer", None)

    worker._ensure_h3_import_path()

    assert sys.path[0] == engines_root


@pytest.mark.parametrize("concurrent_install", [False, True])
def test_kaggle_pulls_synced_h3_library_when_missing(monkeypatch, tmp_path, concurrent_install):
    scripts = tmp_path / "scripts"
    monkeypatch.setattr(worker, "REMOTE_SCRIPTS", scripts)
    monkeypatch.delattr(worker, "__file__")
    monkeypatch.setenv("COLAB_ULTRA_KAGGLE", "1")
    monkeypatch.setenv("COLAB_ULTRA_DRIVE_TOKEN", "private-drive-token")
    monkeypatch.setitem(sys.modules, "h3_infer", None)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    file = {"id": "init", "name": "__init__.py", "size": "4", "mimeType": "text/x-python"}
    models = {"id": "models", "name": "models", "mimeType": worker.DRIVE_FOLDER_MIME}
    responses = iter([{"files": [{"id": key}]} for key in ("hub", "scripts", "package")] + [
        {"files": [file], "nextPageToken": "more"}, {"files": [models]}, {"files": [file]},
    ])

    def request(method, url, *, params, stream=False, **_kwargs):
        assert method == "GET"
        if stream:
            assert params == {"alt": "media"} and url.endswith("/init")
            response = MagicMock()
            response.__enter__.return_value = response
            response.iter_content.return_value = [b"pass"]
            return response
        return Mock(**{"json.return_value": next(responses)})

    http = Mock(side_effect=request)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=http))
    if concurrent_install:
        def replace_with_concurrent_install(package, target):
            assert target == scripts / "h3_infer"
            assert (package / "__init__.py").read_bytes() == b"pass"
            target.mkdir()
            (target / "__init__.py").write_bytes(b"# concurrent worker")
            raise OSError("Directory not empty")

        monkeypatch.setattr(Path, "replace", replace_with_concurrent_install)

    worker._ensure_h3_import_path()
    assert str(scripts) in sys.path
    init = scripts / "h3_infer" / "__init__.py"
    assert init.read_bytes() == (b"# concurrent worker" if concurrent_install else b"pass")
    if not concurrent_install:
        assert (scripts / "h3_infer" / "models" / "__init__.py").read_bytes() == b"pass"
    assert any(entry.kwargs["params"].get("pageToken") == "more" for entry in http.call_args_list)


def test_model_scope_for_int8_convrot_dit():
    assert worker._model_scope("dit", "int8_convrot") == {
        worker.IMAGE_GROUP: {
            key: worker.BF16_FILES[key]
            for key in ("vae/config.json", "vae/diffusion_pytorch_model.safetensors", "scheduler/scheduler_config.json")
        },
        worker.INT8_CONVROT_GROUP: worker.INT8_CONVROT_FILES,
    }


def test_dit_dry_run_uses_configured_variant(monkeypatch, tmp_path, capsys):
    code = tmp_path / "code"
    code.mkdir()
    (code / "pipeline_worker.json").write_text('{"dit_variant": "int8_convrot"}', encoding="utf-8")
    monkeypatch.chdir(code)

    assert worker.main(["--role", "dit", "--worker-id", "dit-worker", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Model staging: minimax-h3-image/int8-convrot: " + worker.INT8_CONVROT_FILENAME in out
    assert "transformer/" not in out and "pdd_" not in out and "processor/" not in out


def test_dry_run_direct_and_in_colab_kernel(monkeypatch, tmp_path, capsys):
    code = tmp_path / "code"
    code.mkdir()
    (code / "pipeline_worker.json").write_text(json.dumps({"embed_queue": "text-queue"}), encoding="utf-8")
    source = Path(worker.__file__).resolve()
    monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", "https://private-token@invalid.example")
    monkeypatch.setenv("COLAB_ULTRA_WORKER_ROLE", "dit")
    monkeypatch.setenv("COLAB_ULTRA_WORKER_ID", "env-worker")
    result = subprocess.run([sys.executable, "-B", "-S", str(source), "--role", "embed", "--worker-id", "t4",
                             "--dry-run"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "role=embed worker_id=t4" in result.stdout
    assert "text-queue" in result.stdout and "Upstream queues: []" in result.stdout
    assert "processor/tokenizer.json" in result.stdout
    assert "minimax-h3-qwen3vl-32b/int8-convrot: qwen3vl_32b_minimax_h3_int8_convrot.safetensors" in result.stdout
    assert "transformer/" not in result.stdout and "private-token" not in result.stdout
    assert not result.stderr

    monkeypatch.setenv("COLAB_ULTRA_WORKER_ROLE", "vae")
    monkeypatch.setenv("CU_DRY_RUN", "1")
    monkeypatch.setattr(sys, "argv", ["ipykernel_launcher.py", "--role", "dit", "--kernel-only"])
    monkeypatch.chdir(code)
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), {"__name__": "__main__"})
    out = capsys.readouterr().out
    assert "role=vae worker_id=env-worker" in out
    assert "Upstream queues: ['text-queue', 'h3_pipe_dit']" in out
    assert "vae/config.json, vae/diffusion_pytorch_model.safetensors" in out
    assert "processor/" not in out and "transformer/" not in out and "private-token" not in out


def test_embed_vae_dry_run(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    assert worker.main(["--role", "embed_vae", "--worker-id", "kaggle", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "role=embed_vae worker_id=kaggle" in out
    assert "With 2+ GPUs, embed runs with tail TE layers CPU-offloaded concurrently with VAE on GPU 1." in out
    assert "VAE on GPU 0 starts after embed exits successfully." in out
    assert "With fewer than two GPUs, embed runs sequentially before VAE on GPU 0." in out
    assert "processor/tokenizer.json" in out and "vae/config.json" in out
    assert "minimax-h3-qwen3vl-32b/int8-convrot: qwen3vl_32b_minimax_h3_int8_convrot.safetensors" in out
    assert "transformer/" not in out
    assert "embed_vae_queue" not in out


def test_worker_script_path_uses_project_code_then_synced_scripts(monkeypatch, tmp_path):
    code, scripts = tmp_path / "code", tmp_path / "scripts"
    code.mkdir()
    scripts.mkdir()
    project_script, synced_script = (directory / "h3_pipeline_worker.py" for directory in (code, scripts))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delattr(worker, "__file__")
    monkeypatch.setattr(worker, "REMOTE_SCRIPTS", scripts)
    with pytest.raises(RuntimeError, match="h3_pipeline_worker.py"):
        worker._worker_script_path(code)
    synced_script.touch()
    assert worker._worker_script_path(code) == synced_script.resolve()
    project_script.touch()
    assert worker._worker_script_path(code) == project_script.resolve()


@pytest.mark.parametrize(("gpu_count", "mask", "override"), [
    (2, None, False),
    (2, "  ALL  ", False),
    (2, "0,1", True),
    (1, "0", False),
])
def test_embed_vae_launch_order_and_environment(monkeypatch, tmp_path, gpu_count, mask, override):
    code = tmp_path / "code"
    code.mkdir()
    monkeypatch.chdir(code)
    redis_url = "https://private-redis-token@example.test"
    monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", redis_url)
    if mask is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    if override:
        monkeypatch.setenv("COLAB_ULTRA_TE_OFFLOAD_LAYERS", "16")
        monkeypatch.setenv("COLAB_ULTRA_VAE_TILE_BATCH_SIZE", "10")
    monkeypatch.setattr(worker, "_cu_drive_tok", "private-drive-token", raising=False)
    sleep = Mock()
    monkeypatch.setattr(worker.time, "sleep", sleep)
    staged = tmp_path / "staged"
    stage = Mock(return_value=staged)
    monkeypatch.setattr(worker, "_stage_models", stage)
    active_mask = None if mask and mask.strip().lower() == "all" else mask

    def device_count():
        popen.assert_not_called()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == active_mask
        return gpu_count

    count = Mock(side_effect=device_count)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=count)))
    embed, vae0, vae1 = Mock(), Mock(), Mock()
    embed.wait.return_value = vae0.poll.return_value = 0
    vae1.poll.side_effect = [None, None, 0]

    def embed_poll():
        assert popen.call_count == 2
        return 0 if sleep.called else None

    embed.poll.side_effect = embed_poll

    monkeypatch.setenv("COLAB_ULTRA_KAGGLE", "1")
    ensure_h3_import_path = Mock()
    monkeypatch.setattr(worker, "_ensure_h3_import_path", ensure_h3_import_path)

    def spawn(args, *, env):
        stage.assert_called_once()
        count.assert_called_once_with()
        ensure_h3_import_path.assert_called_once_with()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == mask
        suffix = args[6].removeprefix("kaggle-")
        if suffix == "embed":
            return embed
        if suffix == "vae-1":
            embed.poll.assert_not_called()
            embed.wait.assert_not_called()
            return vae1
        if suffix == "vae-0":
            assert embed.poll.call_count == 2
            assert vae1.poll.call_count == 1
        else:
            assert suffix == "vae"
            embed.wait.assert_called_once_with()
        return vae0

    popen = Mock(side_effect=spawn)
    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    assert worker.main(["--role", "embed_vae", "--worker-id", "kaggle"]) == 0
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == mask
    stage.assert_called_once_with({
        worker.IMAGE_GROUP: {key: url for key, url in worker.BF16_FILES.items() if key.startswith(("processor/", "vae/"))},
        ("minimax-h3-qwen3vl-32b", "int8-convrot"): {
            "qwen3vl_32b_minimax_h3_int8_convrot.safetensors":
                "hf://Comfy-Org/MiniMax-H3-Image/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        },
    })
    expected_children = [("embed", "embed", active_mask)]
    if gpu_count >= 2:
        expected_children.extend([("vae", "vae-1", "1"), ("vae", "vae-0", "0")])
        embed.wait.assert_not_called()
        assert sleep.call_args_list == [call(0.25), call(0.25)]
    else:
        expected_children.append(("vae", "vae", "0"))
        sleep.assert_not_called()

    for entry, (role, suffix, gpu) in zip(popen.call_args_list, expected_children, strict=True):
        assert entry.args[0] == [sys.executable, "-u", str(Path(worker.__file__).resolve()), "--role", role,
                                 "--worker-id", f"kaggle-{suffix}"]
        env = entry.kwargs["env"]
        assert env.get("CUDA_VISIBLE_DEVICES") == gpu
        assert env["COLAB_ULTRA_REDIS_URL"] == redis_url
        assert env["COLAB_ULTRA_DRIVE_TOKEN"] == "private-drive-token"
        assert env["COLAB_ULTRA_STAGED_MODELS"] == str(staged)
        assert env.get("COLAB_ULTRA_VAE_DEVICE") == ("cuda" if role == "vae" else None)
        offload = "16" if override else "14" if suffix == "embed" and gpu_count >= 2 else None
        tiles = "10" if override else "8" if suffix == "vae-1" else None
        assert env.get("COLAB_ULTRA_TE_OFFLOAD_LAYERS") == offload
        assert env.get("COLAB_ULTRA_VAE_TILE_BATCH_SIZE") == tiles
    vae0.poll.assert_called_once_with()
    for child in (embed, vae0, vae1):
        child.terminate.assert_not_called()


@pytest.mark.parametrize(("failing_child", "status", "expected"), [("embed", 3, 3), ("vae-1", -9, 1)])
def test_embed_vae_concurrent_failure_terminates_sibling(monkeypatch, tmp_path, failing_child, status, expected):
    embed, vae1 = Mock(), Mock()
    failed, sibling = (embed, vae1) if failing_child == "embed" else (vae1, embed)
    failed.poll.return_value = status
    sibling.poll.return_value = None
    sibling.wait.side_effect = [subprocess.TimeoutExpired("sibling", 5), 0]
    launch = Mock(side_effect=[embed, vae1])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    monkeypatch.setattr(worker.subprocess, "Popen", launch)

    assert worker._launch_embed_vae(tmp_path / "worker.py", tmp_path, "worker", "https://example.test", tmp_path) == expected
    assert launch.call_count == 2
    failed.terminate.assert_not_called()
    assert sibling.method_calls[-4:] == [call.terminate(), call.wait(timeout=5), call.kill(), call.wait()]


def test_embed_vae_cleans_up_if_polling_is_interrupted(monkeypatch, tmp_path):
    embed, vae1 = Mock(), Mock()
    for child in (embed, vae1):
        child.poll.return_value = None
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    monkeypatch.setattr(worker.subprocess, "Popen", Mock(side_effect=[embed, vae1]))
    monkeypatch.setattr(worker.time, "sleep", Mock(side_effect=KeyboardInterrupt))

    with pytest.raises(KeyboardInterrupt):
        worker._launch_embed_vae(tmp_path / "worker.py", tmp_path, "worker", "https://example.test", tmp_path)
    for child in (embed, vae1):
        assert child.method_calls == [call.poll(), call.poll(), call.terminate(), call.wait(timeout=5)]


@pytest.mark.parametrize(("failure", "expected"), [(3, 3), (-15, 1), (KeyboardInterrupt, None)])
def test_embed_vae_sequential_embed_failure_skips_vae(monkeypatch, tmp_path, failure, expected):
    embed = Mock()
    embed.poll.return_value = None
    launch = Mock(return_value=embed)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1)))
    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    args = (tmp_path / "worker.py", tmp_path, "worker", "https://example.test", tmp_path)
    if failure is KeyboardInterrupt:
        embed.wait.side_effect = [KeyboardInterrupt, 0]
        with pytest.raises(KeyboardInterrupt):
            worker._launch_embed_vae(*args)
        assert embed.method_calls == [call.wait(), call.poll(), call.terminate(), call.wait(timeout=5)]
    else:
        embed.wait.return_value = failure
        assert worker._launch_embed_vae(*args) == expected
        assert embed.method_calls == [call.wait()]
    launch.assert_called_once()


class MockTensor:
    def __init__(self, value, dtype=None):
        self.value = value
        self.dtype = dtype

    def to(self, dtype):
        return MockTensor(self.value, dtype)

    def cpu(self):
        return self


def test_mocked_pipeline_handoff(monkeypatch, tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "pipeline_worker.json").write_text(json.dumps({"run_name": "images", "vae_tile_batch_size": 2}),
                                               encoding="utf-8")
    monkeypatch.chdir(code)
    redis_url = "https://fake-token@example.test"
    monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", redis_url)
    monkeypatch.setattr(worker, "_install_redis", Mock(side_effect=AssertionError("REST requires no Redis package")))
    monkeypatch.setattr(worker.time, "sleep", Mock())
    events = []
    monkeypatch.setattr(worker.os, "sync", lambda: events.append("sync"))

    def save(payload, path):
        assert Path("/tmp").resolve() in path.resolve().parents
        assert tmp_path not in path.parents
        path.write_bytes(pickle.dumps(payload))

    def load(path, *, map_location, weights_only):
        assert (map_location, weights_only) == ("cpu", True)
        return pickle.loads(path.read_bytes())

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bfloat16", load=load, save=save))
    client = FakeRedis()
    queue_names = {role: worker.DEFAULTS[f"{role}_queue"] for role in worker.STAGE_ROLES}

    class ObservedQueue(queue.TaskQueue):
        def __init__(self, url, name):
            assert url == redis_url
            super().__init__(url, name, client=client)

        def push(self, tasks):
            if self.name != queue_names["embed"]:
                (task,) = tasks
                rel = task.get("latents_rel", task["prompt_embeds_rel"])
                assert (tmp_path / rel).stat().st_size > 0
                events.append(("push", self.name, task["slug"]))
            return super().push(tasks)

        def ack(self, msg_id, slug, worker_id, *, output):
            assert (tmp_path / output).stat().st_size > 0
            events.append(("ack", self.name, slug))
            return super().ack(msg_id, slug, worker_id, output=output)

        def report_progress(self, slug, worker_id, step):
            events.append(("progress", self.name, slug, step))
            return super().report_progress(slug, worker_id, step)

    ensure = Mock()
    remote = {"queue": SimpleNamespace(TaskQueue=ObservedQueue), "recipe": recipe,
              "models": SimpleNamespace(ensure=ensure, check_relative=models.check_relative)}
    monkeypatch.setattr(worker, "_import_remote", remote.__getitem__)
    embed_model, vae_model = object(), object()
    load_embed = Mock(return_value=embed_model)
    load_vae = Mock(return_value=(vae_model, 2))
    encode = Mock(return_value=(MockTensor("embedding", "bfloat16"), MockTensor("tags")))
    monkeypatch.setattr(worker, "_load_embed", load_embed)
    monkeypatch.setattr(worker, "_load_vae", load_vae)
    monkeypatch.setattr(worker, "_encode_prompt", encode)

    class DiT:
        resident = False

        @contextmanager
        def keep_dit_resident(self):
            events.append("dit_enter")
            self.resident = True
            try:
                yield
            finally:
                self.resident = False
                events.append("dit_exit")

        def __call__(self, path, *, height, width, seed, output_type, return_stats, step_callback):
            assert self.resident and output_type == "latent" and return_stats
            assert (width, height, seed) == (65, 33, 123)
            payload = load(path, map_location="cpu", weights_only=True)
            assert payload["prompt"] == "mountains"
            assert payload["prompt_embeds"].dtype == "bfloat16"
            assert payload["text_token_tags"].value == "tags"
            for step in range(6):
                step_callback(step, None, None)
            return MockTensor("latents"), SimpleNamespace(step_ms=tuple(range(6)), peak_vram_bytes=789)

    load_dit = Mock(return_value=(DiT(), "sageattention"))
    monkeypatch.setattr(worker, "_load_dit", load_dit)

    class Image:
        def save(self, path, *, compress_level):
            assert compress_level == 3
            assert path.parent.parent == tmp_path / "runs" / "images"
            assert not (path.parent.parent / path.name).exists()
            path.write_bytes(b"mock PNG")

    def decode(vae, latents, output_size, tile_batch_size):
        assert vae is vae_model and latents.value == "latents"
        assert output_size == [96, 64] and tile_batch_size == 2
        return Image()

    monkeypatch.setattr(worker, "_decode_image", decode)

    def upload(local_path, relative):
        path = Path(local_path)
        assert path == tmp_path / relative and path.stat().st_size > 0
        if path.suffix in (".png", ".json"):
            assert path.with_suffix(".json").is_file() and path.with_suffix(".png").is_file()
        events.append(("upload", relative))

    monkeypatch.setattr(worker, "_cu_upload_large_to_drive", upload, raising=False)
    queues = {role: ObservedQueue(redis_url, name) for role, name in queue_names.items()}
    slugs = ("first", "second")
    queues["embed"].push([{"slug": slug, "prompt": "mountains", "width": 65, "height": 33, "seed": 123}
                          for slug in slugs])
    worker_ids = {"embed": "t4", "dit": "l4", "vae": "cpu"}
    for role, worker_id in worker_ids.items():
        assert worker.main(["--role", role, "--worker-id", worker_id]) == 0
        assert queues[role].status()["counts"] == {"done": 2}
        assert queues[role].status()["workers"][worker_id]["status"] == "stopped"
        assert queues[role].status()["all_done"]
        if role != "vae":
            downstream = "dit" if role == "embed" else "vae"
            assert queues[downstream].status()["counts"] == {"queued": 2}

    assert ensure.call_count == 4
    (processor, text_weights, dit_weights, vae_weights) = ensure.call_args_list
    bf16_dir = processor.kwargs["to"]
    assert processor.args[:2] == worker.IMAGE_GROUP
    assert all(key.startswith("processor/") for key in processor.args[2])
    assert text_weights.args == (*worker.TE_GROUP, worker.TE_FILES)
    assert dit_weights.args[2] == {key: url for key, url in worker.BF16_FILES.items() if not key.startswith("processor/")}
    assert set(vae_weights.args[2]) == {"vae/config.json", "vae/diffusion_pytorch_model.safetensors"}
    load_embed.assert_called_once_with(worker._settings(code), bf16_dir, text_weights.kwargs["to"])
    load_dit.assert_called_once_with(worker._settings(code), bf16_dir)
    load_vae.assert_called_once_with(bf16_dir, worker._settings(code))
    encode.assert_called_once_with(embed_model, "mountains", [])

    embed_outputs = queues["embed"].status()["tasks"]
    embed_relative = embed_outputs["first"]["output"]
    assert embed_relative == embed_outputs["second"]["output"]
    assert Path(embed_relative).parent == Path("runs/h3_embeds_cache")
    assert events.count("sync") == 7  # one embedding, two latents, two images, two JSON files
    assert events.count("dit_enter") == events.count("dit_exit") == 1
    for index, slug in enumerate(slugs):
        run_dir = tmp_path / "runs" / "images"
        image = run_dir / f"{slug}.png"
        assert image.read_bytes() == b"mock PNG"
        assert events.index(("upload", f"runs/images/{slug}.png")) < events.index(("upload", f"runs/images/{slug}.json"))
        assert events.index(("upload", f"runs/images/{slug}.json")) < events.index(("ack", queue_names["vae"], slug))
        assert queues["vae"].status()["tasks"][slug]["output"] == str(image)
        stats = json.loads((run_dir / f"{slug}.json").read_text(encoding="utf-8"))
        assert stats["embed_cache_hit"] is bool(index)
        assert [stats[key] for key in ("embed_worker", "dit_worker", "vae_worker")] == ["t4", "l4", "cpu"]
        assert stats["output_size"] == [96, 64] and stats["seed"] == 123
        assert stats["step_ms"] == list(range(6)) and all(isinstance(ms, float) for ms in stats["step_ms"])
        assert stats["peak_vram_bytes"] == 789 and stats["active_attention_backend"] == "sageattention"
        for upstream, downstream in (("embed", "dit"), ("dit", "vae")):
            push_index = events.index(("push", queue_names[downstream], slug))
            assert push_index < events.index(("ack", queue_names[upstream], slug))
            if upstream == "dit":
                assert events[push_index - 1] == "sync"
            elif index == 0:
                assert events[push_index - 2:push_index] == ["sync", ("upload", embed_relative)]
        dit_steps = [event[3] for event in events
                     if isinstance(event, tuple) and event[:3] == ("progress", queue_names["dit"], slug)]
        assert dit_steps == [f"dit_step_{step}/6" for step in range(1, 7)]
        assert ("progress", queue_names["vae"], slug, "decoding_vae") in events


def test_edit_embeddings_use_reference_content(monkeypatch, tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    for name in ("reference.png", "copy.png"):
        (code / name).write_bytes(b"same image bytes")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bfloat16"))
    encode = Mock(return_value=(MockTensor("embedding"), MockTensor("tags")))
    monkeypatch.setattr(worker, "_encode_prompt", encode)

    def save(dest, payload):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(pickle.dumps(payload))

    monkeypatch.setattr(worker, "_save_pt", save)
    model, progress = object(), Mock()
    settings = worker._settings(code)
    task = {"slug": "edit", "prompt": "Lighten <Picture 1>", "edit_image": "reference.png"}
    embed_path, metadata = worker._embed_task(model, settings, code, task, progress)
    assert Path(embed_path).parent == Path("runs/h3_embeds_cache")
    assert not metadata["embed_cache_hit"]
    encode.assert_called_once_with(model, task["prompt"], [code / "reference.png"])
    cached_path, metadata = worker._embed_task(model, settings, code, {**task, "edit_image": ["copy.png"]}, progress)
    assert metadata["embed_cache_hit"] and cached_path == embed_path
    (code / "reference.png").write_bytes(b"different image bytes")
    new_path, metadata = worker._embed_task(model, settings, code, task, progress)
    assert not metadata["embed_cache_hit"] and new_path != embed_path


@pytest.mark.parametrize("upload_error", [None, RuntimeError("Drive unavailable")])
def test_embed_live_upload_only_on_cache_miss(monkeypatch, tmp_path, capsys, upload_error):
    code = tmp_path / "code"
    code.mkdir()
    (code / "pipeline_worker.json").write_text('{"embeds_dir": "runs/custom/"}', encoding="utf-8")
    settings = worker._settings(code)
    task = {"slug": "upload", "prompt": "mountains"}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        bfloat16="bfloat16", save=lambda payload, path: path.write_bytes(pickle.dumps(payload)),
    ))
    monkeypatch.setattr(worker.os, "sync", Mock())
    monkeypatch.setattr(worker, "_encode_prompt", Mock(return_value=(MockTensor("embedding"), MockTensor("tags"))))

    def upload(local_path, relative):
        assert isinstance(local_path, str) and Path(local_path) == tmp_path / relative
        assert pickle.loads(Path(local_path).read_bytes())["prompt"] == task["prompt"]
        worker.os.sync.assert_called_once()
        if upload_error:
            raise upload_error
        return False

    uploader = Mock(side_effect=upload)
    monkeypatch.setattr(worker, "_cu_upload_large_to_drive", uploader, raising=False)
    relative, metadata = worker._embed_task(None, settings, code, task, Mock())
    assert Path(relative).parent == Path("runs/custom")
    assert not metadata["embed_cache_hit"]
    cached_relative, metadata = worker._embed_task(None, settings, code, task, Mock())
    assert cached_relative == relative and metadata["embed_cache_hit"]
    uploader.assert_called_once_with(str(tmp_path / relative), relative)
    expected = "[Warn] live Drive upload skipped (RuntimeError): Drive unavailable\n" if upload_error else ""
    assert capsys.readouterr().out == expected


@pytest.fixture
def dit_stage(monkeypatch, tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "pipeline_worker.json").write_text('{"drive_wait_sec": 0}', encoding="utf-8")
    monkeypatch.chdir(code)
    redis_url = "https://dit-token@example.test"
    monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", redis_url)
    monkeypatch.setattr(worker.time, "sleep", Mock())
    monkeypatch.setattr(worker.os, "sync", Mock())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        save=lambda payload, path: path.write_bytes(pickle.dumps(payload)),
    ))
    queues = {role: Mock() for role in worker.STAGE_ROLES}
    queues["embed"].status.return_value = {"all_done": True}
    queues["dit"].status.return_value = {"unclaimed": 0}
    named_queues = {worker.DEFAULTS[f"{role}_queue"]: queue for role, queue in queues.items()}
    modules = {"queue": SimpleNamespace(TaskQueue=lambda _, name: named_queues[name]),
               "models": models, "recipe": recipe}
    monkeypatch.setattr(worker, "_import_remote", modules.__getitem__)
    monkeypatch.setattr(worker, "_stage_models", Mock(return_value=tmp_path))
    pipe = Mock()
    monkeypatch.setattr(worker, "_load_dit", Mock(return_value=(pipe, "sdpa")))
    embed_rel = "runs/h3_embeds_cache/encoded.pt"
    embed_path = tmp_path / embed_rel
    embed_path.parent.mkdir(parents=True)
    embed_path.write_bytes(b"embeds")
    return pipe, queues, {"prompt_embeds_rel": embed_rel, "width": 65, "height": 33, "seed": 123}


def test_dit_streams_edit_batch_before_next_image(monkeypatch, tmp_path, dit_stage):
    pipe, queues, common = dit_stage
    dit, vae = queues["dit"], queues["vae"]
    first_ref, last_ref = (tmp_path / "code" / name for name in ("first.png", "last.png"))
    for path in (first_ref, last_ref):
        path.write_bytes(b"reference")
    tasks = [
        ("1-0", {**common, "slug": "first", "edit_image": "first.png", "seed": 1}),
        ("2-0", {**common, "slug": "bad-dit", "seed": 2}),
        ("3-0", {**common, "slug": "last", "edit_image": ["last.png", "first.png"], "seed": 3}),
        ("4-0", {**common, "slug": "bad-ref", "edit_image": "last.png", "seed": 4}),
    ]
    dit.claim_batch.side_effect = [tasks, [], []]
    monkeypatch.setattr(sys.modules["torch"], "inference_mode", nullcontext, raising=False)
    events = []
    state = None

    @contextmanager
    def residency(model):
        nonlocal state
        assert state is None
        state = model
        events.append(f"{model} enter")
        try:
            yield
        finally:
            state = None
            events.append(f"{model} exit")

    pipe._vae_on_device = Mock(side_effect=lambda: residency("vae"))
    pipe.keep_dit_resident.side_effect = lambda: residency("dit")
    pipe._generation_sizes = Mock(return_value=((96, 64), (1, 2, 1, 3, 4)))

    def init_latents(shape, seed, rand_device, latents):
        assert state == "vae" and (shape, rand_device, latents) == ((1, 2, 1, 3, 4), "cpu", None)
        events.append(("initial", seed))
        return seed, f"initial {seed}"

    def encode(refs, generator, rand_device, tiling, noise):
        assert state == "vae" and (rand_device, tiling, noise) == ("cpu", None, None)
        events.append(("encode", generator))
        if generator == 4:
            raise ValueError("reference decoding failed")
        expected = [str(first_ref)] if generator == 1 else [str(last_ref), str(first_ref)]
        assert refs == expected
        return [f"encoded {generator}"]

    def run(embeds, *, height, width, seed, latents, output_type, return_stats, step_callback, ref_latents=None):
        assert state == "dit" and return_stats and output_type == "latent"
        assert (height, width) == (33, 65) and embeds == tmp_path / common["prompt_embeds_rel"]
        assert latents == f"initial {seed}"
        assert ref_latents == ([f"encoded {seed}"] if seed != 2 else None)
        events.append(("run", seed))
        if seed == 2:
            raise RuntimeError("DiT failed")
        step_callback(0, None, None)
        return MockTensor(f"final {seed}"), SimpleNamespace(step_ms=(10,), peak_vram_bytes=None)

    def push(tasks):
        (task,) = tasks
        path = tmp_path / task["latents_rel"]
        assert pickle.loads(path.read_bytes())["latents"].value == f"final {task['seed']}"
        events.append(("push", task["slug"]))

    pipe._initial_latents = Mock(side_effect=init_latents)
    pipe._encode_references = Mock(side_effect=encode)
    pipe.side_effect = run
    vae.push.side_effect = push
    dit.ack.side_effect = lambda msg_id, slug, *_args, **_kwargs: events.append(("ack", slug))

    assert worker.main(["--role", "dit", "--worker-id", "l4"]) == 0
    assert events == [
        "vae enter", ("initial", 1), ("encode", 1), ("initial", 2), ("initial", 3), ("encode", 3),
        ("initial", 4), ("encode", 4), "vae exit", "dit enter", ("run", 1), ("push", "first"),
        ("ack", "first"), ("run", 2), ("run", 3), ("push", "last"), ("ack", "last"), "dit exit",
    ]
    assert pipe._generation_sizes.call_args_list == [call(33, 65, None)] * len(tasks)
    pipe._vae_on_device.assert_called_once_with()
    pipe.keep_dit_resident.assert_called_once_with()
    pipe.generate_batch.assert_not_called()
    assert dit.fail.call_args_list == [call("4-0", "bad-ref", "l4", "ValueError: reference decoding failed"),
                                      call("2-0", "bad-dit", "l4", "RuntimeError: DiT failed")]


def test_dit_mixed_batches_release_and_restore_residency(monkeypatch, tmp_path, capsys, dit_stage):
    pipe, queues, common = dit_stage
    dit, vae = queues["dit"], queues["vae"]
    front = tmp_path / "code" / "front.png"
    back = tmp_path / "data" / "back.png"
    back.parent.mkdir()
    for path in (front, back):
        path.write_bytes(b"reference")
    dit.claim_batch.side_effect = [
        [("1-0", {**common, "slug": "first"}), ("2-0", {**common, "slug": "bad-text", "width": 0})],
        [("3-0", {**common, "slug": "second"})],
        [("4-0", {**common, "slug": "missing", "edit_image": "missing.png"}),
         ("5-0", {**common, "slug": "edit", "edit_image": "front.png"}),
         ("6-0", {**common, "slug": "mixed-text", "seed": 25}),
         ("7-0", {**common, "slug": "multi-edit", "edit_image": ["back.png", "front.png"]})],
        [("8-0", {**common, "slug": "last"})],
        [], [],
    ]
    resident = False
    clock = 0.0
    events = []
    monkeypatch.setattr(worker.time, "perf_counter", lambda: clock)
    worker.os.sync.side_effect = lambda: events.append("sync")
    dit.ack.side_effect = lambda *args, **kwargs: events.append(("ack", args[1]))
    vae.push.side_effect = lambda tasks: events.append(("push", tasks[0]["slug"]))

    @contextmanager
    def residency():
        nonlocal resident
        assert not resident
        resident = True
        events.append("enter")
        try:
            yield
        finally:
            resident = False
            events.append("exit")

    pipe.keep_dit_resident.side_effect = residency

    def text_call(path, *, height, width, seed, output_type, return_stats, step_callback):
        assert resident and output_type == "latent" and return_stats
        assert path == tmp_path / common["prompt_embeds_rel"]
        assert (height, width, seed) == (33, 65, 123)
        events.append("text")
        step_callback(0, None, None)
        return MockTensor("text"), SimpleNamespace(step_ms=(100,), peak_vram_bytes=None)

    def edit_call(items, *, output_type, return_stats, step_callback):
        nonlocal clock
        assert not resident and output_type == "latent" and return_stats
        base = {"prompt_embeds": tmp_path / common["prompt_embeds_rel"], "width": 65, "height": 33, "seed": 123}
        assert items == [{**base, "edit_images": [str(front)]}, {**base, "seed": 25},
                         {**base, "edit_images": [str(back), str(front)]}]
        events.append("batch")
        for i in range(len(items)):
            step_callback(i, i, None, None)
        clock += 1.5
        return [(MockTensor("edit"), SimpleNamespace(step_ms=(100, 500), peak_vram_bytes=789)),
                (MockTensor("mixed-text"), SimpleNamespace(step_ms=(), peak_vram_bytes=None)),
                (MockTensor("multi-edit"), SimpleNamespace(step_ms=(250,), peak_vram_bytes=123))]

    pipe.side_effect = text_call
    pipe.generate_batch.side_effect = edit_call
    assert worker.main(["--role", "dit", "--worker-id", "l4"]) == 0

    assert [event for event in events if isinstance(event, str) and event != "sync"] == [
        "enter", "text", "text", "exit", "batch", "enter", "text", "exit",
    ]
    assert pipe.generate_batch.call_count == 1
    assert dit.claim_batch.call_args_list == [call("l4", count=4, reclaim_idle_ms=300_000, block_ms=2000)] * 6
    assert dit.fail.call_args_list == [
        call("2-0", "bad-text", "l4", "ValueError: Output height and width must be positive"),
        call("4-0", "missing", "l4", f"FileNotFoundError: Reference image 'missing.png' missing under {tmp_path}"),
    ]
    slugs = ("first", "second", "edit", "mixed-text", "multi-edit", "last")
    expected_acks = []
    for msg_id, slug, handoff in zip((1, 3, 5, 6, 7, 8), slugs, vae.push.call_args_list, strict=True):
        relative = f"data/h3_latents_cache/pipe_demo/{slug}.pt"
        (task,) = handoff.args[0]
        assert (task["slug"], task["latents_rel"], task["dit_worker"]) == (slug, relative, "l4")
        payload = pickle.loads((tmp_path / relative).read_bytes())
        assert payload["output_size"] == [96, 64] and payload["seed"] == task["seed"]
        assert payload["active_attention_backend"] == task["active_attention_backend"] == "sdpa"
        if slug in ("edit", "mixed-text", "multi-edit"):
            assert payload["latents"].value == slug
        expected_acks.append(call(f"{msg_id}-0", slug, "l4", output=relative))
        push = events.index(("push", slug))
        assert events[push - 1:push + 2] == ["sync", ("push", slug), ("ack", slug)]
    assert dit.ack.call_args_list == expected_acks
    edit_task, mixed_task, multi_task = [handoff.args[0][0] for handoff in vae.push.call_args_list[2:5]]
    assert (edit_task["dit_sec"], mixed_task["dit_sec"], multi_task["dit_sec"]) == (0.6, 1.5, 0.25)
    assert edit_task["step_ms"] == [100.0, 500.0] and edit_task["peak_vram_bytes"] == 789
    assert dit.report_progress.call_args_list == [call(slug, "l4", f"dit_step_{step}/6")
                                                  for slug, step in zip(slugs, (1, 1, 1, 2, 3, 1), strict=True)]
    assert "[dit l4] mixed-text dit_step_2/6" in capsys.readouterr().out


def test_dit_batch_failure_redacts_and_worker_stop_releases_residency(tmp_path, capsys, dit_stage):
    pipe, queues, common = dit_stage
    (tmp_path / "code" / "reference.png").write_bytes(b"reference")
    dit = queues["dit"]
    dit.claim_batch.side_effect = [
        [("1-0", {**common, "slug": "edit", "edit_image": "reference.png"}),
         ("2-0", {**common, "slug": "mixed-text"})],
        [("3-0", {**common, "slug": "next-text"})],
        RuntimeError("claim interrupted"),
    ]
    pipe.generate_batch.side_effect = ValueError("dit-token")
    pipe.return_value = MockTensor("text"), SimpleNamespace(step_ms=(100,), peak_vram_bytes=None)
    events = []

    @contextmanager
    def residency():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    pipe.keep_dit_resident.side_effect = residency
    with pytest.raises(SystemExit, match="claim interrupted"):
        worker.main(["--role", "dit", "--worker-id", "l4"])
    assert events == ["enter", "exit"]
    assert dit.fail.call_args_list == [call("1-0", "edit", "l4", "ValueError: [redacted]"),
                                      call("2-0", "mixed-text", "l4", "ValueError: [redacted]")]
    dit.ack.assert_called_once_with("3-0", "next-text", "l4", output="data/h3_latents_cache/pipe_demo/next-text.pt")
    queues["vae"].push.assert_called_once()
    stderr = capsys.readouterr().err
    assert stderr.count("Traceback") == 1 and "ValueError: [redacted]" in stderr and "dit-token" not in stderr


def test_task_failure_redacts_credentials_and_continues(monkeypatch, capsys):
    redis_url = "https://rest%2Ftoken@example.test"
    own, downstream = Mock(), Mock()
    own.claim_batch.side_effect = [[("1-0", {"slug": "bad"})], [("2-0", {"slug": "good"})], [], []]
    own.status.return_value = {"unclaimed": 0}
    handler = Mock(side_effect=[ValueError(f"{redis_url} rest/token rest%2Ftoken"), ("file.pt", {"next": True})])
    monkeypatch.setattr(worker.time, "sleep", Mock())

    assert worker._run_loop("embed", "t4", {"embed": own, "dit": downstream}, worker.DEFAULTS,
                            handler, worker._redis_redactor(redis_url)) == 0
    own.fail.assert_called_once_with("1-0", "bad", "t4", "ValueError: [redacted] [redacted] [redacted]")
    downstream.push.assert_called_once_with([{"slug": "good", "next": True, "embed_worker": "t4"}])
    own.ack.assert_called_once_with("2-0", "good", "t4", output="file.pt")
    stderr = capsys.readouterr().err
    assert "Traceback" in stderr and "ValueError" in stderr
    assert all(secret not in stderr for secret in (redis_url, "rest/token", "rest%2Ftoken"))
