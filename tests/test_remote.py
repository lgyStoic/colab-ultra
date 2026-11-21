import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from colab_ultra import staging
from colab_ultra.remote import bg, binaries

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stage(tmp_path, monkeypatch):
    # Exercise the package exactly as the VM imports it after sync-remote.
    target = staging.sync_remote(tmp_path)
    monkeypatch.syspath_prepend(str(target.parent))
    module = importlib.import_module("cu_remote.stage")
    monkeypatch.setattr(module, "DRIVE", tmp_path / "mount")
    monkeypatch.setattr(module, "NVME", tmp_path / "nvme")
    return module


def test_plans(stage, tmp_path, monkeypatch):
    dest = stage.DRIVE / "MyDrive" / "tile"
    hf = stage.plan("hf://org/model/sub/file", dest, concurrency=64)
    assert hf.source == "https://huggingface.co/org/model/resolve/main/sub/file"
    assert hf.copy_to_drive and hf.work_dir == stage.NVME and hf.concurrency == 16
    assert stage.plan(hf.source, dest).source == hf.source
    assert stage.plan("https://example.test/w", tmp_path / "w", backend="hf_transfer").concurrency == 64
    drive = stage.plan("drive://models/w", tmp_path / "w")
    assert drive.source == str(stage.DRIVE / "MyDrive/models/w") and drive.backend == "copy"
    assert stage.plan(drive.source, tmp_path / "v").backend == "copy"
    api = Mock()
    api.list_repo_files.return_value = ["a.txt", "sub/a.gguf", "sub/b.gguf"]
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api))
    found = stage.repo_plans("org/model", "*.gguf", tmp_path)
    assert [p.dest.relative_to(tmp_path).as_posix() for p in found] == ["sub/a.gguf", "sub/b.gguf"]


def test_two_stage_reject_partial_and_skip(stage, tmp_path, monkeypatch):
    dest = stage.DRIVE / "MyDrive/model"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"broken-but-large")
    marker = Path(f"{dest}.aria2")
    marker.touch()
    item = stage.plan("https://example.test/weights", dest, min_bytes=3)
    copy = Mock(wraps=stage.shutil.copy2)
    monkeypatch.setattr(stage.shutil, "copy2", copy)

    def download(_, temp):
        assert temp.is_relative_to(stage.NVME) and not dest.exists() and not marker.exists()
        temp.write_bytes(b"complete")

    fetch = Mock(side_effect=download)
    monkeypatch.setattr(stage, "_download", fetch)
    assert stage.run(item).read_bytes() == b"complete"
    assert stage.run(item) == dest
    fetch.assert_called_once()
    assert copy.call_args.args[1].is_relative_to(dest.parent)


@pytest.mark.parametrize("partial", [True, False])
def test_failed_download_is_not_published(stage, tmp_path, monkeypatch, partial):
    def download(_, temp):
        temp.write_bytes(b"correct size" if partial else b"abc")
        if partial:
            Path(f"{temp}.aria2").touch()
    monkeypatch.setattr(stage, "_download", download)
    plan = stage.plan("https://example.test/f", tmp_path / "f", min_bytes=3)
    with pytest.raises(IOError, match="Incomplete"):
        stage.run(plan)
    assert not plan.dest.exists() and not list(tmp_path.rglob("*.aria2"))


def test_hf_transfer_benchmark(stage, tmp_path, monkeypatch):
    def download(**kw):
        assert kw["max_files"] == 64 and os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
        Path(kw["filename"]).write_bytes(b"12345")
    monkeypatch.setitem(sys.modules, "hf_transfer", SimpleNamespace(download=download))
    plan = stage.plan("hf://org/repo/w", tmp_path / "f", backend="hf_transfer", min_bytes=0)
    result = stage.benchmark([plan])
    assert result["bytes"] == 5 and result["MB_per_second"] > 0
    with pytest.raises(ValueError, match="fresh"):
        stage.benchmark([plan])


def test_sync_verifies_before_removing_legacy(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    scripts = hub / "scripts"
    scripts.mkdir(parents=True)
    old = [scripts / name for name in ("fast_stage_nvme.py", "vram_policy.py")]
    for p in old:
        p.touch()
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "__init__.py").touch()
    with monkeypatch.context() as patch:
        patch.setattr(staging, "RUNTIME", incomplete)
        with pytest.raises((OSError, FileNotFoundError)):
            staging.sync_remote(hub)
    assert all(p.exists() for p in old)
    target = staging.sync_remote(hub)
    assert not any(p.exists() for p in old)
    assert all((target / f.name).read_bytes() == f.read_bytes() for f in staging.runtime_files())
    assert not any(f.name.startswith("._") or f.name == "__pycache__" for f in target.iterdir())


def test_binary_cache_and_encoding(tmp_path, monkeypatch):
    import uuid

    name = "cu-test-" + uuid.uuid4().hex
    built = tmp_path / name
    built.write_bytes(b"example")
    build = Mock(return_value={name: built})
    local = binaries.ensure_binary(name, tmp_path / "cache", build)
    try:
        assert local.stat().st_mode & 0o777 == 0o755
        assert binaries.ensure_binary(name, tmp_path / "cache", build).read_bytes() == b"example"
        build.assert_called_once()
    finally:
        local.unlink()
    src = tmp_path / "tile.png"
    dest = src.with_suffix(".webp")
    src.touch()
    def encode(cmd, check):
        assert cmd[cmd.index("-lossless") + 1] == cmd[cmd.index("-compression_level") + 1] == "1"
        Path(cmd[-1]).write_bytes(b"encoded")
    monkeypatch.setattr(bg.subprocess, "run", encode)
    with bg.encoders() as enqueue:
        enqueue(src, dest)
    assert dest.read_bytes() == b"encoded" and not src.exists()


@pytest.mark.parametrize("name, feature", [("minimax_h3_diffsynth", "dit_nf4=True"),
                                          ("minimax_h3_sdcpp_server", "--max-vram 17")])
@pytest.mark.parametrize("env_flag", [True, False])
def test_recipe_dry_run(tmp_path, name, feature, env_flag):
    code = tmp_path / "code"
    code.mkdir()
    cfg = {"run_name": "smoke", "max_vram": "17", "vae_fp32": True, "dit_nf4": True,
           "prompts": [{"slug": "tile", "prompt": "a red kite", "edit_image": "input.png"}]}
    (code / "prompts.json").write_text(json.dumps(cfg))
    cmd = [sys.executable, str(ROOT / "recipes" / (name + ".py"))]
    env = os.environ.copy()
    if env_flag:
        env["CU_DRY_RUN"] = "1"
    else:
        cmd.append("--dry-run")
    output = subprocess.check_output(cmd, cwd=tmp_path, env=env, text=True)
    assert "a red kite" in output and "input.png" in output and feature in output
    assert "aria2" in output and not (tmp_path / "runs").exists()
    (code / "prompts.json").write_text('{"prompts": [{}]}')
    bad = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert bad.returncode and "expected a prompt" in bad.stderr


def test_unchanged_workflows_import_legacy_names(tmp_path, monkeypatch):
    """Exercise the exec bootstrap without starting a heartbeat or a VM."""
    from colab_ultra.session import _job_source

    scripts = staging.sync_remote(tmp_path).parent
    monkeypatch.syspath_prepend(str(scripts))
    for name in ("fast_stage_nvme", "vram_policy"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    code = _job_source("", "workflow", "test", "").split("_hb_stop =")[0]
    exec(code.replace("/content/drive/MyDrive/ColabUltra/scripts", str(scripts)), {})
    from fast_stage_nvme import stage_dual_channel
    from vram_policy import build_vram_plan

    assert stage_dual_channel(tmp_path, [], []) >= 0
    assert callable(build_vram_plan)
