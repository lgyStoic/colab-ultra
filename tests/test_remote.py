import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from colab_ultra import session, staging
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
    monkeypatch.delenv("DRIVE_ACCESS_TOKEN", raising=False)
    dest = stage.DRIVE / "MyDrive" / "tile"
    hf = stage.plan("hf://org/model/sub/file", dest, concurrency=64)
    assert hf.source == "https://huggingface.co/org/model/resolve/main/sub/file"
    assert hf.copy_to_drive and hf.work_dir == stage.NVME and hf.backend == "hf_transfer" and hf.concurrency == 64
    assert stage.plan(hf.source, dest).source == hf.source
    assert stage.plan("https://example.test/w", tmp_path / "w").concurrency == 64
    assert stage.plan("https://example.test/w", tmp_path / "w", backend="aria2", concurrency=64).concurrency == 16
    drive = stage.plan("drive://models/w", tmp_path / "w")
    assert drive.source == str(stage.DRIVE / "MyDrive/models/w") and drive.backend == "copy"
    assert stage.plan(drive.source, tmp_path / "v").backend == "copy"
    api = Mock()
    api.list_repo_files.return_value = ["a.txt", "sub/a.gguf", "sub/b.gguf"]
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api))
    found = stage.repo_plans("org/model", "*.gguf", tmp_path)
    assert [p.dest.relative_to(tmp_path).as_posix() for p in found] == ["sub/a.gguf", "sub/b.gguf"]


def test_colab_drive_plan_uses_hf_transfer_with_token(stage, monkeypatch):
    monkeypatch.delenv("COLAB_ULTRA_KAGGLE", raising=False)
    source = "drive://ColabUltra/models/foo.safetensors"
    dest = "/content/models/foo.safetensors"
    monkeypatch.setenv("DRIVE_ACCESS_TOKEN", "tok")

    item = stage.plan(source, dest)
    assert item.backend == "hf_transfer"
    assert item.concurrency == 64

    monkeypatch.delenv("DRIVE_ACCESS_TOKEN")
    assert stage.plan(source, dest).backend == "copy"


@pytest.mark.parametrize("failure_at", ["lookup", "hf_transfer", "aria2", "size"])
def test_drive_download_falls_back_to_fuse(stage, tmp_path, monkeypatch, failure_at):
    monkeypatch.delenv("COLAB_ULTRA_KAGGLE", raising=False)
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "0")
    monkeypatch.setenv("DRIVE_ACCESS_TOKEN", "tok")
    backend = "aria2" if failure_at == "aria2" else "hf_transfer"
    item = stage.plan("drive://model.bin", tmp_path / "model.bin", backend=backend, min_bytes=1)
    source = Path(item.source)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"complete")
    temp = tmp_path / "download"
    marker = temp.with_suffix(".aria2")
    media = "https://www.googleapis.com/drive/v3/files/example?alt=media"

    lookup = Mock(return_value=(media, source.stat().st_size))
    if failure_at == "lookup":
        lookup.side_effect = OSError("Drive lookup failed")

    def download(**kwargs):
        Path(kwargs["filename"]).write_bytes(b"partial")
        if failure_at == "hf_transfer":
            raise Exception("hf_transfer failed")

    def aria2(cmd, **kwargs):
        temp.write_bytes(b"partial")
        marker.touch()
        raise subprocess.CalledProcessError(1, cmd)

    copy2 = stage.shutil.copy2

    def copy(source, dest):
        assert not dest.exists() and not marker.exists()
        return copy2(source, dest)

    transfer = Mock(side_effect=download)
    copy_fuse = Mock(side_effect=copy)
    monkeypatch.setattr(stage, "_drive_media", lookup)
    monkeypatch.setitem(sys.modules, "hf_transfer", SimpleNamespace(download=transfer))
    monkeypatch.setattr(stage.shutil, "copy2", copy_fuse)
    if backend == "aria2":
        monkeypatch.setattr(stage, "ensure_aria2", lambda: None)
        monkeypatch.setattr(stage.subprocess, "run", aria2)

    stage._download(item, temp)

    lookup.assert_called_once_with(item.source, {"Authorization": "Bearer tok"})
    if failure_at in {"hf_transfer", "size"}:
        transfer.assert_called_once()
        assert transfer.call_args.kwargs["url"] == media
        assert transfer.call_args.kwargs["max_files"] == 64
    copy_fuse.assert_called_once_with(item.source, temp)
    assert temp.read_bytes() == b"complete"
    assert stage.valid(temp, item.min_bytes)


@pytest.mark.parametrize("fuse_state", ["kaggle", "missing", "undersized", "incomplete"])
def test_drive_download_raises_without_valid_colab_fuse(stage, tmp_path, monkeypatch, fuse_state):
    monkeypatch.delenv("COLAB_ULTRA_KAGGLE", raising=False)
    monkeypatch.setenv("DRIVE_ACCESS_TOKEN", "tok")
    item = stage.plan("drive://model.bin", tmp_path / "model.bin", min_bytes=1)
    source = Path(item.source)
    source.parent.mkdir(parents=True)
    if fuse_state != "missing":
        source.write_bytes(b"x" if fuse_state == "undersized" else b"complete")
    if fuse_state == "incomplete":
        Path(f"{source}.aria2").touch()
    if fuse_state == "kaggle":
        monkeypatch.setenv("COLAB_ULTRA_KAGGLE", "1")
        assert stage.valid(source, item.min_bytes)

    failure = OSError("Drive lookup failed")
    monkeypatch.setattr(stage, "_drive_media", Mock(side_effect=failure))
    copy_fuse = Mock()
    monkeypatch.setattr(stage.shutil, "copy2", copy_fuse)

    with pytest.raises(OSError) as raised:
        stage._download(item, tmp_path / "download")

    assert raised.value is failure
    copy_fuse.assert_not_called()


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


def test_aria2_without_headers_omits_conf_path(stage, tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(stage, "ensure_aria2", lambda: None)
    seen = []
    temp = tmp_path / "weights.bin.part"

    def fake_run(cmd, **kwargs):
        seen.append((list(cmd), kwargs))
        temp.write_bytes(b"weights")

    monkeypatch.setattr(stage.subprocess, "run", fake_run)
    item = stage.plan("https://example.test/weights.bin", tmp_path / "weights.bin", backend="aria2", min_bytes=0)
    stage._download(item, temp)
    assert len(seen) == 1
    cmd, kwargs = seen[0]
    assert cmd[0] == "aria2c"
    assert not any(arg.startswith("--conf-path") for arg in cmd)
    assert "input" not in kwargs


def test_aria2_with_hf_token_uses_temp_conf_file(stage, tmp_path, monkeypatch):
    token = "hf_secret_test_token"
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setattr(stage, "ensure_aria2", lambda: None)
    seen = []
    temp = tmp_path / "model.gguf.part"

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        conf_flags = [arg for arg in cmd if arg.startswith("--conf-path=")]
        assert len(conf_flags) == 1
        conf_path = Path(conf_flags[0].split("=", 1)[1])
        assert conf_path.parent == temp.parent and conf_path.exists()
        assert f"header=Authorization: Bearer {token}" in conf_path.read_text(encoding="utf-8")
        temp.write_bytes(b"gguf")

    monkeypatch.setattr(stage.subprocess, "run", fake_run)
    item = stage.plan("hf://org/repo/model.gguf", tmp_path / "model.gguf", backend="aria2", min_bytes=0)
    stage._download(item, temp)
    assert len(seen) == 1
    assert all(token not in arg for arg in seen[0])
    assert not list(tmp_path.glob("*.aria2.conf"))


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


def test_sync_remote_syncs_h3_infer(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "models").mkdir(parents=True)
    (source / "__pycache__").mkdir()
    (source / "__init__.py").write_text("# package\n")
    (source / "models" / "dit.py").write_text("# model\n")
    (source / "models" / "._dit.py").touch()
    (source / "__pycache__" / "ignored.py").touch()
    monkeypatch.setenv("H3_INFER_SOURCE", str(source))

    scripts = tmp_path / "scripts"
    target = scripts / "h3_infer"
    (target / "models").mkdir(parents=True)
    (target / "models" / "dit.py").write_text("# old\n")
    (target / "models" / "stale.py").touch()
    (target / "keep.txt").touch()

    assert staging.sync_remote(tmp_path) == scripts / "cu_remote"
    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*.py")) == [
        "__init__.py", "models/dit.py",
    ]
    assert (target / "__init__.py").read_text() == "# package\n"
    assert (target / "models" / "dit.py").read_text() == "# model\n"
    assert (target / "keep.txt").is_file()


def test_sync_remote_syncs_bundled_h3_infer_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("H3_INFER_SOURCE", raising=False)
    sync_tree = Mock(wraps=staging._sync_python_tree)
    monkeypatch.setattr(staging, "_sync_python_tree", sync_tree)

    scripts = staging.sync_remote(tmp_path).parent
    source = ROOT / "engines" / "h3_infer"
    target = scripts / "h3_infer"
    sync_tree.assert_called_with(source, target, scripts, prune_stale=True)
    assert (target / "models" / "dit.py").read_bytes() == (source / "models" / "dit.py").read_bytes()


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


@pytest.mark.parametrize("env_flag", [True, False])
def test_recipe_dry_run(tmp_path, env_flag):
    code = tmp_path / "code"
    code.mkdir()
    cfg = {"run_name": "smoke", "max_vram": "17", "vae_fp32": True, "dit_nf4": True,
           "prompts": [{"slug": "tile", "prompt": "a red kite", "edit_image": "input.png"}]}
    (code / "prompts.json").write_text(json.dumps(cfg))
    cmd = [sys.executable, str(ROOT / "recipes" / "minimax_h3_infer.py")]
    env = os.environ.copy()
    if env_flag:
        env["CU_DRY_RUN"] = "1"
    else:
        cmd.append("--dry-run")
    output = subprocess.check_output(cmd, cwd=tmp_path, env=env, text=True)
    assert "a red kite" in output and "input.png" in output
    assert "compile_mode=regional" in output and "attention_backend=auto" in output
    assert "hf_transfer" in output and not (tmp_path / "runs").exists()
    (code / "prompts.json").write_text('{"prompts": [{}]}')
    bad = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert bad.returncode and "expected a prompt" in bad.stderr


@pytest.mark.parametrize("overrides, backend", [
    ({"use_sage_attention": False}, "sdpa"),
    ({"sage_attn": False}, "sdpa"),
    ({"attention_backend": "flash_attn", "use_sage_attention": False}, "flash_attn"),
])
def test_recipe_dry_run_attention_override(tmp_path, overrides, backend):
    code = tmp_path / "code"
    code.mkdir()
    cfg = {"prompts": [{"prompt": "a red kite"}], **overrides}
    (code / "prompts.json").write_text(json.dumps(cfg))
    cmd = [sys.executable, str(ROOT / "recipes" / "minimax_h3_infer.py"), "--dry-run"]
    output = subprocess.check_output(cmd, cwd=tmp_path, text=True)
    assert f"attention_backend={backend}" in output


def test_template_dry_run(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    cfg = {"run_name": "smoke", "prompts": [{"slug": "tile", "prompt": "a red kite"}]}
    (code / "prompts.json").write_text(json.dumps(cfg))
    cmd = [sys.executable, str(ROOT / "recipes" / "_template.py"), "--dry-run"]
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert "a red kite" in result.stdout and not (tmp_path / "runs").exists()


@pytest.mark.parametrize("existing", [None, "existing"])
@pytest.mark.parametrize("fails", [False, True])
def test_job_source_drive_token_lifecycle(tmp_path, monkeypatch, existing, fails):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    monkeypatch.setenv("COLAB_ULTRA_EXEC_LOG", str(tmp_path / "exec.log"))
    monkeypatch.setenv("DRIVE_ACCESS_TOKEN", existing or "placeholder")
    if existing is None:
        monkeypatch.delenv("DRIVE_ACCESS_TOKEN")

    code = f'assert os.environ["DRIVE_ACCESS_TOKEN"] == {existing or "tok"!r}'
    if fails:
        code += '\nraise RuntimeError("job failed")'
    source = session._job_source(code, "drive.py", "drive.py", "", drive_token="tok")
    source = source.replace("/content/", str(tmp_path) + "/")
    if fails:
        with pytest.raises(RuntimeError, match="job failed"):
            exec(source, {})
    else:
        exec(source, {})

    assert os.environ.get("DRIVE_ACCESS_TOKEN") == existing


def test_colab_exec_forwards_short_lived_drive_token(monkeypatch):
    token = Mock(return_value="tok")
    wrap = Mock(return_value="wrapped code")
    run = Mock(return_value=(0, "", ""))
    monkeypatch.setattr(session, "get_short_lived_drive_token", token)
    monkeypatch.setattr(session, "_load_local_hf_token", lambda: None)
    monkeypatch.setattr(session, "_load_kaggle_sessions", lambda: {})
    monkeypatch.setattr(session, "_job_source", wrap)
    monkeypatch.setattr(session, "run_detached", run)
    args = SimpleNamespace(code="print(1)", file=None, session="ultra")

    session.cmd_exec(args)

    token.assert_called_once_with()
    assert wrap.call_args.kwargs["drive_token"] == "tok"
    run.assert_called_once_with(args, "wrapped code")


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
