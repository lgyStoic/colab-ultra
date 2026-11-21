import builtins
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import URLError

import pytest

from colab_ultra import cli, models_cli, session
from colab_ultra.remote import models, stage


@pytest.fixture
def variant(tmp_path):
    path = tmp_path / "models/demo/v1"
    (path / "encoder").mkdir(parents=True)
    (path / "encoder/config").write_bytes(b"abc")
    (path / "weight.bin").write_bytes(b"12345")
    models.write_manifest(path, {"model": "demo", "variant": "v1", "files": {
        "encoder/config": {"bytes": 3}, "weight.bin": {"bytes": 5}}})
    return path


def model_cmd(hub, action, *opts):
    cli.main(["models", action, "--hub", str(hub), *opts])


def test_list_stat_only(tmp_path, variant, capsys, monkeypatch):
    root = tmp_path / "models"
    for name in (".cache", "__pycache__", "._folder"):
        (variant / name).mkdir()
        (variant / name / "hidden").touch()
    for name in (".DS_Store", "._weight.bin"):
        (variant / name).touch()
    for name in ("Bad", ".cache", "__pycache__", "._model"):
        (root / name / "v1").mkdir(parents=True)
    (variant.parent / "INVALID").mkdir()
    (variant / "link").symlink_to(variant / "weight.bin")
    (variant / "nested").symlink_to(variant)
    (variant.parent / "linked").symlink_to(variant)
    (root / "mirror").symlink_to(variant.parent)
    (root / "other/v2").mkdir(parents=True)
    (root / "other/v3").mkdir()
    (root / "other/v3/manifest.json").write_text("{")
    # A model payload must never be opened, including via Path.read_*().
    def guard(open_file):
        def checked(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).is_relative_to(root):
                assert Path(path).name == "manifest.json", path
            return open_file(path, *args, **kwargs)
        return checked
    monkeypatch.setattr(io, "open", guard(io.open))
    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    model_cmd(tmp_path, "ls", "--json")
    listing = json.loads(capsys.readouterr().out)
    assert [(r["model"], r["variant"], r["manifest"]) for r in listing["variants"]] == [
        ("demo", "v1", "OK"), ("other", "v2", "MISSING"), ("other", "v3", "MISMATCH")]
    assert listing["totals"] == {"variants": 3, "files": 2, "bytes": 8}
    model_cmd(tmp_path, "ls")
    rendered = capsys.readouterr().out
    assert all(label in rendered for label in (
        "model/variant", "files", "size (GB)", "manifest", "Total", "OK", "MISSING", "MISMATCH"))
    (variant / "weight.bin").unlink()
    model_cmd(tmp_path, "ls", "--json")
    assert json.loads(capsys.readouterr().out)["variants"][0]["manifest"] == "MISMATCH"


@pytest.mark.parametrize("detected", [False, True])
def test_default_hub(tmp_path, monkeypatch, capsys, detected):
    hub = tmp_path / ("Drive" if detected else "ColabUltra")
    (hub / "models/demo/v1").mkdir(parents=True)
    lookup = Mock(return_value=str(hub) if detected else None)
    monkeypatch.setattr(models_cli, "detect_local_drive_hub", lookup)
    cli.main(["models", "ls", "--json"])
    assert json.loads(capsys.readouterr().out)["totals"]["variants"] == 1
    lookup.assert_called_once_with(create=False)


def test_verify_one_or_all(tmp_path, variant, capsys):
    (tmp_path / "models/other/v1").mkdir(parents=True)
    model_cmd(tmp_path, "verify", "demo/v1")
    assert "demo/v1  OK" in capsys.readouterr().out
    (variant / "encoder/config").unlink()
    (variant / "weight.bin").write_bytes(b"x")
    (variant / "new").touch()
    with pytest.raises(SystemExit) as exc:
        model_cmd(tmp_path, "verify", "--json")
    assert exc.value.code == 1
    demo, other = json.loads(capsys.readouterr().out)["variants"]
    assert demo["manifest"] == "MISMATCH" and other["manifest"] == "MISSING"
    assert demo["issues"] == [{"kind": "missing", "path": "encoder/config"},
                              {"kind": "untracked", "path": "new"},
                              {"kind": "size_mismatch", "path": "weight.bin", "expected": 5, "actual": 1}]
    assert other["issues"][0]["kind"] == "invalid_manifest"


@pytest.mark.parametrize("invalid", [None, "{", "[]", {"model": "wrong"}, {"variant": "wrong"},
                                     {"files": {"a": {"bytes": -5}}}, {"files": {"../escape": {"bytes": 0}}}])
def test_verify_invalid_manifest(tmp_path, variant, capsys, invalid):
    path = variant / "manifest.json"
    if invalid is None:
        path.unlink()
    elif isinstance(invalid, str):
        path.write_text(invalid)
    else:
        manifest = models.read_manifest(variant)
        path.write_text(json.dumps({**manifest, **invalid}))
    with pytest.raises(SystemExit) as exc:
        model_cmd(tmp_path, "verify", "demo/v1", "--json")
    assert exc.value.code == 1
    row, = json.loads(capsys.readouterr().out)["variants"]
    assert row["issues"][0]["kind"] == "invalid_manifest"


@pytest.mark.parametrize("target", ["demo", "demo/v1/x", "../demo", "demo/../v1", "/abs/v1",
                                    "demo/", "Demo/v1", "demo/missing", "linked/v1", "demo/link"])
@pytest.mark.parametrize("action", ["verify", "rm"])
def test_refuse_target(tmp_path, variant, target, action):
    (variant.parent.parent / "linked").symlink_to(variant.parent)
    (variant.parent / "link").symlink_to(variant)
    with pytest.raises(SystemExit) as exc:
        model_cmd(tmp_path, action, target, *(["--yes"] if action == "rm" else []))
    assert exc.value.code and (variant / "weight.bin").exists()


def test_rm_preview_and_empty_parent(tmp_path, variant, capsys):
    model_cmd(tmp_path, "rm", "demo/v1")
    assert (variant / "weight.bin").exists()
    out = capsys.readouterr().out
    assert "Dry run" in out and "2 files" in out and "GB" in out
    (variant.parent / "v2").mkdir()
    model_cmd(tmp_path, "rm", "demo/v1", "--yes")
    assert not variant.exists() and variant.parent.exists()
    model_cmd(tmp_path, "rm", "demo/v2", "--yes")
    assert not variant.parent.exists() and (tmp_path / "models").is_dir()


@pytest.mark.parametrize("failure", [ValueError("missing"), OSError("short"), URLError("timeout"),
                                     subprocess.CalledProcessError(7, "fetch")])
@pytest.mark.parametrize("default_dest", [True, False])
def test_kaggle_stages_one_copy(tmp_path, monkeypatch, failure, default_dest):
    monkeypatch.setenv("COLAB_ULTRA_KAGGLE", "1")
    monkeypatch.setenv("DRIVE_ACCESS_TOKEN", "test")
    monkeypatch.setattr(stage, "DRIVE", tmp_path / "drive")
    monkeypatch.setattr(stage, "NVME", tmp_path / "nvme")
    vdir = models.variant_dir("demo", "v1")
    target = vdir if default_dest else tmp_path / "target"
    target.mkdir(parents=True)
    (target / "ready").write_bytes(b"ready")
    files = {"ready": "https://test/ready", "nested/drive": "https://test/drive",
             "missing": "hf://org/repo/source"}
    plan = Mock(wraps=stage.plan)
    monkeypatch.setattr(stage, "plan", plan)

    def transfer(item, temp):
        assert not item.copy_to_drive and item.work_dir == item.dest.parent
        assert item.dest.is_relative_to(target) and item.force
        if item.source == str(vdir / "missing"):
            temp.write_bytes(b"partial")
            raise failure
        temp.write_bytes(b"fetched")

    monkeypatch.setattr(stage, "_download", transfer)
    paths = models.ensure("demo", "v1", files, to=None if default_dest else target, revision="v2")
    assert paths == {rel: target / rel for rel in files}
    assert paths["ready"].read_bytes() == b"ready"
    assert paths["missing"].read_bytes() == paths["nested/drive"].read_bytes() == b"fetched"
    assert not (vdir / "manifest.json").exists()
    if not default_dest:
        assert not vdir.exists()
    requests = [c.args[0] for c in plan.call_args_list]
    drive = "drive://ColabUltra/models/demo/v1/"
    origin = "https://huggingface.co/org/repo/resolve/v2/source"
    assert sorted(requests) == sorted([drive + "nested/drive", drive + "missing", origin])
    assert requests.index(drive + "missing") < requests.index(origin)


def test_kaggle_native_drive_download_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_ULTRA_KAGGLE", "1")
    monkeypatch.setenv("DRIVE_ACCESS_TOKEN", "test")
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    monkeypatch.setattr(stage, "_drive_media", lambda *args: ("https://drive.test/media", 3))
    transfers = []

    def download(url, filename, headers, **kwargs):
        transfers.append(url)
        if url == "https://drive.test/media":
            assert headers["Authorization"] == "Bearer test"
            raise Exception("native download failed")
        assert "Authorization" not in headers
        Path(filename).write_bytes(b"abc")

    monkeypatch.setitem(sys.modules, "hf_transfer", SimpleNamespace(download=download))
    paths = models.ensure("demo", "v1", {"cfg": "https://origin.test/cfg"}, to=tmp_path)
    assert transfers == ["https://drive.test/media", "https://origin.test/cfg"]
    assert paths["cfg"].read_bytes() == b"abc"


def test_kaggle_propagates_fallback_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_ULTRA_KAGGLE", "1")
    monkeypatch.setenv("DRIVE_ACCESS_TOKEN", "test")
    run = Mock(side_effect=[ValueError("missing"), OSError("origin failed")])
    monkeypatch.setattr(stage, "run", run)
    with pytest.raises(OSError, match="origin failed"):
        models.ensure("demo", "v1", {"a": "https://example.test/a"}, to=tmp_path)
    assert run.call_count == 2 and list(tmp_path.iterdir()) == []


@pytest.fixture
def local_job(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_ULTRA_EXEC_LOG", str(tmp_path / "exec.log"))
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTHONUNBUFFERED", "COLAB_ULTRA_KAGGLE", "DRIVE_ACCESS_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    def execute(source):
        exec(source.replace("/content/", str(tmp_path) + "/"), {})
    return execute


@pytest.mark.parametrize("fails", [False, True])
def test_tee_latest(tmp_path, local_job, capsys, fails):
    logfile = tmp_path / "exec.log"
    logfile.write_text("obsolete output")
    stdout, stderr = sys.stdout, sys.stderr
    code = """print("normal")
print("error output", file=sys.stderr, flush=True)
assert sys.stdout.encoding and isinstance(sys.stdout.isatty(), bool)
sys.stdout.write("partial")"""
    if fails:
        code += '\nraise ValueError("script failed")'
    source = session._job_source(code, "script.py", "demo/script.py", "")
    if fails:
        with pytest.raises(ValueError, match="script failed"):
            local_job(source)
    else:
        local_job(source)
    assert sys.stdout is stdout and sys.stderr is stderr
    assert not (tmp_path / ".task_heartbeat.json").exists()
    out = capsys.readouterr()
    assert out.out == "normal\npartial" and out.err == "error output\n"
    lines = logfile.read_text().splitlines()
    assert lines[0].startswith("[colab-ultra] started ") and lines[0].endswith(" task=demo/script.py")
    assert lines[-1].startswith("[colab-ultra] finished ")
    assert lines[-1].endswith(f" task=demo/script.py rc={int(fails)}")
    for line in (lines[0], lines[-1]):
        assert datetime.fromisoformat(line.split()[2]).utcoffset() == timedelta(0)
    assert "obsolete output" not in lines and lines[1:3] == ["normal", "error output"]


def test_unavailable_log(tmp_path, monkeypatch, local_job, capsys):
    monkeypatch.setenv("COLAB_ULTRA_EXEC_LOG", str(tmp_path))  # opening a directory for write fails
    local_job(session._job_source('print("still running")', "script", "task", ""))
    assert capsys.readouterr().out == "still running\n"


def test_latest_exec_owns_log(tmp_path, local_job):
    logfile = tmp_path / "exec.log"
    with logfile.open("w") as previous_job:
        local_job(session._job_source('print("new output")', "new", "new", ""))
        previous_job.write("old job completed late\n")
    lines = logfile.read_text().splitlines()
    assert lines[1] == "new output" and lines[-1].endswith("task=new rc=0")


@pytest.mark.parametrize("token,fails", [("short-lived", False), ("short-lived", True), ("", False)])
def test_kaggle_exec_env(monkeypatch, local_job, token, fails):
    monkeypatch.setattr(session, "_load_kaggle_sessions", lambda: {"ultra": {"proxy_url": "url"}})
    monkeypatch.setattr(session, "assignments", lambda: [])
    monkeypatch.setattr(session, "kaggle_active", lambda sess: True)
    monkeypatch.setattr(session, "detect_local_drive_hub", lambda: None)
    monkeypatch.setattr(session, "get_short_lived_drive_token", lambda: token)
    monkeypatch.setattr(session, "kaggle_runtime_code", lambda: "")
    def remote(url, code, **kwargs):
        local_job(code)
        return 0, "", ""
    monkeypatch.setattr(session, "jupyter_exec_ws", remote)
    code = ('assert os.environ["COLAB_ULTRA_KAGGLE"] == "1"\n'
            f'assert os.environ.get("DRIVE_ACCESS_TOKEN") == {token or None!r}')
    if fails:
        with pytest.raises(RuntimeError, match="job failure"):
            cli.main(["exec", code + '\nraise RuntimeError("job failure")'])
    else:
        cli.main(["exec", code])
    assert "DRIVE_ACCESS_TOKEN" not in os.environ


@pytest.mark.parametrize("provider", ["colab", "kaggle"])
@pytest.mark.parametrize("lines,tail", [(2, "two\nthree"), (0, "one\ntwo\nthree")])
def test_logs(provider, lines, tail, monkeypatch, capsys):
    kaggle = provider == "kaggle"
    monkeypatch.setattr(session, "_load_kaggle_sessions", lambda: {"job": {"proxy_url": "ku"}} if kaggle else {})
    resolve = Mock(return_value=("cu", "ct"))
    monkeypatch.setattr(session, "resolve_proxy", resolve)
    active = Mock(side_effect=AssertionError("healthy proxy does not need a second check"))
    monkeypatch.setattr(session, "kaggle_active", active)
    hb = {"active": True, "task": "demo/script.py"}
    def read(url, token, path):
        assert (url, token) == (("ku", "") if kaggle else ("cu", "ct"))
        return {"/content/.cu_last_exec.log": b"one\ntwo\nthree",
                "/content/.task_heartbeat.json": json.dumps(hb).encode()}[path]
    monkeypatch.setattr(session, "jupyter_get_bytes", read)
    args = ["logs", "-s", "job", "-n", str(lines)]
    cli.main([*args, "--json"])
    assert json.loads(capsys.readouterr().out) == {
        "ok": True, "session": "job", "provider": provider, "active_task": hb,
        "exists": True, "lines_total": 3, "log": tail}
    cli.main(args)
    assert capsys.readouterr().out == "Active task: demo/script.py\n" + tail + "\n"
    assert resolve.call_count == (0 if kaggle else 2)


@pytest.mark.parametrize("raw", [None, b""])
def test_logs_before_first_output(monkeypatch, capsys, raw):
    monkeypatch.setattr(session, "_load_kaggle_sessions", lambda: {})
    monkeypatch.setattr(session, "resolve_proxy", lambda _: ("url", "tok"))
    monkeypatch.setattr(session, "jupyter_get_bytes",
                        lambda url, tok, path: raw if path.endswith(".log") else b"{")
    cli.main(["logs", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["exists"] is (raw is not None) and data["active_task"] is None
    assert data["log"] == "" and data["lines_total"] == 0
    if raw is None:
        cli.main(["logs"])
        assert "No exec log yet" in capsys.readouterr().out


@pytest.mark.parametrize("url,active", [("", False), ("url", False), ("", True), ("url", True)])
def test_missing_kaggle_log_checks_session(monkeypatch, url, active, capsys):
    vm = {"proxy_url": url}
    monkeypatch.setattr(session, "_load_kaggle_sessions", lambda: {"ultra": vm})
    check = Mock(return_value=active)
    monkeypatch.setattr(session, "kaggle_active", check)
    monkeypatch.setattr(session, "jupyter_get_bytes", Mock(return_value=None))
    if url and active:
        cli.main(["logs"])
        assert "No exec log yet" in capsys.readouterr().out
    else:
        with pytest.raises(SystemExit, match="no proxy URL" if active else "Lost Kaggle session"):
            cli.main(["logs"])
    check.assert_called_once_with(vm)
