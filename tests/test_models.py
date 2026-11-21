import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from colab_ultra import cli, session, staging
from colab_ultra.remote import models, stage
from tools import migrate_models as migration


@pytest.fixture
def mount(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "DRIVE", tmp_path / "drive")
    monkeypatch.setattr(stage, "NVME", tmp_path / "nvme")
    monkeypatch.delenv("DRIVE_ACCESS_TOKEN", raising=False)
    return tmp_path / "drive/MyDrive/ColabUltra/models"


def test_variant_names(mount):
    assert models.variant_dir("a0._-z", "gguf-q4_k_m") == mount / "a0._-z/gguf-q4_k_m"
    for bad in ("", ".", "..", "/", "a/b", "_abc", ".hidden", "Bf16", "a ", "a\n", "é"):
        with pytest.raises(ValueError):
            models.variant_dir(bad, "bf16")
        with pytest.raises(ValueError):
            models.variant_dir("model", bad)


@pytest.mark.parametrize("key", ["", "/file", "a//b", "../x", "x/../y", "./a", "a/", "a\\b",
                                 "manifest.json", "manifest.json/subdir"])
def test_bad_file_keys_are_rejected_before_writes(mount, key):
    with pytest.raises(ValueError):
        models.ensure("demo", "bf16", {"valid": "https://a.test/v", key: "https://a.test/f"})
    assert not mount.exists()


def test_ensure_download_reuse_metadata_and_nvme(mount, tmp_path, monkeypatch):
    files = {"vae/weights.gguf": "hf://org/repo/weights.gguf",
             "sidecar.json": "https://huggingface.co/org/repo/resolve/tag%2Fv1/config.json"}
    api = Mock()
    digest = "a" * 64

    def metadata(repo, paths, *, revision, expand):
        assert repo == "org/repo" and revision == "tag/v1" and expand
        return [SimpleNamespace(path=p, lfs=SimpleNamespace(sha256=digest) if p.endswith("gguf") else None)
                for p in paths]

    api.get_paths_info.side_effect = metadata
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api))
    weight = b"x" * (stage.MIN_BYTES + 1)
    fetch = Mock(side_effect=lambda plan, temp: temp.write_bytes(weight if plan.min_bytes else b"{}"))
    original = stage._download
    monkeypatch.setattr(stage, "_download", fetch)
    paths = models.ensure("demo", "bf16", files, revision="tag/v1")
    vdir = mount / "demo/bf16"
    assert paths == {rel: vdir / rel for rel in files}
    assert fetch.call_count == 2
    assert {call.args[0].source for call in fetch.call_args_list} == {
        "https://huggingface.co/org/repo/resolve/tag%2Fv1/weights.gguf", files["sidecar.json"]}
    manifest = models.read_manifest(vdir)
    assert manifest["files"] == {
        "vae/weights.gguf": {"source": files["vae/weights.gguf"], "bytes": len(weight), "sha256": digest},
        "sidecar.json": {"source": files["sidecar.json"], "bytes": 2, "sha256": None}}
    api.get_paths_info.assert_called_once()
    manifest["files"]["vae/weights.gguf"].update(source=None, sha256=None)
    models.write_manifest(vdir, manifest)

    models.ensure("demo", "bf16", {"vae/weights.gguf": files["vae/weights.gguf"]}, revision="tag/v1")
    assert fetch.call_count == 2
    assert models.read_manifest(vdir)["files"]["vae/weights.gguf"]["sha256"] == digest
    assert models.read_manifest(vdir)["files"]["vae/weights.gguf"]["source"] == files["vae/weights.gguf"]
    assert "sidecar.json" in models.read_manifest(vdir)["files"]

    # A size mismatch above the normal staging floor must still cause a download.
    paths["vae/weights.gguf"].write_bytes(weight + b"?")
    api.get_paths_info.side_effect = ConnectionError("offline metadata")
    models.ensure("demo", "bf16", files, revision="tag/v1")
    assert fetch.call_count == 3 and paths["vae/weights.gguf"].stat().st_size == len(weight)
    assert models.read_manifest(vdir)["files"]["vae/weights.gguf"]["sha256"] is None

    monkeypatch.setattr(stage, "_download", original)
    plans = Mock(wraps=stage.plan)
    monkeypatch.setattr(stage, "plan", plans)
    local = tmp_path / "local"
    copied = models.ensure("demo", "bf16", files, to=local, revision="tag/v1")
    assert copied == {rel: local / rel for rel in files}
    assert copied["vae/weights.gguf"].read_bytes() == weight
    assert copied["sidecar.json"].read_text() == "{}"
    assert {call.args[0] for call in plans.call_args_list if call.args[0].startswith("drive://")} == {
        f"drive://ColabUltra/models/demo/bf16/{rel}" for rel in files}


def test_describe(mount, tmp_path, capsys):
    dest = tmp_path / "local"
    models.describe("demo", "bf16", {"part/w.pt": "hf://org/repo/w.pt"}, to=dest)
    out = capsys.readouterr().out
    assert str(mount / "demo/bf16/part/w.pt") in out and str(dest / "part/w.pt") in out
    assert "hf_transfer" in out and "https://huggingface.co/org/repo/resolve/main/w.pt" in out
    assert not mount.exists() and not dest.exists()


def put(hub, rel, data=b"abc"):
    path = hub / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def tree(hub):
    return {str(p.relative_to(hub)): (p.stat().st_size, p.stat().st_mtime_ns) for p in hub.rglob("*")}


@pytest.fixture
def hub(tmp_path):
    hub = tmp_path / "hub"
    primary = "models/MiniMaxAI/MiniMax-H3-Image"
    mirror = "huizhou-h3/models/MiniMax/MiniMax-H3-Image"
    for base in (primary, mirror):
        put(hub, base + "/processor/vocab.json")
        put(hub, base + "/vae/diffusion_pytorch_model.safetensors")
    put(hub, "models/h3gguf/diffusion_pytorch_model.safetensors")
    put(hub, primary + "/.gitattributes")
    put(hub, "models/minimax-h3-image/bf16/.gitattributes")
    put(hub, "models/MiniMaxAI/weights_verified.json")
    put(hub, "huizhou-h3/models/download_manifest.json")
    put(hub, "moss-distill/models/MOSS-Image-SFT/text_encoder/config.json")
    for skip in (".cache/info", "__pycache__/a.pyc", "._weights", ".DS_Store"):
        put(hub, "moss-distill/models/MOSS-Image-SFT/" + skip)
    for d in ("models/llm", "huizhou16k/models", "moss-distill/models/checkpoints"):
        (hub / d).mkdir(parents=True)
    return hub


def test_migrate_inventory_and_abort(hub, tmp_path, capsys):
    conflict = put(hub, "models/minimax-h3-image/bf16/processor/vocab.json", b"x")
    # Include a conflict between two sources before either exists at its destination.
    put(hub, "huizhou-h3/models/MiniMax/MiniMax-H3-Image/vae/diffusion_pytorch_model.safetensors", b"x")
    snapshot = tree(hub)
    report = tmp_path / "report.json"
    assert migration.main(["--hub", str(hub), "--report", str(report)]) == 0
    assert tree(hub) == snapshot
    data = json.loads(report.read_text())
    assert all(data["totals"][action]["count"] > 0 for action in migration.ACTIONS)
    assert data["totals"]["CONFLICT"]["count"] == 3
    assert {x["src"] for x in data["items"] if x["action"] == "UNMAPPED"} == {
        "models/MiniMaxAI/weights_verified.json", "huizhou-h3/models/download_manifest.json"}
    assert not any(migration.ignored(s) for x in data["items"] for s in Path(x["src"]).parts)
    assert (str(conflict.relative_to(hub)) in capsys.readouterr().out)
    assert migration.main(["--hub", str(hub), "--apply"]) != 0
    assert tree(hub) == snapshot


def test_migrate_apply_and_repeat(hub):
    vdir = hub / "models/minimax-h3-image/bf16"
    saved = {"source": "https://example.test/attr", "bytes": 3, "sha256": None}
    models.write_manifest(vdir, {"model": "minimax-h3-image", "variant": "bf16", "files": {".gitattributes": saved}})
    primary = hub / "models/MiniMaxAI/MiniMax-H3-Image"
    original_inode = (primary / "processor/vocab.json").stat().st_ino
    assert migration.main(["--hub", str(hub), "--apply"]) == 0
    assert (vdir / "processor/vocab.json").stat().st_ino == original_inode
    entries = models.read_manifest(vdir)["files"]
    assert entries[".gitattributes"] == saved
    assert entries["vae/diffusion_pytorch_model.safetensors"] == {"bytes": 3, "source": None, "sha256": None}
    assert not primary.exists() and not (hub / "models/h3gguf").exists()
    assert not (hub / "huizhou-h3/models/MiniMax").exists()
    assert not (hub / "models/llm").exists() and not (hub / "huizhou16k/models").exists()
    assert (hub / "huizhou-h3/models/download_manifest.json").exists()
    assert (hub / "moss-distill/models/MOSS-Image-SFT/.cache/info").exists()
    assert not (hub / "moss-distill/models/checkpoints").exists()
    assert all(i.action == "UNMAPPED" for i in migration.scan(hub))
    snapshot = tree(hub)
    assert migration.main(["--hub", str(hub), "--apply"]) == 0
    assert tree(hub) == snapshot


@pytest.mark.parametrize("project,dest", [(None, "models/foo/bf16"), ("proj", "models"), (None, "datasets")])
def test_pull_cli_and_generated_code(project, dest, monkeypatch):
    # Run the generated transfer body with the remote package and provisioning mocked.
    for name in ("cmd_up", "cmd_down", "cmd_exec"):
        monkeypatch.setattr(session, name, Mock())
    source = "https://example.test/file.txt"
    argv = ["pull", source, "--dest", dest]
    if project:
        argv.extend(["-p", project])
    cli.main(argv)
    args = session.cmd_exec.call_args.args[0]
    assert args.project == project and args.code == staging.pull_code(source, project, dest)
    vm_stage, vm_models = Mock(wraps=stage), Mock(wraps=models)
    vm_stage.run_many = Mock()
    vm_models.ensure = Mock()
    monkeypatch.setitem(sys.modules, "cu_remote", SimpleNamespace(models=vm_models, stage=vm_stage))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    exec(args.code.split("import time\nos.sync()")[0], {})
    if not project and dest.startswith("models/"):
        vm_models.ensure.assert_called_once_with("foo", "bf16", {"file.txt": source})
    else:
        target = Path("/content/drive/MyDrive/ColabUltra") / (project or "") / dest
        vm_stage.plan.assert_called_once_with(source, target / "file.txt", min_bytes=0)
    session.cmd_down.assert_called_once()


@pytest.mark.parametrize("dest", ["models", "models/foo", "models/foo/../bad", "models/Foo/bf16"])
def test_pull_rejects_invalid_destination_before_allocation(dest, monkeypatch):
    monkeypatch.setattr(session, "cmd_up", Mock())
    with pytest.raises(ValueError):
        staging.pull_code("org/repo", dest=dest)
    argv = ["pull", "org/repo"]
    if dest != "models":
        argv += ["--dest", dest]
    with pytest.raises(SystemExit):
        cli.main(argv)
    session.cmd_up.assert_not_called()


def test_shared_repo_pull_preserves_subdirectories(monkeypatch):
    api = Mock()
    api.list_repo_files.return_value = ["sub/config.json", "sub/weights.gguf"]
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api))
    vm_models = Mock(wraps=models)
    vm_models.ensure = Mock()
    monkeypatch.setitem(sys.modules, "cu_remote", SimpleNamespace(models=vm_models, stage=stage))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    code = staging.pull_code("org/repo", dest="models/foo/bf16/release", filename="*.gguf")
    exec(code.split("import time\nos.sync()")[0], {})
    vm_models.ensure.assert_called_once_with("foo", "bf16", {
        "release/sub/weights.gguf": "https://huggingface.co/org/repo/resolve/main/sub/weights.gguf"})


@pytest.mark.parametrize("source,dest", [
    ("models/h3gguf/minimax_h3_image_pruned-UD-Q4_K_XL.gguf",
     "models/minimax-h3-image/gguf-ud-q4_k_xl/minimax_h3_image_pruned-UD-Q4_K_XL.gguf"),
    ("models/h3gguf/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
     "models/minimax-h3-qwen3vl-32b/gguf-q4_k_m/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"),
    ("huizhou-h3/models/Comfy-Org/MiniMax-H3-Image/text_encoders/q.safetensors",
     "models/minimax-h3-image/comfy-int8-convrot/text_encoders/q.safetensors"),
    ("moss-distill/models/lodestones_8step/time_embedding.safetensors",
     "models/moss-image-sft/lora-lodestones-8step/time_embedding.safetensors"),
])
def test_other_migration_variants(tmp_path, source, dest):
    put(tmp_path, source)
    assert migration.scan(tmp_path)[0] == migration.Item("MOVE", source, dest, 3)


def test_report_cannot_truncate_a_payload(hub):
    payload = hub / "models/h3gguf/diffusion_pytorch_model.safetensors"
    original = payload.read_bytes()
    assert migration.main(["--hub", str(hub), "--report", str(payload)]) != 0
    assert payload.read_bytes() == original
