#!/usr/bin/env python3
"""Run one H3 pipeline stage after `colab-ultra sync-remote`.

Submit prompt tasks to the embed queue and run embed, dit, and vae workers
against the same project. Configure the queues in code/pipeline_worker.json.
Set COLAB_ULTRA_WORKER_ROLE and COLAB_ULTRA_WORKER_ID for `exec -f`, or use
--role and --worker-id when running this file directly. For Upstash REST, put
the token in the Redis URL's userinfo. The DiT stage accepts prompts with or
without reference images.
Use embed_vae on Kaggle to run embed alongside a VAE worker on GPU 1,
then start a VAE worker on GPU 0 once embed exits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import lzma
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import zlib
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

STAGE_ROLES = ("embed", "dit", "vae")
ROLES = (*STAGE_ROLES, "embed_vae")
WORKER_ENV_FILE = Path("/tmp/.cu_worker_env.json")
REDIS_URL_FILE = Path("/tmp/.cu_redis_url")
REMOTE_SCRIPTS = Path("/content/drive/MyDrive/ColabUltra/scripts")
H3_CACHE_DIR = Path("/content/.cache/h3_infer")
DRIVE_COMPILE_CACHE_DIR = Path("/content/drive/MyDrive/ColabUltra/.cache/h3_infer")
_DRIVE_COMPILE_CACHE_PARTS = ("ColabUltra", ".cache", "h3_infer")
_COMPILE_CACHE_SUBDIRS = ("triton", "inductor")
_ACTIVE_COMPILE_CACHE: tuple[Path | None, int] = (None, 0)
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DEFAULTS = {
    "run_name": "pipe_demo",
    "embed_queue": "h3_pipe_embed",
    "dit_queue": "h3_pipe_dit",
    "vae_queue": "h3_pipe_vae",
    "embeds_dir": "runs/h3_embeds_cache",
    "dit_batch_size": 4,
    "dit_variant": "bf16",
    "reclaim_idle_ms": 300_000,
    "poll_interval_sec": 2.0,
    "drive_wait_sec": 90.0,
    "attention_backend": "auto",
    "compile_mode": "regional",
    "te_offload_layers": 0,
    "vae_device": "auto",
    "vae_tile_batch_size": None,
    "width": 2048,
    "height": 2048,
    "output_format": "png",
    "seed_base": 100,
}

# exec -f receives this source alone. Keep the model manifests in sync with minimax_h3_infer.py.
IMAGE_GROUP = ("minimax-h3-image", "bf16")
TE_GROUP = ("minimax-h3-qwen3vl-32b", "int8-convrot")
TE_FILENAME = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
TE_FILES = {TE_FILENAME: f"hf://Comfy-Org/MiniMax-H3-Image/text_encoders/{TE_FILENAME}"}
TE_SPLIT_LAYER_IDX = 22
TE_NUM_LAYERS = 50
EMBED_VAE_OFFLOAD_LAYERS = 14
EMBED_VAE_CONCURRENT_VAE_TILE_BATCH_SIZE = 8
CANVAS_MULTIPLE = 32
VISION_MIN_SHORT_EDGE = 1024
VISION_MAX_SHORT_EDGE = 1024
TEXT_TAG = 1
IMAGE_TAG = 0
MAX_SEQ_LEN = 4096
BF16_FILES = {
    name: f"hf://MiniMaxAI/MiniMax-H3-Image/{name}"
    for name in (
        "pdd_config.json",
        "pdd_fused_heads.safetensors",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer/model.safetensors.index.json",
        *(f"transformer/model-{i:05d}-of-00012.safetensors" for i in range(1, 13)),
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "processor/chat_template.json",
        "processor/merges.txt",
        "processor/preprocessor_config.json",
        "processor/tokenizer.json",
        "processor/tokenizer_config.json",
        "processor/video_preprocessor_config.json",
        "processor/vocab.json",
    )
}
INT8_CONVROT_GROUP = ("minimax-h3-image", "int8-convrot")
INT8_CONVROT_FILENAME = "minimax_h3_image_pruned_int8_convrot.safetensors"
INT8_CONVROT_FILES = {
    INT8_CONVROT_FILENAME: f"hf://Comfy-Org/MiniMax-H3-Image/diffusion_models/{INT8_CONVROT_FILENAME}",
}
SAVE_OPTIONS = {
    "png": {"compress_level": 3},
    "jpg": {"quality": 95},
    "jpeg": {"quality": 95},
    "webp": {"lossless": True, "quality": 100, "method": 4},
}


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def resolve_worker(role=None, worker_id=None):
    role = role or os.environ.get("COLAB_ULTRA_WORKER_ROLE")
    worker_id = worker_id or os.environ.get("COLAB_ULTRA_WORKER_ID")
    if not role or not worker_id:
        worker_env = _read_json(WORKER_ENV_FILE)
        role = role or worker_env.get("role")
        worker_id = worker_id or worker_env.get("worker_id")
    worker_id = worker_id or os.environ.get("COLAB_ULTRA_SESSION") or socket.gethostname()
    if role not in ROLES:
        raise ValueError(f"Set a worker role from {', '.join(ROLES)}")
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise ValueError("worker_id must be a nonempty string")
    return role, worker_id


def resolve_redis_url(code: Path) -> str:
    value = os.environ.get("COLAB_ULTRA_REDIS_URL", "").strip()
    if value:
        return value
    for path in (REDIS_URL_FILE, code / ".redis_url"):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if value:
            return value
    raise ValueError("Set COLAB_ULTRA_REDIS_URL, /tmp/.cu_redis_url, or code/.redis_url")


def _redis_redactor(url):
    secrets = {url, unquote(url)}
    try:
        parts = urlsplit(url)
        secrets.add(parts.password or parts.username or "")
        secrets.update(value for key, value in parse_qsl(parts.query)
                       if key.lower() in {"password", "token", "auth"})
    except ValueError:
        pass
    secrets.update(unquote(secret) for secret in tuple(secrets))
    pattern = re.compile("|".join(re.escape(secret) for secret in sorted(secrets, key=len, reverse=True) if secret))
    return lambda message: pattern.sub("[redacted]", message)


def _safe_name(value):
    return (isinstance(value, str) and value.strip() not in {"", ".", ".."}
            and not any(char in "/\\" or ord(char) < 32 or ord(char) == 127 for char in value))


def _settings(code):
    cfg = _read_json(code / "pipeline_worker.json")
    settings = {key: cfg.get(key, default) for key, default in DEFAULTS.items()}
    names = [f"{role}_queue" for role in STAGE_ROLES]
    for key in ("run_name", *names):
        if not _safe_name(settings[key]):
            raise ValueError(f"{key} must be a safe single-segment name")
    if len({settings[key] for key in names}) != len(STAGE_ROLES):
        raise ValueError("Stage queues must have distinct names")
    embeds_dir = settings["embeds_dir"]
    if not isinstance(embeds_dir, str) or not all(_safe_name(part) for part in embeds_dir.rstrip("/").split("/")):
        raise ValueError("embeds_dir must be a safe relative path")
    settings["dit_batch_size"] = int(settings["dit_batch_size"])
    if settings["dit_batch_size"] < 1:
        raise ValueError("dit_batch_size must be positive")
    dit_variant = os.environ.get("COLAB_ULTRA_DIT_VARIANT", "").strip() or settings["dit_variant"]
    if dit_variant not in ("bf16", "int8_convrot"):
        raise ValueError("dit_variant must be bf16 or int8_convrot")
    settings["dit_variant"] = dit_variant
    for key in ("reclaim_idle_ms", "height", "width", "seed_base"):
        settings[key] = int(settings[key])
    offload_layers = os.environ.get("COLAB_ULTRA_TE_OFFLOAD_LAYERS", "").strip()
    settings["te_offload_layers"] = int(offload_layers or settings["te_offload_layers"])
    if not 0 <= settings["te_offload_layers"] <= TE_NUM_LAYERS:
        raise ValueError("te_offload_layers must be between 0 and 50")
    vae_tile_batch_size = os.environ.get("COLAB_ULTRA_VAE_TILE_BATCH_SIZE", "").strip()
    if vae_tile_batch_size or "vae_tile_batch_size" in cfg:
        settings["vae_tile_batch_size"] = int(vae_tile_batch_size or cfg["vae_tile_batch_size"])
        if settings["vae_tile_batch_size"] < 1:
            raise ValueError("vae_tile_batch_size must be positive")
    vae_device = os.environ.get("COLAB_ULTRA_VAE_DEVICE", "").strip() or _read_json(WORKER_ENV_FILE).get(
        "vae_device", settings["vae_device"],
    )
    if not isinstance(vae_device, str) or not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", vae_device):
        raise ValueError("vae_device must be auto, cpu, cuda, or cuda:<int>")
    settings["vae_device"] = vae_device
    for key, minimum in (("poll_interval_sec", 0.001), ("drive_wait_sec", 0.0)):
        value = float(settings[key])
        if not math.isfinite(value) or value < minimum:
            raise ValueError(f"{key} must be finite and at least {minimum}")
        settings[key] = value
    if settings["reclaim_idle_ms"] < 0 or min(settings[key] for key in ("width", "height")) < 1:
        raise ValueError("Dimensions must be positive, reclaim_idle_ms must be nonnegative")
    settings["output_format"] = _output_format(settings["output_format"])
    return settings


def _output_format(value):
    fmt = str(value).lower().lstrip(".")
    if fmt not in SAVE_OPTIONS:
        raise ValueError(f"Unsupported output_format: {fmt!r}")
    return fmt


def _import_remote(module):
    scripts = str(REMOTE_SCRIPTS)
    if REMOTE_SCRIPTS.is_dir() and scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        return importlib.import_module(f"cu_remote.{module}")
    except ModuleNotFoundError as exc:
        if exc.name != "cu_remote":
            raise
    source = globals().get("__file__")
    if source and Path(source).name == "h3_pipeline_worker.py":
        repo = str(Path(source).resolve().parents[1])
        if repo not in sys.path:
            sys.path.insert(0, repo)
    return importlib.import_module(f"colab_ultra.remote.{module}")


def _pip_install(*packages):
    uv = shutil.which("uv")
    if uv and subprocess.run(
        [uv, "pip", "install", "--system", "--python", sys.executable, "-q", *packages]
    ).returncode == 0:
        importlib.invalidate_caches()
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    importlib.invalidate_caches()


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _install_redis():
    major = re.match(r"\d+", _package_version("redis"))
    if major is None or int(major[0]) < 5:
        _pip_install("redis>=5.0")


def _model_scope(role, dit_variant="bf16"):
    if role in ("embed", "embed_vae"):
        prefixes = ("processor/", "vae/") if role == "embed_vae" else ("processor/",)
        return {
            IMAGE_GROUP: {key: url for key, url in BF16_FILES.items() if key.startswith(prefixes)},
            TE_GROUP: TE_FILES,
        }
    if role == "dit":
        if dit_variant == "int8_convrot":
            return {
                IMAGE_GROUP: {
                    key: url for key, url in BF16_FILES.items()
                    if key.startswith("vae/") or key == "scheduler/scheduler_config.json"
                },
                INT8_CONVROT_GROUP: INT8_CONVROT_FILES,
            }
        return {IMAGE_GROUP: {key: url for key, url in BF16_FILES.items() if not key.startswith("processor/")}}
    return {IMAGE_GROUP: {key: url for key, url in BF16_FILES.items() if key.startswith("vae/")}}


def _stage_models(scope):
    content_free = shutil.disk_usage("/content").free if Path("/content").exists() else 0
    tmp_free = shutil.disk_usage("/tmp").free
    nvme_root = Path("/content/models") if content_free >= min(35 * 1024**3, tmp_free) else Path("/tmp/models")
    print(f"[Disk] staging root={nvme_root}", flush=True)
    models = _import_remote("models")
    for group, files in scope.items():
        models.ensure(*group, files, to=nvme_root.joinpath(*group))
    return nvme_root


def _nonempty_file(path):
    try:
        return path.is_file() and path.stat().st_size > 0
    except FileNotFoundError:
        return False


def wait_for_drive_file(path: Path, timeout_sec: float = 90.0, poll_sec: float = 1.0, *, pull=None) -> Path:
    deadline = time.monotonic() + timeout_sec
    while True:
        # Listing the directory refreshes Drive FUSE's cached entries.
        try:
            os.listdir(path.parent)
        except OSError:
            pass
        if _nonempty_file(path):
            return path
        remaining = deadline - time.monotonic()
        if pull is not None and remaining > 0:
            pull(deadline)
            if _nonempty_file(path):
                return path
            remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FileNotFoundError(f"Drive file missing or empty after {timeout_sec:g}s: {path}")
        time.sleep(min(poll_sec, remaining))


def _wait_for_artifact(project_root, relative, timeout_sec):
    _import_remote("models").check_relative(relative)
    path = project_root / relative
    token = _drive_token()
    if not token:
        return wait_for_drive_file(path, timeout_sec=timeout_sec)

    def pull(deadline):
        try:
            drive = _Drive(token)
            parent = drive.folder(("ColabUltra", project_root.name, *Path(relative).parts[:-1]), deadline=deadline)
            entry = drive.find(parent, path.name, deadline=deadline) if parent else None
            if entry and int(entry.get("size", 0)) > 0:
                with _atomic_output(path) as staged:
                    drive.download(entry, staged, deadline=deadline)
        except (OSError, ValueError, KeyError):
            # Drive may have listed the name before its media became available.
            pass

    return wait_for_drive_file(path, timeout_sec=timeout_sec, pull=pull)


def _drive_token():
    return os.environ.get("COLAB_ULTRA_DRIVE_TOKEN") or os.environ.get("DRIVE_ACCESS_TOKEN") or globals().get("_cu_drive_tok")


class _Drive:
    def __init__(self, token):
        import requests

        self.http = requests
        self.headers = {"Authorization": f"Bearer {token}"}

    def request(self, method, url=DRIVE_FILES_URL, *, deadline=None, timeout=15, headers=None, **kwargs):
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("Drive artifact deadline expired")
        response = self.http.request(method, url, headers=self.headers if headers is None else headers,
                                     timeout=timeout, **kwargs)
        response.raise_for_status()
        return response

    def find(self, parent, name, *, folder=False, deadline=None):
        name = name.replace("\\", "\\\\").replace("'", "\\'")
        operator = "=" if folder else "!="
        query = f"'{parent}' in parents and name = '{name}' and mimeType {operator} '{DRIVE_FOLDER_MIME}' and trashed = false"
        params = {"q": query, "fields": "files(id,size)", "orderBy": "modifiedTime desc", "pageSize": 1}
        entries = self.request("GET", params=params, deadline=deadline).json().get("files", [])
        return entries[0] if entries else None

    def folder(self, parts, *, create=False, deadline=None):
        parent = "root"
        for name in parts:
            entry = self.find(parent, name, folder=True, deadline=deadline)
            if entry is None:
                if not create:
                    return None
                entry = self.request("POST", params={"fields": "id"}, json={
                    "name": name, "mimeType": DRIVE_FOLDER_MIME, "parents": [parent],
                }).json()
            parent = entry["id"]
        return parent

    def download(self, entry, dest, *, deadline=None):
        with self.request("GET", f"{DRIVE_FILES_URL}/{entry['id']}", params={"alt": "media"},
                          stream=True, deadline=deadline) as response, dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Drive artifact deadline expired")
                handle.write(chunk)
        if dest.stat().st_size != int(entry["size"]):
            raise OSError("Incomplete Drive artifact")

    def download_python_tree(self, parent, dest):
        dest.mkdir()
        params = {"q": f"'{parent}' in parents and trashed = false", "pageSize": 100,
                  "fields": "nextPageToken,files(id,name,mimeType,size)"}
        while True:
            page = self.request("GET", params=params).json()
            for entry in page.get("files", []):
                name = entry["name"]
                if not _safe_name(name) or name.startswith((".", "__pycache__")):
                    continue
                if entry["mimeType"] == DRIVE_FOLDER_MIME:
                    self.download_python_tree(entry["id"], dest / name)
                elif name.endswith(".py"):
                    self.download(entry, dest / name)
            if not page.get("nextPageToken"):
                return
            params["pageToken"] = page["nextPageToken"]

    def upload(self, path, parts):
        parent = self.folder(parts[:-1], create=True)
        name = parts[-1]
        previous = self.find(parent, name)
        url = "https://www.googleapis.com/upload/drive/v3/files"
        size = str(path.stat().st_size)
        if previous:
            method, url, metadata = "PATCH", f"{url}/{previous['id']}", {"name": name}
        else:
            method, metadata = "POST", {"name": name, "parents": [parent]}
        session = self.request(method, url, params={"uploadType": "resumable"}, json=metadata,
                               headers={**self.headers, "X-Upload-Content-Length": size})
        with path.open("rb") as handle:
            self.request("PUT", session.headers["Location"], data=handle,
                         headers={"Content-Length": size}, timeout=120)


def _publish_artifact(path, project_root, relative):
    uploader = globals().get("_cu_upload_large_to_drive")
    token = _drive_token()
    try:
        if callable(uploader):
            uploader(str(path), relative)
        elif token and (os.environ.get("COLAB_ULTRA_KAGGLE") == "1"
                        or not Path(path).resolve().is_relative_to(Path("/content/drive"))):
            _Drive(token).upload(path, ("ColabUltra", project_root.name, *Path(relative).parts))
    except Exception as exc:
        detail = "" if token else f": {exc}"
        print(f"[Warn] live Drive upload skipped ({type(exc).__name__}){detail}", flush=True)


@contextmanager
def _atomic_output(dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as staging:
        staged = Path(staging) / dest.name
        yield staged
        staged.replace(dest)
    os.sync()


def _configure_compile_cache_env(local_dir: Path) -> None:
    for var, name in (("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor")):
        cache_dir = local_dir / name
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(cache_dir)


def _compile_cache_archive(torch_module) -> Path | None:
    if os.environ.get("COLAB_ULTRA_KAGGLE") == "1":
        if not _drive_token():
            return None
    elif not DRIVE_COMPILE_CACHE_DIR.parent.parent.is_dir():
        return None
    major, minor = torch_module.cuda.get_device_capability(0)
    torch_ver = getattr(torch_module, "__version__", "unknown")
    triton_ver = _package_version("triton") or "unknown"
    torch_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", torch_ver)
    triton_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", triton_ver)
    return DRIVE_COMPILE_CACHE_DIR / f"sm{major}{minor}_pt{torch_tag}_tr{triton_tag}.tar"


def _count_cache_files(local_dir: Path) -> int:
    return sum(
        not path.is_symlink() and path.is_file()
        for name in _COMPILE_CACHE_SUBDIRS
        if not (local_dir / name).is_symlink()
        for path in (local_dir / name).rglob("*")
    )


def _extract_compile_cache(archive: Path, local_dir: Path) -> int:
    if local_dir.is_symlink():
        raise tarfile.ExtractError("compile cache directory is a symlink")
    count = 0
    with tarfile.open(archive, "r:*") as cache_tar:
        members = [(member, Path(member.name)) for member in cache_tar.getmembers()]
        for member, path in members:
            if (path.is_absolute() or not path.parts or ".." in path.parts or "\0" in member.name
                    or path.parts[0] not in _COMPILE_CACHE_SUBDIRS
                    or not (member.isdir() or (len(path.parts) > 1 and member.isfile()))):
                raise tarfile.ExtractError("unsafe compile cache member")
            if any(local_dir.joinpath(*path.parts[:depth]).is_symlink()
                   for depth in range(1, len(path.parts) + 1)):
                raise tarfile.ExtractError("compile cache path contains a symlink")

        for member, path in members:
            target = local_dir / path
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            output = tempfile.NamedTemporaryFile(dir=target.parent, delete=False)
            staged = Path(output.name)
            try:
                with output, cache_tar.extractfile(member) as source:
                    shutil.copyfileobj(source, output)
                staged.chmod(0o600 | (member.mode & 0o100))
                staged.replace(target)
            finally:
                staged.unlink(missing_ok=True)
            count += 1
    return count


def _restore_compile_cache(archive: Path | None, local_dir: Path) -> int:
    try:
        if archive is None:
            return 0
        if _nonempty_file(archive):
            count = _extract_compile_cache(archive, local_dir)
        else:
            if os.environ.get("COLAB_ULTRA_KAGGLE") != "1" or not (token := _drive_token()):
                return 0
            drive = _Drive(token)
            parent = drive.folder(_DRIVE_COMPILE_CACHE_PARTS)
            entry = drive.find(parent, archive.name) if parent else None
            if not entry or int(entry.get("size", 0)) <= 0:
                return 0
            with tempfile.TemporaryDirectory(dir="/tmp") as staging:
                local_archive = Path(staging) / archive.name
                drive.download(entry, local_archive)
                count = _extract_compile_cache(local_archive, local_dir)
    except (OSError, tarfile.TarError, EOFError, lzma.LZMAError, zlib.error, ValueError, KeyError) as exc:
        print(f"[Warn] compile cache restore failed ({type(exc).__name__})", flush=True)
        return 0
    if count:
        print(f"[Cache] restored {count} compile cache files from {archive.name}", flush=True)
    return count


def _save_compile_cache(archive: Path | None, local_dir: Path, initial_files: int) -> bool:
    try:
        count = _count_cache_files(local_dir)
        if archive is None or count == 0:
            return False
        token = None
        if os.environ.get("COLAB_ULTRA_KAGGLE") == "1":
            if (0 < initial_files and count <= initial_files) or not (token := _drive_token()):
                return False
        elif _nonempty_file(archive) and count <= initial_files:
            return False

        def cache_member(member):
            if (member.isdir() or member.isfile()) and not (local_dir / member.name).is_symlink():
                return member
            return None

        with tempfile.TemporaryDirectory(dir="/tmp") as staging:
            local_archive = Path(staging) / archive.name
            with tarfile.open(local_archive, "w", dereference=True) as cache_tar:
                for name in _COMPILE_CACHE_SUBDIRS:
                    if (local_dir / name).is_dir():
                        cache_tar.add(local_dir / name, arcname=name, filter=cache_member)
            if token:
                _Drive(token).upload(local_archive, (*_DRIVE_COMPILE_CACHE_PARTS, archive.name))
            else:
                with _atomic_output(archive) as output:
                    shutil.copyfile(local_archive, output)
    except (OSError, tarfile.TarError) as exc:
        print(f"[Warn] compile cache save failed ({type(exc).__name__})", flush=True)
        return False
    print(f"[Cache] saved {count} compile cache files to {archive.name}", flush=True)
    return True


def _save_active_compile_cache() -> None:
    global _ACTIVE_COMPILE_CACHE

    archive, initial_files = _ACTIVE_COMPILE_CACHE
    try:
        _save_compile_cache(archive, H3_CACHE_DIR, initial_files)
    finally:
        _ACTIVE_COMPILE_CACHE = (None, 0)


def _save_pt(dest, payload):
    import torch

    with tempfile.TemporaryDirectory(dir="/tmp") as local:
        path = Path(local) / "payload.pt"
        torch.save(payload, path)
        with _atomic_output(dest) as staged:
            shutil.copyfile(path, staged)


def _install_embed_dependencies():
    from packaging.version import Version

    missing = []
    for name, minimum in (("transformers", "4.57.0"), ("safetensors", "0.4.3")):
        installed = _package_version(name)
        if not installed or Version(installed) < Version(minimum):
            missing.append(f"{name}>={minimum}")
    if not _package_version("pillow"):
        missing.append("pillow")
    if missing:
        _pip_install(*missing)


@contextmanager
def _without_optional_transformers_backends():
    # Transformers can eagerly import optional integrations during model import.
    hidden = [name for name in ("accelerate", "bitsandbytes") if name not in sys.modules]
    for name in hidden:
        sys.modules[name] = None
    try:
        yield
    finally:
        for name in hidden:
            sys.modules.pop(name, None)


def _h3_qwen3vl_config(config_class):
    return config_class(
        architectures=["Qwen3VLForConditionalGeneration"],
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        text_config={
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "dtype": "float16",
            "eos_token_id": 151645,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "initializer_range": 0.02,
            "intermediate_size": 25600,
            "max_position_embeddings": 262144,
            "num_attention_heads": 64,
            "num_hidden_layers": TE_NUM_LAYERS,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
            "rope_theta": 5000000,
            "use_cache": False,
            "vocab_size": 151936,
        },
        vision_config={
            "deepstack_visual_indexes": [8, 16, 24],
            "depth": 27,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 4304,
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 5120,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    )


def _pin_cpu_tensor(tensor, enabled):
    if enabled and hasattr(tensor, "pin_memory"):
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor
    return tensor


def _install_embed_device_hooks(text_model, embed_device, dev0, dev1, offload_layers=0):
    import torch

    if embed_device != dev0:
        def lookup_on_cpu(_module, args):
            return (args[0].to(embed_device),)

        def return_to_gpu(_module, _args, output):
            return output.to(dev0)

        text_model.embed_tokens.register_forward_pre_hook(lookup_on_cpu)
        text_model.embed_tokens.register_forward_hook(return_to_gpu)

    if offload_layers > 0 and str(dev0) != "cpu":
        offload_start = len(text_model.layers) - offload_layers
        for idx in range(offload_start, len(text_model.layers)):
            layer = text_model.layers[idx]
            exec_dev = dev1 if dev1 is not None and idx >= TE_SPLIT_LAYER_IDX else dev0
            cpu_params = [(param, param.data) for _, param in layer.named_parameters()]

            def load_layer(_module, _args, cpu_params=cpu_params, exec_dev=exec_dev):
                for param, cpu_tensor in cpu_params:
                    param.data = cpu_tensor.to(exec_dev, non_blocking=True)

            def release_layer(_module, _args, _output, cpu_params=cpu_params):
                for param, cpu_tensor in cpu_params:
                    param.data = cpu_tensor

            layer.register_forward_pre_hook(load_layer)
            layer.register_forward_hook(release_layer, always_call=True)

    if dev1 is None:
        return

    cross_cache = {}

    def move_cached(value):
        if isinstance(value, torch.Tensor):
            if value.device == dev1:
                return value
            if value not in cross_cache:
                cross_cache[value] = value.to(dev1)
            return cross_cache[value]
        if isinstance(value, tuple):
            return tuple(move_cached(item) for item in value)
        return value

    def move_layer_inputs(_module, args, kwargs):
        # Hidden states cross once at the split. Later layers reuse the copies
        # of the position embeddings, attention mask and position IDs.
        if args:
            args = (args[0].to(dev1), *(move_cached(value) for value in args[1:]))
        kwargs = {
            key: value.to(dev1) if key == "hidden_states" else move_cached(value)
            for key, value in kwargs.items()
        }
        return args, kwargs

    def clear_cross_cache(_module, _args, _output):
        cross_cache.clear()

    for layer in text_model.layers[TE_SPLIT_LAYER_IDX:]:
        layer.register_forward_pre_hook(move_layer_inputs, with_kwargs=True)
    text_model.register_forward_hook(clear_cross_cache, always_call=True)


def _load_embed(settings, bf16_dir, te_dir):
    global _ACTIVE_COMPILE_CACHE

    _install_embed_dependencies()

    import torch

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count > 0:
        _configure_compile_cache_env(H3_CACHE_DIR)
        archive = _compile_cache_archive(torch)
        _ACTIVE_COMPILE_CACHE = (archive, _restore_compile_cache(archive, H3_CACHE_DIR))

    _ensure_h3_import_path()
    from h3_infer.ops import Int8ConvRotLinear
    from safetensors import safe_open

    dev0 = torch.device("cuda:0" if gpu_count else "cpu")
    dev1 = torch.device("cuda:1") if gpu_count >= 2 else None
    # FP16 keeps the ConvRot kernel and visual tower on the verified T4 path.
    dtype = torch.float16 if gpu_count else torch.float32
    offload_tokens = gpu_count >= 2 or (
        gpu_count == 1 and torch.cuda.get_device_properties(dev0).total_memory < 32 * 1024**3
    )
    embed_device = torch.device("cpu") if offload_tokens else dev0
    with _without_optional_transformers_backends():
        from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLModel

        processor = AutoProcessor.from_pretrained(str(bf16_dir / "processor"))
        config = _h3_qwen3vl_config(Qwen3VLConfig)
        with torch.device("meta"):
            model = Qwen3VLModel(config)
    text_model = model.language_model
    text_model.norm = torch.nn.Identity()
    offload_layers = settings.get("te_offload_layers", 0) if gpu_count > 0 else 0
    offload_start = len(text_model.layers) - offload_layers

    vision_rotary = model.visual.rotary_pos_emb
    if hasattr(vision_rotary, "config"):
        rotary_arg = config.vision_config
    else:
        rotary_arg = config.vision_config.hidden_size // config.vision_config.num_heads // 2
    with torch.device(dev0):
        model.visual.rotary_pos_emb = type(vision_rotary)(rotary_arg)
        text_model.rotary_emb = type(text_model.rotary_emb)(config.text_config)

    with safe_open(str(te_dir / TE_FILENAME), framework="pt", device="cpu") as weights:
        keys = set(weights.keys())

        def parameter(key, device, dtype, pin=False):
            tensor = weights.get_tensor(key).to(device=device, dtype=dtype)
            return torch.nn.Parameter(_pin_cpu_tensor(tensor, pin), requires_grad=False)

        text_model.embed_tokens.weight = parameter("model.embed_tokens.weight", embed_device, dtype)
        for name, _ in list(model.visual.named_parameters()):
            parent_path, _, attr = name.rpartition(".")
            parent = model.visual.get_submodule(parent_path)
            setattr(parent, attr, parameter(f"visual.{name}", dev0, dtype))

        for idx, layer in enumerate(text_model.layers):
            exec_dev = dev1 if dev1 is not None and idx >= TE_SPLIT_LAYER_IDX else dev0
            store_dev = torch.device("cpu") if idx >= offload_start else exec_dev
            pin_layer = gpu_count > 0 and idx >= offload_start
            prefix = f"model.layers.{idx}"
            for path in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
                layer.get_submodule(path).weight = parameter(f"{prefix}.{path}.weight", store_dev, dtype, pin_layer)

            for parent_name, projections in (
                ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                ("mlp", ("gate_proj", "up_proj", "down_proj")),
            ):
                parent = getattr(layer, parent_name)
                for name in projections:
                    linear = getattr(parent, name)
                    path = f"{prefix}.{parent_name}.{name}"
                    meta_key = f"{path}.comfy_quant"
                    quant = json.loads(bytes(weights.get_tensor(meta_key).tolist())) if meta_key in keys else {}
                    with torch.device("meta"):
                        projection = Int8ConvRotLinear(
                            linear.in_features, linear.out_features,
                            convrot=bool(quant.get("convrot", True)),
                            convrot_groupsize=int(quant.get("convrot_groupsize", 256)),
                            bias=False,
                        )
                    projection.weight = parameter(f"{path}.weight", store_dev, torch.int8, pin_layer)
                    scale = weights.get_tensor(f"{path}.weight_scale").to(device=store_dev, dtype=torch.float32)
                    if scale.numel() == 1:
                        scale = scale.reshape(1, 1).expand(projection.out_features, 1).contiguous()
                    else:
                        scale = scale.reshape(projection.out_features, 1).contiguous()
                    projection.weight_scale = torch.nn.Parameter(
                        _pin_cpu_tensor(scale, pin_layer), requires_grad=False,
                    )
                    setattr(parent, name, projection)

    meta_left = [name for name, tensor in (*model.named_parameters(), *model.named_buffers()) if tensor.is_meta]
    assert not meta_left, f"Unmaterialized meta tensors: {meta_left[:10]}"
    _install_embed_device_hooks(text_model, embed_device, dev0, dev1, offload_layers=offload_layers)
    print(f"[Embed] split={TE_SPLIT_LAYER_IDX} offload_layers={offload_layers}", flush=True)
    return model.eval(), processor


def _resize_vision_image(image):
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    short_edge = min(width, height)
    scale = min(max(short_edge, VISION_MIN_SHORT_EDGE), VISION_MAX_SHORT_EDGE) / short_edge
    size = tuple(max(CANVAS_MULTIPLE, round(dim * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                 for dim in (width, height))
    return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)


def _encode_prompt(te_bundle, prompt, edit_paths):
    import torch

    model, processor = te_bundle
    tokenizer = processor.tokenizer
    vision_weight = next(model.visual.parameters())
    dev0 = vision_weight.device
    token_ids, tags = [], []
    pixel_values = image_grid_thw = mm_token_type_ids = None
    if edit_paths:
        from PIL import Image

        images = []
        for path in edit_paths:
            with Image.open(path) as image:
                images.append(_resize_vision_image(image))
        vision = processor.image_processor(images=images, return_tensors="pt")
        merge_area = int(processor.image_processor.merge_size) ** 2
        start, pad, end = (tokenizer.convert_tokens_to_ids(token)
                           for token in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"))
        for idx, grid in enumerate(vision["image_grid_thw"]):
            label_ids = tokenizer(f"Picture {idx + 1}: ", add_special_tokens=False)["input_ids"][:MAX_SEQ_LEN]
            vision_ids = [start] + [pad] * (int(grid.prod()) // merge_area) + [end]
            token_ids.extend(label_ids + vision_ids)
            tags.extend([TEXT_TAG] * len(label_ids) + [IMAGE_TAG] * len(vision_ids))
        pixel_values = vision["pixel_values"].to(device=dev0, dtype=vision_weight.dtype)
        image_grid_thw = vision["image_grid_thw"].to(dev0)

    prompt_ids = tokenizer(prompt or " ", add_special_tokens=False)["input_ids"][:MAX_SEQ_LEN]
    token_ids.extend(prompt_ids)
    tags.extend([TEXT_TAG] * len(prompt_ids))
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=dev0)
    if pixel_values is not None:
        mm_token_type_ids = (input_ids == model.config.image_token_id).to(dtype=torch.int32)

    with torch.inference_mode():
        output = model(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            pixel_values=pixel_values, image_grid_thw=image_grid_thw, mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=False, use_cache=False, return_dict=True,
        )
        embeds = output.last_hidden_state[0].to(device="cpu", dtype=torch.bfloat16)
    return embeds, torch.tensor(tags, dtype=torch.long, device="cpu")


def _edit_image_specs(task):
    specs = task.get("edit_image")
    if specs is None or specs == "":
        return []
    if isinstance(specs, str):
        return [specs]
    if not isinstance(specs, list) or any(not isinstance(spec, str) or not spec for spec in specs):
        raise ValueError("edit_image must be a path or a list of paths")
    return specs


def _embed_task(te_bundle, settings, code, task, progress):
    import torch

    start = time.perf_counter()
    prompt = task["prompt"]
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    specs = _edit_image_specs(task)
    project_root = code.parent
    edit_paths = []
    edit_digests = []
    if specs:
        resolve_image = _import_remote("recipe").resolve_image
        for spec in specs:
            path = resolve_image(spec, code, project_root)
            edit_paths.append(path)
            digest = hashlib.sha256()
            with path.open("rb") as image_file:
                for chunk in iter(partial(image_file.read, 1024 * 1024), b""):
                    digest.update(chunk)
            edit_digests.append(digest.hexdigest())
    key = json.dumps({"prompt": prompt, "edits": edit_digests}, sort_keys=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    relative = f"{settings['embeds_dir'].rstrip('/')}/{digest[:16]}.pt"
    dest = project_root / relative
    cache_hit = _nonempty_file(dest)
    if not cache_hit:
        progress("encoding_prompt")
        try:
            embeds, tags = _encode_prompt(te_bundle, prompt, edit_paths)
        finally:
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and cuda.is_available():
                cuda.empty_cache()
        _save_pt(dest, {"prompt": prompt, "prompt_embeds": embeds, "text_token_tags": tags})
        _publish_artifact(dest, project_root, relative)
    return relative, {"prompt_embeds_rel": relative, "embed_sec": time.perf_counter() - start,
                      "embed_cache_hit": cache_hit}


def _ensure_h3_import_path():
    """Use the synced library on Colab or the bundled package when run locally."""
    try:
        importlib.import_module("h3_infer")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "h3_infer":
            raise
    source = globals().get("__file__")
    if source and Path(source).name == "h3_pipeline_worker.py":
        engines_root = Path(source).resolve().parents[1] / "engines"
        if (engines_root / "h3_infer").is_dir():
            if str(engines_root) not in sys.path:
                sys.path.insert(0, str(engines_root))
            return
    if _nonempty_file(REMOTE_SCRIPTS / "h3_infer" / "__init__.py"):
        if str(REMOTE_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(REMOTE_SCRIPTS))
        importlib.invalidate_caches()
        return

    token = _drive_token()
    if os.environ.get("COLAB_ULTRA_KAGGLE") == "1" and token:
        # Kaggle's bootstrap materializes cu_remote only. Pull the library published by sync-remote.
        try:
            drive = _Drive(token)
            parent = drive.folder(("ColabUltra", "scripts", "h3_infer"))
            if parent is None:
                raise FileNotFoundError("h3_infer")
            REMOTE_SCRIPTS.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=REMOTE_SCRIPTS) as temp:
                package = Path(temp) / "h3_infer"
                drive.download_python_tree(parent, package)
                if not _nonempty_file(package / "__init__.py"):
                    raise FileNotFoundError("h3_infer/__init__.py")
                package.replace(REMOTE_SCRIPTS / "h3_infer")
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"Could not load the Drive h3_infer library ({type(exc).__name__}). "
                               "Run colab-ultra sync-remote with h3_infer available.") from None
        if str(REMOTE_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(REMOTE_SCRIPTS))
        importlib.invalidate_caches()


def _has_attention_kernel(module, function):
    try:
        return callable(getattr(importlib.import_module(module), function, None))
    except (ImportError, OSError, RuntimeError):
        return False


def _load_dit(settings, bf16_dir):
    global _ACTIVE_COMPILE_CACHE

    _configure_compile_cache_env(H3_CACHE_DIR)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The DiT worker requires CUDA")
    archive = _compile_cache_archive(torch)
    _ACTIVE_COMPILE_CACHE = (archive, _restore_compile_cache(archive, H3_CACHE_DIR))
    major = torch.cuda.get_device_capability(0)[0]
    if major >= 8 and _package_version("sageattention") != "1.0.6":
        try:
            _pip_install("sageattention==1.0.6")
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"[Warn] sageattention installation failed ({type(exc).__name__})", flush=True)

    _ensure_h3_import_path()
    from h3_infer import MiniMaxH3ImagePipeline
    from h3_infer.ops.attention import recommended_attention_backend

    requested = settings["attention_backend"]
    active = "sdpa"
    if requested == "auto":
        active = recommended_attention_backend(major)
    elif requested == "sageattention" and major >= 8 and (
        _has_attention_kernel("sageattention", "sageattn_varlen")
        or _has_attention_kernel("sageattention", "sageattn")
    ):
        active = "sageattention"
    elif requested == "flash_attn" and major in (8, 9) and _has_attention_kernel("flash_attn", "flash_attn_varlen_func"):
        active = "flash_attn"
    print(f"[Attn] requested={requested} active={active}", flush=True)
    dit_checkpoint = None
    if settings["dit_variant"] == "int8_convrot":
        dit_checkpoint = bf16_dir.parents[1].joinpath(*INT8_CONVROT_GROUP, INT8_CONVROT_FILENAME)
    pipe = MiniMaxH3ImagePipeline.from_pretrained(
        bf16_dir, dit_checkpoint=dit_checkpoint, device="cuda", attention_backend=active, offload_dit=True,
        prune_adaln="auto", staged_vram=True, resident_blocks="auto", compile_mode=settings["compile_mode"],
    )
    return pipe, active


def _wait_for_edit_image(spec, code, project_root, timeout_sec):
    resolve_image = _import_remote("recipe").resolve_image
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            path = resolve_image(spec, code, project_root)
            break
        except FileNotFoundError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            # Refresh the candidate directories before retrying recipe resolution.
            for folder in (code, project_root, project_root / "data", project_root / "runs"):
                try:
                    os.listdir((folder / Path(spec).expanduser()).parent)
                except OSError:
                    pass
            time.sleep(min(1.0, remaining))
    return wait_for_drive_file(path, timeout_sec=max(0.0, deadline - time.monotonic()))


class _DiTRunner:
    def __init__(self, pipe, active_attn, settings, code):
        self.pipe = pipe
        self.active_attn = active_attn
        self.settings = settings
        self.code = code
        self.project_root = code.parent
        self.resident = None

    def _enter_resident(self):
        if self.resident is None:
            resident = self.pipe.keep_dit_resident()
            resident.__enter__()
            self.resident = resident

    def close(self):
        if self.resident is not None:
            resident = self.resident
            self.resident = None
            resident.__exit__(None, None, None)

    def _prepare_item(self, task):
        if not _safe_name(task["slug"]):
            raise ValueError("slug must be a safe single-segment name")
        specs = _edit_image_specs(task)
        item = {key: int(task.get(key, self.settings[key])) for key in ("height", "width")}
        if min(item.values()) < 1:
            raise ValueError("Output height and width must be positive")
        item["seed"] = int(task.get("seed", self.settings["seed_base"]))
        timeout = self.settings["drive_wait_sec"]
        item["prompt_embeds"] = _wait_for_artifact(self.project_root, task["prompt_embeds_rel"], timeout)
        if specs:
            item["edit_images"] = [
                str(_wait_for_edit_image(spec, self.code, self.project_root, timeout)) for spec in specs
            ]
        return item

    def _save_latents(self, slug, item, result):
        latents, stats = result
        metadata = {
            "step_ms": [float(value) for value in stats.step_ms],
            "peak_vram_bytes": int(stats.peak_vram_bytes) if stats.peak_vram_bytes is not None else None,
            "active_attention_backend": self.active_attn,
        }
        output_size = [(item[key] + 31) // 32 * 32 for key in ("width", "height")]
        relative = f"data/h3_latents_cache/{self.settings['run_name']}/{slug}.pt"
        _save_pt(self.project_root / relative, {
            "latents": latents.cpu(), "output_size": output_size, "seed": item["seed"], **metadata,
        })
        return relative, {"latents_rel": relative, **metadata}

    def __call__(self, tasks, progress, complete, fail):
        if any(task.get("edit_image") for _, task in tasks):
            self._run_batch(tasks, progress, complete, fail)
            return

        for msg_id, task in tasks:
            try:
                start = time.perf_counter()
                item = self._prepare_item(task)
                self._enter_resident()
                self._run_item(msg_id, task, item, progress, complete, start)
            except Exception as exc:
                fail([(msg_id, task)], exc)

    def _run_item(self, msg_id, task, item, progress, complete, start):
        slug = task["slug"]

        def report_step(step, _latents, _vel):
            progress(slug, f"dit_step_{step + 1}/6")

        embeds = item.pop("prompt_embeds")
        result = self.pipe(embeds, **item, output_type="latent", return_stats=True, step_callback=report_step)
        relative, fields = self._save_latents(slug, item, result)
        complete(msg_id, task, relative, {**fields, "dit_sec": time.perf_counter() - start})

    def _run_streaming_batch(self, valid_tasks, batch_items, progress, complete, fail):
        import torch

        pending = dict(valid_tasks)
        try:
            with torch.inference_mode(), self.pipe._vae_on_device():
                for (msg_id, task), item in zip(valid_tasks, batch_items):
                    try:
                        _, shape = self.pipe._generation_sizes(item["height"], item["width"], None)
                        generator, item["latents"] = self.pipe._initial_latents(shape, item["seed"], "cpu", None)
                        if item.get("edit_images"):
                            item["ref_latents"] = self.pipe._encode_references(
                                item.pop("edit_images"), generator, "cpu", None, None,
                            )
                    except Exception as exc:
                        del pending[msg_id]
                        fail([(msg_id, task)], exc)
            if pending:
                self._enter_resident()
        except Exception as exc:
            fail(pending.items(), exc)
            return

        for (msg_id, task), item in zip(valid_tasks, batch_items):
            if msg_id not in pending:
                continue
            try:
                self._run_item(msg_id, task, item, progress, complete, time.perf_counter())
            except Exception as exc:
                fail([(msg_id, task)], exc)

    def _run_batch(self, tasks, progress, complete, fail):
        valid_tasks = []
        batch_items = []
        for msg_id, task in tasks:
            try:
                item = self._prepare_item(task)
            except Exception as exc:
                fail([(msg_id, task)], exc)
            else:
                valid_tasks.append((msg_id, task))
                batch_items.append(item)
        if not batch_items:
            return
        self.close()
        streaming_methods = ("_vae_on_device", "_generation_sizes", "_initial_latents", "_encode_references")
        pipe_attributes = dir(self.pipe)
        if all(name in pipe_attributes and callable(getattr(self.pipe, name)) for name in streaming_methods):
            self._run_streaming_batch(valid_tasks, batch_items, progress, complete, fail)
            return

        def report_batch_step(item_idx, step, _latents, _vel):
            progress(valid_tasks[item_idx][1]["slug"], f"dit_step_{step + 1}/6")

        try:
            start = time.perf_counter()
            results = self.pipe.generate_batch(batch_items, output_type="latent", return_stats=True,
                                               step_callback=report_batch_step)
            batch_sec = time.perf_counter() - start
            if len(results) != len(batch_items):
                raise RuntimeError("DiT batch returned a different number of results than tasks")
        except Exception as exc:
            fail(valid_tasks, exc)
            return

        for (msg_id, task), item, result in zip(valid_tasks, batch_items, results):
            try:
                relative, fields = self._save_latents(task["slug"], item, result)
                step_ms = fields["step_ms"]
                dit_sec = round(sum(step_ms) / 1000.0 if step_ms else batch_sec, 3)
                complete(msg_id, task, relative, {**fields, "dit_sec": dit_sec})
            except Exception as exc:
                fail([(msg_id, task)], exc)


def _resolve_vae_device(requested):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _load_vae(bf16_dir, settings):
    import torch

    _ensure_h3_import_path()
    from h3_infer.config import MiniMaxH3VAEConfig
    from h3_infer.models.vae import MiniMaxH3ImageVAE
    from h3_infer.runtime.loader import load_safetensors

    device = _resolve_vae_device(settings["vae_device"])
    dtype = torch.float32
    tile_batch_size = settings["vae_tile_batch_size"]
    if device.type == "cuda":
        if torch.cuda.get_device_capability(device)[0] >= 8:
            dtype = torch.bfloat16
        if tile_batch_size is None:
            vram = torch.cuda.get_device_properties(device).total_memory
            tile_batch_size = 64 if vram >= 18 * 1024**3 else 16
    elif tile_batch_size is None:
        tile_batch_size = 4
    dtype_name = str(dtype).removeprefix("torch.")
    print(f"[VAE] device={device} dtype={dtype_name} tile_batch_size={tile_batch_size}", flush=True)
    config = MiniMaxH3VAEConfig.from_json(bf16_dir / "vae/config.json")
    with torch.device("meta"):
        vae = MiniMaxH3ImageVAE(config, attention_backend="sdpa", stats_device=device).eval()
    load_safetensors(vae, bf16_dir / "vae/diffusion_pytorch_model.safetensors", device=device,
                     dtype_override=dtype, ignore_keys=("decoder.mask_token",))
    return vae, tile_batch_size


def _decode_image(vae, latents, output_size, tile_batch_size):
    import torch
    from PIL import Image
    from h3_infer.models.vae import IMAGENET_MEAN, IMAGENET_STD

    weight = getattr(getattr(vae, "post_quant_conv", None), "weight", None)
    device, dtype = getattr(weight, "device", "cpu"), vae.latents_std.dtype
    with torch.inference_mode():
        latents = latents[:, :, 0].to(device=device, dtype=dtype) * vae.latents_std + vae.latents_mean
        pixels = vae.decode(latents, tiling=None, tile_batch_size=tile_batch_size)
        mean = torch.tensor(IMAGENET_MEAN, dtype=dtype, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=dtype, device=device).view(1, 3, 1, 1)
        pixels = (pixels.to(dtype=dtype) * std + mean).clamp(0, 1)
        rgb = (pixels[0].permute(1, 2, 0) * 255).to(dtype=torch.uint8).cpu()
        image = Image.fromarray(rgb.numpy())
    if image.size != tuple(output_size):
        image = image.resize(tuple(output_size), Image.Resampling.LANCZOS)
    return image


def _vae_task(vae, settings, project_root, task, progress):
    import torch

    start = time.perf_counter()
    slug = task["slug"]
    fmt = _output_format(task.get("output_format", settings["output_format"]))
    latents_path = _wait_for_artifact(project_root, task["latents_rel"], settings["drive_wait_sec"])
    progress("decoding_vae")
    payload = torch.load(latents_path, map_location="cpu", weights_only=True)
    image = _decode_image(vae, payload["latents"], payload["output_size"], settings["vae_tile_batch_size"])
    run_name = settings["run_name"]
    run_dir = project_root / "runs" / run_name
    out_path = run_dir / f"{slug}.{fmt}"
    with _atomic_output(out_path) as staged:
        image.save(staged, **SAVE_OPTIONS[fmt])
    stats = {
        "slug": slug, "file": out_path.name, "vae_worker": task["vae_worker"],
        "vae_sec": time.perf_counter() - start,
        **{key: task[key] for key in ("embed_worker", "embed_sec", "embed_cache_hit", "dit_worker", "dit_sec")
           if key in task},
        **{key: payload[key] for key in ("output_size", "seed", "step_ms", "peak_vram_bytes", "active_attention_backend")},
    }
    stats_path = run_dir / f"{slug}.json"
    with _atomic_output(stats_path) as staged:
        staged.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for path in (out_path, stats_path):
        _publish_artifact(path, project_root, f"runs/{run_name}/{path.name}")
    return str(out_path), {}


def should_exit_worker(own_status, upstream_statuses) -> bool:
    return own_status.get("unclaimed", 0) == 0 and all(status.get("all_done") is True for status in upstream_statuses)


def _run_loop(role, worker_id, queues, settings, process, redact):
    index = STAGE_ROLES.index(role)
    my_queue = queues[role]
    upstreams = [queues[upstream] for upstream in STAGE_ROLES[:index]]
    next_queue = queues[STAGE_ROLES[index + 1]] if index + 1 < len(STAGE_ROLES) else None
    batch_size = settings["dit_batch_size"] if role == "dit" else 1
    poll_sec = settings["poll_interval_sec"]

    def progress(slug, step):
        my_queue.report_progress(slug, worker_id, step)
        print(redact(f"[{role} {worker_id}] {slug} {step}"), flush=True)

    def complete(msg_id, task, output, fields):
        if next_queue is not None:
            next_queue.push([{**task, **fields, f"{role}_worker": worker_id}])
        my_queue.ack(msg_id, task["slug"], worker_id, output=output)

    def fail(tasks, exc):
        print(redact(traceback.format_exc()), file=sys.stderr, flush=True)
        error = redact(f"{type(exc).__name__}: {exc}")
        for msg_id, task in tasks:
            my_queue.fail(msg_id, task["slug"], worker_id, error)

    empty_polls = 0
    while True:
        tasks = my_queue.claim_batch(
            worker_id, count=batch_size, reclaim_idle_ms=settings["reclaim_idle_ms"], block_ms=int(poll_sec * 1000),
        )
        if not tasks:
            # Each upstream publishes its handoff before acknowledging its input.
            upstream_statuses = [queue.status() for queue in upstreams]
            empty_polls = empty_polls + 1 if should_exit_worker(my_queue.status(), upstream_statuses) else 0
            if empty_polls >= 2:
                my_queue.set_worker_status(worker_id, "stopped")
                return 0
            time.sleep(poll_sec)
            continue
        empty_polls = 0
        if role == "dit":
            process(tasks, progress, complete, fail)
            continue
        for msg_id, task in tasks:
            slug = task["slug"]
            try:
                if not _safe_name(slug):
                    raise ValueError("slug must be a safe single-segment name")
                output, fields = process({**task, f"{role}_worker": worker_id}, partial(progress, slug))
                complete(msg_id, task, output, fields)
            except Exception as exc:
                fail([(msg_id, task)], exc)


def _worker_script_path(code):
    name = "h3_pipeline_worker.py"
    source = globals().get("__file__")
    if source and Path(source).name == name and Path(source).is_file():
        return Path(source).resolve()
    for directory in (code, REMOTE_SCRIPTS, Path.cwd()):
        path = directory / name
        if path.is_file():
            return path.resolve()
    raise RuntimeError(f"Place {name} in {code} or {REMOTE_SCRIPTS} to launch embed_vae children.")


def _launch_embed_vae(script, code, worker_id, redis_url, model_root):
    import torch

    env = {**os.environ, "COLAB_ULTRA_REDIS_URL": redis_url, "COLAB_ULTRA_STAGED_MODELS": str(model_root)}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(code), env.get("PYTHONPATH"))))
    token = _drive_token()
    if token:
        env["COLAB_ULTRA_DRIVE_TOKEN"] = token
    visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    all_devices = visible_devices is not None and visible_devices.strip().lower() == "all"
    if all_devices:
        # CUDA expects an unset mask to expose all GPUs.
        env.pop("CUDA_VISIBLE_DEVICES")
        os.environ.pop("CUDA_VISIBLE_DEVICES")
    try:
        gpu_count = torch.cuda.device_count()
    finally:
        if all_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    children = []

    def launch(role, suffix, child_env):
        child = subprocess.Popen(
            [sys.executable, "-u", str(script), "--role", role, "--worker-id", f"{worker_id}-{suffix}"],
            env=child_env,
        )
        children.append(child)
        return child

    try:
        vae0_env = {**env, "CUDA_VISIBLE_DEVICES": "0", "COLAB_ULTRA_VAE_DEVICE": "cuda"}
        if gpu_count >= 2:
            embed_env = {**env}
            embed_env.setdefault("COLAB_ULTRA_TE_OFFLOAD_LAYERS", str(EMBED_VAE_OFFLOAD_LAYERS))
            vae1_env = {**env, "CUDA_VISIBLE_DEVICES": "1", "COLAB_ULTRA_VAE_DEVICE": "cuda"}
            vae1_env.setdefault("COLAB_ULTRA_VAE_TILE_BATCH_SIZE", str(EMBED_VAE_CONCURRENT_VAE_TILE_BATCH_SIZE))
            embed_proc = launch("embed", "embed", embed_env)
            launch("vae", "vae-1", vae1_env)
            vae0_launched = False
        else:
            embed_proc = launch("embed", "embed", env)
            status = embed_proc.wait()
            children.remove(embed_proc)
            if status != 0:
                return status if status > 0 else 1
            launch("vae", "vae", vae0_env)
            vae0_launched = True

        while children:
            for child in children[:]:
                status = child.poll()
                if status is not None:
                    children.remove(child)
                    if status != 0:
                        return status if status > 0 else 1
                    if child is embed_proc and not vae0_launched:
                        launch("vae", "vae-0", vae0_env)
                        vae0_launched = True
            if children:
                time.sleep(0.25)
        return 0
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one stage of the MiniMax-H3 pipeline")
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--worker-id")
    parser.add_argument("--dry-run", action="store_true", help="Show queue and model settings")
    args = parser.parse_args(argv)
    cwd = Path.cwd()
    code = cwd if cwd.name == "code" else cwd / "code"
    try:
        role, worker_id = resolve_worker(args.role, args.worker_id)
        settings = _settings(code)
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        parser.error(str(exc))
    queue_names = {stage: settings[f"{stage}_queue"] for stage in STAGE_ROLES}
    scope = _model_scope(role, settings["dit_variant"])
    if args.dry_run or os.environ.get("CU_DRY_RUN") == "1":
        print(f"Dry run: role={role} worker_id={worker_id} run_name={settings['run_name']}")
        print(f"Queues: {queue_names}")
        if role == "embed_vae":
            print("With 2+ GPUs, embed runs with tail TE layers CPU-offloaded concurrently with VAE on GPU 1.")
            print("VAE on GPU 0 starts after embed exits successfully.")
            print("With fewer than two GPUs, embed runs sequentially before VAE on GPU 0.")
        else:
            upstreams = [queue_names[stage] for stage in STAGE_ROLES[:STAGE_ROLES.index(role)]]
            print(f"Upstream queues: {upstreams}")
        for group, files in scope.items():
            print(f"Model staging: {'/'.join(group)}: {', '.join(files)}")
        return 0

    try:
        redis_url = resolve_redis_url(code)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Redis configuration failed ({type(exc).__name__}). "
                         "Check COLAB_ULTRA_REDIS_URL, /tmp/.cu_redis_url, or code/.redis_url.") from None
    redact = _redis_redactor(redis_url)
    try:
        if not redis_url.startswith(("http://", "https://")):
            _install_redis()
        if role == "embed_vae":
            script = _worker_script_path(code)
            return _launch_embed_vae(script, code, worker_id, redis_url, _stage_models(scope))
        task_queue = _import_remote("queue").TaskQueue
        queues = {stage: task_queue(redis_url, name) for stage, name in queue_names.items()}
        staged_models = os.environ.get("COLAB_ULTRA_STAGED_MODELS")
        nvme_root = Path(staged_models) if staged_models else _stage_models(scope)
        bf16_dir = nvme_root.joinpath(*IMAGE_GROUP)
        if role == "embed":
            try:
                te_bundle = _load_embed(settings, bf16_dir, nvme_root.joinpath(*TE_GROUP))
                process = partial(_embed_task, te_bundle, settings, code)
                return _run_loop(role, worker_id, queues, settings, process, redact)
            finally:
                _save_active_compile_cache()
        if role == "dit":
            try:
                pipe, active = _load_dit(settings, bf16_dir)
                runner = _DiTRunner(pipe, active, settings, code)
                try:
                    return _run_loop(role, worker_id, queues, settings, runner, redact)
                finally:
                    runner.close()
            finally:
                _save_active_compile_cache()
        vae, tile_batch_size = _load_vae(bf16_dir, settings)
        process = partial(_vae_task, vae, {**settings, "vae_tile_batch_size": tile_batch_size}, code.parent)
        return _run_loop(role, worker_id, queues, settings, process, redact)
    except Exception as exc:
        raise SystemExit(redact(f"H3 {role} worker failed ({type(exc).__name__}): {exc}")) from None


if __name__ == "__main__":
    # Colab evaluates this source in a kernel whose argv belongs to IPython.
    argv = sys.argv[1:] if Path(sys.argv[0]).name == "h3_pipeline_worker.py" else []
    result = main(argv)
    if result:
        raise SystemExit(result)
