"""Stage model files on VM-local storage and publish completed files atomically."""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

DRIVE = Path("/content/drive")
NVME = Path("/content")
MIN_BYTES = 1_000_000
_install_lock = threading.Lock()
_drive_copy = threading.Lock()


@dataclass(frozen=True)
class Plan:
    """Resolved single-file transfer, inspectable without network access."""
    source: str
    dest: Path
    backend: str
    work_dir: Path
    copy_to_drive: bool
    concurrency: int
    min_bytes: int
    force: bool = False


def hf_url(repo, filename, revision="main"):
    """Form a Hugging Face resolve URL for a specific file."""
    return f"https://huggingface.co/{repo}/resolve/{quote(revision, safe='')}/{quote(filename)}"


def plan(source, dest, *, backend="hf_transfer", concurrency=None, min_bytes=MIN_BYTES, force=False):
    """Resolve a file source and choose the local download and publication paths."""
    dest = Path(dest).resolve()
    source = str(source)
    drive_api = source.startswith("drive://") and bool(os.environ.get("DRIVE_ACCESS_TOKEN"))
    if backend not in {"aria2", "hf_transfer"} or min_bytes < 0:
        raise ValueError("Invalid staging backend or min_bytes")
    if source.startswith("hf://"):
        owner, repo, filename = source[5:].split("/", 2)
        source = hf_url(f"{owner}/{repo}", filename)
    elif source.startswith("drive://"):
        source = str(DRIVE / "MyDrive" / source[8:].lstrip("/"))
    if Path(source).is_relative_to(DRIVE):
        if not drive_api:
            backend = "copy"
    elif urlsplit(source).scheme not in {"http", "https"}:
        raise ValueError(f"Expected a Drive path or HF/HTTP URL: {source}")
    count = int(concurrency if concurrency is not None else (64 if backend == "hf_transfer" else 16))
    if count < 1:
        raise ValueError("concurrency must be positive")
    on_drive = dest.is_relative_to(DRIVE.resolve())
    if os.environ.get("COLAB_ULTRA_KAGGLE") == "1" and os.environ.get("DRIVE_ACCESS_TOKEN"):
        # Kaggle's mirrored Drive tree is on local disk, so an atomic rename suffices.
        on_drive = False
    return Plan(source, dest, backend, NVME if on_drive else dest.parent, on_drive,
                min(count, 16) if backend == "aria2" else count, min_bytes, force)


def valid(path, min_bytes=MIN_BYTES):
    """A resumable aria2 fragment can have the full preallocated size."""
    path = Path(path)
    return path.is_file() and not Path(f"{path}.aria2").exists() and path.stat().st_size > min_bytes


def ensure_aria2():
    """Install aria2 once per VM, preferring the existing Drive package cache."""
    with _install_lock:
        if shutil.which("aria2c"):
            return
        cache = DRIVE / "MyDrive/ColabUltra/bin/aria2_debs"
        debs = sorted(cache.glob("*.deb"))
        if debs and subprocess.run(["dpkg", "-i", *debs]).returncode == 0 and shutil.which("aria2c"):
            return
        subprocess.run(["apt-get", "-qq", "update"], check=True)
        subprocess.run(["apt-get", "-qq", "install", "-y", "--no-install-recommends", "aria2"], check=True)


def _drive_media(source, headers):
    """Resolve an exact MyDrive-relative path to its media URL and known size."""
    parent = "root"
    parts = Path(source).relative_to(DRIVE / "MyDrive").parts
    if not parts:
        raise ValueError("Expected a path to a Drive file")
    for part in parts:
        escaped = part.replace("\\", "\\\\").replace("'", "\\'")
        query = urlencode({"q": f"'{parent}' in parents and name = '{escaped}' and trashed = false",
                           "fields": "files(id,size)", "pageSize": 2})
        request = Request(f"https://www.googleapis.com/drive/v3/files?{query}", headers=headers)
        with urlopen(request, timeout=20) as response:
            files = json.load(response)["files"]
        if len(files) != 1:
            raise ValueError(f"Expected one Drive entry named {part!r}, found {len(files)}")
        parent = files[0]["id"]
    return f"https://www.googleapis.com/drive/v3/files/{parent}?alt=media", int(files[0]["size"])


def _download(item, temp):
    """Write to a private local file. Publish only after the worker succeeds."""
    if item.backend == "copy":
        if not valid(item.source, item.min_bytes):
            raise IOError(f"Incomplete Drive source: {item.source}")
        shutil.copy2(item.source, temp)
        return
    url = item.source
    headers = {}
    expected = None
    if Path(url).is_relative_to(DRIVE):
        headers["Authorization"] = "Bearer " + os.environ["DRIVE_ACCESS_TOKEN"]
        url, expected = _drive_media(url, headers)
    elif urlsplit(url).hostname == "huggingface.co":
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = "Bearer " + token
    if item.backend == "aria2":
        ensure_aria2()
        count = str(item.concurrency)
        cmd = ["aria2c", "-x", count, "-s", count, "-k", "4M", "--file-allocation=none",
               "--auto-file-renaming=false", "-d", str(temp.parent), "-o", temp.name]
        conf_path = temp.with_name(f"{temp.name}.aria2.conf") if headers else None
        try:
            if conf_path is not None:
                # Pass authorization through a restricted config file, never in argv or logs.
                conf_path.write_text("".join(f"header={k}: {v}\n" for k, v in headers.items()), encoding="utf-8")
                conf_path.chmod(0o600)
                subprocess.run([*cmd, f"--conf-path={conf_path}", url], check=True)
            else:
                subprocess.run([*cmd, url], check=True)
        finally:
            if conf_path is not None:
                conf_path.unlink(missing_ok=True)
    else:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        with _install_lock:
            try:
                import hf_transfer
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf_transfer"], check=True)
                import hf_transfer

        try:
            hf_transfer.download(url=url, filename=str(temp), max_files=item.concurrency, headers=headers,
                                 chunk_size=10 * 1024 * 1024, parallel_failures=min(3, item.concurrency - 1), max_retries=5)
        except Exception as exc:
            # The native downloader reports HTTP transfer failures as plain Exceptions.
            if expected is not None and os.environ.get("COLAB_ULTRA_KAGGLE") == "1":
                raise OSError(f"Drive download failed for {item.dest}") from exc
            raise
    if expected is not None and temp.stat().st_size != expected:
        raise IOError(f"Expected {expected} bytes for {item.dest}")


def run(item):
    """Stage one plan, with sequential Drive writes and no reusable partial output."""
    dest = item.dest
    marker = Path(f"{dest}.aria2")
    if not item.force and valid(dest, item.min_bytes):
        return dest
    if marker.exists():
        dest.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    item.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=item.work_dir) as work:
        temp = Path(work) / "download"
        _download(item, temp)
        if not valid(temp, item.min_bytes):
            raise IOError(f"Incomplete transfer for {dest}")
        if item.copy_to_drive:
            with _drive_copy, tempfile.TemporaryDirectory(dir=dest.parent) as out:
                copy = Path(out) / dest.name
                shutil.copy2(temp, copy)
                if copy.stat().st_size != temp.stat().st_size:
                    raise IOError(f"Incomplete Drive copy for {dest}")
                copy.replace(dest)
        else:
            temp.replace(dest)
    return dest


def run_many(plans, workers=6):
    """Overlap independent transfers and propagate every worker failure."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, plans))


def repo_plans(repo, pattern, dest, *, revision="main", **options):
    """Expand an HF repo glob, retaining its directory structure."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, revision=revision)
    matches = [f for f in files if fnmatch.fnmatchcase(f, pattern)]
    if not matches:
        raise FileNotFoundError(f"No files match {repo}/{pattern} at {revision}")
    return [plan(hf_url(repo, f, revision), Path(dest) / f, **options) for f in matches]


def benchmark(plans, workers=1):
    """Measure uncached end-to-end throughput including any final Drive copy."""
    plans = list(plans)
    if not plans or any(p.dest.exists() for p in plans):
        raise ValueError("Supply plans with fresh benchmark destinations")
    start = time.monotonic()
    files = run_many(plans, workers)
    seconds = time.monotonic() - start
    size = sum(f.stat().st_size for f in files)
    return {"bytes": size, "seconds": seconds, "MB_per_second": size / 1e6 / seconds}


def stage_dual_channel(dest_root, drive_files=None, hf_urls=None, concurrency=4, splits=16):
    """Stage the legacy tuple lists used by the unchanged image workflow."""
    start = time.monotonic()
    root = Path(dest_root)
    plans = [plan(url, root / folder / name, concurrency=splits, min_bytes=0)
             for folder, name, url in hf_urls or ()]
    for folder, source in drive_files or ():
        url = source if source.startswith(("/content/drive/", "drive://")) else "drive://" + source
        plans.append(plan(url, root / folder / Path(source).name, concurrency=splits, min_bytes=0))
    run_many(plans, workers=concurrency)
    return time.monotonic() - start
