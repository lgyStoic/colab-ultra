"""Shared Drive model variants and their optional VM-local copies."""
from __future__ import annotations

import json
import re
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

from . import stage

_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
_WEIGHTS = {".safetensors", ".gguf", ".bin", ".pt", ".pth"}
_variant_locks = defaultdict(threading.Lock)


def library_dir() -> Path:
    return stage.DRIVE / "MyDrive/ColabUltra/models"


def variant_dir(model: str, variant: str) -> Path:
    for name in (model, variant):
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"Invalid model or variant name: {name!r}")
    return library_dir() / model / variant


def check_relative(path: str) -> None:
    if (not isinstance(path, str) or "\\" in path or "\x00" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))):
        raise ValueError(f"Expected a clean relative path: {path!r}")


def read_manifest(vdir):
    path = vdir / "manifest.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("model") != vdir.parent.name
                or data.get("variant") != vdir.name or not isinstance(data.get("files"), dict)):
            raise ValueError(f"Invalid model manifest: {path}")
        return data
    return {"model": vdir.parent.name, "variant": vdir.name, "files": {}}


def write_manifest(vdir, data):
    vdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=vdir) as work:
        temp = Path(work) / "manifest.json"
        temp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        temp.replace(vdir / "manifest.json")


def _hf_location(url):
    parsed = urlsplit(url)
    parts = parsed.path.lstrip("/").split("/", 4)
    if parsed.hostname == "huggingface.co" and len(parts) == 5 and parts[2] == "resolve":
        return "/".join(parts[:2]), unquote(parts[3]), unquote(parts[4])
    return None


def _source_url(source, revision):
    if source.startswith("hf://"):
        owner, repo, filename = source[5:].split("/", 2)
        if not owner or not repo or not filename:
            raise ValueError(f"Incomplete HF source: {source!r}")
        return stage.hf_url(f"{owner}/{repo}", filename, revision)
    parsed = urlsplit(source)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Expected an HF or HTTP(S) file source: {source!r}")
    return source


def _sources(files, revision):
    urls = {}
    for rel, source in files.items():
        check_relative(rel)
        if rel.split("/")[0] == "manifest.json" or any(str(p) in files for p in Path(rel).parents):
            raise ValueError(f"File path collision: {rel!r}")
        urls[rel] = _source_url(source, revision)
    return urls


def _minimum(rel):
    return stage.MIN_BYTES if Path(rel).suffix.lower() in _WEIGHTS else 0


def _fill_hashes(entries, pending):
    # One metadata request per repository/revision. Regular Git files have no LFS hash.
    for (repo, revision), paths in pending.items():
        try:
            from huggingface_hub import HfApi
            infos = HfApi().get_paths_info(repo, list(paths), revision=revision, expand=True)
            for info in infos:
                for rel in paths.get(info.path, ()):
                    entries[rel]["sha256"] = getattr(info.lfs, "sha256", None)
        except Exception:
            continue


def ensure(model, variant, files, *, to=None, revision="main") -> dict[str, Path]:
    """Fetch missing variant files, then optionally stage them at the same paths under to."""
    vdir = variant_dir(model, variant)
    urls = _sources(files, revision)
    with _variant_locks[vdir]:
        manifest = read_manifest(vdir)
        entries = manifest["files"].copy()
        downloads = {}
        for rel, url in urls.items():
            dest = vdir / rel
            size = entries.get(rel, {}).get("bytes")
            if not size or not stage.valid(dest, 0) or size != dest.stat().st_size:
                downloads[rel] = stage.plan(url, dest, min_bytes=_minimum(rel), force=True)
        if downloads:
            stage.run_many(downloads.values())

        pending = defaultdict(lambda: defaultdict(list))
        for rel in urls:
            old = entries.get(rel, {})
            reused = rel not in downloads
            entry = {"source": (old.get("source") if reused else None) or files[rel],
                     "bytes": (vdir / rel).stat().st_size,
                     "sha256": old.get("sha256") if reused else None}
            entries[rel] = entry
            if not entry["sha256"]:
                # Use the recorded origin when a caller supplies an alternate mirror.
                location = _hf_location(_source_url(entry["source"], revision))
                if location:
                    repo, ref, path = location
                    pending[repo, ref][path].append(rel)
        _fill_hashes(entries, pending)
        if entries != manifest["files"] or not (vdir / "manifest.json").exists():
            # Another worker can publish unrelated files during a long transfer.
            latest = read_manifest(vdir)
            latest["files"].update({rel: entries[rel] for rel in files})
            write_manifest(vdir, latest)

        root = Path(to) if to is not None else vdir
        if to is not None and root.resolve() != vdir.resolve():
            copies = [stage.plan(f"drive://ColabUltra/models/{model}/{variant}/{rel}", root / rel,
                                 min_bytes=_minimum(rel), force=True) for rel in files]
            stage.run_many(copies)
    return {rel: root / rel for rel in files}


def describe(model, variant, files, *, to=None):
    """Print source, library and optional local paths without any transfers."""
    vdir = variant_dir(model, variant)
    print(f"{model}/{variant} | {vdir}")
    for rel, url in _sources(files, "main").items():
        plan = stage.plan(url, vdir / rel, min_bytes=_minimum(rel))
        destination = f" -> {Path(to) / rel}" if to is not None else ""
        print(f"  {plan.backend}: {plan.source} -> {plan.dest}{destination}")
