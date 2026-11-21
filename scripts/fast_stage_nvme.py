#!/usr/bin/env python3
"""Dual-channel concurrent NVMe weight stager for Colab (Drive v3 API + HF CDN via aria2c).

Key guarantees:
1. Auto-discovers Colab's native DriveFS auth server (--metadata_server_auth_uri in /proc/*/cmdline),
   with fallback to 127.0.0.1:8012 (colab-dfs-ext) and DRIVE_ACCESS_TOKEN env var.
2. Resolves exact Google Drive relative paths (e.g. 'ColabUltra/huizhou-h3/models/.../vae/diffusion_pytorch_model.safetensors')
   by walking folder IDs from 'root' so duplicate filenames across models NEVER collide.
3. Checks for residual '.aria2' control files before skipping existing files so partial downloads resume cleanly.
4. Prints periodic aria2c progress summaries (--summary-interval=10) instead of running as a silent black box.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path


def ensure_mydrive_symlink() -> None:
    if os.path.isdir("/content/drive/My Drive") and not os.path.exists("/content/drive/MyDrive"):
        try:
            os.symlink("/content/drive/My Drive", "/content/drive/MyDrive")
        except OSError:
            pass


def ensure_aria2() -> None:
    if shutil.which("aria2c"):
        return

    # 1. Fast path (~1s): install from Drive-cached Ubuntu .deb bundle (no apt-get update needed)
    deb_cache_dir = Path("/content/drive/MyDrive/ColabUltra/bin/aria2_debs")
    cached_debs = sorted(deb_cache_dir.glob("*.deb")) if deb_cache_dir.is_dir() else []
    if cached_debs:
        print(f"[Stage] Installing aria2 from Drive cache ({len(cached_debs)} debs)...", flush=True)
        res = subprocess.run(
            ["dpkg", "-i", *[str(p) for p in cached_debs]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode == 0 and shutil.which("aria2c"):
            return
        print(f"[Stage] Drive deb cache fallback ({res.stderr.strip()[:200]}), using apt...", flush=True)

    # 2. Clean apt fallback (~3-5s): skip /etc/apt/sources.list.d/ (avoids 6-min r2u.stat.illinois.edu stall)
    print("[Stage] Installing aria2 via official Ubuntu repo (skipping slow 3rd-party apt sources)...", flush=True)
    apt_fast_update = [
        "apt-get", "-qq",
        "-o", "Dir::Etc::sourcelist=sources.list",
        "-o", "Dir::Etc::sourceparts=-",
        "-o", "APT::Get::List-Cleanup=0",
        "update",
    ]
    if subprocess.run(apt_fast_update).returncode != 0:
        subprocess.check_call(["apt-get", "-qq", "update"])
    subprocess.check_call(
        ["apt-get", "-qq", "-o", "APT::Keep-Downloaded-Packages=true", "install", "-y", "--no-install-recommends", "aria2"]
    )

    # 3. Populate Drive cache for 1-second future cold starts
    try:
        deb_cache_dir.mkdir(parents=True, exist_ok=True)
        for pat in ("aria2_*.deb", "libaria2-0_*.deb", "libcares2_*.deb"):
            for deb in Path("/var/cache/apt/archives").glob(pat):
                dst = deb_cache_dir / deb.name
                if not dst.exists():
                    shutil.copy2(deb, dst)
    except Exception:
        pass


def discover_colab_drive_auth_uris() -> list[str]:
    """Discover metadata_server_auth_uri from running Colab DriveFS process, plus port 8012 fallback."""
    uris: list[str] = []
    for cmd_path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            raw = Path(cmd_path).read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
            m = re.search(r"--metadata_server_auth_uri=(\S+)", raw)
            if m:
                uri = m.group(1).rstrip("/")
                if not uri.startswith(("http://", "https://")):
                    uri = f"http://{uri}"
                if uri not in uris:
                    uris.append(uri)
        except OSError:
            continue
    fallback = "http://127.0.0.1:8012/computeMetadata/v1"
    if fallback not in uris:
        uris.append(fallback)
    return uris


def get_colab_drive_token() -> str:
    env_tok = os.environ.get("DRIVE_ACCESS_TOKEN", "").strip()
    if env_tok:
        return env_tok

    errors = []
    for base_uri in discover_colab_drive_auth_uris():
        token_url = f"{base_uri}/instance/service-accounts/default/token"
        try:
            req = urllib.request.Request(token_url, headers={"Metadata-Flavor": "Google"})
            data = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
            if data.get("access_token"):
                return data["access_token"]
        except Exception as exc:
            errors.append(f"{base_uri}: {exc}")

    raise RuntimeError(
        "Unable to obtain Google Drive OAuth token from Colab DriveFS metadata server or DRIVE_ACCESS_TOKEN. "
        f"Make sure Google Drive is mounted first (`colab-ultra up` / `colab drivemount`). Tried: {'; '.join(errors)}"
    )


def _drive_api_get(token: str, params: dict) -> dict:
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"https://www.googleapis.com/drive/v3/files?{qs}",
        headers={"Authorization": f"Bearer {token}"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8"))


def resolve_drive_entry(token: str, drive_spec: str) -> dict:
    """Resolve either a MyDrive-relative path ('ColabUltra/proj/models/a/b.safetensors') or a unique bare filename."""
    clean = drive_spec.strip().lstrip("/")
    for prefix in ("content/drive/MyDrive/", "content/drive/My Drive/", "MyDrive/", "My Drive/"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]

    if "/" in clean:
        parts = [p for p in clean.split("/") if p]
        parent_id = "root"
        for idx, part in enumerate(parts):
            escaped = part.replace("\\", "\\\\").replace("'", "\\'")
            is_last = idx == len(parts) - 1
            mime_clause = (
                "and mimeType != 'application/vnd.google-apps.folder'"
                if is_last
                else "and mimeType = 'application/vnd.google-apps.folder'"
            )
            q = f"'{parent_id}' in parents and name = '{escaped}' and trashed = false {mime_clause}"
            data = _drive_api_get(token, {"q": q, "fields": "files(id,name,size)", "pageSize": 10})
            files = data.get("files", [])
            if not files:
                raise FileNotFoundError(f"Drive path segment '{part}' not found while resolving '{drive_spec}'")
            if is_last:
                return files[0]
            parent_id = files[0]["id"]

    escaped = clean.replace("\\", "\\\\").replace("'", "\\'")
    matches: list[dict] = []
    page_token = None
    while True:
        params = {
            "q": f"name = '{escaped}' and trashed = false and mimeType != 'application/vnd.google-apps.folder'",
            "fields": "nextPageToken,files(id,name,size,parents)",
            "pageSize": 100,
        }
        if page_token:
            params["pageToken"] = page_token
        data = _drive_api_get(token, params)
        matches.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    if not matches:
        raise FileNotFoundError(f"Drive v3 API did not find '{drive_spec}'")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous bare filename '{drive_spec}' matched {len(matches)} files in Google Drive! "
            f"Pass its MyDrive-relative path instead (e.g. 'ColabUltra/<project>/models/.../{clean}')."
        )
    return matches[0]


def _is_complete_local_file(dest: Path, expected_size: int = 0) -> bool:
    if not dest.exists():
        return False
    if Path(f"{dest}.aria2").exists():
        return False
    actual = dest.stat().st_size
    if expected_size > 0:
        return actual == expected_size
    return actual > 0


def stage_dual_channel(
    dest_root: Path,
    drive_files: list[tuple[str, str]] | None = None,
    hf_urls: list[tuple[str, str, str]] | None = None,
    concurrency: int = 4,
    splits: int = 16,
) -> float:
    """Concurrently stage Drive v3 files and HF CDN URLs onto local NVMe.

    Args:
        dest_root: Local NVMe root directory (e.g. Path('/content/models')).
        drive_files: List of (dest_subdir, drive_rel_path_or_unique_name) to pull via Drive v3 API.
        hf_urls: List of (dest_subdir, out_filename, url) to pull via HF/HTTP CDN.
        concurrency: Max concurrent files per channel (-j).
        splits: Connections per file (-x / -s).

    Returns:
        Elapsed seconds.
    """
    ensure_mydrive_symlink()
    ensure_aria2()
    t0 = time.time()
    drive_files = drive_files or []
    hf_urls = hf_urls or []

    drive_lines: list[str] = []
    hf_lines: list[str] = []
    total_bytes = 0
    token = ""

    if drive_files:
        token = get_colab_drive_token()
        for subdir, drive_spec in drive_files:
            meta = resolve_drive_entry(token, drive_spec)
            fname = meta["name"]
            sz = int(meta.get("size", 0))
            dest_dir = dest_root / subdir if subdir else dest_root
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / fname
            if _is_complete_local_file(dest, expected_size=sz):
                print(f"[Stage] Already staged: {dest} ({sz / (1024**3):.2f} GiB)", flush=True)
                continue
            total_bytes += sz
            print(f"[Stage] Queued Drive v3: {drive_spec} -> {dest} ({sz / (1024**3):.2f} GiB)", flush=True)
            drive_lines.append(
                f"https://www.googleapis.com/drive/v3/files/{meta['id']}?alt=media\n"
                f"  dir={dest_dir}\n"
                f"  out={fname}\n"
            )

    for subdir, fname, url in hf_urls:
        dest_dir = dest_root / subdir if subdir else dest_root
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / fname
        if _is_complete_local_file(dest, expected_size=0):
            print(f"[Stage] Already staged: {dest} ({dest.stat().st_size / (1024**3):.2f} GiB)", flush=True)
            continue
        print(f"[Stage] Queued HF CDN: {fname} -> {dest}", flush=True)
        hf_lines.append(f"{url}\n  dir={dest_dir}\n  out={fname}\n")

    if not drive_lines and not hf_lines:
        print("[Stage] All requested files are already present and verified on local NVMe.", flush=True)
        return 0.0

    procs: list[tuple[str, subprocess.Popen]] = []
    drive_list = Path("/tmp/aria2_ch_drive.txt")
    hf_list = Path("/tmp/aria2_ch_hf.txt")
    auth_conf = Path("/tmp/aria2_drive_auth.conf")

    common_flags = [
        f"-j{concurrency}",
        f"-x{splits}",
        f"-s{splits}",
        "-k16M",
        "--file-allocation=none",
        "--continue=true",
        "--console-log-level=warn",
        "--summary-interval=10",
    ]

    try:
        if drive_lines:
            drive_list.write_text("".join(drive_lines), encoding="utf-8")
            auth_conf.write_text(f"header=Authorization: Bearer {token}\n", encoding="utf-8")
            os.chmod(auth_conf, 0o600)
            procs.append((
                "Drive-v3",
                subprocess.Popen(["aria2c", f"--conf-path={auth_conf}", "-i", str(drive_list), *common_flags]),
            ))

        if hf_lines:
            hf_list.write_text("".join(hf_lines), encoding="utf-8")
            procs.append((
                "HF-CDN",
                subprocess.Popen(["aria2c", "-i", str(hf_list), *common_flags]),
            ))

        for name, proc in procs:
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"{name} aria2c exited with code {rc}")
    finally:
        for p in (auth_conf, drive_list, hf_list):
            if p.exists():
                p.unlink()

    elapsed = time.time() - t0
    if total_bytes > 0:
        gib = total_bytes / (1024**3)
        print(f"[Stage] Done in {elapsed:.2f}s ({gib:.2f} GiB Drive-known, {(gib * 1024) / max(elapsed, 0.1):.1f} MiB/s)", flush=True)
    else:
        print(f"[Stage] Done in {elapsed:.2f}s", flush=True)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage Drive/HF model weights to Colab local NVMe via aria2c.")
    parser.add_argument("--dest-root", default="/content/models", help="Local NVMe destination root")
    parser.add_argument(
        "--drive",
        action="append",
        default=[],
        help="dest_subdir|MyDrive_relative_path (or dest_subdir:unique_filename) to pull via Drive v3 API",
    )
    parser.add_argument("--hf", action="append", default=[], help="dest_subdir|filename|url to pull via HF CDN")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--splits", type=int, default=16)
    args = parser.parse_args()

    drive_files = []
    for item in args.drive:
        if "|" in item:
            subdir, spec = item.split("|", 1)
        elif ":" in item:
            subdir, spec = item.split(":", 1)
        else:
            subdir, spec = "", item
        drive_files.append((subdir, spec))

    hf_urls = []
    for item in args.hf:
        parts = item.split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"Expected --hf format 'subdir|filename|url', got: {item}")
        hf_urls.append((parts[0], parts[1], parts[2]))

    stage_dual_channel(
        dest_root=Path(args.dest_root),
        drive_files=drive_files,
        hf_urls=hf_urls,
        concurrency=args.concurrency,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()
