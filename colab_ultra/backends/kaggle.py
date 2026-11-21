#!/usr/bin/env python3
"""
Kaggle Free Dual-T4 (2x 16GB = 30 GiB VRAM) Backend for Colab Ultra.

Provides a 100% transparent, zero-usage-difference execution layer over Kaggle's
official KernelsService REST API + Official Kaggle Jupyter Server proxy:
  - Quota inspection (`GetAcceleratorQuotaStatistics`: 30h/week GPU, 20h/week TPU)
  - Headless session provisioning (`CreateKernelSession` + `GetOperation`)
  - Seamless `/content/drive/MyDrive/ColabUltra/<project>/{code,models,data,runs}`
    filesystem layout inside the Kaggle container
  - Automatic bidirectional sync with `~/ColabUltra/<project>` (via incremental
    Jupyter Contents tarballs + optional Cloud-to-Cloud Google Drive v3 API for
    multi-GB models/checkpoints at Google Cloud backbone wire speed)
  - Real-time Jupyter WebSocket `exec` streaming + keepalive + graceful `CancelKernelSession`
"""

import io
import json
import os
import re
import tarfile
import time

import requests

from colab_ultra.config import save_config
from colab_ultra.config import DEFAULT_GCLOUD_ADC, DRIVE_ADC_PATH, KAGGLE_SESSIONS_PATH

from .jupyter import (
    jupyter_ensure_kernel_id,
    jupyter_exec_ws,
    jupyter_get_bytes,
    jupyter_get_kernels,
    jupyter_put_bytes,
)

KAGGLE_API_BASE = "https://api.kaggle.com/v1"
def is_kaggle_gpu_spec(gpu_choice: str, force_kaggle: bool = False) -> bool:
    if force_kaggle:
        return True
    g = (gpu_choice or "").strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    return g in ("T4X2", "2XT4", "DUALT4", "KAGGLE", "KAGGLET4", "P100")


def load_kaggle_token() -> str | None:
    """Load Kaggle Bearer API token (`KGAT_...`) from env or `~/.kaggle/access_token`."""
    env_tok = (os.environ.get("KAGGLE_API_TOKEN") or "").strip()
    if env_tok:
        if os.path.isfile(env_tok):
            try:
                with open(env_tok, "r", encoding="utf-8") as f:
                    return f.read().strip() or None
            except Exception:
                pass
        return env_tok

    tok_file = os.path.expanduser("~/.kaggle/access_token")
    if os.path.isfile(tok_file):
        try:
            with open(tok_file, "r", encoding="utf-8") as f:
                val = f.read().strip()
                if val:
                    return val
        except Exception:
            pass
    return None


def _kaggle_headers(token: str | None = None) -> dict:
    tok = token or load_kaggle_token()
    if not tok:
        raise RuntimeError(
            "未检测到 Kaggle API Token (~/.kaggle/access_token)。请先在 https://www.kaggle.com/settings/api 生成 Token。"
        )
    return {
        "Authorization": f"Bearer {tok}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "kaggle-api/v1.8.0 colab-ultra",
    }


def _parse_proto_seconds(val) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s.endswith("s"):
        s = s[:-1]
    try:
        return float(s)
    except Exception:
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        return float(m.group(1)) if m else 0.0


def get_kaggle_quota(timeout: float = 8.0) -> dict:
    """Query Kaggle weekly free GPU (T4 x2) and TPU quota in hours."""
    tok = load_kaggle_token()
    if not tok:
        return {"configured": False}
    try:
        resp = requests.post(
            f"{KAGGLE_API_BASE}/kernels.KernelsApiService/GetAcceleratorQuotaStatistics",
            headers=_kaggle_headers(tok),
            json={},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        gq = data.get("gpuQuota") or {}
        tq = data.get("tpuQuota") or {}

        g_tot = _parse_proto_seconds(gq.get("totalTimeAllowed"))
        g_used = _parse_proto_seconds(gq.get("timeUsed"))
        g_res = _parse_proto_seconds(gq.get("timeReserved"))
        t_tot = _parse_proto_seconds(tq.get("totalTimeAllowed"))
        t_used = _parse_proto_seconds(tq.get("timeUsed"))

        return {
            "configured": True,
            "gpu_total_hours": round(g_tot / 3600.0, 2),
            "gpu_used_hours": round(g_used / 3600.0, 2),
            "gpu_remaining_hours": round(max(0.0, g_tot - g_used) / 3600.0, 2),
            "gpu_reserved_seconds": round(g_res, 1),
            "tpu_total_hours": round(t_tot / 3600.0, 2),
            "tpu_remaining_hours": round(max(0.0, t_tot - t_used) / 3600.0, 2),
            "refresh_time": data.get("quotaRefreshTime", ""),
        }
    except Exception as e:
        return {"configured": True, "error": str(e)}


def _load_kaggle_sessions() -> dict:
    if not os.path.isfile(KAGGLE_SESSIONS_PATH):
        return {}
    try:
        with open(KAGGLE_SESSIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_kaggle_sessions(sessions: dict) -> None:
    """Publish the private registry atomically."""
    save_config(sessions, KAGGLE_SESSIONS_PATH)


def resolve_carrier_slug(cfg: dict) -> str:
    """Find or remember the user's private Kaggle carrier notebook slug."""
    slug = (cfg.get("kaggle_carrier_slug") or "").strip()
    if slug and "/" in slug:
        return slug

    resp = requests.post(
        f"{KAGGLE_API_BASE}/kernels.KernelsApiService/ListKernels",
        headers=_kaggle_headers(),
        json={"mine": True, "pageSize": 30},
        timeout=15,
    )
    resp.raise_for_status()
    kernels = (resp.json() or {}).get("kernels") or []
    preferred = None
    fallback = None
    for k in kernels:
        ref = k.get("ref") or ""
        if not ref and k.get("author") and k.get("slug"):
            ref = f"{k['author']}/{k['slug']}"
        if not ref or "/" not in ref:
            continue
        if "colab-ultra" in ref.lower():
            preferred = ref
            break
        if fallback is None:
            fallback = ref

    chosen = preferred or fallback
    if not chosen:
        raise RuntimeError(
            "账号下未找到可用的私有 Kaggle Notebook 载体。请先在 Kaggle Code 创建一个私有 Notebook (如 colab-ultra-t4x2-probe)。"
        )
    return chosen


# ---------------------------------------------------------------------------
# Jupyter Proxy REST + WebSocket helpers (works identically for Kaggle JWE URL)
# ---------------------------------------------------------------------------


# Google Drive Cloud-to-Cloud Short-Lived Token Helper (Zero Long-Lived Secrets on VM)
# ---------------------------------------------------------------------------

_DRIVE_TOKEN_STATUS = {"checked_at": 0.0, "ok": False, "token": None, "expires_at": 0.0}


def get_short_lived_drive_token() -> str | None:
    """
    If local Google ADC has `https://www.googleapis.com/auth/drive` scope,
    mint a short-lived 1-hour access token locally so Kaggle VMs can transfer
    multi-GB models/checkpoints directly with Google Drive over Google Cloud's
    internal network without exposing refresh_token or touching Mac bandwidth.
    """
    now = time.time()
    if _DRIVE_TOKEN_STATUS["ok"] and _DRIVE_TOKEN_STATUS["token"] and _DRIVE_TOKEN_STATUS["expires_at"] > now + 180:
        return _DRIVE_TOKEN_STATUS["token"]
    if not _DRIVE_TOKEN_STATUS["ok"] and (now - _DRIVE_TOKEN_STATUS["checked_at"] < 60):
        return None

    _DRIVE_TOKEN_STATUS["checked_at"] = now
    try:
        import google.auth.transport.requests
        from google.oauth2.credentials import Credentials

        for cand in (DRIVE_ADC_PATH, DEFAULT_GCLOUD_ADC):
            if not os.path.isfile(cand):
                continue
            with open(cand, "r", encoding="utf-8") as f:
                info = json.load(f)
            creds = Credentials.from_authorized_user_info(info)
            creds.refresh(google.auth.transport.requests.Request())
            tok = creds.token
            r = requests.get(
                "https://www.googleapis.com/drive/v3/about?fields=user(emailAddress)",
                headers={"Authorization": f"Bearer {tok}"},
                timeout=6,
            )
            if r.status_code == 200:
                _DRIVE_TOKEN_STATUS.update({
                    "ok": True,
                    "token": tok,
                    "expires_at": now + 3000,
                })
                return tok
    except Exception:
        pass

    _DRIVE_TOKEN_STATUS["ok"] = False
    return None


# ---------------------------------------------------------------------------
# Transparent `/content/drive/MyDrive/ColabUltra/<project>` Sync Engine
# ---------------------------------------------------------------------------

_IGNORE_NAMES = {"__pycache__", ".git", ".DS_Store", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache"}


def _should_skip(name: str) -> bool:
    return name in _IGNORE_NAMES or name.startswith("._")


def sync_project_to_kaggle(proxy_url: str, project_name: str, local_hub: str | None, max_inline_mb: float = 80.0):
    """
    Ensure `/content/drive/MyDrive/ColabUltra/<project>/{code,models,data,runs}` exists on Kaggle,
    incrementally sync local `code/` (and small/medium `data/` files) into the container,
    and if Drive OAuth is active, sync any large cloud `models/` & `data/` directly from Google Drive.
    """
    if not project_name:
        return

    # 1. Fetch existing remote inbound manifest so unchanged files aren't re-uploaded
    remote_manifest_bytes = jupyter_get_bytes(
        proxy_url, "", f"/content/drive/MyDrive/ColabUltra/{project_name}/_cu_in_manifest.json", timeout=10
    )
    remote_manifest = {}
    if remote_manifest_bytes:
        try:
            remote_manifest = json.loads(remote_manifest_bytes.decode("utf-8"))
        except Exception:
            remote_manifest = {}

    new_manifest = dict(remote_manifest)
    changed_files: list[tuple[str, str]] = []  # (abs_local_path, rel_inside_project)
    max_bytes = int(max_inline_mb * 1024 * 1024)

    if local_hub:
        proj_local = os.path.join(local_hub, project_name)
        for sub in ("code", "data"):
            sub_dir = os.path.join(proj_local, sub)
            if not os.path.isdir(sub_dir):
                continue
            for root, dirs, files in os.walk(sub_dir):
                dirs[:] = [d for d in dirs if not _should_skip(d)]
                for fn in files:
                    if _should_skip(fn):
                        continue
                    abs_p = os.path.join(root, fn)
                    try:
                        st = os.stat(abs_p)
                    except OSError:
                        continue
                    if not os.path.isfile(abs_p) or st.st_size > max_bytes:
                        continue
                    rel_p = os.path.relpath(abs_p, proj_local).replace(os.sep, "/")
                    sig = f"{st.st_size}:{int(st.st_mtime)}"
                    if remote_manifest.get(rel_p) != sig:
                        changed_files.append((abs_p, rel_p))
                        new_manifest[rel_p] = sig

    has_tar = False
    if changed_files:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for abs_p, rel_p in changed_files:
                tf.add(abs_p, arcname=rel_p)
            man_bytes = json.dumps(new_manifest).encode("utf-8")
            ti = tarfile.TarInfo(name="_cu_in_manifest.json")
            ti.size = len(man_bytes)
            ti.mtime = int(time.time())
            tf.addfile(ti, io.BytesIO(man_bytes))
        jupyter_put_bytes(proxy_url, "", "/content/_cu_sync_in.tar.gz", buf.getvalue(), timeout=90)
        has_tar = True

    drive_tok = get_short_lived_drive_token()
    unpack_code = f"""
import os, tarfile
_proj = '/content/drive/MyDrive/ColabUltra/{project_name}'
for _s in ('code', 'models', 'data', 'runs'):
    os.makedirs(f'{{_proj}}/{{_s}}', exist_ok=True)
if {has_tar!r} and os.path.isfile('/content/_cu_sync_in.tar.gz'):
    with tarfile.open('/content/_cu_sync_in.tar.gz', 'r:gz') as _tf:
        _tf.extractall(_proj)
    os.remove('/content/_cu_sync_in.tar.gz')
"""
    if drive_tok:
        # Also pull any large files in MyDrive/ColabUltra/<project>/{models,data} directly from Google Drive v3 API
        unpack_code += f"""
_dtok = {drive_tok!r}
try:
    import requests as _rq
    _hdr = {{'Authorization': f'Bearer {{_dtok}}'}}
    def _find_folder(parent_id, name):
        q = f"name = '{{name}}' and '{{parent_id}}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        r = _rq.get('https://www.googleapis.com/drive/v3/files', headers=_hdr, params={{'q': q, 'fields': 'files(id,name)'}}, timeout=15).json()
        fs = r.get('files') or []
        return fs[0]['id'] if fs else None

    def _sync_down_folder(folder_id, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        pt = None
        while True:
            q = f"'{{folder_id}}' in parents and trashed = false"
            r = _rq.get('https://www.googleapis.com/drive/v3/files', headers=_hdr, params={{'q': q, 'fields': 'nextPageToken,files(id,name,mimeType,size)', 'pageSize': 200, 'pageToken': pt}}, timeout=20).json()
            for it in r.get('files') or []:
                nm = it['name']
                if nm.startswith('._'): continue
                dst = os.path.join(local_dir, nm)
                if it['mimeType'] == 'application/vnd.google-apps.folder':
                    _sync_down_folder(it['id'], dst)
                elif not it['mimeType'].startswith('application/vnd.google-apps.'):
                    sz = int(it.get('size') or 0)
                    if os.path.isfile(dst) and os.path.getsize(dst) == sz:
                        continue
                    with _rq.get(f"https://www.googleapis.com/drive/v3/files/{{it['id']}}?alt=media", headers=_hdr, stream=True, timeout=120) as dl:
                        dl.raise_for_status()
                        tmp = dst + '.part'
                        with open(tmp, 'wb') as f:
                            for ch in dl.iter_content(chunk_size=8*1024*1024):
                                if ch: f.write(ch)
                        os.replace(tmp, dst)
            pt = r.get('nextPageToken')
            if not pt: break

    _cu_id = _find_folder('root', 'ColabUltra')
    _pj_id = _find_folder(_cu_id, {project_name!r}) if _cu_id else None
    if _pj_id:
        for _sub in ('models', 'data'):
            _sid = _find_folder(_pj_id, _sub)
            if _sid:
                _sync_down_folder(_sid, f'{{_proj}}/{{_sub}}')
except Exception as _e:
    print(f'[colab-ultra] Drive cloud pull notice: {{_e}}')
"""
    jupyter_exec_ws(proxy_url, unpack_code, timeout=600.0)


def pull_live_tarball_to_local(proxy_url: str, remote_tar_path: str, local_dest_dir: str) -> int:
    """Download a pre-packed `.tar.gz` from `/api/contents/` (works even while kernel is BUSY!) and extract locally."""
    raw = jupyter_get_bytes(proxy_url, "", remote_tar_path, timeout=60.0)
    if not raw:
        return 0
    os.makedirs(local_dest_dir, exist_ok=True)
    count = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            for member in tf.getmembers():
                # Prevent path traversal
                norm = os.path.normpath(member.name)
                if norm.startswith("..") or os.path.isabs(norm):
                    continue
                tf.extract(member, path=local_dest_dir)
                count += 1
    except Exception:
        return 0
    return count


def build_in_vm_sync_snippet(project_name: str | None, drive_token: str | None) -> str:
    """
    Generate Python helper snippet injected into Kaggle `exec` wrapper so:
      1. During training (`_hb_loop` every 20s), modified logs/samples/events in `runs/`
         are packed into `/content/.cu_live_runs.tar.gz` without blocking training.
      2. In `finally:` on completion/exception, ALL modified files in `runs/` & `outputs/`
         are either uploaded directly to Google Drive (if >25MB and `_dtok` available)
         or packed into `/content/.cu_final_runs.tar.gz` for immediate local extraction.
    """
    if not project_name:
        return "def _cu_pack_runs(final=False): pass\n"

    return f"""
_cu_sync_state = {{}}
_cu_drive_tok = {drive_token!r}

def _cu_upload_large_to_drive(local_path, rel_under_proj):
    if not _cu_drive_tok:
        return False
    try:
        import requests as _rq
        _hdr = {{'Authorization': f'Bearer {{_cu_drive_tok}}'}}
        def _ens_folder(pid, name):
            q = f"name = '{{name}}' and '{{pid}}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            r = _rq.get('https://www.googleapis.com/drive/v3/files', headers=_hdr, params={{'q': q, 'fields': 'files(id)'}}, timeout=15).json()
            fs = r.get('files') or []
            if fs: return fs[0]['id']
            cr = _rq.post('https://www.googleapis.com/drive/v3/files', headers={{**_hdr, 'Content-Type': 'application/json'}}, json={{'name': name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [pid]}}, timeout=15).json()
            return cr['id']

        parts = [p for p in rel_under_proj.split('/') if p]
        pid = _ens_folder('root', 'ColabUltra')
        pid = _ens_folder(pid, {project_name!r})
        for seg in parts[:-1]:
            pid = _ens_folder(pid, seg)
        fname = parts[-1]
        sz = os.path.getsize(local_path)
        q = f"name = '{{fname}}' and '{{pid}}' in parents and trashed = false"
        ex = (_rq.get('https://www.googleapis.com/drive/v3/files', headers=_hdr, params={{'q': q, 'fields': 'files(id,size)'}}, timeout=15).json().get('files') or [])
        if ex:
            init_url = f"https://www.googleapis.com/upload/drive/v3/files/{{ex[0]['id']}}?uploadType=resumable"
            init_r = _rq.patch(init_url, headers={{**_hdr, 'Content-Type': 'application/json; charset=UTF-8', 'X-Upload-Content-Length': str(sz)}}, json={{'name': fname}}, timeout=20)
        else:
            init_url = 'https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable'
            init_r = _rq.post(init_url, headers={{**_hdr, 'Content-Type': 'application/json; charset=UTF-8', 'X-Upload-Content-Length': str(sz)}}, json={{'name': fname, 'parents': [pid]}}, timeout=20)
        init_r.raise_for_status()
        up_url = init_r.headers['Location']
        with open(local_path, 'rb') as f:
            put_r = _rq.put(up_url, data=f, headers={{'Content-Length': str(sz)}}, timeout=900)
            put_r.raise_for_status()
        return True
    except Exception as _e:
        print(f'[colab-ultra] Drive direct upload fallback ({{rel_under_proj}}): {{_e}}')
        return False

def _cu_pack_runs(final=False):
    import tarfile
    proj_dir = '/content/drive/MyDrive/ColabUltra/{project_name}'
    runs_dir = os.path.join(proj_dir, 'runs')
    if not os.path.isdir(runs_dir):
        return
    to_tar = []
    max_tar_file = (25 * 1024 * 1024) if (_cu_drive_tok and final) else (220 * 1024 * 1024 if final else 15 * 1024 * 1024)
    for root, dirs, files in os.walk(runs_dir):
        for fn in files:
            if fn.startswith('._') or fn.endswith('.tmp') or fn.endswith('.part'):
                continue
            ap = os.path.join(root, fn)
            try:
                st = os.stat(ap)
            except OSError:
                continue
            rel = os.path.relpath(ap, proj_dir).replace(os.sep, '/')
            sig = f'{{st.st_size}}:{{int(st.st_mtime)}}'
            if _cu_sync_state.get(rel) == sig:
                continue
            if final and st.st_size > max_tar_file and _cu_drive_tok:
                if _cu_upload_large_to_drive(ap, rel):
                    _cu_sync_state[rel] = sig
                    continue
            if st.st_size <= max_tar_file:
                to_tar.append((ap, rel, sig))
            elif final:
                print(f'⚠️ [colab-ultra] 大文件 {{rel}} ({{st.st_size/1048576:.1f}} MB) 超出免授权回传上限；运行 `colab-ultra init --with-drive-scope` 可开启云对云无限大小直传 Google Drive。')
    if not to_tar:
        return
    out_tar = '/content/_cu_final_runs.tar.gz' if final else '/content/_cu_live_runs.tar.gz'
    tmp_tar = out_tar + '.tmp'
    with tarfile.open(tmp_tar, 'w:gz', compresslevel=1 if final else 3) as tf:
        for ap, rel, sig in to_tar:
            try:
                tf.add(ap, arcname=rel)
                if final:
                    _cu_sync_state[rel] = sig
            except Exception:
                pass
    os.replace(tmp_tar, out_tar)
"""


def flush_kaggle_outputs_before_stop(proxy_url: str, local_hub: str | None):
    """Best-effort sync of any modified `runs/` files across all projects on `/content/drive/MyDrive/ColabUltra` before stopping."""
    if not local_hub:
        return
    code = """
import os, tarfile
base = '/content/drive/MyDrive/ColabUltra'
out_tar = '/content/_cu_shutdown_flush.tar.gz'
items = []
if os.path.isdir(base):
    for proj in os.listdir(base):
        pdir = os.path.join(base, proj)
        rdir = os.path.join(pdir, 'runs')
        if not os.path.isdir(rdir):
            continue
        for root, _, files in os.walk(rdir):
            for fn in files:
                if fn.startswith('._') or fn.endswith(('.tmp', '.part')):
                    continue
                ap = os.path.join(root, fn)
                try:
                    sz = os.path.getsize(ap)
                except OSError:
                    continue
                rel = os.path.relpath(ap, base).replace(os.sep, '/')
                if sz <= 200 * 1024 * 1024:
                    items.append((ap, rel))
if items:
    with tarfile.open(out_tar + '.tmp', 'w:gz', compresslevel=1) as tf:
        for ap, rel in items:
            try: tf.add(ap, arcname=rel)
            except Exception: pass
    os.replace(out_tar + '.tmp', out_tar)
print('FLUSHED:', len(items))
"""
    try:
        ks = jupyter_get_kernels(proxy_url, "", timeout=6.0)
        if ks and not any(k.get("execution_state") == "busy" for k in ks):
            jupyter_exec_ws(proxy_url, code, timeout=45.0)
            pull_live_tarball_to_local(proxy_url, "/content/_cu_shutdown_flush.tar.gz", local_hub)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# High-Level Session Provisioning, Inspection & Shutdown
# ---------------------------------------------------------------------------


def spin_up_kaggle_session(
    session_name: str,
    gpu_choice: str,
    cfg: dict,
    probe_local_path: str,
    quiet: bool = False,
    console=None,
) -> dict:
    """Provision a Kaggle T4x2 interactive session and initialize `/content/drive/MyDrive/ColabUltra`."""
    sessions = _load_kaggle_sessions()
    existing = sessions.get(session_name)
    if existing and existing.get("proxy_url"):
        if jupyter_get_kernels(existing["proxy_url"], "", timeout=6.0) is not None:
            if console and not quiet:
                console.print(f"♻️ [green]复用已在运行的 Kaggle 免费双卡会话 ({session_name}, sid={existing['session_id']})[/green]")
            return existing

    g_norm = (gpu_choice or "T4X2").upper().replace("-", "").replace("_", "")
    if g_norm == "P100":
        machine_shape = "NvidiaTeslaP100"
        acc_label = "P100"
        shape_label = "1xP100-16GB (Free)"
    elif g_norm in ("TPU", "V5E8", "V38"):
        machine_shape = "Tpu1VmV38"
        acc_label = "TPU-v5e8"
        shape_label = "TPU-20h/wk (Free)"
    else:
        machine_shape = "NvidiaTeslaT4"
        acc_label = "T4x2"
        shape_label = "2xT4-30GB (Free)"

    slug = resolve_carrier_slug(cfg)
    if console and not quiet:
        console.print(f"⚡ [bold cyan]正在启动 Kaggle 免费 {acc_label} ({shape_label}) 实例 ({session_name}, 0 CU/h)...[/bold cyan]")

    hdrs = _kaggle_headers()
    resp = requests.post(
        f"{KAGGLE_API_BASE}/kernels.KernelsApiService/CreateKernelSession",
        headers=hdrs,
        json={
            "slug": slug,
            "language": "python",
            "kernelType": "notebook",
            "machineShape": machine_shape,
            "enableInternet": True,
        },
        timeout=25,
    )
    resp.raise_for_status()
    op = resp.json()
    op_name = op["name"]
    sid = int((op.get("metadata") or {}).get("kernelSessionId") or op_name.split("/")[-1])

    # Record session_id immediately so `colab-ultra down` can always clean up even if interrupted
    sessions[session_name] = {
        "name": session_name,
        "provider": "kaggle",
        "session_id": sid,
        "endpoint": f"kaggle-{sid}",
        "slug": slug,
        "accelerator": acc_label,
        "shape": shape_label,
        "proxy_url": "",
        "proxy_token": "",
        "created_at": time.time(),
    }
    _save_kaggle_sessions(sessions)

    for _ in range(60):
        if op.get("done"):
            break
        time.sleep(3.0)
        r2 = requests.post(
            f"{KAGGLE_API_BASE}/common.OperationsService/GetOperation",
            headers=hdrs,
            json={"name": op_name},
            timeout=15,
        )
        r2.raise_for_status()
        op = r2.json()

    if not op.get("done") or op.get("error"):
        err = op.get("error") or "Timed out waiting for CreateKernelSession"
        try:
            requests.post(
                f"{KAGGLE_API_BASE}/kernels.KernelsApiService/CancelKernelSession",
                headers=hdrs,
                json={"kernelSessionId": sid},
                timeout=10,
            )
        except Exception:
            pass
        sessions.pop(session_name, None)
        _save_kaggle_sessions(sessions)
        raise RuntimeError(f"Kaggle 会话启动失败: {err}")

    proxy_url = (op.get("response") or {}).get("jupyterUrl", "").rstrip("/")
    sessions[session_name]["proxy_url"] = proxy_url
    _save_kaggle_sessions(sessions)

    # Bootstrap `/content` symlink -> `/kaggle/working` (even if `/content` pre-exists in Kaggle's Docker image)
    kid = jupyter_ensure_kernel_id(proxy_url, "")
    boot_env_code = """
import os, shutil
kw = '/kaggle/working'
os.makedirs(kw, exist_ok=True)
if os.path.islink('/content'):
    if os.path.realpath('/content') != os.path.realpath(kw):
        os.unlink('/content')
        os.symlink(kw, '/content')
elif os.path.isdir('/content'):
    for _it in os.listdir('/content'):
        _src, _dst = os.path.join('/content', _it), os.path.join(kw, _it)
        if not os.path.exists(_dst):
            shutil.move(_src, _dst)
    try:
        shutil.rmtree('/content')
        os.symlink(kw, '/content')
    except Exception:
        os.makedirs(f'{kw}/drive', exist_ok=True)
        if not os.path.exists('/content/drive'):
            os.symlink(f'{kw}/drive', '/content/drive')
else:
    os.symlink(kw, '/content')
hub = '/content/drive/MyDrive/ColabUltra'
for sub in ('models', 'datasets', 'outputs/images', 'outputs/training', 'hf_cache', 'torch_cache', 'pip_cache', 'workspace', 'emergency_backups'):
    os.makedirs(f'{hub}/{sub}', exist_ok=True)
for lk, tg in (('workspace', f'{hub}/workspace'), ('outputs', f'{hub}/outputs')):
    lp = f'/content/{lk}'
    if not os.path.exists(lp) and not os.path.islink(lp):
        os.symlink(tg, lp)
os.environ['COLAB_ULTRA_ROOT'] = hub
os.environ['HF_HOME'] = f'{hub}/hf_cache'
os.environ['TORCH_HOME'] = f'{hub}/torch_cache'
"""
    jupyter_exec_ws(proxy_url, boot_env_code, kernel_id=kid, timeout=30.0)

    if probe_local_path and os.path.isfile(probe_local_path):
        with open(probe_local_path, "rb") as pf:
            probe_bytes = pf.read()
        jupyter_put_bytes(proxy_url, "", "/content/_cu_probe.py", probe_bytes, timeout=30.0)
        probe_boot = (
            "import subprocess, shutil, os\n"
            "if os.path.isfile('/content/_cu_probe.py'):\n"
            "    shutil.copy2('/content/_cu_probe.py', '/content/.colab_probe.py')\n"
            "subprocess.run(['python3', '/content/_cu_probe.py', '--once'], check=False)\n"
            "subprocess.Popen(['python3', '/content/_cu_probe.py'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
        )
        jupyter_exec_ws(proxy_url, probe_boot, kernel_id=kid, timeout=20.0)

    return sessions[session_name]


def fetch_kaggle_telemetry(proxy_url: str, probe_local_path: str = "", auto_deploy: bool = True) -> dict:
    raw = (
        jupyter_get_bytes(proxy_url, "", "/content/_cu_telemetry.json", timeout=10.0)
        or jupyter_get_bytes(proxy_url, "", "/content/.colab_telemetry.json", timeout=10.0)
    )
    data = None
    if raw:
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = None
    if data and (time.time() - data.get("unix_time", 0)) < 25:
        return data

    if auto_deploy and probe_local_path and os.path.isfile(probe_local_path):
        try:
            ks = jupyter_get_kernels(proxy_url, "", timeout=6.0)
            if ks and not any(k.get("execution_state") == "busy" for k in ks):
                with open(probe_local_path, "rb") as pf:
                    jupyter_put_bytes(proxy_url, "", "/content/_cu_probe.py", pf.read(), timeout=20.0)
                boot = (
                    "import subprocess, shutil, os\n"
                    "if os.path.isfile('/content/_cu_probe.py'):\n"
                    "    shutil.copy2('/content/_cu_probe.py', '/content/.colab_probe.py')\n"
                    "subprocess.run(['python3', '/content/_cu_probe.py', '--once'], check=False)\n"
                    "subprocess.Popen(['python3', '/content/_cu_probe.py'], "
                    "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
                )
                jupyter_exec_ws(proxy_url, boot, timeout=15.0)
                raw2 = (
                    jupyter_get_bytes(proxy_url, "", "/content/_cu_telemetry.json", timeout=10.0)
                    or jupyter_get_bytes(proxy_url, "", "/content/.colab_telemetry.json", timeout=10.0)
                )
                if raw2:
                    data = json.loads(raw2.decode("utf-8"))
        except Exception:
            pass

    return data or {}


def list_active_kaggle_vms(probe_local_path: str = "", auto_deploy: bool = True) -> list[dict]:
    """Return active Kaggle sessions formatted identically to Colab VM dicts."""
    sessions = _load_kaggle_sessions()
    if not sessions:
        return []

    vms = []
    for s_name, info in list(sessions.items()):
        p_url = info.get("proxy_url") or ""
        if not p_url:
            continue

        kernels = jupyter_get_kernels(p_url, "", timeout=8.0)
        if kernels is None:
            # A failed proxy check does not establish that billing stopped.
            continue

        kernel_state = "idle"
        last_activity = ""
        for k in kernels:
            if k.get("execution_state") == "busy":
                kernel_state = "busy"
            last_activity = k.get("last_activity", last_activity)

        telemetry = fetch_kaggle_telemetry(p_url, probe_local_path=probe_local_path, auto_deploy=auto_deploy)
        vms.append({
            "name": s_name,
            "provider": "kaggle",
            "session_id": info.get("session_id"),
            "endpoint": info.get("endpoint") or f"kaggle-{info.get('session_id')}",
            "accelerator": info.get("accelerator", "T4x2"),
            "shape": info.get("shape", "2xT4-30GB (Free)"),
            "proxy_url": p_url,
            "proxy_token": "",
            "kernel_state": kernel_state,
            "last_activity": last_activity,
            "telemetry": telemetry,
        })

    return vms


def shut_down_kaggle_session(vm: dict, local_hub: str | None = None, flush_outputs: bool = True) -> bool:
    """Flush outputs to `~/ColabUltra` and cancel the Kaggle session (`0s` reserved)."""
    p_url = vm.get("proxy_url") or ""
    sid = vm.get("session_id")
    if not sid:
        ep = str(vm.get("endpoint") or "")
        if ep.startswith("kaggle-") and ep[7:].isdigit():
            sid = int(ep[7:])

    if flush_outputs and p_url:
        flush_kaggle_outputs_before_stop(p_url, local_hub)

    if not sid:
        raise ValueError(f"Missing Kaggle session ID for {vm['name']}")
    response = requests.post(f"{KAGGLE_API_BASE}/kernels.KernelsApiService/CancelKernelSession",
                             headers=_kaggle_headers(), json={"kernelSessionId": int(sid)}, timeout=15)
    if response.status_code != 404:
        response.raise_for_status()
    return True


def kaggle_active(vm):
    """Check the specific Jupyter endpoint. Treat ambiguous transport failures as unknown."""
    if vm.get("proxy_url"):
        response = requests.get(f"{vm['proxy_url'].rstrip('/')}/api/kernels", timeout=8)
        if response.status_code in (404, 410):
            return False
        response.raise_for_status()
        if not isinstance(response.json(), list):
            raise ValueError(f"Unexpected Kaggle kernel status for {vm['name']}")
        return True
    # Interrupted creations have a carrier slug but may lack a proxy.
    owner, slug = vm["slug"].split("/", 1)
    response = requests.post(f"{KAGGLE_API_BASE}/kernels.KernelsApiService/GetKernelSessionStatus",
                             headers=_kaggle_headers(), json={"userName": owner, "kernelSlug": slug}, timeout=10)
    response.raise_for_status()
    status = response.json().get("status", "QUEUED")
    return str(status).upper() not in {"2", "3", "5", "COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}


def forget_kaggle_session(vm):
    """Drop a verified stopped session only if the local name still matches its ID."""
    sessions = _load_kaggle_sessions()
    if sessions.get(vm["name"], {}).get("session_id") == vm["session_id"]:
        sessions.pop(vm["name"], None)
        _save_kaggle_sessions(sessions)
