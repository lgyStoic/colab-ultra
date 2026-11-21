import base64
import json
import sys
import time
import uuid

import requests
from colab_cli.auth import AuthProvider
from colab_cli.common import state as colab_state
from colab_cli.runtime import ColabRuntime

from colab_ultra.backends.jupyter import jupyter_headers

from .jupyter import jupyter_exec_ws, jupyter_get_kernels
from colab_ultra.retry import retry, transient

colab_state.auth_provider = AuthProvider.ADC


def backup_vm_outputs_to_drive(proxy_url: str, proxy_token: str):
    """If Drive is mounted and /content/outputs exists, copy to emergency_backups before shutdown."""
    try:
        rt = ColabRuntime(proxy_url, proxy_token)
        code = (
            "import os, shutil, time\n"
            "if os.path.isdir('/content/drive/MyDrive') and os.path.isdir('/content/outputs'):\n"
            "    dst = f'/content/drive/MyDrive/ColabUltra/emergency_backups/backup_{int(time.time())}'\n"
            "    os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
            "    shutil.copytree('/content/outputs', dst, dirs_exist_ok=True)\n"
        )
        rt.execute_code(code, timeout=6)
        rt.stop()
    except Exception:
        pass

def fetch_vm_kernels(proxy_url: str, proxy_token: str) -> list:
    """Read the backend-neutral kernel inventory for telemetry."""
    return jupyter_get_kernels(proxy_url, proxy_token) or []


def restart_vm_kernel(proxy_url: str, proxy_token: str) -> bool:
    """Restart the Jupyter kernel on the VM to release all leaked GPU VRAM without losing the VM or Drive mount."""
    kernels = fetch_vm_kernels(proxy_url, proxy_token)
    if not kernels:
        return False
    kid = kernels[0].get("id")
    if not kid:
        return False
    url = f"{proxy_url.rstrip('/')}/api/kernels/{kid}/restart"
    params, headers = jupyter_headers(proxy_url, proxy_token)
    try:
        resp = requests.post(url, params=params, headers=headers, timeout=15)
        time.sleep(2.0)
        return resp.status_code in (200, 204)
    except Exception:
        return False


def resolve_proxy(name):
    """Resolve only the named live assignment, even if another VM is the sole survivor."""

    colab_state._sessions = None
    try:
        sessions, assignments = retry(colab_state.sync_sessions)
    except (SystemExit, Exception) as exc:
        if "active colab session" in str(exc).lower() or "session not found" in str(exc).lower():
            raise SystemExit(lost_session_message(name)) from exc
        raise
    local = sessions.get(name)
    for assignment in assignments:
        if assignment.endpoint == (local.endpoint if local else name):
            proxy = assignment.runtime_proxy_info
            return proxy.url, proxy.token
    raise SystemExit(lost_session_message(name))


def lost_session_message(name):
    """Tell the operator how to recover without targeting a different assignment."""
    return (f"Active Colab session not found: {name}. Run colab-ultra status, "
            f"then colab-ultra up -s {name} --colab with your accelerator flags.")



def assignments():
    """Return active assignments with local names, without probing their kernels."""
    names = {s.endpoint: s.name for s in colab_state.store.list().values()}
    return [{"provider": "colab", "endpoint": a.endpoint, "name": names.get(a.endpoint, "web-session"),
             "accelerator": a.accelerator.value, "proxy_url": a.runtime_proxy_info.url,
             "proxy_token": a.runtime_proxy_info.token} for a in colab_state.client.list_assignments()]


def run_detached(args, wrapped_code):
    """Submit once and collect incremental logs from the detached job."""
    timeout_sec = args.timeout

    p_url, p_tok = resolve_proxy(args.session)
    last_token_refresh = time.monotonic()
    if args.restart_kernel:
        restart_vm_kernel(p_url, p_tok)

    def _short_ws_exec(code, call_timeout=25.0):
        nonlocal p_url, p_tok, last_token_refresh
        if time.monotonic() - last_token_refresh > 2400:
            p_url, p_tok = resolve_proxy(args.session)
            last_token_refresh = time.monotonic()
        try:
            return jupyter_exec_ws(p_url, code, proxy_token=p_tok, timeout=call_timeout)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None) or getattr(exc, "status_code", None)
            if status in (404, 410) or "active colab session" in str(exc).lower():
                raise SystemExit(lost_session_message(args.session)) from exc
            raise

    job_id = uuid.uuid4().hex[:10]
    remote_py = f"/tmp/_cu_job_{job_id}.py"
    remote_log = f"/tmp/_cu_job_{job_id}.log"
    remote_rc = f"/tmp/_cu_job_{job_id}.rc"
    remote_pid_file = f"/tmp/_cu_job_{job_id}.pid"
    src_b64 = base64.b64encode(wrapped_code.encode("utf-8")).decode("ascii")

    launch_code = (
        "import base64, json, os, shlex, subprocess, sys\n"
        f"_py = {remote_py!r}\n"
        f"_log = {remote_log!r}\n"
        f"_rc = {remote_rc!r}\n"
        f"_pidf = {remote_pid_file!r}\n"
        f"with open(_py, 'wb') as _f: _f.write(base64.b64decode({src_b64!r}))\n"
        "for _p in (_log, _rc, _pidf):\n"
        "    if os.path.exists(_p):\n"
        "        try: os.remove(_p)\n"
        "        except OSError: pass\n"
        "_cmd = f'nohup {shlex.quote(sys.executable)} -u {shlex.quote(_py)} > {shlex.quote(_log)} 2>&1; echo $? > {shlex.quote(_rc)}'\n"
        "_proc = subprocess.Popen(['bash', '-lc', _cmd], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "with open(_pidf, 'w') as _f: _f.write(str(_proc.pid))\n"
        "print(json.dumps({'launched': True, 'pid': _proc.pid}))\n"
    )

    t0 = time.time()
    is_json_mode = bool(getattr(args, "json", False))
    out_chunks: list[str] = []
    err_str = ""
    ret_code = 1
    remote_pgid: int | None = None

    l_rc, l_out, l_err = _short_ws_exec(launch_code, call_timeout=30.0)
    if l_rc != 0:
        if "active colab session" in (l_err or l_out).lower():
            raise SystemExit(lost_session_message(args.session))
        raise SystemExit(f"[colab-ultra] Failed to launch remote job: {l_err or l_out}")
    for line in l_out.splitlines():
        try:
            info = json.loads(line.strip())
            if info.get("launched"):
                remote_pgid = int(info.get("pid", 0)) or None
        except Exception:
            pass

    if not remote_pgid:
        raise SystemExit(f"Launch acknowledgement missing for {job_id}. Inspect {remote_log} before running again.")
    deadline = t0 + float(timeout_sec)
    offset = 0
    consecutive_errors = 0

    try:
        while True:
            now = time.time()
            if now >= deadline:
                err_str = f"[colab-ultra] Execution timed out after {timeout_sec}s\n"
                if not is_json_mode:
                    sys.stderr.write(err_str)
                    sys.stderr.flush()
                ret_code = 124
                break

            poll_code = (
                "import base64, json, os\n"
                f"_log = {remote_log!r}\n"
                f"_rc = {remote_rc!r}\n"
                f"_off = {offset}\n"
                "_chunk = b''\n"
                "if os.path.exists(_log):\n"
                "    with open(_log, 'rb') as _f:\n"
                "        _f.seek(_off)\n"
                "        _chunk = _f.read(262144)\n"
                "        _off = _f.tell()\n"
                "_done = False\n"
                "_code = None\n"
                "if os.path.exists(_rc):\n"
                "    try:\n"
                "        _txt = open(_rc, 'r').read().strip()\n"
                "        if _txt != '':\n"
                "            _code = int(_txt)\n"
                "            _done = True\n"
                "    except Exception:\n"
                "        pass\n"
                "print(json.dumps({'offset': _off, 'b64': base64.b64encode(_chunk).decode('ascii'), 'done': _done, 'rc': _code}))\n"
            )

            try:
                p_rc, p_out, p_err = _short_ws_exec(poll_code, call_timeout=25.0)
                if p_rc != 0:
                    raise RuntimeError(p_err or p_out or "poll returned non-zero")
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors >= 5 or not transient(exc.__cause__ or exc):
                    err_str = f"[colab-ultra] Unable to read {remote_log}. Check colab-ultra status and job output: {exc}\n"
                    if not is_json_mode:
                        sys.stderr.write(err_str)
                        sys.stderr.flush()
                    ret_code = 1
                    break
                time.sleep(2 ** (consecutive_errors - 1))
                continue

            rec = None
            for line in reversed(p_out.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        rec = json.loads(line)
                        break
                    except Exception:
                        pass

            if rec is None:
                time.sleep(2.0)
                continue

            new_offset = int(rec.get("offset", offset))
            raw_b64 = rec.get("b64") or ""
            if raw_b64:
                text = base64.b64decode(raw_b64).decode("utf-8", errors="replace")
                out_chunks.append(text)
                if is_json_mode:
                    sys.stderr.write(text)
                    sys.stderr.flush()
                else:
                    sys.stdout.write(text)
                    sys.stdout.flush()
            offset = new_offset

            # If we hit the 256KB per-poll cap, drain immediately without sleeping
            if len(raw_b64) >= 340000:
                continue

            if rec.get("done"):
                ret_code = int(rec.get("rc", 0))
                break

            elapsed_so_far = time.time() - t0
            if elapsed_so_far < 15.0:
                time.sleep(1.0)
            elif elapsed_so_far < 60.0:
                time.sleep(2.5)
            else:
                time.sleep(4.0)
    except KeyboardInterrupt:
        if remote_pgid:
            try:
                _short_ws_exec(f"import os, signal\ntry: os.killpg({remote_pgid}, signal.SIGTERM)\nexcept Exception: pass\n", call_timeout=10.0)
            except Exception:
                pass
        ret_code = 130
    finally:
        try:
            _short_ws_exec(
                f"import os\nfor _p in ({remote_py!r}, {remote_rc!r}, {remote_pid_file!r}):\n"
                "    if os.path.exists(_p):\n"
                "        try: os.remove(_p)\n"
                "        except OSError: pass\n",
                call_timeout=10.0,
            )
        except Exception:
            pass

    out_str = "".join(out_chunks)
    return ret_code, out_str, err_str
