"""Shared Jupyter contents, kernel and execution transport."""
import base64
import json
import os
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests
import websocket

from colab_ultra.retry import retry

KAGGLE_PROXY_HOST = "jupyter-proxy.kaggle.net"

def is_kaggle_proxy(proxy_url: str) -> bool:
    return bool(proxy_url and KAGGLE_PROXY_HOST in proxy_url)


def jupyter_headers(proxy_url: str, proxy_token: str = "") -> tuple[dict, dict]:
    if is_kaggle_proxy(proxy_url):
        return {}, {}
    params = {"authuser": "0", "colab-runtime-proxy-token": proxy_token}
    headers = {
        "Authorization": f"Bearer {proxy_token}",
        "X-Colab-Runtime-Proxy-Token": proxy_token,
    }
    return params, headers


_PROXY_ROOT_IS_SLASH: dict[str, bool] = {}


def _jupyter_root_is_slash(proxy_url: str) -> bool:
    """Detect once per Kaggle proxy URL whether `/api/contents/` is rooted at `/` or `/kaggle/working`."""
    base = proxy_url.rstrip("/")
    if base in _PROXY_ROOT_IS_SLASH:
        return _PROXY_ROOT_IS_SLASH[base]
    try:
        r = requests.get(f"{base}/api/contents/kaggle", params={"content": "0"}, timeout=6.0)
        is_slash = (r.status_code == 200)
    except Exception:
        is_slash = False
    _PROXY_ROOT_IS_SLASH[base] = is_slash
    return is_slash


def map_contents_path(proxy_url: str, remote_path: str) -> str:
    """
    Map a `/content/...` VM path to Jupyter `/api/contents/<rel>` path.
    - If Jupyter root is `/` (Colab or Kaggle `/`-rooted image) -> `content/...`
    - If Jupyter root is `/kaggle/working` (symlinked as `/content`) -> strip leading `content/`
    """
    clean = remote_path.lstrip("/")
    if is_kaggle_proxy(proxy_url) and not _jupyter_root_is_slash(proxy_url):
        if clean == "content":
            return ""
        if clean.startswith("content/"):
            return clean[len("content/"):]
    return clean


def jupyter_get_kernels(proxy_url: str, proxy_token: str = "", timeout: float = 10.0) -> list | None:
    url = f"{proxy_url.rstrip('/')}/api/kernels"
    params, headers = jupyter_headers(proxy_url, proxy_token)
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 403, 404, 502, 503, 504):
            return None
    except Exception:
        return None
    return None


def jupyter_ensure_kernel_id(proxy_url: str, proxy_token: str = "") -> str:
    """Select a kernel or create one after an authoritative empty listing."""
    params, headers = jupyter_headers(proxy_url, proxy_token)
    url = f"{proxy_url.rstrip('/')}/api/kernels"
    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    kernels = response.json()
    if kernels:
        return kernels[0]["id"]
    response = requests.post(url, params=params, headers=headers, json={"name": "python3"}, timeout=12)
    response.raise_for_status()
    return response.json()["id"]


def jupyter_put_bytes(proxy_url: str, proxy_token: str, remote_path: str, data: bytes, timeout: float = 60.0):
    rel = map_contents_path(proxy_url, remote_path)
    url = f"{proxy_url.rstrip('/')}/api/contents/{rel}"
    params, headers = jupyter_headers(proxy_url, proxy_token)
    payload = {
        "name": os.path.basename(rel),
        "path": rel,
        "type": "file",
        "format": "base64",
        "content": base64.b64encode(data).decode("ascii"),
    }
    resp = requests.put(url, params=params, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()


def jupyter_get_bytes(proxy_url: str, proxy_token: str, remote_path: str, timeout: float = 60.0) -> bytes | None:
    rel = map_contents_path(proxy_url, remote_path)
    url = f"{proxy_url.rstrip('/')}/api/contents/{rel}"
    params, headers = jupyter_headers(proxy_url, proxy_token)
    try:
        r = requests.get(
            url,
            params={**params, "content": "1", "type": "file", "format": "base64"},
            headers=headers,
            timeout=timeout,
        )
        if r.status_code != 200:
            r = requests.get(url, params={**params, "content": "1"}, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        body = r.json()
        content = body.get("content", "")
        fmt = body.get("format", "")
        if fmt == "base64":
            return base64.b64decode(content)
        if isinstance(content, (dict, list)):
            return json.dumps(content).encode("utf-8")
        return str(content).encode("utf-8")
    except Exception:
        return None


def jupyter_exec_ws(
    proxy_url: str,
    code: str,
    proxy_token: str = "",
    kernel_id: str | None = None,
    timeout: float = 3600.0,
    stream_stdout: bool = False,
    stream_to_stderr: bool = False,
    on_periodic_tick=None,
    tick_interval_sec: float = 20.0,
) -> tuple[int, str, str]:
    """
    Execute Python code over standard Jupyter WebSocket (`/api/kernels/<id>/channels`),
    streaming stdout/stderr, maintaining WebSocket ping keepalives, and returning
    `(exit_code, stdout_str, stderr_str)`.
    """
    sid = uuid.uuid4().hex

    def connect():
        kid = kernel_id or jupyter_ensure_kernel_id(proxy_url, proxy_token)
        ws_base = proxy_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_base}/api/kernels/{kid}/channels?session_id={sid}"
        params, headers = jupyter_headers(proxy_url, proxy_token)
        if params:
            ws_url += "&" + urlencode({"colab-runtime-proxy-token": proxy_token})
            headers["X-Colab-Client-Agent"] = "colab-cli"
        ws = websocket.create_connection(ws_url, header=headers or None, timeout=30)
        try:
            ws.settimeout(5.0)
        except Exception:
            ws.close()
            raise
        return ws

    # A failed send can still reach Jupyter. Only channel preparation can be replayed.
    ws = retry(connect)
    msg_id = uuid.uuid4().hex
    req = {
        "header": {
            "msg_id": msg_id,
            "username": "colab-ultra",
            "session": sid,
            "msg_type": "execute_request",
            "version": "5.3",
            "date": datetime.now(timezone.utc).isoformat(),
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "buffers": [],
        "channel": "shell",
    }

    out_chunks: list[str] = []
    err_chunks: list[str] = []
    exit_code = 0
    reply_received = False
    idle_received = False

    t0 = time.time()
    last_ping = t0
    last_tick = t0

    try:
        try:
            ws.send(json.dumps(req))
        except Exception as exc:
            raise RuntimeError("Task submission uncertain. Check colab-ultra status and job output before resubmitting.") from exc
        while True:
            now = time.time()
            if now - t0 > timeout:
                err_chunks.append(f"\n[colab-ultra] Execution timed out after {timeout}s\n")
                exit_code = 124
                break

            if now - last_ping >= 20.0:
                last_ping = now
                try:
                    ws.ping()
                except Exception:
                    pass

            if on_periodic_tick and (now - last_tick >= tick_interval_sec):
                last_tick = now
                try:
                    on_periodic_tick()
                except Exception:
                    pass

            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                raise RuntimeError("Jupyter connection lost after submission; check colab-ultra status "
                                   "and the job output before running again") from e
            if not raw:
                raise RuntimeError("Jupyter closed after submission; check the job output before running again")
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
                continue

            mtype = msg.get("msg_type") or (msg.get("header") or {}).get("msg_type") or ""
            content = msg.get("content") or {}

            if mtype == "stream":
                sname = content.get("name", "stdout")
                txt = content.get("text", "")
                if sname == "stderr":
                    err_chunks.append(txt)
                    if stream_stdout or stream_to_stderr:
                        import sys
                        sys.stderr.write(txt)
                        sys.stderr.flush()
                else:
                    out_chunks.append(txt)
                    if stream_stdout:
                        import sys
                        sys.stdout.write(txt)
                        sys.stdout.flush()
                    elif stream_to_stderr:
                        import sys
                        sys.stderr.write(txt)
                        sys.stderr.flush()
            elif mtype in ("execute_result", "display_data"):
                txt = (content.get("data") or {}).get("text/plain", "")
                if txt:
                    out_chunks.append(txt + "\n")
                    if stream_stdout:
                        print(txt, flush=True)
            elif mtype == "error":
                exit_code = 1
                tb = "\n".join(content.get("traceback") or [])
                if not tb:
                    tb = f"{content.get('ename', 'Error')}: {content.get('evalue', '')}"
                # Strip ANSI escape sequences only for clean JSON logs if needed, keep readable stderr
                err_chunks.append(tb + "\n")
                if stream_stdout:
                    import sys
                    sys.stderr.write(tb + "\n")
                    sys.stderr.flush()
            elif mtype == "execute_reply":
                reply_received = True
                if content.get("status") == "error":
                    exit_code = 1
                if idle_received:
                    break
            elif mtype == "status":
                if content.get("execution_state") == "idle":
                    idle_received = True
                    if reply_received:
                        break
    finally:
        try:
            ws.close()
        except Exception:
            pass

    return exit_code, "".join(out_chunks), "".join(err_chunks)
