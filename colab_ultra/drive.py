import json
import os
import sys

from colab_cli.runtime import ColabRuntime
from colab_ultra.backends.colab import colab_state
from colab_ultra.backends.jupyter import jupyter_exec_ws
from colab_ultra.compat.colab_cli_patch import patch_kernel_client
from colab_ultra.retry import retry


def detect_local_drive_hub(*, create=True) -> str | None:
    """Detect macOS Google Drive for Desktop mount (FileProvider) and maintain ~/ColabUltra symlink."""
    import glob as _glob

    shortcut = os.path.expanduser("~/ColabUltra")
    if os.path.isdir(shortcut):
        return os.path.realpath(shortcut)
    cloud_root = os.path.expanduser("~/Library/CloudStorage")
    candidates = []
    if os.path.isdir(cloud_root):
        for gd in sorted(_glob.glob(os.path.join(cloud_root, "GoogleDrive-*"))):
            for sub in ("My Drive", "我的云端硬盘", "MyDrive"):
                p = os.path.join(gd, sub)
                if os.path.isdir(p):
                    candidates.append(p)
    if not candidates:
        return None
    hub = os.path.join(candidates[0], "ColabUltra")
    if not create:
        return hub if os.path.isdir(hub) else None
    try:
        os.makedirs(hub, exist_ok=True)
        shortcut = os.path.expanduser("~/ColabUltra")
        if os.path.islink(shortcut) or not os.path.exists(shortcut):
            if os.path.islink(shortcut) and os.readlink(shortcut) != hub:
                os.unlink(shortcut)
            if not os.path.exists(shortcut):
                os.symlink(hub, shortcut)
    except Exception:
        pass
    return hub


def scripts_target(hub=None):
    """Locate the scripts directory inside the user's Drive hub."""
    from pathlib import Path

    root = Path(hub) if hub else Path.home() / "ColabUltra"
    if not root.is_dir():
        raise FileNotFoundError(f"Drive hub unavailable: {root}. Run colab-ultra init.")
    return root / "scripts"


def _propagate(client, endpoint, auth_type):
    """Perform Colab's two-step XSRF-protected credential propagation."""
    url = f"{client.colab_domain}/tun/m/credentials-propagation/{endpoint}"
    params = dict(authuser="0", authtype=auth_type, version="2",
                  dryrun="true", propagate="true", record="false")
    headers = {"X-Colab-Client-Agent": "colab-cli", "Accept": "application/json"}

    def request(method):
        response = client.session.request(method, url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        return json.loads(response.text.removeprefix(")]}'").strip())

    headers["x-goog-colab-token"] = retry(lambda: request("GET"))["token"]
    if auth_type == "dfs_ephemeral":
        dry = request("POST")
        if not dry.get("success") and dry.get("unauthorized_redirect_uri"):
            print("Authorize Google Drive at", dry["unauthorized_redirect_uri"], file=sys.stderr)
            input("Press Enter after granting Drive access: ")
    params["dryrun"] = "false"
    if not request("POST").get("success"):
        raise RuntimeError(f"{auth_type} credential propagation was declined")


def mount_drive(name, path="/content/drive"):
    """Mount with persistent credentials, falling back to interactive Drive auth."""
    session = colab_state.get_session(name)
    try:
        _propagate(colab_state.client, session.endpoint, "dfs_persistent")
        code = f"from google.colab import drive\ndrive._mount({path!r}, ephemeral=False)"
        rc, _, err = jupyter_exec_ws(session.url, code, proxy_token=session.token, timeout=120)
        if rc:
            raise RuntimeError(err)
        return
    except Exception as exc:
        print(f"Persistent Drive mount unavailable ({type(exc).__name__}), attempting interactive mount.", file=sys.stderr)

    def authorize(message, ws):
        request = message.get("content", {}).get("request", {})
        if request.get("authType") != "dfs_ephemeral":
            return False
        _propagate(colab_state.client, session.endpoint, "dfs_ephemeral")
        value = {"type": "colab_reply", "colab_msg_id": message["metadata"]["colab_msg_id"]}
        reply = ws.session.msg("input_reply", {"value": value})
        reply["parent_header"] = message["header"]
        ws.stdin_channel.send(reply)
        return True

    patch_kernel_client()
    runtime = ColabRuntime(session.url, session.token)
    runtime.colab_request_hook = authorize
    try:
        outputs = runtime.execute_code(f"from google.colab import drive\ndrive.mount({path!r})",
                                       allow_stdin=True, timeout=600)
        for output in outputs:
            if output.get("output_type") == "error":
                raise RuntimeError(output.get("evalue", "Drive mount failed"))
    finally:
        runtime.stop()


def initialize_drive(colab_bin, name, quiet=False):
    """Initialize the hub and disambiguate a hung execution channel from a failed mount."""
    from pathlib import Path

    import requests

    from colab_ultra.backends.colab import resolve_proxy
    from colab_ultra.backends.jupyter import jupyter_headers
    from colab_ultra.config import BASE_DIR
    from colab_ultra.retry import transient

    mount_drive(name)
    script = os.path.join(BASE_DIR, "workflows", "init_20tb_drive.py")
    code = Path(script).read_text(encoding="utf-8")
    url, token = resolve_proxy(name)
    caught = None
    rc = 0
    err = ""
    try:
        rc, _, err = jupyter_exec_ws(url, code, proxy_token=token, timeout=60, stream_stdout=not quiet)
    except Exception as exc:
        caught = exc
        if not (isinstance(exc, TimeoutError) or transient(exc)
                or (exc.__cause__ is not None and transient(exc.__cause__))
                or "after submission" in str(exc)):
            raise
    if caught is None and rc == 0:
        return
    try:
        params, headers = jupyter_headers(url, token)
        result = requests.get(f"{url.rstrip('/')}/api/contents/content/drive/MyDrive",
                              params={**params, "content": "1"}, headers=headers, timeout=10)
        result.raise_for_status()
        listing = result.json()
    except Exception as verify_exc:
        detail = caught if caught is not None else f"exit code {rc}: {err}".strip()
        raise RuntimeError(f"Drive initialization failed for '{name}' ({detail})") from verify_exc
    if listing.get("type") == "directory" and isinstance(listing.get("content"), list):
        return
    raise RuntimeError(f"Drive initialization failed for '{name}': MyDrive is not listable")
