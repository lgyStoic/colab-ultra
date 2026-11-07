#!/usr/bin/env python3
"""
100% Standalone Colab Ultra Telemetry & Smart Breaker Daemon for Linux VPS.
Zero dependency on local laptop/desktop! Even if the local machine is powered off or offline:
  1. Queries /tun/m/ccu-info & /tun/m/assignments directly via Google Colaboratory API.
  2. If any active Colab VM lacks a running hardware probe (.colab_telemetry.json),
     the VPS automatically uploads /usr/local/share/colab_probe.py and launches it
     inside the Colab VM via Jupyter Kernel WebSocket.
  3. Reads live nvidia-smi GPU utilization %, VRAM, power draw (W), temp (°C), CPU, and 20TB Drive stats.
  4. Tracks low-load / deadlock duration and automatically backs up /content/outputs to
     20TB Google Drive and trips the breaker (unassigns the VM) after the configured timeout.
"""

import json
import os
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone

SENTINEL_CFG_PATH = "/etc/colab-sentinel.json"
_sentinel_cfg = {}
if os.path.exists(SENTINEL_CFG_PATH):
    try:
        with open(SENTINEL_CFG_PATH, "r", encoding="utf-8") as _f:
            _sentinel_cfg = json.load(_f)
    except Exception:
        pass

ADC_PATH = os.environ.get("COLAB_ADC_PATH", _sentinel_cfg.get("adc_path", "/etc/colab-adc.json"))
LOCAL_ADC_FALLBACK = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
PROBE_SCRIPT_PATH = _sentinel_cfg.get("probe_script_path", "/usr/local/share/colab_probe.py")
TOKEN_CACHE_PATH = "/var/tmp/colab_oauth_token.json"
IDLE_STATE_PATH = "/var/tmp/colab_breaker_state.json"
PUSHED_STATS_PATH = "/tmp/colab_stats.json"
OUT_PATH = os.environ.get("COLAB_STATS_OUT", _sentinel_cfg.get("out_path", "/var/www/html/colab_stats.json"))

COLAB_DOMAIN = "https://colab.research.google.com"
XSSI_PREFIX = ")]}'\n"

# Smart Breaker thresholds (default: 5 min warning, 10 min auto-trip)
GPU_MIN_UTIL_PCT = float(_sentinel_cfg.get("gpu_min_util_pct", 5.0))
CPU_MIN_UTIL_PCT = float(_sentinel_cfg.get("cpu_min_util_pct", 55.0))
ALERT_AFTER_SEC = int(float(_sentinel_cfg.get("alert_after_minutes", 5.0)) * 60)
STOP_AFTER_SEC = int(float(_sentinel_cfg.get("stop_after_minutes", 10.0)) * 60)

SHAPE_MAP = {0: "Standard", 1: "High-RAM", "0": "Standard", "1": "High-RAM", "HIGH_RAM": "High-RAM"}


def get_access_token() -> str:
    """Get or refresh Google OAuth2 access token using standard library urllib."""
    now = time.time()
    if os.path.exists(TOKEN_CACHE_PATH):
        try:
            with open(TOKEN_CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("expires_at", 0) > now + 60 and cached.get("access_token"):
                return cached["access_token"]
        except Exception:
            pass

    adc_file = ADC_PATH if os.path.exists(ADC_PATH) else LOCAL_ADC_FALLBACK
    with open(adc_file, "r", encoding="utf-8") as f:
        creds = json.load(f)

    data = urllib.parse.urlencode({
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token",
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        tok = json.loads(resp.read().decode("utf-8"))

    access_token = tok["access_token"]
    expires_in = int(tok.get("expires_in", 3600))
    try:
        with open(TOKEN_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"access_token": access_token, "expires_at": now + expires_in}, f)
        os.chmod(TOKEN_CACHE_PATH, 0o600)
    except Exception:
        pass
    return access_token


def colab_get(path: str, token: str, timeout: float = 8.0) -> dict:
    url = f"{COLAB_DOMAIN}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "X-Colab-Client-Agent": "colab-cli",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    if raw.startswith(XSSI_PREFIX):
        raw = raw[len(XSSI_PREFIX):]
    return json.loads(raw) if raw.strip() else {}


def proxy_request_json(
    proxy_url: str,
    proxy_token: str,
    subpath: str,
    method: str = "GET",
    payload: dict = None,
    timeout: float = 5.0,
):
    url = proxy_url.rstrip("/") + "/" + subpath.lstrip("/")
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Authorization": f"Bearer {proxy_token}",
        "X-Colab-Runtime-Proxy-Token": proxy_token,
        "Accept": "application/json",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw.strip() else {}


def run_code_on_vm_from_vps(proxy_url: str, proxy_token: str, code: str) -> bool:
    """Execute Python code on the Colab VM directly from the VPS via Jupyter Kernel WebSocket."""
    try:
        import websocket
        kernels = proxy_request_json(proxy_url, proxy_token, "/api/kernels", method="GET")
        if not isinstance(kernels, list) or len(kernels) == 0:
            k_resp = proxy_request_json(proxy_url, proxy_token, "/api/kernels", method="POST", payload={"name": "python3"})
            kernel_id = k_resp.get("id")
        else:
            kernel_id = kernels[0].get("id")

        if not kernel_id:
            return False

        ws_base = proxy_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
        session_id = uuid.uuid4().hex
        ws_url = f"{ws_base}/api/kernels/{kernel_id}/channels?session_id={session_id}"
        headers = [
            f"Authorization: Bearer {proxy_token}",
            f"X-Colab-Runtime-Proxy-Token: {proxy_token}",
        ]
        msg_id = uuid.uuid4().hex
        msg = {
            "header": {
                "msg_id": msg_id,
                "username": "colab-ultra-vps",
                "session": session_id,
                "msg_type": "execute_request",
                "version": "5.3",
                "date": datetime.now(timezone.utc).isoformat(),
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": True,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": False,
            },
            "buffers": [],
            "channel": "shell",
        }
        ws = websocket.create_connection(ws_url, header=headers, timeout=6)
        try:
            ws.send(json.dumps(msg))
            time.sleep(0.6)
        finally:
            ws.close()
        return True
    except Exception:
        return False


def ensure_probe_deployed_from_vps(proxy_url: str, proxy_token: str) -> dict:
    """Upload /usr/local/share/colab_probe.py to Colab VM and start it if telemetry is missing/stale."""
    if not os.path.exists(PROBE_SCRIPT_PATH):
        return {}
    try:
        with open(PROBE_SCRIPT_PATH, "r", encoding="utf-8") as f:
            probe_code = f.read()
        proxy_request_json(
            proxy_url,
            proxy_token,
            "/api/contents/content/.colab_probe.py",
            method="PUT",
            payload={"type": "file", "format": "text", "content": probe_code},
            timeout=6.0,
        )
        launch_snippet = (
            "import subprocess, os\n"
            "subprocess.Popen(['python3', '/content/.colab_probe.py'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
        )
        run_code_on_vm_from_vps(proxy_url, proxy_token, launch_snippet)
        time.sleep(1.5)
        item = proxy_request_json(proxy_url, proxy_token, "/api/contents/content/.colab_telemetry.json?content=1")
        content = item.get("content")
        if isinstance(content, str):
            return json.loads(content)
        elif isinstance(content, dict):
            return content
    except Exception:
        pass
    return {}


def backup_and_unassign(endpoint: str, proxy_url: str, proxy_token: str, token: str) -> bool:
    """Back up /content/outputs and loose /content files to 20TB Google Drive (if mounted) and shut down VM."""
    if proxy_url and proxy_token:
        backup_snippet = (
            "import os, shutil, time, glob\n"
            "if os.path.exists('/content/drive/MyDrive'):\n"
            "    base = '/content/drive/MyDrive/ColabUltra/emergency_backups'\n"
            "    if os.path.exists('/content/outputs') and not os.path.islink('/content/outputs'):\n"
            "        dst = f'{base}/backup_{int(time.time())}'\n"
            "        os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
            "        shutil.copytree('/content/outputs', dst, dirs_exist_ok=True)\n"
            "    rescue = f'{base}/latest_root_files'\n"
            "    os.makedirs(rescue, exist_ok=True)\n"
            "    for ext in ('*.py', '*.ipynb', '*.json', '*.yaml', '*.sh', '*.log', '*.txt', '*.safetensors', '*.pt'):\n"
            "        for p in glob.glob(f'/content/{ext}'):\n"
            "            if os.path.isfile(p) and not os.path.basename(p).startswith('.colab_'):\n"
            "                shutil.copy2(p, os.path.join(rescue, os.path.basename(p)))\n"
        )
        run_code_on_vm_from_vps(proxy_url, proxy_token, backup_snippet)
        time.sleep(2.0)

    try:
        path = f"/tun/m/unassign/{endpoint}?authuser=0"
        info = colab_get(path, token)
        xsrf = info.get("token", "")
        url = f"{COLAB_DOMAIN}{path}"
        req = urllib.request.Request(
            url,
            data=b"",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "X-Colab-Client-Agent": "colab-cli",
                "X-Goog-Colab-Token": xsrf,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception:
        return False


def main():
    now = time.time()
    token = get_access_token()
    ccu = colab_get("/tun/m/ccu-info?authuser=0", token)
    assign_resp = colab_get("/tun/m/assignments?authuser=0", token)
    assignments = assign_resp.get("assignments") or []

    pushed_by_ep = {}
    if os.path.exists(PUSHED_STATS_PATH):
        try:
            with open(PUSHED_STATS_PATH, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            if now - p_data.get("unix_time", 0) < 90:
                for v in p_data.get("vms", []):
                    if v.get("endpoint"):
                        pushed_by_ep[v["endpoint"]] = v
        except Exception:
            pass

    idle_state = {}
    if os.path.exists(IDLE_STATE_PATH):
        try:
            with open(IDLE_STATE_PATH, "r", encoding="utf-8") as f:
                idle_state = json.load(f)
        except Exception:
            idle_state = {}

    new_idle_state = {}
    vm_list = []

    for a in assignments:
        ep = a.get("endpoint", "")
        acc = a.get("accelerator", "NONE")
        shape_raw = a.get("machineShape", 0)
        shape = SHAPE_MAP.get(shape_raw, "Standard")
        proxy_info = a.get("runtimeProxyInfo") or {}
        proxy_url = proxy_info.get("url", "")
        proxy_token = proxy_info.get("token", "")

        kernel_state = "idle"
        telemetry = {}
        if proxy_url and proxy_token:
            try:
                kernels = proxy_request_json(proxy_url, proxy_token, "/api/kernels")
                if isinstance(kernels, list):
                    for k in kernels:
                        if k.get("execution_state") == "busy":
                            kernel_state = "busy"
            except Exception:
                pass

            try:
                item = proxy_request_json(proxy_url, proxy_token, "/api/contents/content/.colab_telemetry.json?content=1")
                content = item.get("content")
                if isinstance(content, str):
                    telemetry = json.loads(content)
                elif isinstance(content, dict):
                    telemetry = content
            except Exception:
                telemetry = {}

            if not telemetry or (now - telemetry.get("unix_time", 0)) > 30:
                telemetry = ensure_probe_deployed_from_vps(proxy_url, proxy_token)

        pushed_vm = pushed_by_ep.get(ep, {})
        gpu = telemetry.get("gpu") or {}
        sys_info = telemetry.get("system") or {}
        drive = telemetry.get("drive") or {}
        hb = telemetry.get("task_heartbeat") or pushed_vm.get("task_heartbeat")

        gpu_avail = bool(gpu.get("available")) if gpu else bool(pushed_vm.get("gpu_available"))
        gpu_name = gpu.get("name") or pushed_vm.get("gpu_name") or (acc if acc != "NONE" else "CPU")
        gpu_util = round(float(gpu.get("gpu_util_pct", pushed_vm.get("gpu_util_pct", 0.0))), 1) if gpu_avail else 0.0
        vram_used = round(float(gpu.get("vram_used_mib", pushed_vm.get("vram_used_mib", 0.0))), 0) if gpu_avail else 0.0
        vram_total = round(float(gpu.get("vram_total_mib", pushed_vm.get("vram_total_mib", 0.0))), 0) if gpu_avail else 0.0
        power_w = round(float(gpu.get("power_draw_w", pushed_vm.get("power_draw_w", 0.0))), 1) if gpu_avail else 0.0
        power_lim = round(float(gpu.get("power_limit_w", pushed_vm.get("power_limit_w", 0.0))), 1) if gpu_avail else 0.0
        temp_c = round(float(gpu.get("temp_c", pushed_vm.get("temp_c", 0.0))), 0) if gpu_avail else 0.0
        cpu_util = round(float(sys_info.get("cpu_util_est_pct", pushed_vm.get("cpu_util_pct", 0.0))), 1)
        ram_used = round(float(sys_info.get("ram_used_gb", pushed_vm.get("ram_used_gb", 0.0))), 1)
        ram_total = round(float(sys_info.get("ram_total_gb", pushed_vm.get("ram_total_gb", 0.0))), 1)
        drive_mounted = bool(drive.get("mounted", pushed_vm.get("drive_mounted", False)))
        drive_free_tb = round(float(drive.get("free_tb", pushed_vm.get("drive_free_tb") or 0.0)), 2) if drive_mounted else None

        has_probe = bool(gpu or sys_info or pushed_vm)
        min_active_watts = 75.0 if power_lim >= 250.0 else 28.0
        power_is_active = (power_w >= min_active_watts) if power_w > 0 else False
        power_is_dead_idle = (power_w > 0 and power_w < min_active_watts)

        if not has_probe:
            is_low = (kernel_state == "idle")
            reason = f"内核 {kernel_state.upper()} (正在挂载功耗探针)" if not is_low else "内核空闲 (IDLE)"
        elif power_is_active and gpu_util >= GPU_MIN_UTIL_PCT:
            is_low = False
            reason = f"显卡高功耗时真负载 ({power_w:.0f}W, GPU {gpu_util:.1f}%)"
        elif power_is_active:
            is_low = False
            reason = f"显卡功耗活跃 ({power_w:.0f}W >= {min_active_watts:.0f}W)"
        elif not power_is_dead_idle and gpu_util >= GPU_MIN_UTIL_PCT:
            is_low = False
            reason = f"GPU 活跃计算中 ({gpu_util:.1f}%)"
        elif kernel_state == "busy" and cpu_util >= CPU_MIN_UTIL_PCT:
            is_low = False
            reason = f"CPU/云盘密集处理中 (CPU {cpu_util:.1f}%, GPU {power_w:.0f}W)"
        elif hb and hb.get("age_seconds", 999) < 45 and hb.get("active", True):
            is_low = False
            reason = f"任务心跳活跃 ({hb.get('task', 'job')})"
        else:
            is_low = True
            vram_gb = vram_used / 1024.0
            if power_is_dead_idle and gpu_util >= 20.0:
                reason = f"疑似通信死锁假忙 (GPU显示{gpu_util:.0f}% 但真实功耗仅 {power_w:.0f}W < {min_active_watts:.0f}W)"
            elif vram_gb >= 3.0 and kernel_state == "busy":
                reason = f"疑似进程卡死/死锁 (功耗仅 {power_w:.0f}W, 显存 {vram_gb:.1f}GB, 内核 BUSY)"
            elif vram_gb >= 3.0:
                reason = f"模型驻留但空载待机 (功耗仅 {power_w:.0f}W, 显存 {vram_gb:.1f}GB)"
            else:
                reason = f"待机底噪空转 (功耗 {power_w:.0f}W < {min_active_watts:.0f}W, GPU {gpu_util:.1f}%)"

        prev = idle_state.get(ep, {"low_load_sec": 0, "last_ts": now})
        elapsed = max(1, min(120, int(now - prev.get("last_ts", now))))
        low_sec = (prev.get("low_load_sec", 0) + elapsed) if is_low else 0

        if low_sec >= STOP_AFTER_SEC:
            unassigned = backup_and_unassign(ep, proxy_url, proxy_token, token)
            if unassigned:
                continue

        new_idle_state[ep] = {"low_load_sec": low_sec, "last_ts": now}

        vm_list.append({
            "name": pushed_vm.get("name", "colab-vm"),
            "endpoint": ep,
            "accelerator": acc,
            "shape": shape,
            "kernel_state": kernel_state,
            "gpu_available": gpu_avail,
            "gpu_name": gpu_name,
            "gpu_util_pct": gpu_util,
            "vram_used_mib": vram_used,
            "vram_total_mib": vram_total,
            "power_draw_w": power_w,
            "power_limit_w": power_lim,
            "temp_c": temp_c,
            "cpu_util_pct": cpu_util,
            "ram_used_gb": ram_used,
            "ram_total_gb": ram_total,
            "drive_mounted": drive_mounted,
            "drive_free_tb": drive_free_tb,
            "is_low_load": is_low,
            "status_reason": reason,
            "low_load_sec": low_sec,
            "stop_after_sec": STOP_AFTER_SEC,
            "task_heartbeat": hb,
        })

    try:
        with open(IDLE_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(new_idle_state, f)
    except Exception:
        pass

    balance = float(
        ccu.get("currentBalance")
        or ccu.get("paidBalance")
        or ccu.get("paidComputeUnitsBalance")
        or 0.0
    )
    burn = float(ccu.get("consumptionRateHourly", 0.0))
    rem_hours = round(balance / burn, 1) if burn > 0 else None

    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "unix_time": int(now),
        "source": "vps-autonomous-breaker",
        "paid_balance": round(balance, 2),
        "burn_rate_hourly": round(burn, 2),
        "estimated_hours_left": rem_hours,
        "eligible_gpus": ccu.get("eligibleGpus") or ["H100", "G4", "A100", "L4", "T4"],
        "eligible_tpus": ccu.get("eligibleTpus") or ["V6E1", "V5E1"],
        "active_vms_count": len(vm_list),
        "vms": vm_list,
        "breaker_config": {
            "gpu_min_util_pct": GPU_MIN_UTIL_PCT,
            "alert_after_minutes": ALERT_AFTER_SEC // 60,
            "stop_after_minutes": STOP_AFTER_SEC // 60,
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tmp_path = OUT_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, OUT_PATH)


if __name__ == "__main__":
    main()
