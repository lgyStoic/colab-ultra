#!/usr/bin/env python3
"""
Colab Ultra Smart Breaker & Hardware Load Monitor (智能电闸与负载巡检中枢)

Features:
1. Full-Account Discovery: Detects ALL active Colab VMs (both CLI and browser-opened).
2. Out-of-Band Hardware Telemetry: Samples GPU Util %, VRAM, Power (W), Temp (°C),
   CPU %, RAM, 20TB Google Drive space, and Kernel busy/idle state without blocking training.
3. Deadlock & Low-Load Detection: Distinguishes active compute, CPU/Drive I/O, idle waiting,
   and hung/deadlocked processes (high VRAM occupied + 0% GPU utilization).
4. Two-Stage Protection:
   - Stage 1 (Low-Load Alert): macOS native notification + voice alert + Webhook push.
   - Stage 2 (Auto-Breaker Trip): Emergency backup to 20TB Google Drive + automatic VM release.
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

try:
    import colab_cli  # noqa: F401
except ImportError:
    _uv_py = os.path.expanduser("~/.local/share/uv/tools/google-colab-cli/bin/python")
    if os.path.exists(_uv_py) and os.path.realpath(sys.executable) != os.path.realpath(_uv_py):
        os.execv(_uv_py, [_uv_py, __file__, *sys.argv[1:]])
    raise SystemExit("Missing google-colab-cli. Install with: uv tool install google-colab-cli")

import requests
import websocket
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from colab_cli.auth import AuthProvider
from colab_cli.common import state as colab_state
from colab_cli.client import Accelerator, Variant, resolve_assign_shape
from colab_cli.runtime import ColabRuntime

colab_state.auth_provider = AuthProvider.ADC
console = Console()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if os.path.join(BASE_DIR, "colab_ultra") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "colab_ultra"))

from kaggle_backend import (  # noqa: E402
    DRIVE_ADC_PATH,
    _load_kaggle_sessions,
    build_in_vm_sync_snippet,
    cancel_all_orphan_kaggle_sessions,
    get_kaggle_quota,
    get_short_lived_drive_token,
    is_kaggle_gpu_spec,
    is_kaggle_proxy,
    jupyter_exec_ws,
    jupyter_headers,
    list_active_kaggle_vms,
    load_kaggle_token,
    pull_live_tarball_to_local,
    shut_down_kaggle_session,
    spin_up_kaggle_session,
    sync_project_to_kaggle,
)

PROBE_LOCAL_PATH = os.path.join(BASE_DIR, "colab_ultra", "telemetry_probe.py")
VPS_SENTINEL_LOCAL_PATH = os.path.join(BASE_DIR, "colab_ultra", "vps_sentinel.py")
CONFIG_PATH = os.path.expanduser(
    os.environ.get("COLAB_ULTRA_CONFIG", "~/.config/colab-ultra/config.json")
)
LOG_DIR = os.path.join(os.path.dirname(CONFIG_PATH), "logs")

DEFAULT_CONFIG = {
    "poll_interval_sec": 15,
    "gpu_min_util_pct": 5.0,
    "cpu_min_util_pct": 55.0,
    "cpu_only_min_util_pct": 8.0,
    "alert_after_minutes": 5.0,
    "stop_after_minutes": 10.0,
    "enable_macos_notification": True,
    "enable_voice_alert": True,
    "webhook_url": "",  # Optional Bark / Feishu / Telegram / Custom webhook URL
    "auto_backup_to_drive_before_stop": True,
    "drive_hub_root": "/content/drive/MyDrive/ColabUltra",
    "kaggle_carrier_slug": "garryling/colab-ultra-t4x2-probe",
    "prefer_kaggle_for_t4": True,
    "vps_host": "",
    "vps_ssh_key": "~/.ssh/id_ed25519",
    "vps_remote_stats_path": "/tmp/colab_stats.json",
    "local_stats_out": "",
}


def resolve_colab_bin() -> str:
    return shutil.which("colab") or os.path.expanduser("~/.local/bin/colab")


def load_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    else:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        try:
            os.chmod(CONFIG_PATH, 0o600)
        except Exception:
            pass
    return cfg


def log_breaker_event(event_type: str, details: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "breaker_events.jsonl")
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_type,
        **details,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def send_alert(title: str, message: str, cfg: dict, voice_text: str = None):
    """Send multi-channel alert: Terminal + macOS Notification + Voice + Webhook."""
    console.print(Panel(f"[bold yellow]{message}[/bold yellow]", title=f"🚨 {title}", border_style="yellow"))
    log_breaker_event("ALERT", {"title": title, "message": message})

    # 1. macOS Native Notification
    if cfg.get("enable_macos_notification", True) and sys.platform == "darwin":
        try:
            safe_title = title.replace('"', '\\"')
            safe_msg = message.replace('"', '\\"').replace("\n", " ")
            script = f'display notification "{safe_msg}" with title "{safe_title}" sound name "Glass"'
            subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    # 2. macOS Voice Alert (TTS)
    if cfg.get("enable_voice_alert", True) and voice_text and sys.platform == "darwin":
        try:
            subprocess.Popen(["say", voice_text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    # 3. Webhook (Bark / Feishu / Custom)
    webhook_url = cfg.get("webhook_url", "").strip()
    if webhook_url:
        try:
            requests.post(
                webhook_url,
                json={
                    "title": title,
                    "text": message,
                    "msg_type": "text",
                    "content": {"text": f"{title}\n{message}"},
                },
                timeout=5,
            )
        except Exception:
            pass


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


def start_remote_probe_via_runtime(proxy_url: str, proxy_token: str):
    kernels = fetch_vm_kernels(proxy_url, proxy_token)
    kid = kernels[0].get("id") if kernels else None
    rt = ColabRuntime(proxy_url, proxy_token, kernel_id=kid)
    try:
        boot_code = (
            "import subprocess\n"
            "subprocess.run(['python3', '/content/.colab_probe.py', '--once'], check=False)\n"
            "subprocess.Popen(['python3', '/content/.colab_probe.py'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
        )
        rt.execute_code(boot_code, timeout=12)
    finally:
        rt.stop()


def upload_file_to_vm(proxy_url: str, proxy_token: str, local_path: str, remote_path: str):
    with open(local_path, "rb") as f:
        b64_content = base64.b64encode(f.read()).decode("ascii")
    url = f"{proxy_url.rstrip('/')}/api/contents/{remote_path.lstrip('/')}"
    params = {"authuser": "0", "colab-runtime-proxy-token": proxy_token}
    headers = {
        "Authorization": f"Bearer {proxy_token}",
        "X-Colab-Runtime-Proxy-Token": proxy_token,
    }
    payload = {
        "name": os.path.basename(remote_path),
        "path": remote_path.lstrip("/"),
        "type": "file",
        "format": "base64",
        "content": b64_content,
    }
    resp = requests.put(url, params=params, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()


def fetch_vm_kernels(proxy_url: str, proxy_token: str) -> list:
    url = f"{proxy_url.rstrip('/')}/api/kernels"
    params, headers = jupyter_headers(proxy_url, proxy_token)
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


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


def fetch_vm_telemetry(proxy_url: str, proxy_token: str, auto_deploy: bool = True) -> dict:
    """Fetch /content/.colab_telemetry.json from the VM; deploy probe if missing or stale."""
    url = f"{proxy_url.rstrip('/')}/api/contents/content/.colab_telemetry.json"
    params = {"content": "1", "authuser": "0", "colab-runtime-proxy-token": proxy_token}
    headers = {
        "Authorization": f"Bearer {proxy_token}",
        "X-Colab-Runtime-Proxy-Token": proxy_token,
    }

    def _read():
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 200:
                body = r.json()
                content = body.get("content", "")
                if body.get("format") == "base64":
                    content = base64.b64decode(content).decode("utf-8")
                data = json.loads(content) if isinstance(content, str) else content
                return data
        except Exception:
            pass
        return None

    data = _read()
    now = time.time()
    if data and (now - data.get("unix_time", 0)) < 20:
        return data

    if auto_deploy and os.path.exists(PROBE_LOCAL_PATH):
        try:
            upload_file_to_vm(proxy_url, proxy_token, PROBE_LOCAL_PATH, "content/.colab_probe.py")
            start_remote_probe_via_runtime(proxy_url, proxy_token)
            data = _read()
        except Exception:
            pass

    return data or {}


def inspect_all_vms():
    """Query account CCU info and all active VMs (both Colab and Kaggle Free T4x2) with full hardware telemetry."""
    ccu = colab_state.client.get_consumption_user_info()
    local_sessions, assignments = colab_state.sync_sessions()
    name_by_endpoint = {s.endpoint: s.name for s in local_sessions.values()}

    vms = []
    for a in assignments:
        proxy_url = a.runtime_proxy_info.url
        proxy_token = a.runtime_proxy_info.token
        name = name_by_endpoint.get(a.endpoint, "web-session")

        kernels = fetch_vm_kernels(proxy_url, proxy_token)
        kernel_state = "idle"
        last_activity = ""
        for k in kernels:
            if k.get("execution_state") == "busy":
                kernel_state = "busy"
            last_activity = k.get("last_activity", last_activity)

        telemetry = fetch_vm_telemetry(proxy_url, proxy_token, auto_deploy=True)
        vms.append({
            "name": name,
            "provider": "colab",
            "endpoint": a.endpoint,
            "accelerator": a.accelerator.value,
            "shape": a.machine_shape.name,
            "proxy_url": proxy_url,
            "proxy_token": proxy_token,
            "kernel_state": kernel_state,
            "last_activity": last_activity,
            "telemetry": telemetry,
        })

    try:
        kaggle_vms = list_active_kaggle_vms(probe_local_path=PROBE_LOCAL_PATH, auto_deploy=True)
        vms.extend(kaggle_vms)
    except Exception:
        pass

    return ccu, vms


def sync_dashboard_json(ccu, vms, idle_tracker: dict = None, cfg: dict = None):
    """Export hardware-agnostic Colab status + GPU telemetry JSON and sync to optional web dashboard."""
    if cfg is None:
        cfg = load_config()
    balance = float(ccu.paid_compute_units_balance or 0.0)
    burn = float(ccu.consumption_rate_hourly or 0.0)
    rem_hours = round(balance / burn, 1) if burn > 0 else None

    vm_list = []
    for vm in vms:
        t = vm.get("telemetry", {})
        gpu = t.get("gpu", {})
        sys_info = t.get("system", {})
        drive = t.get("drive", {})
        hb = t.get("task_heartbeat")
        ep = vm["endpoint"]
        low_sec = (idle_tracker or {}).get(ep, {}).get("low_load_sec", 0)
        is_low, reason = evaluate_vm_load(vm, cfg)
        stop_sec = int(cfg.get("stop_after_minutes", 25) * 60)

        vm_list.append({
            "name": vm["name"],
            "provider": vm.get("provider", "colab"),
            "endpoint": ep,
            "accelerator": vm["accelerator"],
            "shape": vm["shape"],
            "kernel_state": vm["kernel_state"],
            "gpu_available": bool(gpu.get("available")),
            "gpu_count": int(gpu.get("gpu_count", 1 if gpu.get("available") else 0) or 0),
            "gpu_name": gpu.get("name") or vm["accelerator"],
            "gpu_util_pct": round(float(gpu.get("gpu_util_pct", 0.0)), 1) if gpu.get("available") else 0.0,
            "vram_used_mib": round(float(gpu.get("vram_used_mib", 0.0)), 0) if gpu.get("available") else 0.0,
            "vram_total_mib": round(float(gpu.get("vram_total_mib", 0.0)), 0) if gpu.get("available") else 0.0,
            "power_draw_w": round(float(gpu.get("power_draw_w", 0.0)), 1) if gpu.get("available") else 0.0,
            "power_limit_w": round(float(gpu.get("power_limit_w", 0.0)), 1) if gpu.get("available") else 0.0,
            "temp_c": round(float(gpu.get("temp_c", 0.0)), 0) if gpu.get("available") else 0.0,
            "cpu_cores": int(sys_info.get("cpu_cores", 0) or 0),
            "cpu_util_pct": round(float(sys_info.get("cpu_util_est_pct", 0.0)), 1),
            "io_mb_s": round(float(sys_info.get("io_mb_s", 0.0) or 0.0), 2),
            "ram_used_gb": round(float(sys_info.get("ram_used_gb", 0.0)), 1),
            "ram_total_gb": round(float(sys_info.get("ram_total_gb", 0.0)), 1),
            "drive_mounted": bool(drive.get("mounted")),
            "drive_free_tb": round(float(drive.get("free_tb", 0.0)), 2) if drive.get("mounted") else None,
            "is_low_load": is_low,
            "status_reason": reason,
            "low_load_sec": low_sec,
            "stop_after_sec": stop_sec,
            "task_heartbeat": hb,
            "training": t.get("training"),
        })

    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "unix_time": int(time.time()),
        "source": "colab-ultra-probe",
        "paid_balance": round(balance, 2),
        "burn_rate_hourly": round(burn, 2),
        "estimated_hours_left": rem_hours,
        "eligible_gpus": list(getattr(ccu, "eligible_gpus", []) or ["H100", "G4", "A100", "L4", "T4"]),
        "eligible_tpus": list(getattr(ccu, "eligible_tpus", []) or ["V6E1", "V5E1"]),
        "active_vms_count": len(vm_list),
        "vms": vm_list,
        "breaker_config": {
            "gpu_min_util_pct": cfg.get("gpu_min_util_pct", 5.0),
            "alert_after_minutes": cfg.get("alert_after_minutes", 10),
            "stop_after_minutes": cfg.get("stop_after_minutes", 25),
        },
    }

    local_out = os.path.expanduser(cfg.get("local_stats_out", "") or "")
    vps_host = (cfg.get("vps_host", "") or "").strip()
    vps_remote_path = cfg.get("vps_remote_stats_path", "/tmp/colab_stats.json") or "/tmp/colab_stats.json"
    ssh_key = os.path.expanduser(cfg.get("vps_ssh_key", "~/.ssh/id_ed25519") or "~/.ssh/id_ed25519")

    if local_out:
        try:
            os.makedirs(os.path.dirname(local_out), exist_ok=True)
            with open(local_out + ".tmp", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(local_out + ".tmp", local_out)

            if vps_host and os.path.exists(ssh_key):
                subprocess.Popen(
                    [
                        "scp", "-q",
                        "-i", ssh_key,
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "ConnectTimeout=4",
                        local_out,
                        f"{vps_host}:{vps_remote_path}",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass


def render_dashboard(ccu, vms, idle_tracker: dict = None):
    """Render a rich terminal dashboard of CCU balance, Kaggle free quota, and real-time VM hardware loads."""
    sync_dashboard_json(ccu, vms, idle_tracker)
    balance = ccu.paid_compute_units_balance
    burn = ccu.consumption_rate_hourly
    rem_hours = (balance / burn) if burn > 0 else float("inf")
    rem_str = f"{rem_hours:.1f} 小时" if burn > 0 else "∞ (0 消耗休眠中)"

    kq = get_kaggle_quota()
    if kq.get("configured") and "gpu_remaining_hours" in kq:
        kg_str = f"🎁 [bold cyan]Kaggle 免费双卡:[/bold cyan] [bold green]{kq['gpu_remaining_hours']:.1f}h[/bold green] / {kq['gpu_total_hours']:.0f}h   "
    else:
        kg_str = ""

    header_text = (
        f"💳 [bold cyan]剩余算力 (CCU):[/bold cyan] [bold green]{balance:.2f}[/bold green] 点   "
        f"🔥 [bold cyan]当前燃烧速率:[/bold cyan] [bold {'red' if burn > 0 else 'green'}]{burn:.2f} CU/小时[/]   "
        f"{kg_str}"
        f"⏳ [bold cyan]预计可跑:[/bold cyan] {rem_str}   "
        f"🖥️ [bold cyan]在线机器:[/bold cyan] {len(vms)} 台"
    )
    console.print(Panel(header_text, title="⚡ Colab Ultra 智能电闸与算力监控台 (Colab + Kaggle T4x2)", border_style="cyan"))

    if not vms:
        console.print("[dim]😴 当前没有任何运行中的云端实例，算力电闸处于【断电休眠】状态（0 扣费）。[/dim]")
        return

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("会话 / 节点", style="cyan")
    table.add_column("硬件规格", style="bold")
    table.add_column("GPU 利用率", justify="right")
    table.add_column("显存 (VRAM)", justify="right")
    table.add_column("功耗 / 温度", justify="right")
    table.add_column("CPU / 内存", justify="right")
    table.add_column("20TB 云盘", justify="center")
    table.add_column("内核 / 负载状态", justify="left")

    for vm in vms:
        t = vm["telemetry"]
        gpu = t.get("gpu", {})
        sys_info = t.get("system", {})
        drive = t.get("drive", {})
        hb = t.get("task_heartbeat")

        if gpu.get("available"):
            g_util = gpu.get("gpu_util_pct", 0.0)
            v_used = gpu.get("vram_used_mib", 0.0) / 1024.0
            v_tot = gpu.get("vram_total_mib", 1.0) / 1024.0
            pwr = gpu.get("power_draw_w", 0.0)
            temp = gpu.get("temp_c", 0.0)

            util_color = "green" if g_util >= 20 else ("yellow" if g_util >= 5 else "red")
            gpu_util_str = f"[{util_color}]{g_util:5.1f}%[/{util_color}]"
            vram_str = f"{v_used:.1f} / {v_tot:.1f} GB"
            pwr_str = f"{pwr:.0f}W | {temp:.0f}°C"
        else:
            gpu_util_str = "[dim]N/A[/dim]"
            vram_str = "[dim]N/A[/dim]"
            pwr_str = "[dim]N/A[/dim]"

        cpu_cores = sys_info.get("cpu_cores", 0)
        cpu_pct = sys_info.get("cpu_util_est_pct", 0.0)
        ram_u = sys_info.get("ram_used_gb", 0.0)
        ram_t = sys_info.get("ram_total_gb", 0.0)
        core_prefix = f"{cpu_cores}核 " if cpu_cores else ""
        cpu_str = f"{core_prefix}{cpu_pct:.0f}% | {ram_u:.1f}/{ram_t:.0f}G"

        if drive.get("mounted"):
            if drive.get("mode") == "kaggle_synced":
                free_gb = drive.get("free_gb", 0.0)
                drive_str = f"[green]双向同步 ({free_gb:.0f}G空)[/green]"
            else:
                free_tb = drive.get("free_tb", 0.0)
                drive_str = f"[green]已挂载 ({free_tb:.1f}T空)[/green]"
        else:
            drive_str = "[yellow]未挂载[/yellow]"

        k_state = vm["kernel_state"].upper()
        idle_sec = (idle_tracker or {}).get(vm["endpoint"], {}).get("low_load_sec", 0)
        if idle_sec > 0:
            status_str = f"[yellow]⚠️ 低负载 ({idle_sec // 60}m{idle_sec % 60}s)[/yellow] [{k_state}]"
        else:
            status_str = f"[green]🟢 高效运行中[/green] [{k_state}]"

        if hb and hb.get("age_seconds", 999) < 60:
            status_str += f"\n[dim]📌 {hb.get('task', '')}: {hb.get('status', '')}[/dim]"

        acc_label = "CPU" if vm["accelerator"] in ("NONE", "CPU") else vm["accelerator"]
        table.add_row(
            f"{vm['name']}\n[dim]{vm['endpoint'][:18]}..[/dim]",
            f"{acc_label}\n[dim]{vm['shape']}[/dim]",
            gpu_util_str,
            vram_str,
            pwr_str,
            cpu_str,
            drive_str,
            status_str,
        )

    console.print(table)


def evaluate_vm_load(vm: dict, cfg: dict) -> tuple[bool, str]:
    """
    Determine whether the VM is actively doing meaningful work.
    Returns (is_low_load: bool, reason: str).
    """
    t = vm.get("telemetry", {})
    gpu = t.get("gpu", {})
    sys_info = t.get("system", {})
    hb = t.get("task_heartbeat")

    accel_upper = str(vm.get("accelerator", "") or "").upper()
    is_cpu_tier = accel_upper in ("NONE", "CPU") or accel_upper.startswith("TPU")
    gpu_avail = bool(gpu.get("available"))
    gpu_count = int(gpu.get("gpu_count", 1) or 1) if gpu_avail else 0
    gpu_util = gpu.get("gpu_util_pct", 0.0) if gpu_avail else 0.0
    vram_used_gb = (gpu.get("vram_used_mib", 0.0) / 1024.0) if gpu_avail else 0.0
    power_w = gpu.get("power_draw_w", 0.0) if gpu_avail else 0.0
    power_lim = gpu.get("power_limit_w", 0.0) if gpu_avail else 0.0
    max_card_power_w = float(gpu.get("max_card_power_w", power_w) or power_w) if gpu_avail else 0.0
    max_card_lim_w = float(gpu.get("max_card_power_limit_w", power_lim) or power_lim) if gpu_avail else 0.0
    cpu_cores = sys_info.get("cpu_cores", 0) or 2
    cpu_util = sys_info.get("cpu_util_est_pct", 0.0)
    load_1m = sys_info.get("load_1m", 0.0)
    io_mb_s = float(sys_info.get("io_mb_s", 0.0) or 0.0)
    ram_used_gb = sys_info.get("ram_used_gb", 0.0)
    ram_total_gb = sys_info.get("ram_total_gb", 0.0)
    k_state = vm.get("kernel_state", "idle")

    hb_fresh = bool(hb and hb.get("age_seconds", 999) < 45 and hb.get("active", True))
    hb_desc = ""
    hb_elapsed_sec = 0.0
    if hb_fresh:
        t_name = hb.get("task", "job")
        l_pid = hb.get("local_pid")
        s_at = hb.get("started_at")
        hb_elapsed_sec = max(0.0, time.time() - float(s_at)) if s_at else 0.0
        pid_part = f", PID {l_pid}" if l_pid else ""
        dur_part = f", 已运行 {hb_elapsed_sec / 60:.1f}m" if hb_elapsed_sec > 0 else ""
        hb_desc = f"{t_name}{pid_part}{dur_part}"

    # If a GPU VM returned empty telemetry (probe timed out during heavy I/O or boot), report honestly instead of pretending it's a 2-core CPU box
    if not is_cpu_tier and not t:
        suffix = f" | 任务: {hb_desc}" if hb_desc else ""
        return (k_state != "busy"), f"{accel_upper or 'GPU'} 探针采样超时/重连中 (内核 {k_state.upper()}{suffix})"

    # Branch A: CPU-Only or TPU-Host Data Processing Node
    if is_cpu_tier or not gpu_avail:
        if hb_fresh:
            return False, f"CPU 任务执行中 ({hb_desc}, CPU {cpu_util:.1f}%, I/O {io_mb_s:.1f}MB/s)"
        cpu_only_min = float(cfg.get("cpu_only_min_util_pct", 8.0))
        if cpu_util >= cpu_only_min or load_1m >= 0.4 or io_mb_s >= 2.0:
            return False, f"CPU 数据处理活跃 ({cpu_cores}核 CPU {cpu_util:.1f}%, I/O {io_mb_s:.1f}MB/s, 内存 {ram_used_gb:.1f}/{ram_total_gb:.0f}GB)"
        if k_state == "busy":
            return False, f"CPU 内核忙碌执行中 ({cpu_cores}核 CPU {cpu_util:.1f}%, 内存 {ram_used_gb:.1f}/{ram_total_gb:.0f}GB)"
        return True, f"CPU 节点空载待机 ({cpu_cores}核 CPU {cpu_util:.1f}%, 内存 {ram_used_gb:.1f}/{ram_total_gb:.0f}GB, 内核 IDLE)"

    # Branch B: GPU Node — verified by physical power draw (Watts)
    # On multi-GPU (e.g. 2xT4), check per-card active wattage so P0 idle (27W * 2 = 54W) doesn't masquerade as active compute
    min_active_watts = 75.0 if max_card_lim_w >= 250.0 else (30.0 if gpu_count > 1 else 28.0)
    effective_power_w = max_card_power_w if gpu_count > 1 else power_w
    power_is_active = (effective_power_w >= min_active_watts) if effective_power_w > 0 else False
    power_is_dead_idle = (effective_power_w > 0 and effective_power_w < min_active_watts)
    task_suffix = f" [{hb_desc}]" if hb_desc else ""

    # 1. Active GPU compute verified by physical power draw (Watts)
    if power_is_active:
        return False, f"显卡高功耗真负载 ({power_w:.0f}W, GPU {gpu_util:.1f}%){task_suffix}"
    if not power_is_dead_idle and gpu_util >= cfg["gpu_min_util_pct"]:
        return False, f"GPU 活跃计算中 ({gpu_util:.1f}%){task_suffix}"

    # 2. Active CPU / Disk I/O on a GPU node (e.g. weight staging / pip install / preprocessing)
    if k_state == "busy" and (cpu_util >= cfg.get("cpu_min_util_pct", 55.0) or io_mb_s >= 2.0):
        return False, f"CPU/云盘读写活跃 (CPU {cpu_util:.1f}%, I/O {io_mb_s:.1f}MB/s, GPU {power_w:.0f}W){task_suffix}"

    # 2b. Fresh task heartbeat exemption ONLY during initial 10-min setup/staging OR when mild CPU/IO is present.
    # Never let a hung script's background heartbeat disable the GPU wattage breaker forever!
    if hb_fresh and (hb_elapsed_sec < 600.0 or cpu_util >= 10.0 or io_mb_s >= 1.0):
        return False, f"任务准备/加载中 ({hb_desc}, GPU {power_w:.0f}W, CPU {cpu_util:.1f}%, I/O {io_mb_s:.1f}MB/s)"

    # 3. Otherwise, the GPU node IS in a low-load / idle wattage state! Classify why:
    if hb_fresh:
        return True, f"疑似任务挂起/卡死 ({hb_desc}, GPU仅 {power_w:.0f}W, CPU {cpu_util:.1f}%, I/O {io_mb_s:.1f}MB/s)"
    if power_is_dead_idle and gpu_util >= 20.0:
        return True, f"疑似通信死锁假忙 (GPU显示{gpu_util:.0f}% 但单卡最高功耗仅 {effective_power_w:.0f}W < {min_active_watts:.0f}W)"
    elif vram_used_gb >= 3.0 and k_state == "busy":
        return True, f"疑似进程卡死/死锁 (总功耗 {power_w:.0f}W, 显存 {vram_used_gb:.1f}GB, 内核 BUSY)"
    elif vram_used_gb >= 3.0 and k_state == "idle":
        return True, f"模型驻留但空载待机 (总功耗 {power_w:.0f}W, 显存 {vram_used_gb:.1f}GB)"
    else:
        return True, f"待机底噪空转 (总功耗 {power_w:.0f}W, GPU {gpu_util:.1f}%)"


def watch_loop(args):
    """Continuous Smart Breaker daemon loop."""
    cfg = load_config()
    if args.gpu_threshold is not None:
        cfg["gpu_min_util_pct"] = args.gpu_threshold
    if args.alert_minutes is not None:
        cfg["alert_after_minutes"] = args.alert_minutes
    if args.stop_minutes is not None:
        cfg["stop_after_minutes"] = args.stop_minutes
    if args.interval is not None:
        cfg["poll_interval_sec"] = args.interval

    idle_tracker = {}  # endpoint -> {"low_load_sec": int, "alerted": bool, "last_check": float}

    console.print(
        Panel(
            f"⚡ [bold green]Colab Ultra 智能电闸守护已启动[/bold green]\n"
            f"• 轮询间隔: 每 [bold]{cfg['poll_interval_sec']} 秒[/bold] 采样一次 GPU/CPU/显存/功耗\n"
            f"• 低负载判定线: GPU 利用率 < [bold]{cfg['gpu_min_util_pct']}%[/bold] 且 CPU < [bold]{cfg['cpu_min_util_pct']}%[/bold]\n"
            f"• 一级报警阈值: 持续低负载达 [bold yellow]{cfg['alert_after_minutes']} 分钟[/bold yellow] 触发声音+弹窗告警\n"
            f"• 二级拉闸阈值: 持续低负载达 [bold red]{cfg['stop_after_minutes']} 分钟[/bold red] 自动备份并切断电源释放实例",
            title="🛡️ Smart Breaker Configuration",
            border_style="green",
        )
    )

    while True:
        loop_start = time.time()
        try:
            ccu, vms = inspect_all_vms()
            active_endpoints = set()

            for vm in vms:
                ep = vm["endpoint"]
                active_endpoints.add(ep)
                now = time.time()

                if ep not in idle_tracker:
                    idle_tracker[ep] = {"low_load_sec": 0, "alerted": False, "last_ts": now}

                elapsed = max(1, int(now - idle_tracker[ep]["last_ts"]))
                idle_tracker[ep]["last_ts"] = now

                is_low, reason = evaluate_vm_load(vm, cfg)
                if not is_low:
                    if idle_tracker[ep]["low_load_sec"] > 0:
                        console.print(f"[green]✅ [{vm['name']}] 恢复高负载 ({reason})，空闲计时器已清零。[/green]")
                    idle_tracker[ep]["low_load_sec"] = 0
                    idle_tracker[ep]["alerted"] = False
                else:
                    idle_tracker[ep]["low_load_sec"] += elapsed
                    low_sec = idle_tracker[ep]["low_load_sec"]
                    alert_sec = cfg["alert_after_minutes"] * 60
                    stop_sec = cfg["stop_after_minutes"] * 60
                    prov_label = "Kaggle" if vm.get("provider") == "kaggle" else "Colab"

                    # Stage 1: Alert
                    if low_sec >= alert_sec and not idle_tracker[ep]["alerted"]:
                        idle_tracker[ep]["alerted"] = True
                        remain_min = max(0.0, (stop_sec - low_sec) / 60.0)
                        msg = (
                            f"实例 [{vm['name']}] ({prov_label} {vm['accelerator']}) 已连续低负载 {low_sec/60:.1f} 分钟！\n"
                            f"原因: {reason}\n"
                            f"当前消耗: {ccu.consumption_rate_hourly:.2f} CU/h | 剩余额度: {ccu.paid_compute_units_balance:.2f} CU\n"
                            f"若继续无负载，将在 {remain_min:.1f} 分钟后自动拉闸关机！"
                        )
                        send_alert(
                            title=f"{prov_label} {vm['accelerator']} 低负载预警",
                            message=msg,
                            cfg=cfg,
                            voice_text=f"注意，{prov_label} {vm['accelerator']} 机器负载过低，请检查任务状态。",
                        )

                    # Stage 2: Auto-Breaker Trip (Shut down VM)
                    if low_sec >= stop_sec:
                        send_alert(
                            title=f"🔌 {prov_label} {vm['accelerator']} 智能电闸已拉闸",
                            message=(
                                f"实例 [{vm['name']}] ({prov_label} {vm['accelerator']}) 连续低负载达 {low_sec/60:.1f} 分钟 ({reason})。\n"
                                f"正在执行安全备份并强制释放机器，停止算力/免费时长消耗！"
                            ),
                            cfg=cfg,
                            voice_text=f"{prov_label} {vm['accelerator']} 空闲超时，智能电闸已自动拉闸关机。",
                        )
                        if vm.get("provider") == "kaggle":
                            shut_down_kaggle_session(
                                vm,
                                local_hub=detect_local_drive_hub(),
                                flush_outputs=cfg.get("auto_backup_to_drive_before_stop", True),
                            )
                        else:
                            if cfg.get("auto_backup_to_drive_before_stop", True):
                                backup_vm_outputs_to_drive(vm["proxy_url"], vm["proxy_token"])
                            colab_state.client.unassign(ep)
                            if vm["name"] != "web-session":
                                colab_state.store.remove(vm["name"])
                        log_breaker_event("BREAKER_TRIPPED", {"name": vm["name"], "endpoint": ep, "reason": reason})
                        idle_tracker.pop(ep, None)

            # Prune closed endpoints
            for old_ep in list(idle_tracker.keys()):
                if old_ep not in active_endpoints:
                    idle_tracker.pop(old_ep, None)

            render_dashboard(ccu, vms, idle_tracker)

            if args.once:
                break

        except KeyboardInterrupt:
            console.print("\n[yellow]👋 智能电闸守护已手动停止。[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]巡检异常: {e}[/red]")
            if args.once:
                break

        sleep_time = max(1.0, cfg["poll_interval_sec"] - (time.time() - loop_start))
        time.sleep(sleep_time)


def detect_local_drive_hub() -> str | None:
    """Detect macOS Google Drive for Desktop mount (FileProvider) and maintain ~/ColabUltra symlink."""
    import glob as _glob

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


def build_agent_status_payload(ccu, vms, cfg: dict = None) -> dict:
    """Build deterministic, Agent-friendly JSON payload with wattage diagnostics."""
    if cfg is None:
        cfg = load_config()
    balance = float(ccu.paid_compute_units_balance or 0.0)
    burn = float(ccu.consumption_rate_hourly or 0.0)
    rem_hours = round(balance / burn, 1) if burn > 0 else None

    clean_vms = []
    for vm in vms:
        is_low, reason = evaluate_vm_load(vm, cfg)
        t = vm.get("telemetry", {})
        gpu = t.get("gpu", {})
        sys_info = t.get("system", {})
        drive = t.get("drive", {})
        hb = t.get("task_heartbeat")
        active_task = None
        if hb and hb.get("age_seconds", 999) < 45 and hb.get("active", True):
            s_at = hb.get("started_at")
            active_task = {
                "task": hb.get("task"),
                "local_pid": hb.get("local_pid"),
                "elapsed_sec": round(max(0.0, time.time() - float(s_at)), 1) if s_at else None,
            }
        clean_vms.append({
            "name": vm["name"],
            "provider": vm.get("provider", "colab"),
            "endpoint": vm["endpoint"],
            "accelerator": vm["accelerator"],
            "shape": vm["shape"],
            "gpu_count": int(gpu.get("gpu_count", 1) or 1) if gpu.get("available") else 0,
            "kernel_state": vm["kernel_state"],
            "is_low_load": is_low,
            "status_reason": reason,
            "active_task": active_task,
            "power_draw_w": gpu.get("power_draw_w", 0.0),
            "power_limit_w": gpu.get("power_limit_w", 0.0),
            "gpu_util_pct": gpu.get("gpu_util_pct", 0.0),
            "vram_used_mib": gpu.get("vram_used_mib", 0.0),
            "vram_total_mib": gpu.get("vram_total_mib", 0.0),
            "temp_c": gpu.get("temp_c", 0.0),
            "cpu_cores": int(sys_info.get("cpu_cores", 0) or 0),
            "cpu_util_pct": sys_info.get("cpu_util_est_pct", 0.0),
            "ram_used_gb": sys_info.get("ram_used_gb", 0.0),
            "ram_total_gb": sys_info.get("ram_total_gb", 0.0),
            "drive_mounted": bool(drive.get("mounted")),
            "drive_mode": drive.get("mode", "fuse"),
            "drive_free_tb": drive.get("free_tb"),
            "workspace_path": "/content/workspace",
            "outputs_path": "/content/outputs",
        })

    local_hub = detect_local_drive_hub()
    kq = get_kaggle_quota()
    return {
        "ok": True,
        "paid_balance": round(balance, 2),
        "burn_rate_hourly": round(burn, 2),
        "estimated_hours_left": rem_hours,
        "kaggle_free_gpu_hours_left": kq.get("gpu_remaining_hours"),
        "kaggle_free_gpu_total_hours": kq.get("gpu_total_hours"),
        "kaggle_gpu_reserved_seconds": kq.get("gpu_reserved_seconds"),
        "kaggle_quota_refresh_time": kq.get("refresh_time"),
        "local_drive_mounted": bool(local_hub),
        "local_drive_hub": local_hub,
        "remote_drive_hub": "/content/drive/MyDrive/ColabUltra",
        "active_vms_count": len(clean_vms),
        "vms": clean_vms,
    }


def _spin_up_single_session(colab_bin: str, session_name: str, args):
    """Create a single Colab session, mount Drive, and initialize workspace."""
    tpu_choice = (getattr(args, "tpu", None) or "").strip().lower()
    if getattr(args, "cpu", False):
        gpu_choice = "CPU"
    else:
        gpu_choice = (args.gpu or "T4").upper()

    is_cpu = (gpu_choice in ("CPU", "NONE", "OFF")) and not tpu_choice
    # For CPU data processing, default to 8-vCPU / 50GiB High-RAM (0.26 CU/h) unless --std-mem is requested
    use_high_mem = bool(args.high_mem or (is_cpu and not getattr(args, "std_mem", False)))

    cmd = [colab_bin, "new", "-s", session_name]
    if tpu_choice:
        cmd.extend(["--tpu", tpu_choice])
        hw_label = f"TPU-{tpu_choice.upper()} (超高核CPU宿主机)"
    elif not is_cpu:
        cmd.extend(["--gpu", gpu_choice])
        hw_label = f"{gpu_choice}{' High-RAM' if use_high_mem else ''}"
    else:
        hw_label = "CPU High-RAM (8核 50GB, 0.26 CU/h)" if use_high_mem else "CPU Standard (2核 12GB, 0.08 CU/h)"

    # G4 only supports STANDARD shape on Colab; passing --high-mem fails allocation
    if use_high_mem and not tpu_choice and gpu_choice != "G4":
        cmd.append("--high-mem")

    if not getattr(args, "json", False):
        console.print(f"⚡ [bold cyan]正在合闸启动 Colab {hw_label} 实例 ({session_name})...[/bold cyan]")
    subprocess.check_call(
        cmd,
        stdout=subprocess.DEVNULL if getattr(args, "json", False) else None,
        stderr=subprocess.DEVNULL if getattr(args, "json", False) else None,
    )

    if not getattr(args, "no_drive", False):
        if not getattr(args, "json", False):
            console.print(f"📦 [cyan]正在默认自动挂载 20TB Google Drive 并初始化永久工作区 ({session_name})...[/cyan]")
        try:
            subprocess.call(
                [colab_bin, "drivemount", "-s", session_name],
                stdout=subprocess.DEVNULL if getattr(args, "json", False) else None,
                timeout=90,
            )
            init_script = os.path.join(BASE_DIR, "workflows", "init_20tb_drive.py")
            if os.path.exists(init_script):
                subprocess.call(
                    [colab_bin, "exec", "-s", session_name, "--timeout", "45", "-f", init_script],
                    stdout=subprocess.DEVNULL if getattr(args, "json", False) else None,
                    timeout=60,
                )
        except Exception as e:
            if not getattr(args, "json", False):
                console.print(f"[yellow]⚠️ 自动挂载云盘遇到提示 ({session_name}): {e}[/yellow]")


def cmd_up(args):
    """Spin up one or more sessions (Colab GPU/CPU/TPU or Free Kaggle Dual-T4), deploy probe, and initialize Drive workspace."""
    cfg = load_config()
    colab_bin = resolve_colab_bin()
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    session_names = [args.session] if workers == 1 else [f"{args.session}-{i}" for i in range(1, workers + 1)]

    force_colab = bool(getattr(args, "colab", False))
    force_kaggle = bool(getattr(args, "kaggle", False))
    gpu_raw = (getattr(args, "gpu", None) or "T4").strip()
    is_cpu_or_tpu = bool(getattr(args, "cpu", False) or getattr(args, "tpu", None) or gpu_raw.upper() in ("CPU", "NONE", "OFF"))

    use_kaggle = False
    if not force_colab and not is_cpu_or_tpu:
        if force_kaggle or is_kaggle_gpu_spec(gpu_raw):
            use_kaggle = True
        elif (
            gpu_raw.upper() == "T4"
            and not getattr(args, "high_mem", False)
            and workers == 1
            and cfg.get("prefer_kaggle_for_t4", True)
            and load_kaggle_token()
        ):
            kq = get_kaggle_quota()
            if (kq.get("gpu_remaining_hours") or 0.0) >= 0.5 and (kq.get("gpu_reserved_seconds") or 0.0) == 0.0:
                use_kaggle = True

    started_sessions = []
    if use_kaggle:
        for s_name in session_names:
            try:
                spin_up_kaggle_session(
                    session_name=s_name,
                    gpu_choice=gpu_raw if is_kaggle_gpu_spec(gpu_raw) else "T4x2",
                    cfg=cfg,
                    probe_local_path=PROBE_LOCAL_PATH,
                    quiet=bool(getattr(args, "json", False)),
                    console=console,
                )
                started_sessions.append(s_name)
            except Exception as e:
                if not force_kaggle and not is_kaggle_gpu_spec(gpu_raw) and gpu_raw.upper() == "T4":
                    if not getattr(args, "json", False):
                        console.print(f"[yellow]⚠️ Kaggle 免费双卡暂不可用 ({e})，自动回退至 Colab T4...[/yellow]")
                    use_kaggle = False
                    break
                raise SystemExit(f"❌ Kaggle 实例启动失败: {e}")

    if not use_kaggle:
        pre_endpoints = set()
        try:
            _, pre_assigns = colab_state.sync_sessions()
            pre_endpoints = {a.endpoint for a in pre_assigns if getattr(a, "endpoint", None)}
        except Exception:
            pass
        for s_name in session_names:
            try:
                _spin_up_single_session(colab_bin, s_name, args)
                started_sessions.append(s_name)
            except KeyboardInterrupt:
                try:
                    colab_state._sessions = None
                    tracked, post_assigns = colab_state.sync_sessions()
                    tracked_eps = {s.endpoint for s in tracked.values() if getattr(s, "endpoint", None)}
                    for a in post_assigns:
                        ep = getattr(a, "endpoint", None)
                        if ep and ep not in pre_endpoints and ep not in tracked_eps:
                            colab_state.client.unassign(ep)
                except Exception:
                    pass
                raise
            except subprocess.CalledProcessError as e:
                if not getattr(args, "json", False):
                    console.print(f"[yellow]⚠️ 节点 [{s_name}] 启动受限 (已开启 {len(started_sessions)} 台): {e}[/yellow]")
                break

    # Deploy telemetry probe and refresh dashboard (force fresh read after Drive mount)
    ccu, vms = inspect_all_vms()
    for vm in vms:
        if vm["name"] in started_sessions and vm.get("provider") != "kaggle":
            try:
                if os.path.exists(PROBE_LOCAL_PATH):
                    upload_file_to_vm(vm["proxy_url"], vm["proxy_token"], PROBE_LOCAL_PATH, "content/.colab_probe.py")
                    start_remote_probe_via_runtime(vm["proxy_url"], vm["proxy_token"])
                    vm["telemetry"] = fetch_vm_telemetry(vm["proxy_url"], vm["proxy_token"], auto_deploy=False)
            except Exception:
                pass
            init_script = os.path.join(BASE_DIR, "workflows", "init_20tb_drive.py")
            if os.path.exists(init_script):
                try:
                    upload_file_to_vm(vm["proxy_url"], vm["proxy_token"], init_script, "content/init_20tb_drive.py")
                except Exception:
                    pass
    # Record dual-track CCU start snapshot (aligned with colab-ultra-workstation colab_meter)
    _tracker_path = os.path.expanduser("~/.config/colab-cli/session_tracker.json")
    _gpu_rates = {"CPU": 0.08, "NONE": 0.08, "T4": 1.19, "T4X2": 0.0, "L4": 1.71, "G4": 10.05, "A100": 8.71, "H100": 18.05}
    try:
        os.makedirs(os.path.dirname(_tracker_path), exist_ok=True)
        _tdata = json.loads(open(_tracker_path, "r", encoding="utf-8").read()) if os.path.exists(_tracker_path) else {}
        _start_bal = float(ccu.get("paid_balance", 0.0) or 0.0)
        for _sname in started_sessions:
            _vm_match = next((v for v in vms if v["name"] == _sname), None)
            _accel = str((_vm_match or {}).get("accelerator") or getattr(args, "gpu", None) or "CPU").upper()
            _prov = (_vm_match or {}).get("provider", "colab")
            _rate = 0.0 if _prov == "kaggle" else _gpu_rates.get(_accel, float(ccu.get("burn_rate_hourly", 0.0) or 0.0))
            _tdata[_sname] = {
                "session": _sname,
                "gpu": _accel,
                "provider": _prov,
                "rate": _rate,
                "start_time": time.time(),
                "start_balance": _start_bal,
            }
        with open(_tracker_path, "w", encoding="utf-8") as _tf:
            json.dump(_tdata, _tf, indent=2, ensure_ascii=False)
    except Exception:
        pass

    if getattr(args, "json", False):
        sync_dashboard_json(ccu, vms)
        payload = build_agent_status_payload(ccu, vms)
        payload["started_sessions"] = started_sessions
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        render_dashboard(ccu, vms)
        console.print(f"[green]✅ 实例 {started_sessions} 已就绪（硬件探针 + 统一云盘工作区 + 计费打点已激活）！[/green]")


def _resolve_exec_source(args, project_name: str | None) -> tuple[str, str]:
    if args.file:
        src_path = os.path.expanduser(args.file)
        if not os.path.isabs(src_path) and project_name:
            cand = os.path.expanduser(f"~/ColabUltra/{project_name}/code/{args.file}")
            if os.path.exists(cand):
                src_path = cand
        real_src = os.path.realpath(src_path)
        if real_src.startswith(("/tmp/", "/private/tmp/", "/var/folders/")):
            raise SystemExit(
                f"❌ [colab-ultra] 严禁传入 /tmp 临时脚本 ({args.file})！"
                f"请将脚本保存在 ~/ColabUltra/{project_name or '<project>'}/code/ 云盘目录中再执行。"
            )
        with open(src_path, "r", encoding="utf-8") as src_f:
            user_code = src_f.read()
        script_label = os.path.basename(args.file)
    elif args.code:
        user_code = args.code
        script_label = "<colab_ultra_exec>"
    else:
        raise SystemExit("Provide either inline code or -f <script.py>")
    return user_code, script_label


def cmd_exec(args):
    """Agent-First remote code/script execution with automatic post-run hardware telemetry (Colab & Kaggle unified)."""
    project_name = getattr(args, "project", None)
    timeout_sec = str(getattr(args, "timeout", 3600.0) or 3600.0)
    user_code, script_label = _resolve_exec_source(args, project_name)
    task_label = f"{project_name}/{script_label}" if project_name else script_label
    local_pid = os.getpid()

    proj_prelude = ""
    if project_name:
        proj_prelude = (
            f"_proj_root = '/content/drive/MyDrive/ColabUltra/{project_name}'\n"
            "_code_dir = f'{_proj_root}/code'\n"
            "for _sub in ('code', 'models', 'data', 'runs'):\n"
            "    os.makedirs(f'{_proj_root}/{_sub}', exist_ok=True)\n"
            "os.chdir(_code_dir)\n"
            "if _code_dir not in sys.path:\n"
            "    sys.path.insert(0, _code_dir)\n"
        )

    # Check if target session is an active Kaggle Free Dual-T4 session
    k_sessions = _load_kaggle_sessions()
    k_sess = k_sessions.get(args.session)
    if k_sess and k_sess.get("proxy_url") and fetch_vm_kernels(k_sess["proxy_url"], ""):
        p_url = k_sess["proxy_url"]
        local_hub = detect_local_drive_hub()
        if getattr(args, "restart_kernel", False):
            restart_vm_kernel(p_url, "")
        if project_name:
            sync_project_to_kaggle(p_url, project_name, local_hub)

        drive_tok = get_short_lived_drive_token()
        kaggle_sync_defs = build_in_vm_sync_snippet(project_name, drive_tok)
        wrapped_kaggle_code = (
            "import os, sys, gc, json, threading\n"
            "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n"
            "for _a in ('last_traceback', 'last_value', 'last_type', 'last_exc'):\n"
            "    if hasattr(sys, _a):\n"
            "        try: delattr(sys, _a)\n"
            "        except Exception: pass\n"
            "gc.collect()\n"
            "if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():\n"
            "    try: sys.modules['torch'].cuda.empty_cache()\n"
            "    except Exception: pass\n"
            + proj_prelude
            + kaggle_sync_defs
            + "_hb_stop = threading.Event()\n"
            "def _hb_loop():\n"
            "    import subprocess, time\n"
            "    _p = '/content/.task_heartbeat.json'\n"
            "    _tp = '/content/.colab_telemetry.json'\n"
            "    _pr = '/content/.colab_probe.py'\n"
            "    _s0 = time.time()\n"
            "    while not _hb_stop.is_set():\n"
            "        try:\n"
            f"            with open(_p, 'w') as _f: json.dump({{'active': True, 'task': {task_label!r}, 'local_pid': {local_pid}, 'started_at': _s0}}, _f)\n"
            "            if os.path.exists(_pr) and (not os.path.exists(_tp) or time.time() - os.path.getmtime(_tp) > 45.0):\n"
            "                subprocess.Popen(['python3', _pr], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
            "            _cu_pack_runs(final=False)\n"
            "        except Exception: pass\n"
            "        _hb_stop.wait(15.0)\n"
            "    try:\n"
            "        if os.path.exists(_p): os.remove(_p)\n"
            "    except Exception: pass\n"
            "threading.Thread(target=_hb_loop, daemon=True).start()\n"
            f"_cu_src = {user_code!r}\n"
            "try:\n"
            f"    exec(compile(_cu_src, {script_label!r}, 'exec'), globals())\n"
            "finally:\n"
            "    _hb_stop.set()\n"
            "    try: _cu_pack_runs(final=True)\n"
            "    except Exception: pass\n"
            "    try:\n"
            "        if os.path.exists('/content/.task_heartbeat.json'): os.remove('/content/.task_heartbeat.json')\n"
            "    except Exception: pass\n"
        )

        proj_local_dir = os.path.join(local_hub, project_name) if (local_hub and project_name) else None

        def _live_sync_tick():
            if proj_local_dir:
                pull_live_tarball_to_local(p_url, "/content/_cu_live_runs.tar.gz", proj_local_dir)

        t0 = time.time()
        ret_code, out_str, err_str = jupyter_exec_ws(
            p_url,
            wrapped_kaggle_code,
            timeout=float(timeout_sec),
            stream_stdout=not getattr(args, "json", False),
            stream_to_stderr=bool(getattr(args, "json", False)),
            on_periodic_tick=_live_sync_tick if proj_local_dir else None,
            tick_interval_sec=20.0,
        )
        if proj_local_dir:
            pull_live_tarball_to_local(p_url, "/content/_cu_final_runs.tar.gz", proj_local_dir)
        elapsed = round(time.time() - t0, 2)

        if getattr(args, "json", False):
            ccu, vms = inspect_all_vms()
            sync_dashboard_json(ccu, vms)
            payload = build_agent_status_payload(ccu, vms)
            payload.update({
                "ok": ret_code == 0,
                "exit_code": ret_code,
                "elapsed_sec": elapsed,
                "stdout": out_str,
                "stderr": err_str,
            })
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        elif ret_code != 0:
            sys.exit(ret_code)
        return

    # ------------------------------------------------------------------
    # Colab execution engine (nohup detached + short-lived WS log polling)
    # Completely bypasses buggy google-colab-cli wsclient.py hangs & 10s races,
    # streams live stdout, keeps Jupyter kernel alive via short polls,
    # auto-refreshes proxy tokens, and exits immediately with exact rc.
    # ------------------------------------------------------------------
    import uuid

    def _resolve_colab_proxy() -> tuple[str, str]:
        colab_state._sessions = None
        local_sessions, assignments = colab_state.sync_sessions()
        target_ep = None
        if args.session in local_sessions:
            target_ep = local_sessions[args.session].endpoint
        for a in assignments:
            if (target_ep and a.endpoint == target_ep) or (not target_ep and len(assignments) == 1):
                return a.runtime_proxy_info.url, a.runtime_proxy_info.token
        raise SystemExit(f"[colab-ultra] Active Colab session '{args.session}' not found.")

    p_url, p_tok = _resolve_colab_proxy()
    last_token_refresh = time.time()

    if getattr(args, "restart_kernel", False):
        try:
            restart_vm_kernel(p_url, p_tok)
        except Exception:
            pass

    def _short_ws_exec(code: str, call_timeout: float = 25.0, retries: int = 3) -> tuple[int, str, str]:
        nonlocal p_url, p_tok, last_token_refresh
        last_exc: Exception | None = None
        for attempt in range(retries):
            if time.time() - last_token_refresh > 2400.0:
                try:
                    p_url, p_tok = _resolve_colab_proxy()
                    last_token_refresh = time.time()
                except Exception:
                    pass
            try:
                return jupyter_exec_ws(
                    p_url,
                    code,
                    proxy_token=p_tok,
                    timeout=call_timeout,
                    stream_stdout=False,
                    stream_to_stderr=False,
                )
            except Exception as exc:
                last_exc = exc
                time.sleep(min(2.0 * (attempt + 1), 6.0))
                try:
                    p_url, p_tok = _resolve_colab_proxy()
                    last_token_refresh = time.time()
                except Exception:
                    pass
        raise RuntimeError(f"Short WS exec failed after {retries} retries: {last_exc}")

    wrapped_code = (
        "import os, sys, gc, json, threading\n"
        "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n"
        "os.environ.setdefault('PYTHONUNBUFFERED', '1')\n"
        + proj_prelude
        + "_hb_stop = threading.Event()\n"
        "def _hb_loop():\n"
        "    import subprocess, time\n"
        "    _p = '/content/.task_heartbeat.json'\n"
        "    _tp = '/content/.colab_telemetry.json'\n"
        "    _pr = '/content/.colab_probe.py'\n"
        "    _s0 = time.time()\n"
        "    while not _hb_stop.is_set():\n"
        "        try:\n"
        f"            with open(_p, 'w') as _f: json.dump({{'active': True, 'task': {task_label!r}, 'local_pid': {local_pid}, 'started_at': _s0}}, _f)\n"
        "            if os.path.exists(_pr) and (not os.path.exists(_tp) or time.time() - os.path.getmtime(_tp) > 45.0):\n"
        "                subprocess.Popen(['python3', _pr], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
        "        except Exception: pass\n"
        "        _hb_stop.wait(15.0)\n"
        "    try:\n"
        "        if os.path.exists(_p): os.remove(_p)\n"
        "    except Exception: pass\n"
        "threading.Thread(target=_hb_loop, daemon=True).start()\n"
        f"_cu_src = {user_code!r}\n"
        "try:\n"
        f"    exec(compile(_cu_src, {script_label!r}, 'exec'), globals())\n"
        "finally:\n"
        "    _hb_stop.set()\n"
        "    try:\n"
        "        if os.path.exists('/content/.task_heartbeat.json'): os.remove('/content/.task_heartbeat.json')\n"
        "    except Exception: pass\n"
    )

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

    l_rc, l_out, l_err = _short_ws_exec(launch_code, call_timeout=30.0, retries=3)
    if l_rc != 0:
        raise SystemExit(f"[colab-ultra] Failed to launch remote job: {l_err or l_out}")
    for line in l_out.splitlines():
        try:
            info = json.loads(line.strip())
            if info.get("launched"):
                remote_pgid = int(info.get("pid", 0)) or None
        except Exception:
            pass

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
                p_rc, p_out, p_err = _short_ws_exec(poll_code, call_timeout=25.0, retries=3)
                if p_rc != 0:
                    raise RuntimeError(p_err or p_out or "poll returned non-zero")
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    err_str = f"[colab-ultra] Lost connection to Colab VM after 5 consecutive poll failures: {exc}\n"
                    if not is_json_mode:
                        sys.stderr.write(err_str)
                        sys.stderr.flush()
                    ret_code = 1
                    break
                time.sleep(5.0)
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
                _short_ws_exec(f"import os, signal\ntry: os.killpg({remote_pgid}, signal.SIGTERM)\nexcept Exception: pass\n", call_timeout=10.0, retries=1)
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
                retries=1,
            )
        except Exception:
            pass

    out_str = "".join(out_chunks)
    elapsed = round(time.time() - t0, 2)

    if getattr(args, "json", False):
        ccu, vms = inspect_all_vms()
        sync_dashboard_json(ccu, vms)
        payload = build_agent_status_payload(ccu, vms)
        payload.update({
            "ok": ret_code == 0,
            "exit_code": ret_code,
            "elapsed_sec": elapsed,
            "stdout": out_str,
            "stderr": err_str,
        })
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif ret_code != 0:
        sys.exit(ret_code)


def cmd_down(args):
    """Safely back up outputs to Drive and shut down one or all Colab & Kaggle VMs."""
    cfg = load_config()
    ccu, vms = inspect_all_vms()
    if not vms:
        if args.all:
            cancel_all_orphan_kaggle_sessions()
        if getattr(args, "json", False):
            print(json.dumps({"ok": True, "unassigned": [], "active_vms_count": 0}, indent=2, ensure_ascii=False))
        else:
            console.print("[dim]当前没有运行中的云端实例。[/dim]")
        return

    targets = vms if args.all else [v for v in vms if v["name"] == args.session or v["endpoint"].startswith(args.session or "")]
    if not targets and len(vms) == 1 and not args.session:
        targets = vms

    local_hub = detect_local_drive_hub()
    unassigned = []
    for vm in targets:
        prov = vm.get("provider", "colab")
        if not getattr(args, "json", False):
            console.print(f"🔌 [yellow]正在安全拉闸关机: [{vm['name']}] ({prov.upper()} {vm['accelerator']} - {vm['endpoint']})...[/yellow]")
        if prov == "kaggle":
            shut_down_kaggle_session(
                vm,
                local_hub=local_hub,
                flush_outputs=cfg.get("auto_backup_to_drive_before_stop", True),
            )
        else:
            if cfg.get("auto_backup_to_drive_before_stop", True):
                backup_vm_outputs_to_drive(vm["proxy_url"], vm["proxy_token"])
            colab_state.client.unassign(vm["endpoint"])
            if vm["name"] != "web-session":
                colab_state.store.remove(vm["name"])
        log_breaker_event("MANUAL_DOWN", {"name": vm["name"], "provider": prov, "endpoint": vm["endpoint"]})
        unassigned.append({"name": vm["name"], "provider": prov, "endpoint": vm["endpoint"], "accelerator": vm["accelerator"]})
        if not getattr(args, "json", False):
            console.print(f"[green]✅ [{vm['name']}] 已彻底断电释放，停止算力/额度消耗！[/green]")

    if args.all:
        cancel_all_orphan_kaggle_sessions()

    try:
        ccu_after, vms_after = inspect_all_vms()
        sync_dashboard_json(ccu_after, vms_after, cfg=cfg)

        # Dual-track CCU settlement (aligned with colab-ultra-workstation colab_meter)
        _tracker_path = os.path.expanduser("~/.config/colab-cli/session_tracker.json")
        _tdata = {}
        try:
            if os.path.exists(_tracker_path):
                _tdata = json.loads(open(_tracker_path, "r", encoding="utf-8").read())
        except Exception:
            _tdata = {}
        _end_bal = float(ccu_after.get("paid_balance", 0.0) or 0.0)
        for _u in unassigned:
            _rec = _tdata.pop(_u["name"], None)
            if _rec:
                _dur_sec = max(0.0, time.time() - float(_rec.get("start_time", time.time())))
                _rate = float(_rec.get("rate", 0.0) or 0.0)
                _start_bal = float(_rec.get("start_balance", _end_bal) or _end_bal)
                _api_cost = max(0.0, _start_bal - _end_bal)
                _calc_cost = (_dur_sec / 3600.0) * _rate
                _final_cost = _api_cost if _api_cost > 0.005 else _calc_cost
                _u["duration_sec"] = round(_dur_sec, 1)
                _u["duration_str"] = f"{int(_dur_sec // 60)} 分 {int(_dur_sec % 60)} 秒"
                _u["estimated_cost_ccu"] = round(_final_cost, 3)
        try:
            if os.path.exists(_tracker_path):
                with open(_tracker_path, "w", encoding="utf-8") as _tf:
                    json.dump(_tdata, _tf, indent=2, ensure_ascii=False)
        except Exception:
            pass

        if getattr(args, "json", False):
            payload = build_agent_status_payload(ccu_after, vms_after, cfg=cfg)
            payload["unassigned"] = unassigned
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            _rem_rate = float(ccu_after.get("burn_rate_hourly", 0.0) or 0.0)
            console.print("\n[bold green]✅ 算力释放与计费核验报告：[/bold green]")
            console.print(f"• 实例释放状态: Active assignments: {len(vms_after)}")
            console.print(f"• 实时消耗速率: {_rem_rate:.2f} CCU/h")
            for _u in unassigned:
                _dur = _u.get("duration_str", "未知")
                _cost = _u.get("estimated_cost_ccu", 0.0)
                console.print(f"• 已释放会话: [{_u['name']}] {_u['accelerator']} ({_u['provider'].upper()}) | 运行时长: {_dur} | 本次消耗: {_cost:.2f} CCU")
            console.print(f"• 当前账户余额: {_end_bal:.2f} CCU")
    except Exception:
        pass


def cmd_pull(args):
    """Spin up a cheap/free CPU instance, download large files/repos directly into Google Drive via Google Cloud network, and safely down immediately."""
    session = getattr(args, "session", "pull-worker") or "pull-worker"
    project = args.project
    source = args.source
    dest_dir = getattr(args, "dest", "models") or "models"
    filename = getattr(args, "filename", None)

    console.print(f"🚀 [cyan]正在启动云端 CPU 节点 ({session})，走 Google Cloud 内网高速下载至 Drive，不消耗本地带宽...[/cyan]")
    up_args = argparse.Namespace(
        session=session,
        gpu="CPU",
        cpu=True,
        std_mem=True,
        high_mem=False,
        tpu=None,
        workers=1,
        no_drive=False,
        colab=True,
        kaggle=False,
        json=False,
    )
    cmd_up(up_args)

    dl_code = f"""
import os, sys, subprocess
from pathlib import Path

target_dir = Path("/content/drive/MyDrive/ColabUltra/{project}/{dest_dir}")
target_dir.mkdir(parents=True, exist_ok=True)
source = "{source}"
filename = "{filename or ''}"

print(f"Target directory: {{target_dir}}")
if source.startswith("http://") or source.startswith("https://"):
    out_name = filename or source.split("?")[0].split("/")[-1]
    out_path = target_dir / out_name
    print(f"Downloading URL {{source}} -> {{out_path}}...")
    subprocess.check_call(["curl", "-L", "-C", "-", "-o", str(out_path), source])
    print(f"Downloaded {{out_path.name}}: {{out_path.stat().st_size / 1024 / 1024:.2f}} MB")
else:
    subprocess.check_call(["pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download, snapshot_download
    if filename:
        print(f"Downloading file {{filename}} from HF repo {{source}}...")
        hf_hub_download(repo_id=source, filename=filename, local_dir=str(target_dir))
    else:
        print(f"Downloading entire HF repo {{source}}...")
        snapshot_download(repo_id=source, local_dir=str(target_dir))
import time
os.sync()
def _tx_bytes():
    for ln in Path("/proc/net/dev").read_text().splitlines():
        if "eth0:" in ln:
            return int(ln.split("eth0:", 1)[1].split()[8])
    return 0
print("Flushing Drive FUSE write-back cache to Google Drive...", flush=True)
low_cnt = 0
for _ in range(30):
    b0 = _tx_bytes()
    time.sleep(3.0)
    os.sync()
    mb_s = max(0, _tx_bytes() - b0) / (3.0 * 1024 * 1024)
    if mb_s < 2.0:
        low_cnt += 1
        if low_cnt >= 2:
            break
    else:
        low_cnt = 0
        print(f"  [Drive upload] {{mb_s:.1f}} MB/s...", flush=True)
print("✅ Download and Drive flush complete!", flush=True)
"""
    exec_args = argparse.Namespace(
        session=session,
        project=project,
        file=None,
        timeout=3600.0,
        restart_kernel=False,
        code=dl_code,
        json=getattr(args, "json", False),
    )
    try:
        cmd_exec(exec_args)
    finally:
        console.print(f"🔌 [cyan]下载完成，立即释放云端 CPU 节点 ({session})，避免闲置算力扣费...[/cyan]")
        down_args = argparse.Namespace(
            session=session,
            all=False,
            json=getattr(args, "json", False),
        )
        cmd_down(down_args)


def cmd_init(args):
    """One-stop interactive/CLI setup wizard for Colab Ultra (Upstream Patch + OAuth + Kaggle Free T4x2 + Smart Breaker)."""
    cfg = load_config()
    if args.alert_minutes is not None:
        cfg["alert_after_minutes"] = args.alert_minutes
    if args.stop_minutes is not None:
        cfg["stop_after_minutes"] = args.stop_minutes
    if args.vps is not None:
        cfg["vps_host"] = args.vps
    if args.webhook is not None:
        cfg["webhook_url"] = args.webhook

    console.print(
        Panel(
            "🚀 [bold cyan]Colab Ultra 一键初始化向导 (colab-ultra init)[/bold cyan]\n"
            "自动检查：0) 官方 CLI 补丁  1) Google Colab + Kaggle 免费双卡凭证  2) 显卡瓦数断路器配置  3) 远端 VPS 24h 哨兵同步",
            border_style="cyan",
        )
    )

    # Step 0: Auto-patch upstream google-colab-cli bugs (drivemount HTTP 400 + 8x ADC startup speedup)
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, "colab_ultra"))
        from patch_colab_cli import apply_upstream_patches
        patched = apply_upstream_patches()
        if patched:
            console.print(f"🔧 [green]已自动修补官方 google-colab-cli: {', '.join(patched)}[/green]")
        else:
            console.print("✅ [green]步骤 0/3: 官方 google-colab-cli 补丁状态已最新 (含 drivemount 修复与 0.8s 极速启动)[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️ 官方 CLI 补丁检查提示: {e}[/yellow]")

    # Ensure Agent SKILL.md symlinks are installed for Codex / Antigravity / Agents
    for skill_dir in ("~/.codex/skills", "~/.gemini/config/skills", "~/.agents/skills"):
        parent = os.path.expanduser(skill_dir)
        if os.path.isdir(parent):
            link_path = os.path.join(parent, "colab-ultra")
            if not os.path.exists(link_path):
                try:
                    os.symlink(BASE_DIR, link_path)
                except Exception:
                    pass

    adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    scopes_needed = (
        "openid,"
        "https://www.googleapis.com/auth/cloud-platform,"
        "https://www.googleapis.com/auth/userinfo.email,"
        "https://www.googleapis.com/auth/colaboratory"
    )
    if getattr(args, "with_drive_scope", False):
        scopes_needed += ",https://www.googleapis.com/auth/drive"

    if os.path.exists(adc_path) and not args.auth and not getattr(args, "with_drive_scope", False):
        console.print("✅ [green]步骤 1/3: 已检测到本地 Google ADC 凭证[/green]")
    else:
        console.print("[cyan]🔑 正在拉起 Google 浏览器授权...[/cyan]")
        subprocess.call(["gcloud", "auth", "application-default", "login", f"--scopes={scopes_needed}"])

    kq = get_kaggle_quota()
    if kq.get("configured") and "gpu_remaining_hours" in kq:
        console.print(
            f"✅ [green]Kaggle 免费双卡 T4x2 (30GB) 已就绪: 本周剩余 {kq['gpu_remaining_hours']:.1f}h / {kq['gpu_total_hours']:.0f}h (0 CU/h)[/green]"
        )
    else:
        console.print("[yellow]ℹ️ 未检测到 ~/.kaggle/access_token，仅使用 Google Colab 算力池。[/yellow]")

    local_hub = detect_local_drive_hub()
    if local_hub:
        console.print(f"✅ [green]本地云盘挂载已就绪: ~/ColabUltra -> {local_hub}[/green]")
    else:
        console.print(
            "[yellow]ℹ️ 本地尚未检测到 Google Drive for Desktop 挂载 (~/Library/CloudStorage/GoogleDrive-*)。"
            "建议执行 `brew install --cask google-drive` 并登录同一账号以启用本地 ~/ColabUltra 零算力直读直写。[/yellow]"
        )

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        pass
    console.print(
        f"✅ [green]步骤 2/3: 智能电闸规则已保存至 {CONFIG_PATH} (仓库外隔离存储)[/green]\n"
        f"   • 物理功耗判定: A100/H100 < 75W 视为空载 | L4/T4 < 28W (双卡单卡 < 30W) 视为空载\n"
        f"   • 预警时间: {cfg['alert_after_minutes']} 分钟 | 自动备份并拉闸时间: {cfg['stop_after_minutes']} 分钟"
    )

    vps_host = (cfg.get("vps_host", "") or "").strip()
    ssh_key = os.path.expanduser(cfg.get("vps_ssh_key", "~/.ssh/id_ed25519") or "~/.ssh/id_ed25519")
    vps_poller = VPS_SENTINEL_LOCAL_PATH
    if vps_host and os.path.exists(ssh_key) and os.path.exists(vps_poller) and os.path.exists(adc_path):
        console.print(f"🔄 [cyan]步骤 3/3: 正在将最新凭证与 24h 智能电闸哨兵同步至 VPS ({vps_host})...[/cyan]")
        try:
            subprocess.check_call([
                "scp", "-q", "-i", ssh_key, "-o", "StrictHostKeyChecking=no",
                adc_path, PROBE_LOCAL_PATH, vps_poller, f"{vps_host}:/tmp/"
            ])
            subprocess.check_call([
                "ssh", "-i", ssh_key, "-o", "StrictHostKeyChecking=no", vps_host,
                "sudo mv /tmp/application_default_credentials.json /etc/colab-adc.json && "
                "sudo chmod 600 /etc/colab-adc.json && "
                "sudo mv /tmp/telemetry_probe.py /usr/local/share/colab_probe.py && "
                "sudo mv /tmp/vps_sentinel.py /usr/local/bin/update-colab-stats.py && "
                "sudo chmod +x /usr/local/bin/update-colab-stats.py && "
                "sudo /usr/local/bin/update-colab-stats.py"
            ])
            console.print(f"✅ [green]步骤 3/3: VPS ({vps_host}) 24h 独立值班哨兵已同步就绪！[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ VPS 同步失败: {e}[/yellow]")
    else:
        console.print("ℹ️ [dim]步骤 3/3: 未配置远端 VPS 或跳过同步，使用本地守护模式。[/dim]")

    ccu, vms = inspect_all_vms()
    render_dashboard(ccu, vms)


def main():
    parser = argparse.ArgumentParser(description="Colab Ultra: Agent-First Smart Breaker & 20TB Drive Hub (Colab + Free Kaggle T4x2)")
    sub = parser.add_subparsers(dest="cmd")

    # init
    p_init = sub.add_parser("init", help="一键初始化向导（自动修补官方 CLI、配置凭证、瓦数断路器阈值、同步 VPS 哨兵）")
    p_init.add_argument("--auth", action="store_true", help="强制重新运行浏览器 OAuth 授权")
    p_init.add_argument("--with-drive-scope", action="store_true", help="同时授权 Google Drive API 权限以开启 Kaggle 云对云超大模型直传")
    p_init.add_argument("--alert-minutes", type=float, default=None, help="设置低功耗预警分钟数 (默认 5 分钟)")
    p_init.add_argument("--stop-minutes", type=float, default=None, help="设置低功耗自动拉闸分钟数 (默认 10 分钟)")
    p_init.add_argument("--vps", type=str, default=None, help="绑定远程 VPS 地址以实现 24h 离线断电保护")
    p_init.add_argument("--webhook", type=str, default=None, help="设置 Bark/飞书/Telegram 告警 Webhook URL")

    # status / top
    p_status = sub.add_parser("status", help="查看实时算力账单与所有 Colab / Kaggle 机器的 GPU 瓦数/CPU/云盘负载")
    p_status.add_argument("--json", action="store_true", help="输出 Agent 友好的结构化 JSON 诊断数据")

    # watch
    p_watch = sub.add_parser("watch", help="启动智能电闸后台轮询守护（低负载告警 + 超时自动拉闸）")
    p_watch.add_argument("--interval", type=int, default=None, help="轮询间隔秒数 (默认 15s)")
    p_watch.add_argument("--gpu-threshold", type=float, default=None, help="最低 GPU 利用率百分比阈值 (默认 5%%)")
    p_watch.add_argument("--alert-minutes", type=float, default=None, help="低负载多少分钟触发报警 (默认 5 分钟)")
    p_watch.add_argument("--stop-minutes", type=float, default=None, help="低负载多少分钟自动拉闸关机 (默认 10 分钟)")
    p_watch.add_argument("--once", action="store_true", help="仅执行一次巡检与阈值判定后退出")

    # up
    p_up = sub.add_parser("up", help="一键合闸开机（支持免费 Kaggle T4x2 / Colab GPU / 8核CPU / 44核TPU）、统一工作区与探针")
    p_up.add_argument("-s", "--session", default="ultra", help="会话名称 (默认 ultra；多节点时自动命名为 <session>-1..N)")
    p_up.add_argument("--gpu", default="T4", help="硬件型号: T4 (默认优先用免费 Kaggle T4x2), T4x2, L4, G4, A100, H100, CPU")
    p_up.add_argument("--kaggle", action="store_true", default=False, help="强制使用免费 Kaggle 双卡 T4x2 (30GB 显存, 0 CU/h)")
    p_up.add_argument("--colab", action="store_true", default=False, help="强制使用 Google Colab 节点 (跳过免费 Kaggle 优先路由)")
    p_up.add_argument("--cpu", action="store_true", default=False, help="启动纯 CPU 数据处理节点 (默认自动开启 8核50GB 高内存模式，仅 0.26 CU/h)")
    p_up.add_argument("--tpu", default=None, help="启动超高核 TPU 宿主机做大数据处理: v5e1 (24核47GB, 3.09 CU/h) 或 v6e1 (44核172GB, 5.09 CU/h)")
    p_up.add_argument("--high-mem", action="store_true", default=False, help="开启高内存模式 (--cpu 默认自动开启)")
    p_up.add_argument("--std-mem", action="store_true", default=False, help="在 --cpu 模式下强制使用标准内存 (2核12GB, 0.08 CU/h)")
    p_up.add_argument("--workers", type=int, default=1, help="并发拉起多台节点数量 (例如 --cpu --workers 5 可并发拉起 40核250GB 集群)")
    p_up.add_argument("--no-drive", action="store_true", default=False, help="跳过自动挂载 20TB Google Drive")
    p_up.add_argument("--json", action="store_true", help="以结构化 JSON 返回实例就绪与硬件状态 (Agent 专用)")

    # exec (Agent-first remote execution + telemetry return)
    p_exec = sub.add_parser("exec", help="在 Colab 实例中执行代码或脚本，并自动返回最新 GPU 瓦数/CPU核数/显存/云盘状态")
    p_exec.add_argument("-s", "--session", default="ultra", help="会话名称 (默认 ultra)")
    p_exec.add_argument("-p", "--project", default=None, help="项目名称（直接切换到 /content/drive/MyDrive/ColabUltra/<project>/code 下执行）")
    p_exec.add_argument("-f", "--file", default=None, help="要执行的脚本路径")
    p_exec.add_argument("--timeout", type=float, default=3600.0, help="远程执行超时秒数 (默认 3600s)")
    p_exec.add_argument("--restart-kernel", action="store_true", help="执行前自动重启 Jupyter 内核以释放残留显存")
    p_exec.add_argument("code", nargs="?", default=None, help="要直接执行的 Python 代码片段")
    p_exec.add_argument("--json", action="store_true", help="输出包含 stdout/stderr 与实时硬件瓦数的 JSON (Agent 专用)")

    # pull (Zero-burn cloud download of models/data directly to Drive)
    p_pull = sub.add_parser("pull", help="在 0-CU 云端 CPU 实例上极速下载大文件/模型权重至 Google Drive，下完自动拉闸关机")
    p_pull.add_argument("source", help="下载地址 (HTTP/HTTPS) 或 Hugging Face 仓库/文件名")
    p_pull.add_argument("-p", "--project", required=True, help="目标项目名称 (对应 /content/drive/MyDrive/ColabUltra/<project>)")
    p_pull.add_argument("--dest", default="models", help="目标子目录 (默认 models)")
    p_pull.add_argument("--filename", default=None, help="指定文件名 (可选)")
    p_pull.add_argument("-s", "--session", default="pull-worker", help="临时 CPU 会话名称 (默认 pull-worker)")
    p_pull.add_argument("--json", action="store_true", help="输出结构化 JSON (Agent 专用)")

    # down
    p_down = sub.add_parser("down", help="安全备份输出至云盘并拉闸关机（含网页端孤儿实例），停止扣费")
    p_down.add_argument("-s", "--session", default=None, help="指定会话名或节点前缀")
    p_down.add_argument("--all", action="store_true", help="关闭账号下所有运行中的 Colab 机器")
    p_down.add_argument("--json", action="store_true", help="以结构化 JSON 返回关机结果与剩余算力 (Agent 专用)")

    # test-alert
    sub.add_parser("test-alert", help="测试低负载多通道告警（Mac 系统通知 + 语音播报 + Webhook）")

    args = parser.parse_args()
    if args.cmd in (None, "status"):
        ccu, vms = inspect_all_vms()
        if getattr(args, "json", False):
            sync_dashboard_json(ccu, vms)
            print(json.dumps(build_agent_status_payload(ccu, vms), indent=2, ensure_ascii=False))
        else:
            render_dashboard(ccu, vms)
    elif args.cmd == "init":
        cmd_init(args)
    elif args.cmd == "watch":
        watch_loop(args)
    elif args.cmd == "up":
        cmd_up(args)
    elif args.cmd == "exec":
        cmd_exec(args)
    elif args.cmd == "pull":
        cmd_pull(args)
    elif args.cmd == "down":
        cmd_down(args)
    elif args.cmd == "test-alert":
        cfg = load_config()
        send_alert(
            title="Colab A100 低负载预警 (测试)",
            message="实例 [ultra] (A100) GPU 利用率已连续低于 5% (当前 0.0%, 功耗 44W)。若继续无负载将在 15 分钟后自动拉闸关机！",
            cfg=cfg,
            voice_text="注意，Colab A100 机器负载过低，请检查任务状态。",
        )


if __name__ == "__main__":
    main()
