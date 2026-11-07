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
    "alert_after_minutes": 5.0,
    "stop_after_minutes": 10.0,
    "enable_macos_notification": True,
    "enable_voice_alert": True,
    "webhook_url": "",  # Optional Bark / Feishu / Telegram / Custom webhook URL
    "auto_backup_to_drive_before_stop": True,
    "drive_hub_root": "/content/drive/MyDrive/ColabUltra",
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
    rt = ColabRuntime(proxy_url, proxy_token)
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
    payload = {
        "name": os.path.basename(remote_path),
        "path": remote_path.lstrip("/"),
        "type": "file",
        "format": "base64",
        "content": b64_content,
        "chunk": 1,
    }
    resp = requests.put(url, params=params, json=payload, timeout=8)
    resp.raise_for_status()


def fetch_vm_kernels(proxy_url: str, proxy_token: str) -> list:
    url = f"{proxy_url.rstrip('/')}/api/kernels"
    params = {"authuser": "0", "colab-runtime-proxy-token": proxy_token}
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def fetch_vm_telemetry(proxy_url: str, proxy_token: str, auto_deploy: bool = True) -> dict:
    """Fetch /content/.colab_telemetry.json from the VM; deploy probe if missing or stale."""
    url = f"{proxy_url.rstrip('/')}/api/contents/content/.colab_telemetry.json"
    params = {"content": "1", "authuser": "0", "colab-runtime-proxy-token": proxy_token}

    def _read():
        try:
            r = requests.get(url, params=params, timeout=5)
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
    """Query account CCU info and all active VMs with full hardware telemetry."""
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
            "endpoint": a.endpoint,
            "accelerator": a.accelerator.value,
            "shape": a.machine_shape.name,
            "proxy_url": proxy_url,
            "proxy_token": proxy_token,
            "kernel_state": kernel_state,
            "last_activity": last_activity,
            "telemetry": telemetry,
        })

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
            "endpoint": ep,
            "accelerator": vm["accelerator"],
            "shape": vm["shape"],
            "kernel_state": vm["kernel_state"],
            "gpu_available": bool(gpu.get("available")),
            "gpu_name": gpu.get("name") or vm["accelerator"],
            "gpu_util_pct": round(float(gpu.get("gpu_util_pct", 0.0)), 1) if gpu.get("available") else 0.0,
            "vram_used_mib": round(float(gpu.get("vram_used_mib", 0.0)), 0) if gpu.get("available") else 0.0,
            "vram_total_mib": round(float(gpu.get("vram_total_mib", 0.0)), 0) if gpu.get("available") else 0.0,
            "power_draw_w": round(float(gpu.get("power_draw_w", 0.0)), 1) if gpu.get("available") else 0.0,
            "power_limit_w": round(float(gpu.get("power_limit_w", 0.0)), 1) if gpu.get("available") else 0.0,
            "temp_c": round(float(gpu.get("temp_c", 0.0)), 0) if gpu.get("available") else 0.0,
            "cpu_util_pct": round(float(sys_info.get("cpu_util_est_pct", 0.0)), 1),
            "ram_used_gb": round(float(sys_info.get("ram_used_gb", 0.0)), 1),
            "ram_total_gb": round(float(sys_info.get("ram_total_gb", 0.0)), 1),
            "drive_mounted": bool(drive.get("mounted")),
            "drive_free_tb": round(float(drive.get("free_tb", 0.0)), 2) if drive.get("mounted") else None,
            "is_low_load": is_low,
            "status_reason": reason,
            "low_load_sec": low_sec,
            "stop_after_sec": stop_sec,
            "task_heartbeat": hb,
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
    """Render a rich terminal dashboard of CCU balance and real-time VM hardware loads."""
    sync_dashboard_json(ccu, vms, idle_tracker)
    balance = ccu.paid_compute_units_balance
    burn = ccu.consumption_rate_hourly
    rem_hours = (balance / burn) if burn > 0 else float("inf")
    rem_str = f"{rem_hours:.1f} 小时" if burn > 0 else "∞ (0 消耗休眠中)"

    header_text = (
        f"💳 [bold cyan]剩余算力 (CCU):[/bold cyan] [bold green]{balance:.2f}[/bold green] 点   "
        f"🔥 [bold cyan]当前燃烧速率:[/bold cyan] [bold {'red' if burn > 0 else 'green'}]{burn:.2f} CU/小时[/]   "
        f"⏳ [bold cyan]预计可跑:[/bold cyan] {rem_str}   "
        f"🖥️ [bold cyan]在线机器:[/bold cyan] {len(vms)} 台"
    )
    console.print(Panel(header_text, title="⚡ Colab Ultra 智能电闸与算力监控台", border_style="cyan"))

    if not vms:
        console.print("[dim]😴 当前没有任何运行中的 Colab 虚拟机，算力电闸处于【断电休眠】状态（0 扣费）。[/dim]")
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

        cpu_pct = sys_info.get("cpu_util_est_pct", 0.0)
        ram_u = sys_info.get("ram_used_gb", 0.0)
        ram_t = sys_info.get("ram_total_gb", 0.0)
        cpu_str = f"{cpu_pct:.0f}% | {ram_u:.1f}/{ram_t:.0f}G"

        if drive.get("mounted"):
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

        table.add_row(
            f"{vm['name']}\n[dim]{vm['endpoint'][:18]}..[/dim]",
            f"{vm['accelerator']}\n[dim]{vm['shape']}[/dim]",
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

    gpu_util = gpu.get("gpu_util_pct", 0.0) if gpu.get("available") else 0.0
    vram_used_gb = (gpu.get("vram_used_mib", 0.0) / 1024.0) if gpu.get("available") else 0.0
    power_w = gpu.get("power_draw_w", 0.0) if gpu.get("available") else 0.0
    power_lim = gpu.get("power_limit_w", 0.0) if gpu.get("available") else 0.0
    cpu_util = sys_info.get("cpu_util_est_pct", 0.0)
    k_state = vm.get("kernel_state", "idle")

    min_active_watts = 75.0 if power_lim >= 250.0 else 28.0
    power_is_active = (power_w >= min_active_watts) if power_w > 0 else False
    power_is_dead_idle = (power_w > 0 and power_w < min_active_watts)

    # 1. Active GPU compute verified by physical power draw (Watts)
    if power_is_active:
        return False, f"显卡高功耗真负载 ({power_w:.0f}W, GPU {gpu_util:.1f}%)"
    if not power_is_dead_idle and gpu_util >= cfg["gpu_min_util_pct"]:
        return False, f"GPU 活跃计算中 ({gpu_util:.1f}%)"

    # 2. Active CPU / Disk I/O only when kernel is BUSY and CPU >= 55%
    if k_state == "busy" and cpu_util >= 55.0:
        return False, f"CPU/云盘读写活跃 (CPU {cpu_util:.1f}%, GPU {power_w:.0f}W)"

    # 3. Fresh custom task heartbeat (< 30s)
    if hb and hb.get("age_seconds", 999) < 30 and hb.get("active", True):
        return False, f"任务心跳活跃 ({hb.get('task', 'job')})"

    # 4. Otherwise, it IS in a low-load / idle wattage state! Classify why:
    if power_is_dead_idle and gpu_util >= 20.0:
        return True, f"疑似通信死锁假忙 (GPU显示{gpu_util:.0f}% 但真实功耗仅 {power_w:.0f}W < {min_active_watts:.0f}W)"
    elif vram_used_gb >= 3.0 and k_state == "busy":
        return True, f"疑似进程卡死/死锁 (功耗仅 {power_w:.0f}W, 显存 {vram_used_gb:.1f}GB, 内核 BUSY)"
    elif vram_used_gb >= 3.0 and k_state == "idle":
        return True, f"模型驻留但空载待机 (功耗仅 {power_w:.0f}W, 显存 {vram_used_gb:.1f}GB)"
    else:
        return True, f"待机底噪空转 (功耗 {power_w:.0f}W < {min_active_watts:.0f}W, GPU {gpu_util:.1f}%)"


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

                    # Stage 1: Alert
                    if low_sec >= alert_sec and not idle_tracker[ep]["alerted"]:
                        idle_tracker[ep]["alerted"] = True
                        remain_min = max(0.0, (stop_sec - low_sec) / 60.0)
                        msg = (
                            f"实例 [{vm['name']}] ({vm['accelerator']}) 已连续低负载 {low_sec/60:.1f} 分钟！\n"
                            f"原因: {reason}\n"
                            f"当前消耗: {ccu.consumption_rate_hourly:.2f} CU/h | 剩余额度: {ccu.paid_compute_units_balance:.2f} CU\n"
                            f"若继续无负载，将在 {remain_min:.1f} 分钟后自动拉闸关机！"
                        )
                        send_alert(
                            title=f"Colab {vm['accelerator']} 低负载预警",
                            message=msg,
                            cfg=cfg,
                            voice_text=f"注意，Colab {vm['accelerator']} 机器负载过低，请检查任务状态。",
                        )

                    # Stage 2: Auto-Breaker Trip (Shut down VM)
                    if low_sec >= stop_sec:
                        send_alert(
                            title=f"🔌 Colab {vm['accelerator']} 智能电闸已拉闸",
                            message=(
                                f"实例 [{vm['name']}] ({vm['accelerator']}) 连续低负载达 {low_sec/60:.1f} 分钟 ({reason})。\n"
                                f"正在执行安全备份并强制释放机器，停止算力扣费！"
                            ),
                            cfg=cfg,
                            voice_text=f"Colab {vm['accelerator']} 空闲超时，智能电闸已自动拉闸关机。",
                        )
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
        clean_vms.append({
            "name": vm["name"],
            "endpoint": vm["endpoint"],
            "accelerator": vm["accelerator"],
            "shape": vm["shape"],
            "kernel_state": vm["kernel_state"],
            "is_low_load": is_low,
            "status_reason": reason,
            "power_draw_w": gpu.get("power_draw_w", 0.0),
            "power_limit_w": gpu.get("power_limit_w", 0.0),
            "gpu_util_pct": gpu.get("gpu_util_pct", 0.0),
            "vram_used_mib": gpu.get("vram_used_mib", 0.0),
            "vram_total_mib": gpu.get("vram_total_mib", 0.0),
            "temp_c": gpu.get("temp_c", 0.0),
            "cpu_util_pct": sys_info.get("cpu_util_est_pct", 0.0),
            "ram_used_gb": sys_info.get("ram_used_gb", 0.0),
            "ram_total_gb": sys_info.get("ram_total_gb", 0.0),
            "drive_mounted": bool(drive.get("mounted")),
            "drive_free_tb": drive.get("free_tb"),
            "workspace_path": "/content/workspace",
            "outputs_path": "/content/outputs",
        })

    return {
        "ok": True,
        "paid_balance": round(balance, 2),
        "burn_rate_hourly": round(burn, 2),
        "estimated_hours_left": rem_hours,
        "active_vms_count": len(clean_vms),
        "vms": clean_vms,
    }


def cmd_up(args):
    """Spin up a Colab session, deploy telemetry probe, and upload Drive Hub initializer."""
    colab_bin = resolve_colab_bin()
    gpu_choice = (args.gpu or "T4").upper()
    cmd = [colab_bin, "new", "-s", args.session]
    if gpu_choice not in ("CPU", "NONE", "OFF"):
        cmd.extend(["--gpu", gpu_choice])
    if args.high_mem:
        cmd.append("--high-mem")
    if not getattr(args, "json", False):
        console.print(f"⚡ [bold cyan]正在合闸启动 Colab {gpu_choice} 实例 ({args.session})...[/bold cyan]")
    subprocess.check_call(
        cmd,
        stdout=subprocess.DEVNULL if getattr(args, "json", False) else None,
        stderr=subprocess.DEVNULL if getattr(args, "json", False) else None,
    )

    # Auto-mount 20TB Google Drive and run init_20tb_drive.py by default
    if not getattr(args, "no_drive", False):
        if not getattr(args, "json", False):
            console.print(f"📦 [cyan]正在默认自动挂载 20TB Google Drive 并初始化永久工作区 ({args.session})...[/cyan]")
        try:
            subprocess.call(
                [colab_bin, "drivemount", "-s", args.session],
                stdout=subprocess.DEVNULL if getattr(args, "json", False) else None,
            )
            init_script = os.path.join(BASE_DIR, "workflows", "init_20tb_drive.py")
            if os.path.exists(init_script):
                subprocess.call(
                    [colab_bin, "exec", "-s", args.session, "-f", init_script],
                    stdout=subprocess.DEVNULL if getattr(args, "json", False) else None,
                )
                subprocess.call(
                    [colab_bin, "exec", "-s", args.session, "python3 /content/.colab_probe.py --once"],
                    stdout=subprocess.DEVNULL if getattr(args, "json", False) else None,
                )
        except Exception as e:
            if not getattr(args, "json", False):
                console.print(f"[yellow]⚠️ 自动挂载云盘遇到提示: {e}[/yellow]")

    # Deploy telemetry probe and refresh dashboard
    ccu, vms = inspect_all_vms()
    for vm in vms:
        if vm["name"] == args.session:
            init_script = os.path.join(BASE_DIR, "workflows", "init_20tb_drive.py")
            if os.path.exists(init_script):
                upload_file_to_vm(vm["proxy_url"], vm["proxy_token"], init_script, "content/init_20tb_drive.py")
    if getattr(args, "json", False):
        sync_dashboard_json(ccu, vms)
        print(json.dumps(build_agent_status_payload(ccu, vms), indent=2, ensure_ascii=False))
    else:
        render_dashboard(ccu, vms)
        console.print(f"[green]✅ 实例 [{args.session}] 已就绪（硬件探针 + 20TB 云盘永久工作区已激活）！[/green]")


def cmd_exec(args):
    """Agent-First remote code/script execution with automatic post-run hardware telemetry."""
    colab_bin = resolve_colab_bin()
    cmd = [colab_bin, "exec", "-s", args.session]
    if args.file:
        cmd.extend(["-f", args.file])
    elif args.code:
        cmd.append(args.code)
    else:
        raise SystemExit("Provide either inline code or -f <script.py>")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=getattr(args, "json", False), text=True)
    elapsed = round(time.time() - t0, 2)

    if getattr(args, "json", False):
        ccu, vms = inspect_all_vms()
        sync_dashboard_json(ccu, vms)
        payload = build_agent_status_payload(ccu, vms)
        payload.update({
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "elapsed_sec": elapsed,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        })
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif proc.returncode != 0:
        sys.exit(proc.returncode)


def cmd_down(args):
    """Safely back up outputs to Drive (if mounted) and shut down one or all Colab VMs."""
    cfg = load_config()
    ccu, vms = inspect_all_vms()
    if not vms:
        if getattr(args, "json", False):
            print(json.dumps({"ok": True, "unassigned": [], "active_vms_count": 0}, indent=2, ensure_ascii=False))
        else:
            console.print("[dim]当前没有运行中的 Colab 实例。[/dim]")
        return

    targets = vms if args.all else [v for v in vms if v["name"] == args.session or v["endpoint"].startswith(args.session or "")]
    if not targets and len(vms) == 1 and not args.session:
        targets = vms

    unassigned = []
    for vm in targets:
        if not getattr(args, "json", False):
            console.print(f"🔌 [yellow]正在安全拉闸关机: [{vm['name']}] ({vm['accelerator']} - {vm['endpoint']})...[/yellow]")
        if cfg.get("auto_backup_to_drive_before_stop", True):
            backup_vm_outputs_to_drive(vm["proxy_url"], vm["proxy_token"])
        colab_state.client.unassign(vm["endpoint"])
        if vm["name"] != "web-session":
            colab_state.store.remove(vm["name"])
        log_breaker_event("MANUAL_DOWN", {"name": vm["name"], "endpoint": vm["endpoint"]})
        unassigned.append({"name": vm["name"], "endpoint": vm["endpoint"], "accelerator": vm["accelerator"]})
        if not getattr(args, "json", False):
            console.print(f"[green]✅ [{vm['name']}] 已彻底断电释放，停止算力计费！[/green]")

    try:
        ccu_after, vms_after = inspect_all_vms()
        sync_dashboard_json(ccu_after, vms_after, cfg=cfg)
        if getattr(args, "json", False):
            payload = build_agent_status_payload(ccu_after, vms_after, cfg=cfg)
            payload["unassigned"] = unassigned
            print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception:
        pass


def cmd_init(args):
    """One-stop interactive/CLI setup wizard for Colab Ultra (Upstream Patch + OAuth + Smart Breaker + Optional VPS Sentinel)."""
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
            "自动检查：0) 官方 CLI 挂载补丁与加速  1) Google Colab 凭证  2) 显卡瓦数断路器配置  3) 远端 VPS 24h 哨兵同步",
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

    adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    scopes_needed = (
        "openid,"
        "https://www.googleapis.com/auth/cloud-platform,"
        "https://www.googleapis.com/auth/userinfo.email,"
        "https://www.googleapis.com/auth/colaboratory"
    )
    if os.path.exists(adc_path) and not args.auth:
        console.print("✅ [green]步骤 1/3: 已检测到本地 Google ADC 凭证 (Colab 权限正常，云盘将在 Colab 机器内通过官方接口挂载)[/green]")
    else:
        console.print("[cyan]🔑 正在拉起 Google Colaboratory 浏览器授权...[/cyan]")
        subprocess.call(["gcloud", "auth", "application-default", "login", f"--scopes={scopes_needed}"])

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        pass
    console.print(
        f"✅ [green]步骤 2/3: 智能电闸规则已保存至 {CONFIG_PATH} (仓库外隔离存储)[/green]\n"
        f"   • 物理功耗判定: A100/H100 < 75W 视为空载 | L4/T4 < 28W 视为空载\n"
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
    parser = argparse.ArgumentParser(description="Colab Ultra: Agent-First Smart Breaker & 20TB Drive Hub for Google Colab")
    sub = parser.add_subparsers(dest="cmd")

    # init
    p_init = sub.add_parser("init", help="一键初始化向导（自动修补官方 CLI、配置凭证、瓦数断路器阈值、同步 VPS 哨兵）")
    p_init.add_argument("--auth", action="store_true", help="强制重新运行浏览器 OAuth 授权")
    p_init.add_argument("--with-drive-scope", action="store_true", help="兼容保留标志")
    p_init.add_argument("--alert-minutes", type=float, default=None, help="设置低功耗预警分钟数 (默认 5 分钟)")
    p_init.add_argument("--stop-minutes", type=float, default=None, help="设置低功耗自动拉闸分钟数 (默认 10 分钟)")
    p_init.add_argument("--vps", type=str, default=None, help="绑定远程 VPS 地址以实现 24h 离线断电保护")
    p_init.add_argument("--webhook", type=str, default=None, help="设置 Bark/飞书/Telegram 告警 Webhook URL")

    # status / top
    p_status = sub.add_parser("status", help="查看实时算力账单与所有 Colab 机器的 GPU 瓦数/CPU/云盘负载")
    p_status.add_argument("--json", action="store_true", help="输出 Agent 友好的结构化 JSON 诊断数据")

    # watch
    p_watch = sub.add_parser("watch", help="启动智能电闸后台轮询守护（低负载告警 + 超时自动拉闸）")
    p_watch.add_argument("--interval", type=int, default=None, help="轮询间隔秒数 (默认 15s)")
    p_watch.add_argument("--gpu-threshold", type=float, default=None, help="最低 GPU 利用率百分比阈值 (默认 5%%)")
    p_watch.add_argument("--alert-minutes", type=float, default=None, help="低负载多少分钟触发报警 (默认 5 分钟)")
    p_watch.add_argument("--stop-minutes", type=float, default=None, help="低负载多少分钟自动拉闸关机 (默认 10 分钟)")
    p_watch.add_argument("--once", action="store_true", help="仅执行一次巡检与阈值判定后退出")

    # up
    p_up = sub.add_parser("up", help="一键合闸开机、默认挂载 20TB 云盘并注入硬件负载探针")
    p_up.add_argument("-s", "--session", default="ultra", help="会话名称 (默认 ultra)")
    p_up.add_argument("--gpu", default="T4", help="硬件型号: T4, L4, A100, H100, CPU (默认 T4，省钱模式)")
    p_up.add_argument("--high-mem", action="store_true", default=False, help="开启高内存模式 (默认关闭以节省算力点)")
    p_up.add_argument("--no-drive", action="store_true", default=False, help="跳过自动挂载 20TB Google Drive")
    p_up.add_argument("--json", action="store_true", help="以结构化 JSON 返回实例就绪与硬件状态 (Agent 专用)")

    # exec (Agent-first remote execution + telemetry return)
    p_exec = sub.add_parser("exec", help="在 Colab 实例中执行代码或脚本，并自动返回最新 GPU 瓦数/显存/云盘状态")
    p_exec.add_argument("-s", "--session", default="ultra", help="会话名称 (默认 ultra)")
    p_exec.add_argument("-f", "--file", default=None, help="要上传并执行的本地 Python 脚本路径")
    p_exec.add_argument("code", nargs="?", default=None, help="要直接执行的 Python 代码片段")
    p_exec.add_argument("--json", action="store_true", help="输出包含 stdout/stderr 与实时硬件瓦数的 JSON (Agent 专用)")

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
