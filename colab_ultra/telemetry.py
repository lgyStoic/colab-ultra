import base64
import json
import os
import subprocess
import time
from datetime import datetime

import requests
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from colab_ultra.backends.colab import (
    colab_state,
    fetch_vm_kernels,
)
from colab_ultra.backends.jupyter import jupyter_exec_ws
from colab_ultra.backends.kaggle import (
    get_kaggle_quota,
    list_active_kaggle_vms,
)
from colab_ultra.breaker import evaluate_vm_load
from colab_ultra.config import (
    PROBE_LOCAL_PATH,
    load_config,
)
from colab_ultra.drive import detect_local_drive_hub
from colab_ultra.retry import retry

console = Console()


def start_remote_probe_via_runtime(proxy_url: str, proxy_token: str):
    kernels = fetch_vm_kernels(proxy_url, proxy_token)
    kid = kernels[0].get("id") if kernels else None
    boot_code = (
        "import subprocess\n"
        "subprocess.run(['python3', '/content/.colab_probe.py', '--once'], check=False)\n"
        "subprocess.Popen(['python3', '/content/.colab_probe.py'], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
    )
    jupyter_exec_ws(proxy_url, boot_code, proxy_token=proxy_token, kernel_id=kid, timeout=12)


def upload_file_to_vm(proxy_url: str, proxy_token: str, local_path: str, remote_path: str):
    """Upload a local probe or workflow through the shared contents transport."""
    from colab_ultra.backends.jupyter import jupyter_put_bytes
    with open(local_path, "rb") as source:
        jupyter_put_bytes(proxy_url, proxy_token, remote_path, source.read(), timeout=30)


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
            r = retry(lambda: requests.get(url, params=params, headers=headers, timeout=15), attempts=2)
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
