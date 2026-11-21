#!/usr/bin/env python3
"""Low-load monitoring and the symlink-compatible command entry point."""
if __name__ == "__main__":
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, root)
    try:
        import colab_cli  # noqa: F401
    except ImportError:
        python = os.path.expanduser("~/.local/share/uv/tools/google-colab-cli/bin/python")
        if not os.path.isfile(python) or os.path.abspath(sys.executable) == python:
            raise SystemExit("Install google-colab-cli with uv tool install google-colab-cli")
        os.execv(python, [python, sys.argv[0], *sys.argv[1:]])
    from colab_ultra.cli import main
    main()
    raise SystemExit

import time

from rich.console import Console
from rich.panel import Panel

from colab_ultra.alerts import (
    log_breaker_event,
    send_alert,
)
from colab_ultra.config import load_config
from colab_ultra.drive import detect_local_drive_hub

console = Console()


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
    from colab_ultra.telemetry import inspect_all_vms, render_dashboard

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
                        from colab_ultra.session import teardown_verified
                        teardown_verified(vm, detect_local_drive_hub(), cfg.get("auto_backup_to_drive_before_stop", True))
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
