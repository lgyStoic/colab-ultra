#!/usr/bin/env python3
"""
Colab VM Hardware & Workload Telemetry Probe
Runs inside the remote Google Colab VM as a lightweight background daemon.
Periodically samples GPU (nvidia-smi), CPU, RAM, Disk, Google Drive mount status,
and custom task heartbeats, writing atomically to /content/.colab_telemetry.json.
"""
import glob
import json
import os
import subprocess
import time
from datetime import datetime, timezone

TELEMETRY_PATH = "/content/.colab_telemetry.json"
HEARTBEAT_PATH = "/content/.task_heartbeat.json"
DRIVE_ROOT = "/content/drive/MyDrive"


def sample_gpu():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True, timeout=3).strip()
        if not out:
            return {"available": False}

        def _flt(v):
            try:
                return float(v)
            except Exception:
                return 0.0

        cards = []
        for idx, line in enumerate(out.splitlines()):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            cards.append({
                "index": idx,
                "name": parts[0],
                "gpu_util_pct": _flt(parts[1]),
                "mem_util_pct": _flt(parts[2]),
                "vram_used_mib": _flt(parts[3]),
                "vram_total_mib": _flt(parts[4]),
                "temp_c": _flt(parts[5]),
                "power_draw_w": _flt(parts[6]),
                "power_limit_w": _flt(parts[7]),
            })
        if not cards:
            return {"available": False}

        # Also query active compute apps on GPU
        proc_cmd = [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
        proc_out = subprocess.check_output(proc_cmd, text=True, timeout=3).strip()
        processes = []
        if proc_out:
            for pline in proc_out.splitlines():
                pp = [x.strip() for x in pline.split(",")]
                if len(pp) >= 3:
                    processes.append({
                        "pid": int(pp[0]) if pp[0].isdigit() else pp[0],
                        "name": pp[1],
                        "vram_mib": _flt(pp[2]),
                    })

        n_cards = len(cards)
        agg_name = f"{cards[0]['name']} x{n_cards}" if n_cards > 1 else cards[0]["name"]
        return {
            "available": True,
            "gpu_count": n_cards,
            "name": agg_name,
            "gpu_util_pct": round(max(c["gpu_util_pct"] for c in cards), 1),
            "gpu_util_avg_pct": round(sum(c["gpu_util_pct"] for c in cards) / n_cards, 1),
            "mem_util_pct": round(max(c["mem_util_pct"] for c in cards), 1),
            "vram_used_mib": round(sum(c["vram_used_mib"] for c in cards), 1),
            "vram_total_mib": round(sum(c["vram_total_mib"] for c in cards), 1),
            "temp_c": round(max(c["temp_c"] for c in cards), 1),
            "power_draw_w": round(sum(c["power_draw_w"] for c in cards), 1),
            "power_limit_w": round(sum(c["power_limit_w"] for c in cards), 1),
            "max_card_power_w": round(max(c["power_draw_w"] for c in cards), 1),
            "max_card_power_limit_w": round(max(c["power_limit_w"] for c in cards), 1),
            "gpus": cards,
            "processes": processes,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


_PREV_IO_BYTES = None
_PREV_IO_TS = None


def _sample_proc_io_mb_s() -> float:
    global _PREV_IO_BYTES, _PREV_IO_TS
    total_bytes = 0
    now = time.time()
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/io", "r") as f:
                    for line in f:
                        if line.startswith(("rchar:", "wchar:")):
                            total_bytes += int(line.split(":")[1].strip())
            except Exception:
                pass
    except Exception:
        return 0.0
    if _PREV_IO_BYTES is None or _PREV_IO_TS is None or now <= _PREV_IO_TS:
        _PREV_IO_BYTES, _PREV_IO_TS = total_bytes, now
        return 0.0
    dt = now - _PREV_IO_TS
    delta = max(0, total_bytes - _PREV_IO_BYTES)
    _PREV_IO_BYTES, _PREV_IO_TS = total_bytes, now
    return round((delta / dt) / (1024 * 1024), 2)


def sample_cpu_ram():
    cpu_pct = 0.0
    load_1m, load_5m, load_15m = 0.0, 0.0, 0.0
    try:
        with open("/proc/loadavg", "r") as f:
            la = f.read().split()
            load_1m, load_5m, load_15m = float(la[0]), float(la[1]), float(la[2])
        ncpu = os.cpu_count() or 1
        cpu_pct = min(100.0, round((load_1m / ncpu) * 100.0, 1))
    except Exception:
        pass

    ram_total_gb, ram_used_gb = 0.0, 0.0
    try:
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    meminfo[k.strip()] = int(v.strip().split()[0])
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        ram_total_gb = round(total_kb / (1024 * 1024), 2)
        ram_used_gb = round((total_kb - avail_kb) / (1024 * 1024), 2)
    except Exception:
        pass

    return {
        "cpu_util_est_pct": cpu_pct,
        "load_1m": load_1m,
        "load_5m": load_5m,
        "load_15m": load_15m,
        "cpu_cores": os.cpu_count() or 1,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "io_mb_s": _sample_proc_io_mb_s(),
    }


def sample_drive():
    mounted = os.path.ismount("/content/drive") or os.path.isdir(DRIVE_ROOT)
    if not mounted:
        return {"mounted": False}
    try:
        target = "/content/drive" if os.path.exists("/content/drive") else DRIVE_ROOT
        st = os.statvfs(target)
        total_gb = round((st.f_blocks * st.f_frsize) / (1024 ** 3), 1)
        free_gb = round((st.f_bavail * st.f_frsize) / (1024 ** 3), 1)
        total_tb = round(total_gb / 1024.0, 2)
        free_tb = round(free_gb / 1024.0, 2)
        used_tb = round(total_tb - free_tb, 2)
        is_kaggle_sync = os.path.isdir("/kaggle/working") and not os.path.ismount("/content/drive")
        return {
            "mounted": True,
            "mode": "kaggle_synced" if is_kaggle_sync else "fuse",
            "total_tb": total_tb,
            "used_tb": used_tb,
            "free_tb": free_tb,
            "total_gb": total_gb,
            "free_gb": free_gb,
        }
    except Exception as e:
        return {"mounted": True, "error": str(e)}


def sample_task_heartbeat():
    if not os.path.exists(HEARTBEAT_PATH):
        return None
    try:
        with open(HEARTBEAT_PATH, "r") as f:
            data = json.load(f)
        mtime = os.path.getmtime(HEARTBEAT_PATH)
        data["age_seconds"] = round(time.time() - mtime, 1)
        return data
    except Exception:
        return None


def sample_training_logs():
    candidates = []
    # Never use recursive '**' on /content or /content/drive (which walks the entire 20TB Google Drive FUSE mount!)
    patterns = [
        "/content/drive/MyDrive/ColabUltra/*/runs/*/log.jsonl",
        "/content/drive/MyDrive/ColabUltra/*/runs/log.jsonl",
        "/content/drive/MyDrive/ColabUltra/*_runs/*/log.jsonl",
        "/content/outputs/*/log.jsonl",
        "/content/*/log.jsonl",
    ]
    for pat in patterns:
        for p in glob.glob(pat, recursive=False):
            if os.path.isfile(p):
                try:
                    candidates.append((os.path.getmtime(p), p))
                except Exception:
                    pass
    if not candidates:
        return None
    candidates.sort(reverse=True)
    mtime, log_path = candidates[0]
    run_dir = os.path.dirname(log_path)
    run_name = os.path.basename(run_dir)

    total_it = None
    wandb_url = None
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            total_it = cfg.get("iterations")
            wandb_url = cfg.get("wandb_url")
        except Exception:
            pass
    wb_txt = os.path.join(run_dir, "wandb_url.txt")
    if not wandb_url and os.path.isfile(wb_txt):
        try:
            with open(wb_txt, "r", encoding="utf-8") as f:
                wandb_url = f.read().strip()
        except Exception:
            pass

    lines = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = [ln.strip() for ln in f if ln.strip()]
        lines = raw_lines[-30:]
    except Exception:
        return None

    latest_rec = {}
    if lines:
        try:
            latest_rec = json.loads(lines[-1])
        except Exception:
            latest_rec = {}

    it = latest_rec.get("it")
    s_per_it = latest_rec.get("s_per_it")
    pct = round((float(it) / float(total_it)) * 100.0, 1) if (it and total_it) else None
    eta_min = round(((float(total_it) - float(it)) * float(s_per_it)) / 60.0, 1) if (it and total_it and s_per_it and total_it >= it) else None

    samples = []
    sample_dir = os.path.join(run_dir, "samples")
    if os.path.isdir(sample_dir):
        try:
            s_files = [
                (os.path.getmtime(os.path.join(sample_dir, fn)), fn)
                for fn in os.listdir(sample_dir)
                if fn.endswith((".jpg", ".png"))
            ]
            s_files.sort(reverse=True)
            samples = [fn for _, fn in s_files[:6]]
        except Exception:
            pass

    return {
        "run_name": run_name,
        "log_path": log_path,
        "wandb_url": wandb_url,
        "updated_at": datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "updated_ago_sec": round(max(0.0, time.time() - mtime), 1),
        "it": it,
        "total_it": total_it,
        "pct": pct,
        "s_per_it": s_per_it,
        "eta_min": eta_min,
        "latest": latest_rec,
        "samples": samples,
        "recent_lines": lines,
    }


_LAST_DRIVE = {"mounted": False}
_LAST_TRAINING = None
_LAST_SLOW_TS = 0.0


def _update_slow_samples():
    global _LAST_DRIVE, _LAST_TRAINING
    try:
        _LAST_DRIVE = sample_drive()
    except Exception:
        pass
    try:
        _LAST_TRAINING = sample_training_logs()
    except Exception:
        pass


def collect_once():
    global _LAST_SLOW_TS
    import threading

    now = time.time()
    if now - _LAST_SLOW_TS >= 9.0:
        _LAST_SLOW_TS = now
        t = threading.Thread(target=_update_slow_samples, daemon=True)
        t.start()
        t.join(timeout=1.5)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "unix_time": time.time(),
        "gpu": sample_gpu(),
        "system": sample_cpu_ram(),
        "drive": _LAST_DRIVE,
        "task_heartbeat": sample_task_heartbeat(),
        "training": _LAST_TRAINING,
    }
    tmp_path = TELEMETRY_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, TELEMETRY_PATH)
    if os.path.isdir("/kaggle/working"):
        try:
            k_path = "/content/_cu_telemetry.json"
            with open(k_path + ".tmp", "w") as kf:
                json.dump(payload, kf)
            os.replace(k_path + ".tmp", k_path)
        except Exception:
            pass
    return payload


def main():
    import sys
    if "--once" in sys.argv:
        data = collect_once()
        print(json.dumps(data))
        return

    while True:
        try:
            collect_once()
        except Exception:
            pass
        time.sleep(3)


if __name__ == "__main__":
    main()
