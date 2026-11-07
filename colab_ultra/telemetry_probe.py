#!/usr/bin/env python3
"""
Colab VM Hardware & Workload Telemetry Probe
Runs inside the remote Google Colab VM as a lightweight background daemon.
Periodically samples GPU (nvidia-smi), CPU, RAM, Disk, Google Drive mount status,
and custom task heartbeats, writing atomically to /content/.colab_telemetry.json.
"""
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
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        def _flt(v):
            try:
                return float(v)
            except Exception:
                return 0.0

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

        return {
            "available": True,
            "name": parts[0],
            "gpu_util_pct": _flt(parts[1]),
            "mem_util_pct": _flt(parts[2]),
            "vram_used_mib": _flt(parts[3]),
            "vram_total_mib": _flt(parts[4]),
            "temp_c": _flt(parts[5]),
            "power_draw_w": _flt(parts[6]),
            "power_limit_w": _flt(parts[7]),
            "processes": processes,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


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
    }


def sample_drive():
    mounted = os.path.ismount("/content/drive") or os.path.isdir(DRIVE_ROOT)
    if not mounted:
        return {"mounted": False}
    try:
        st = os.statvfs("/content/drive")
        total_tb = round((st.f_blocks * st.f_frsize) / (1024 ** 4), 2)
        free_tb = round((st.f_bavail * st.f_frsize) / (1024 ** 4), 2)
        used_tb = round(total_tb - free_tb, 2)
        return {
            "mounted": True,
            "total_tb": total_tb,
            "used_tb": used_tb,
            "free_tb": free_tb,
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


def collect_once():
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "unix_time": time.time(),
        "gpu": sample_gpu(),
        "system": sample_cpu_ram(),
        "drive": sample_drive(),
        "task_heartbeat": sample_task_heartbeat(),
    }
    tmp_path = TELEMETRY_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, TELEMETRY_PATH)
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
