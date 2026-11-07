#!/usr/bin/env python3
"""
Colab Ultra — High-RAM Multi-Core CPU Data Processing Workflow
Designed for:
  - Standard CPU (2 vCPU / 12 GiB RAM, 0.08 CU/h)
  - High-RAM CPU (8 vCPU AMD EPYC / 50 GiB RAM, 0.26 CU/h)  <-- Recommended (`colab-ultra up --cpu`)
  - 5x Parallel CPU Cluster (40 vCPU / 250 GiB RAM, 1.30 CU/h) (`colab-ultra up --cpu --workers 5`)
  - Monster Single-Node CPU via TPU-v6e1 Host (44 vCPU AMD EPYC Zen4 / 172 GiB RAM, 5.09 CU/h) (`colab-ultra up --tpu v6e1`)
"""
import concurrent.futures
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time


def get_cpu_model() -> str:
    try:
        out = subprocess.check_output("lscpu | grep 'Model name:'", shell=True, text=True)
        return out.split(":", 1)[1].strip()
    except Exception:
        return platform.processor() or "Unknown CPU"


def get_ram_gb() -> tuple[float, float]:
    try:
        mem = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    mem[k.strip()] = int(v.strip().split()[0])
        total_gb = round(mem.get("MemTotal", 0) / (1024 * 1024), 1)
        avail_gb = round(mem.get("MemAvailable", 0) / (1024 * 1024), 1)
        return total_gb, avail_gb
    except Exception:
        return 0.0, 0.0


def _worker_shard_task(shard_id: int, num_records: int = 250_000) -> dict:
    t0 = time.perf_counter()
    h = hashlib.sha256()
    valid_count = 0
    for i in range(num_records):
        raw = f"shard={shard_id}|rec={i}|payload={i * 2654435761 % 1000003}".encode("utf-8")
        digest = hashlib.blake2b(raw, digest_size=16).digest()
        h.update(digest)
        if digest[0] < 200:
            valid_count += 1
    elapsed = time.perf_counter() - t0
    return {
        "shard_id": shard_id,
        "records_processed": num_records,
        "records_kept": valid_count,
        "checksum": h.hexdigest()[:16],
        "elapsed_sec": round(elapsed, 3),
    }


def main():
    cores = os.cpu_count() or 1
    cpu_model = get_cpu_model()
    ram_total_gb, ram_avail_gb = get_ram_gb()
    disk_free_gb = round(shutil.disk_usage("/content").free / (1024 ** 3), 1)
    drive_mounted = os.path.isdir("/content/drive/MyDrive")

    print("=" * 68)
    print(f"🖥️  Colab Ultra CPU Data Node: {cpu_model}")
    print(f"⚙️  vCPU Cores : {cores} cores")
    print(f"🧠 System RAM : {ram_avail_gb:.1f} GB available / {ram_total_gb:.1f} GB total")
    print(f"💾 Local Disk : {disk_free_gb:.1f} GB free | Google Drive Mounted: {drive_mounted}")
    print("=" * 68)

    num_shards = cores * 2
    records_per_shard = 250_000
    total_records = num_shards * records_per_shard

    print(f"🚀 Running parallel CPU data pipeline: {num_shards} shards ({total_records:,} records) across {cores} vCPUs...")
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as pool:
        futures = [pool.submit(_worker_shard_task, sid, records_per_shard) for sid in range(num_shards)]
        results = [f.result() for f in futures]
    total_sec = time.perf_counter() - t0
    throughput = int(total_records / max(total_sec, 1e-6))

    out_dir = "/content/outputs" if os.path.isdir("/content/outputs") else "/content"
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "cpu_data_summary.json")
    summary = {
        "cpu_model": cpu_model,
        "vcpu_cores": cores,
        "ram_total_gb": ram_total_gb,
        "disk_free_gb": disk_free_gb,
        "drive_mounted": drive_mounted,
        "total_records_processed": total_records,
        "elapsed_sec": round(total_sec, 3),
        "throughput_records_per_sec": throughput,
        "shards": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Completed {total_records:,} records in {total_sec:.2f}s ({throughput:,} records/sec)")
    print(f"📦 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
