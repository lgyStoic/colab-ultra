"""Run after colab-ultra sync-remote.

Use colab-ultra exec -s SESSION -p redis_mvp -f recipes/redis_mvp_worker.py.
Set COLAB_ULTRA_REDIS_URL on the worker or put the URL in code/.redis_url.
For Upstash REST, use an HTTPS URL with the token in its userinfo.
Set queue_name, worker_id, batch_size, reclaim_idle_ms or simulate_crash_slug
in code/mvp_worker.json, or override each with its COLAB_ULTRA_* environment variable.
"""

import argparse
import importlib
import importlib.util
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CRASH_DIR = Path("/tmp")


def _install_redis():
    if importlib.util.find_spec("redis") is not None:
        return
    uv = shutil.which("uv")
    if uv:
        result = subprocess.run(
            [uv, "pip", "install", "--system", "-q", "--python", sys.executable, "redis>=5.0"]
        )
        if result.returncode == 0:
            importlib.invalidate_caches()
            return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "redis>=5.0"], check=True)
    importlib.invalidate_caches()


def _load_queue():
    try:
        from cu_remote.queue import TaskQueue
    except ModuleNotFoundError as exc:
        if exc.name != "cu_remote":
            raise
        if Path(sys.argv[0]).name == "redis_mvp_worker.py":
            sys.path.insert(0, str(Path(sys.argv[0]).resolve().parents[1]))
        from colab_ultra.remote.queue import TaskQueue
    return TaskQueue


def _settings(code):
    path = code / "mvp_worker.json"
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    defaults = {
        "queue_name": "redis_mvp",
        "worker_id": os.environ.get("COLAB_ULTRA_SESSION") or socket.gethostname(),
        "batch_size": 2,
        "reclaim_idle_ms": 120_000,
        "simulate_crash_slug": "",
    }
    settings = {**defaults, **{key: cfg[key] for key in defaults if key in cfg}}
    for key in settings:
        value = os.environ.get(f"COLAB_ULTRA_{key.upper()}")
        if value:
            settings[key] = value
    for key in ("batch_size", "reclaim_idle_ms"):
        settings[key] = int(settings[key])
    return settings


def _safe_name(value):
    return (isinstance(value, str) and value not in {"", ".", ".."}
            and "/" not in value and "\\" not in value and "\x00" not in value)


def _write_output(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    output = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False,
    )
    temp = Path(output.name)
    try:
        with output:
            json.dump(result, output)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    os.sync()


def _run(queue, settings, run_dir):
    worker = settings["worker_id"]
    empty_polls = 0
    while True:
        tasks = queue.claim_batch(
            worker, settings["batch_size"], reclaim_idle_ms=settings["reclaim_idle_ms"],
        )
        if not tasks:
            empty_polls += 1
            if empty_polls >= 2 and queue.status()["unclaimed"] == 0:
                queue.set_worker_status(worker, "stopped")
                print(f"[Worker {worker}] queue empty, exiting", flush=True)
                return 0
            continue
        empty_polls = 0
        slugs = ", ".join(task["slug"] for _, task in tasks)
        print(f"[Worker {worker}] claimed {len(tasks)}: {slugs}", flush=True)
        for msg_id, task in tasks:
            slug = task["slug"]
            try:
                sleep_sec = float(task.get("sleep_sec", 2.0))
                valid_sleep = math.isfinite(sleep_sec) and sleep_sec >= 0
            except (TypeError, ValueError, OverflowError):
                valid_sleep = False
            if not _safe_name(slug) or not valid_sleep:
                queue.fail(msg_id, slug, worker, "Invalid slug or sleep_sec")
                continue
            for i in range(1, 5):
                queue.report_progress(slug, worker, f"step {i}/4")
                time.sleep(sleep_sec / 4)
            try:
                if slug == settings["simulate_crash_slug"]:
                    try:
                        (CRASH_DIR / f"crashed_{slug}").touch(exist_ok=False)
                    except FileExistsError:
                        pass
                    else:
                        print(f"[CrashSim] {slug} left pending for reclaim", flush=True)
                        return 1
                path = run_dir / f"{slug}.json"
                _write_output(path, {"slug": slug, "worker": worker, "status": "done"})
            except OSError as exc:
                queue.fail(msg_id, slug, worker, f"Output failed ({type(exc).__name__})")
                continue
            queue.ack(msg_id, slug, worker, output=str(path))
            print(f"[Worker {worker}] done {slug} -> {path}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulate tasks from a Redis Streams queue")
    parser.add_argument("--dry-run", action="store_true", help="Read settings without connecting")
    args = parser.parse_args(argv)
    cwd = Path.cwd()
    code = cwd if cwd.name == "code" else cwd / "code"
    try:
        settings = _settings(code)
        if not _safe_name(settings["queue_name"]):
            raise SystemExit("queue_name must be a single directory name")
        if not isinstance(settings["worker_id"], str):
            raise SystemExit("worker_id must be a string")
        if settings["batch_size"] < 1 or settings["reclaim_idle_ms"] < 0:
            raise SystemExit("batch_size must be positive and reclaim_idle_ms must be non-negative")
        run_dir = code.parent / "runs" / settings["queue_name"]
        if args.dry_run or os.environ.get("CU_DRY_RUN") == "1":
            print("Dry run. Would poll Redis, simulate four steps per task, and write JSON outputs.")
            return 0
        redis_url = os.environ.get("COLAB_ULTRA_REDIS_URL", "").strip()
        if not redis_url and (code / ".redis_url").is_file():
            redis_url = (code / ".redis_url").read_text(encoding="utf-8").strip()
        if not redis_url:
            raise SystemExit("Set COLAB_ULTRA_REDIS_URL or put the Redis URL in code/.redis_url")
        if not redis_url.startswith(("https://", "http://")):
            _install_redis()
        queue = _load_queue()(redis_url, settings["queue_name"])
        return _run(queue, settings, run_dir)
    except Exception as exc:
        raise SystemExit(f"Redis MVP worker failed ({type(exc).__name__})") from None


if __name__ == "__main__":
    # The Colab exec wrapper evaluates this source in a running kernel.
    argv = sys.argv[1:] if Path(sys.argv[0]).name == "redis_mvp_worker.py" else []
    result = main(argv)
    if result:
        raise SystemExit(result)
