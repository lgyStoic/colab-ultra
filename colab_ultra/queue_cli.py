"""Workstation commands for the Redis Streams queue."""

import argparse
import json
import os
import random
from pathlib import Path

from colab_ultra.config import load_config
from colab_ultra.remote.queue import TaskQueue, mask_redis_url


def add_queue_parser(sub):
    parser = sub.add_parser("queue", help="Manage a Redis Streams task queue")
    parser.add_argument("--redis-url", help="Redis URL (or COLAB_ULTRA_REDIS_URL / config)")
    actions = parser.add_subparsers(dest="action", required=True)
    ping = actions.add_parser("ping", help="Check the Redis connection")
    push = actions.add_parser("push", help="Enqueue prompts or demo tasks")
    status = actions.add_parser("status", help="Show queue and worker status")
    clear = actions.add_parser("clear", help="Delete the queue stream and status")
    for action in (ping, push, status, clear):
        action.add_argument("--redis-url", default=argparse.SUPPRESS, help="Override the Redis URL")
    for action in (push, status, clear):
        action.add_argument("name")
    source = push.add_mutually_exclusive_group()
    source.add_argument("--file", type=Path, help="Read prompts from this file (default prompts.json)")
    source.add_argument("--demo", type=int, metavar="N", help="Generate N demo tasks")
    status.add_argument("--json", action="store_true", help="Print JSON")


def _tasks(args):
    if args.demo is not None:
        if args.demo < 1:
            raise SystemExit("--demo must be a positive integer")
        return [{"slug": f"task_{i:02d}", "sleep_sec": round(random.uniform(2, 4), 2)}
                for i in range(1, args.demo + 1)]
    try:
        cfg = json.loads((args.file or Path("prompts.json")).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SystemExit("Unable to read the prompts file as JSON") from None
    if not isinstance(cfg, dict) or not isinstance(cfg.get("prompts"), list):
        raise SystemExit("The prompts file must contain a prompts list")
    if any(not isinstance(prompt, dict) or not isinstance(prompt.get("slug"), str)
           or not prompt["slug"].strip() for prompt in cfg["prompts"]):
        raise SystemExit("Each prompt must have a non-empty string slug")
    defaults = {key: cfg[key] for key in ("out_dir", "run_name") if key in cfg}
    return [{**defaults, **prompt} for prompt in cfg["prompts"]]


def _show_status(status):
    print(f"Queue {status['queue']}  stream_len={status['stream_len']}  "
          f"unclaimed={status['unclaimed']}  pending={status['pending']}  all_done={status['all_done']}")
    print("Counts " + (", ".join(f"{key}={value}" for key, value in sorted(status["counts"].items()))
                       or "(empty)"))
    for section, columns in (("workers", ("status", "last_slug", "step")),
                             ("tasks", ("status", "worker", "step", "output", "error"))):
        print(section.capitalize())
        for name, row in sorted(status[section].items()):
            details = "  ".join(f"{column}={row[column]}" for column in columns if row.get(column))
            print(f"  {name}  {details}")
        if not status[section]:
            print("  (none)")


def cmd_queue(args):
    redis_url = (args.redis_url or "").strip() or os.environ.get("COLAB_ULTRA_REDIS_URL", "").strip()
    if not redis_url:
        redis_url = (load_config().get("redis_url") or "").strip()
    if not redis_url:
        raise SystemExit("Set --redis-url, COLAB_ULTRA_REDIS_URL, or redis_url in the config")
    tasks = _tasks(args) if args.action == "push" else None
    endpoint = mask_redis_url(redis_url)
    try:
        queue = TaskQueue(redis_url, getattr(args, "name", "ping"))
        if args.action == "ping":
            print(f"{endpoint}  {queue.ping():.2f} ms")
        elif args.action == "push":
            print(f"Pushed {queue.push(tasks)} task(s) to {args.name}")
        elif args.action == "status":
            status = queue.status()
            if args.json:
                print(json.dumps(status, indent=2, ensure_ascii=False))
            else:
                _show_status(status)
        elif args.action == "clear":
            queue.clear()
            print(f"Cleared queue {args.name}")
    except ModuleNotFoundError as exc:
        if exc.name == "redis":
            raise SystemExit("Install redis>=5.0 in the colab-ultra Python environment") from None
        raise
    except Exception as exc:
        raise SystemExit(f"Redis queue {args.action} failed for {endpoint} ({type(exc).__name__})") from None
