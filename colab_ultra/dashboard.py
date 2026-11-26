"""Local web dashboard for Colab, Kaggle, and Redis-backed pipelines."""

import ipaddress
import json
import math
import os
import re
import shutil
import socket
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, urlsplit

from colab_ultra.config import load_config
from colab_ultra.drive import detect_local_drive_hub
from colab_ultra.remote.queue import TaskQueue, mask_redis_url
from colab_ultra.telemetry import build_agent_status_payload, inspect_all_vms


_STAGES = ("embed", "dit", "vae")
_DEFAULT_QUEUES = tuple(f"h3_pipe_{stage}" for stage in _STAGES)
_IMAGE_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
}
_STATS_FIELDS = (
    "vae_sec", "embed_sec", "dit_sec", "embed_worker", "dit_worker", "vae_worker",
    "active_attention_backend", "output_size", "peak_vram_bytes", "seed",
)


class _TelemetryCache:
    _TTL = 10

    def __init__(self, background: bool = False):
        self._background = background
        self._lock = threading.Lock()
        self._expires_at = 0.0
        self._payload = None
        self._refreshing = False

    @staticmethod
    def _collect():
        try:
            ccu, vms = inspect_all_vms()
            payload = build_agent_status_payload(ccu, vms)
            gpu_names = {
                vm.get("endpoint"): vm.get("telemetry", {}).get("gpu", {}).get("name")
                for vm in vms
            }
            for vm in payload.get("vms", []):
                vm["gpu_name"] = gpu_names.get(vm.get("endpoint")) or vm.get("accelerator")
            return payload
        except Exception as exc:
            return {"ok": False, "loading": False, "error": f"{type(exc).__name__}: {exc}", "vms": []}

    def _store(self, payload):
        self._payload = payload
        self._expires_at = time.monotonic() + self._TTL

    def _refresh(self):
        payload = self._collect()
        with self._lock:
            self._store(payload)
            self._refreshing = False

    def get(self):
        with self._lock:
            if self._payload is not None and time.monotonic() < self._expires_at:
                return self._payload
            if not self._background:
                self._store(self._collect())
                return self._payload
            if not self._refreshing:
                self._refreshing = True
                threading.Thread(target=self._refresh, name="dashboard-telemetry", daemon=True).start()
            if self._payload is not None:
                return self._payload
            return {"ok": False, "loading": True, "error": None, "vms": []}


_TELEMETRY_CACHE = _TelemetryCache()


def _fetch_queue_status(redis_url, name):
    try:
        return TaskQueue(redis_url, name).status()
    except Exception as exc:
        return {"queue": name, "error": f"Redis status failed ({type(exc).__name__})"}


class _QueueCache:
    _TTL = 3

    def __init__(self, redis_url, queues, *, background: bool = False):
        self._redis_url = redis_url
        self._queues = tuple(queues)
        self._background = background
        self._lock = threading.Lock()
        self._expires_at = 0.0
        self._payload = None
        self._refreshing = False
        self._generation = 0

    def _collect(self):
        if not self._queues:
            return []
        if len(self._queues) == 1:
            return [_fetch_queue_status(self._redis_url, self._queues[0])]
        with ThreadPoolExecutor(max_workers=min(len(self._queues), 8)) as pool:
            return list(pool.map(lambda name: _fetch_queue_status(self._redis_url, name), self._queues))

    def _refresh(self, generation):
        payload = self._collect()
        retrigger = False
        with self._lock:
            if generation == self._generation:
                self._payload = payload
                self._expires_at = time.monotonic() + self._TTL
                self._refreshing = False
            else:
                retrigger = True
                next_gen = self._generation
        if retrigger:
            threading.Thread(
                target=self._refresh, args=(next_gen,), name="dashboard-queues", daemon=True
            ).start()

    def invalidate(self):
        with self._lock:
            self._generation += 1
            self._expires_at = 0.0
            if self._background and not self._refreshing:
                self._refreshing = True
                gen = self._generation
                threading.Thread(
                    target=self._refresh, args=(gen,), name="dashboard-queues", daemon=True
                ).start()

    def get(self):
        with self._lock:
            if self._payload is not None and time.monotonic() < self._expires_at:
                return self._payload
            if not self._background:
                self._payload = self._collect()
                self._expires_at = time.monotonic() + self._TTL
                return self._payload
            if not self._refreshing:
                self._refreshing = True
                gen = self._generation
                threading.Thread(
                    target=self._refresh, args=(gen,), name="dashboard-queues", daemon=True
                ).start()
            if self._payload is not None:
                return self._payload
            return [
                {"queue": name, "loading": True, "counts": {}, "workers": {}, "tasks": {}}
                for name in self._queues
            ]


def _redis_redactor(redis_url):
    endpoint = mask_redis_url(redis_url)
    replacements = {}
    try:
        parts = urlsplit(redis_url)
        secrets = [parts.username, parts.password]
        secrets.extend(value for key, value in parse_qsl(parts.query)
                       if any(word in key.lower() for word in ("token", "password", "secret", "credential")))
        for secret in filter(None, secrets):
            decoded = unquote(secret)
            for variant in {secret, decoded, quote(decoded, safe="")}:
                replacements[variant] = "[redacted]"
    except ValueError:
        pass
    if replacements:
        secrets_pattern = "|".join(map(re.escape, sorted(replacements, key=len, reverse=True)))
        endpoint = re.sub(secrets_pattern, "[redacted]", endpoint)
    if redis_url and redis_url != endpoint:
        replacements[redis_url] = endpoint
        replacements[quote(redis_url, safe="")] = "[redacted]"
    if not replacements:
        return endpoint, lambda value: value
    terms = sorted(replacements, key=len, reverse=True)
    pattern = re.compile("|".join(map(re.escape, terms)))
    return endpoint, lambda value: pattern.sub(lambda match: replacements[match[0]], value)


def _clean_payload(value, redact, *, redact_keys=False):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {
            (redact(key) if redact_keys else key):
            _clean_payload(item, redact, redact_keys=key in {"counts", "tasks", "workers"})
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_clean_payload(item, redact) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _drive_hub(drive_hub):
    return detect_local_drive_hub(create=False) if drive_hub is None else drive_hub


def _runs_root(drive_hub, project):
    if not drive_hub:
        return None
    try:
        hub = Path(drive_hub).resolve()
        root = hub / project / "runs"
        if root.is_relative_to(hub) and root != hub / "runs" and root.resolve() == root:
            return root
    except (OSError, RuntimeError, ValueError):
        pass
    return None


def _resolve_under(root, path):
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Path escapes the runs directory")
    return resolved


def _image_path(run_dir, file_name):
    if Path(file_name).suffix.lower() not in _IMAGE_TYPES:
        raise ValueError("Unsupported image extension")
    image = _resolve_under(run_dir, run_dir / file_name)
    content_type = _IMAGE_TYPES.get(image.suffix.lower())
    if content_type is None:
        raise ValueError("Unsupported image extension")
    return image, content_type


def _read_object(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _image_stats(run_dir, slug):
    try:
        path = _resolve_under(run_dir, run_dir / f"{slug}.json")
    except (OSError, RuntimeError, ValueError):
        return {}
    stats = _read_object(path) if path.suffix.lower() == ".json" else {}
    return {field: stats[field] for field in _STATS_FIELDS if field in stats}


def _scan_gallery(root):
    if root is None:
        return []
    try:
        directories = list(root.iterdir())
    except OSError:
        return []
    gallery = []
    for directory in directories:
        try:
            run_dir = _resolve_under(root, directory)
            mtime = run_dir.stat().st_mtime
            files = list(run_dir.iterdir())
        except (OSError, RuntimeError, ValueError):
            continue
        items = []
        for file in files:
            try:
                image, _ = _image_path(run_dir, file.name)
                if not image.is_file():
                    continue
                image_mtime = image.stat().st_mtime
            except (OSError, RuntimeError, ValueError):
                continue
            items.append({
                "slug": file.stem, "file_name": file.name, "mtime": image_mtime,
                "url": f"/api/image/{quote(directory.name, safe='')}/{quote(file.name, safe='')}",
                "stats": _image_stats(run_dir, file.stem),
            })
        if items:
            items.sort(key=lambda item: (item["mtime"], item["file_name"]), reverse=True)
            gallery.append({"run_name": directory.name, "mtime": mtime, "items": items})
    return sorted(gallery, key=lambda run: (run["mtime"], run["run_name"]), reverse=True)


def build_snapshot(redis_url, queues, project, *, drive_hub=None, telemetry_cache=None, queue_cache=None):
    """Collect queue states, cached compute telemetry, and local run images."""
    endpoint, redact = _redis_redactor(redis_url)
    if queue_cache is not None:
        states = queue_cache.get()
    else:
        states = [_fetch_queue_status(redis_url, name) for name in queues]

    cache = _TELEMETRY_CACHE if telemetry_cache is None else telemetry_cache
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": redact(project),
        "redis_endpoint": endpoint,
        "queues": _clean_payload(states, redact),
        "telemetry": _clean_payload(cache.get(), redact),
        "gallery": _clean_payload(_scan_gallery(_runs_root(_drive_hub(drive_hub), project)), redact),
    }


def _path_segment(segment):
    return segment not in ("", ".", "..") and not any(char in segment for char in "/\\\0")


def create_dashboard_server(host="127.0.0.1", port=8765, *, redis_url,
                            queues, project="h3_pipeline_demo", drive_hub=None,
                            telemetry_cache=None, queue_cache=None):
    """Bind the dashboard without starting its HTTP loop."""
    queues = tuple(queues)
    cache = _TelemetryCache(background=True) if telemetry_cache is None else telemetry_cache
    q_cache = _QueueCache(redis_url, queues, background=True) if queue_cache is None else queue_cache
    hub = _drive_hub(drive_hub) or ""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Request lines and headers can contain credentials.

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def parse_request(self):
            if not super().parse_request():
                return False
            # A loopback listener only accepts loopback browser authorities.
            bound_host = self.server.server_address[0]
            authority = self.headers.get("Host")
            if authority and ipaddress.ip_address(bound_host).is_loopback:
                try:
                    request_host = urlsplit(f"http://{authority}").hostname
                except ValueError:
                    request_host = None
                if request_host not in {bound_host, "localhost"}:
                    self.send_error(400)
                    return False
            return True

        def send_error(self, code, message=None, explain=None):
            # BaseHTTPRequestHandler may include the request line in its error body.
            self.close_connection = True
            self._json({"error": self.responses[code][0]}, code)

        def _headers(self, content_type, length, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()

        def _send(self, body, content_type, status=200):
            self._headers(content_type, len(body), status)
            if getattr(self, "command", None) != "HEAD":
                self.wfile.write(body)

        def _json(self, data, status=200):
            self._send(json.dumps(data, ensure_ascii=True, allow_nan=False).encode("utf-8"),
                       "application/json; charset=utf-8", status)

        def do_GET(self):
            path = self.path.partition("?")[0]
            if path == "/":
                self._send(_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/snapshot":
                self._json(build_snapshot(
                    redis_url, queues, project, drive_hub=hub, telemetry_cache=cache, queue_cache=q_cache,
                ))
            elif path.startswith("/api/image/"):
                self._image(path.removeprefix("/api/image/"))
            else:
                self.send_error(404)

        def _image(self, tail):
            parts = [unquote(part) for part in tail.split("/")]
            if len(parts) != 2 or not all(map(_path_segment, parts)):
                self.send_error(400)
                return
            root = _runs_root(hub, project)
            if root is None:
                self.send_error(404)
                return
            run_name, file_name = parts
            try:
                run_dir = _resolve_under(root, root / run_name)
                image, content_type = _image_path(run_dir, file_name)
                if not image.is_file():
                    raise FileNotFoundError
                source = image.open("rb")
            except ValueError:
                self.send_error(400)
                return
            except (OSError, RuntimeError):
                self.send_error(404)
                return
            with source:
                self._headers(content_type, os.fstat(source.fileno()).st_size)
                shutil.copyfileobj(source, self.wfile)

        def do_POST(self):
            path = self.path.partition("?")[0].split("/")
            if len(path) != 5 or path[:3] != ["", "api", "queue"] or path[4] != "clear":
                self.send_error(404)
                return
            name = unquote(path[3])
            if name not in queues:
                self._json({"error": "Queue is outside the dashboard configuration"}, 400)
                return
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{self.headers.get('Host')}":
                self.send_error(403)
                return
            try:
                TaskQueue(redis_url, name).clear()
            except Exception:
                self._json({"error": "Redis queue clear failed"}, 502)
                return
            if hasattr(q_cache, "invalidate"):
                q_cache.invalidate()
            self._json({"ok": True})

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    server = Server((host, port), Handler)
    if telemetry_cache is None:
        cache.get()  # Start telemetry before the first snapshot request.
    if queue_cache is None:
        q_cache.get()  # Start Redis queue polling before the first snapshot request.
    return server


def add_dashboard_parser(sub):
    parser = sub.add_parser("dashboard", help="Open the local web monitoring dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--project", default="h3_pipeline_demo", help="Drive project to monitor")
    parser.add_argument("--queues", help="Comma-separated Redis queues (defaults to the project's pipeline)")
    parser.add_argument("--no-open", action="store_true", help="Skip opening the browser")
    parser.add_argument("--redis-url", help="Redis URL (or COLAB_ULTRA_REDIS_URL / config)")


def _configured_queues(hub, project, override):
    if override is not None:
        return list(dict.fromkeys(name.strip() for name in override.split(",") if name.strip()))
    settings = _read_object(Path(hub) / project / "code" / "pipeline_worker.json") if hub else {}
    queues = []
    for stage, default in zip(_STAGES, _DEFAULT_QUEUES):
        name = settings.get(f"{stage}_queue")
        queues.append(name.strip() if isinstance(name, str) and name.strip() else default)
    return queues


def cmd_dashboard(args):
    redis_url = (args.redis_url or "").strip() or os.environ.get("COLAB_ULTRA_REDIS_URL", "").strip()
    if not redis_url:
        redis_url = (load_config().get("redis_url") or "").strip()
    hub = _drive_hub(None) or ""
    queues = _configured_queues(hub, args.project, args.queues)
    with create_dashboard_server(args.host, args.port, redis_url=redis_url, queues=queues,
                                 project=args.project, drive_hub=hub) as server:
        browser_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(args.host, args.host)
        if ":" in browser_host:
            browser_host = f"[{browser_host}]"
        url = f"http://{browser_host}:{server.server_port}/"
        print(f"Colab Ultra dashboard at {url}", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Colab Ultra · Execution Monitor</title>
<style>
  :root {
    color-scheme: dark;
    --canvas: #090b10; --panel: #10141d; --raised: #151b27;
    --line: rgba(255, 255, 255, .08); --ink: #edf0f7; --muted: #9ba7bb; --quiet: #8491a7;
    --indigo: #6366f1; --cyan: #22d3ee; --green: #34d399; --amber: #fbbf24; --rose: #fb7185;
    --mono: ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body {
    margin: 0; color: var(--ink); font: 13px/1.45 system-ui, -apple-system, sans-serif;
    font-variant-numeric: tabular-nums;
    background: radial-gradient(ellipse 850px 420px at 42% 0, #161b30 0, transparent 100%), var(--canvas);
  }
  h1, h2, h3, p, dl, dd { margin: 0; }
  h1 { font-size: 17px; letter-spacing: -.045em; white-space: nowrap; }
  h2 { font-size: 16px; font-weight: 650; letter-spacing: -.025em; }
  h3 { font-size: 13px; font-weight: 650; overflow-wrap: anywhere; }
  dt, th, .muted { color: var(--muted); }
  .mono, time { font-family: var(--mono); }
  .wrap { width: min(100%, 1504px); margin-inline: auto; padding-inline: clamp(16px, 2.5vw, 38px); }
  .topbar { border-bottom: 1px solid var(--line); background: rgba(9, 11, 16, .66); }
  .command, .identity, .controls, .live-meta, .spread {
    display: flex; align-items: center; gap: 10px; min-width: 0;
  }
  .command { justify-content: space-between; min-height: 65px; padding-block: 10px; }
  .identity { flex-wrap: wrap; }
  .controls { flex: none; }
  .brand { display: flex; align-items: center; gap: 9px; }
  .brand-icon {
    width: 29px; height: 29px; padding: 5px; flex: none; color: #b9bdff;
    background: linear-gradient(145deg, #292957, #191c32);
    border: 1px solid #3c3d68; border-radius: 8px; box-shadow: inset 0 1px rgba(255, 255, 255, .07);
  }
  .tag, .project, .pill, .syncing {
    width: fit-content; border: 1px solid var(--line); border-radius: 5px; padding: 2px 7px;
    font-size: 11px; line-height: 1.45; overflow-wrap: anywhere;
  }
  .tag { color: var(--muted); }
  .project { color: #c5c8ff; background: #1a1d33; font-family: var(--mono); }
  .live-meta { color: var(--muted); flex-wrap: wrap; gap: 6px 9px; font-size: 11px; }
  .connection { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
  .live-dot, .pill-dot, .key-dot {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%; flex: none; background: currentColor;
  }
  .live-dot { color: var(--amber); }
  .connection[data-state="live"] .live-dot {
    color: var(--green); animation: pulse 2.4s ease-out infinite;
  }
  .connection[data-state="paused"] .live-dot { color: var(--quiet); }
  .connection[data-state="error"] { white-space: normal; overflow-wrap: anywhere; }
  .connection[data-state="error"] .live-dot { color: var(--rose); }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, .5); }
    75%, 100% { box-shadow: 0 0 0 5px rgba(52, 211, 153, 0); }
  }
  button {
    display: inline-flex; align-items: center; justify-content: center; gap: 6px; flex: none;
    color: var(--ink); background: var(--raised); border: 1px solid var(--line); border-radius: 7px;
    min-height: 32px; padding: 5px 10px; font: inherit; font-size: 12px; cursor: pointer;
    transition: border-color .15s, background .15s;
  }
  button svg { width: 13px; height: 13px; fill: none; stroke: currentColor; stroke-width: 1.6; }
  button:hover { background: #20283a; border-color: #465376; }
  button:focus-visible, .scroll:focus-visible { outline: 2px solid var(--indigo); outline-offset: 2px; }
  button:disabled { opacity: .5; cursor: wait; }
  #toggle { background: transparent; color: var(--muted); }
  #toggle:hover { background: var(--raised); color: var(--ink); }
  main.wrap { padding-block: 18px 42px; }
  section { margin-top: 23px; }
  .section-head { margin-bottom: 9px; }
  .spread { justify-content: space-between; }
  .surface {
    min-width: 0; border: 1px solid var(--line); border-radius: 11px; background: var(--panel);
    box-shadow: inset 0 1px rgba(255, 255, 255, .035), 0 9px 24px rgba(0, 0, 0, .13);
  }
  .kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)) minmax(0, 1.4fr); }
  .kpi { min-width: 0; min-height: 67px; padding: 9px 14px; }
  .kpi + .kpi { border-left: 1px solid var(--line); }
  .kpi dt, .eyebrow {
    color: var(--muted); font-size: 10px; font-weight: 600; letter-spacing: .075em; text-transform: uppercase;
  }
  .kpi dd {
    margin-top: 4px; font: 600 19px/1.25 var(--mono); letter-spacing: -.045em; overflow-wrap: anywhere;
  }
  .kpi.endpoint dd { color: #c3c9fb; font-size: 12px; letter-spacing: 0; margin-top: 8px; }
  .syncing { color: var(--amber); background: rgba(251, 191, 36, .07); font: 11px/1.45 var(--mono); }
  .pipeline {
    display: grid; grid-template-columns: minmax(0, 1fr) 48px minmax(0, 1fr) 48px minmax(0, 1fr);
    align-items: center; padding: 12px;
  }
  .stage {
    min-width: 0; padding: 12px 13px; border: 1px solid var(--line); border-radius: 8px;
    background: linear-gradient(145deg, var(--raised), #111622);
    box-shadow: inset 0 1px rgba(255, 255, 255, .035);
  }
  .stage[data-state="active"] {
    border-color: rgba(99, 102, 241, .55);
    background: linear-gradient(135deg, #202441, #121926 65%);
  }
  .stage-name { margin-top: 1px; font-size: 17px; letter-spacing: -.025em; }
  .stage-summary { margin-top: 9px; color: var(--muted); font-size: 11px; }
  .segments, .meter { height: 5px; display: flex; overflow: hidden; border-radius: 8px; background: #252d3e; }
  .segments { margin-top: 10px; }
  .segment { flex: none; background: var(--tone); }
  .queued, .polling { --tone: var(--amber); }
  .running, .active { --tone: var(--cyan); }
  .done { --tone: var(--green); }
  .failed, .error { --tone: var(--rose); }
  .pill {
    display: inline-flex; align-items: center; gap: 5px;
    color: var(--tone, var(--muted)); background: rgba(255, 255, 255, .035);
  }
  .stage .pill { padding: 3px 6px; font-family: var(--mono); font-size: 10px; }
  .connector { display: block; width: 100%; height: 24px; color: #4c586e; overflow: visible; }
  .connector .signal { display: none; stroke: var(--cyan); }
  .connector.active { color: #858aff; }
  .connector.active .signal { display: inline; animation: flow 1.1s linear infinite; }
  @keyframes flow { to { stroke-dashoffset: -18; } }
  .charts { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 10px; }
  .chart { padding: 12px 14px; }
  .chart > .spread { flex-wrap: wrap; gap: 4px 10px; }
  .legend {
    display: flex; flex-wrap: wrap; gap: 4px 9px; margin-top: 8px; color: var(--muted); font-size: 10px;
  }
  .legend span { display: inline-flex; align-items: center; gap: 4px; }
  .key-dot { color: var(--tone); border-radius: 2px; }
  .distribution { display: grid; gap: 8px; margin-top: 8px; }
  .distribution-row { display: grid; grid-template-columns: 45px minmax(0, 1fr) 43px; align-items: center; gap: 8px; }
  .distribution-row svg { width: 100%; height: 10px; display: block; }
  .distribution-row span:last-child { text-align: right; }
  .distribution-row .mono, .sample-note { font-size: 10px; color: var(--muted); }
  .computes { margin-top: 6px; max-height: 126px; overflow: auto; }
  .spark-row {
    display: grid; grid-template-columns: minmax(0, .8fr) repeat(2, minmax(0, 1fr));
    gap: 10px; align-items: center; padding: 6px 1px;
  }
  .spark-row + .spark-row { border-top: 1px solid var(--line); }
  .spark-name { color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
  .spark-heading { display: flex; justify-content: space-between; gap: 3px; font: 10px/1.4 var(--mono); color: var(--muted); }
  .spark-heading strong { color: var(--tone); font-weight: 600; }
  .spark { width: 100%; height: 25px; display: block; overflow: visible; }
  .vram { --tone: var(--cyan); --fill: linear-gradient(90deg, #2563eb, #22d3ee); }
  .cpu { --tone: #a5a3ff; --fill: linear-gradient(90deg, #6366f1, #a5a3ff); }
  .ram { --fill: linear-gradient(90deg, #0d9488, #34d399); }
  .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; align-items: start; }
  .card { padding: 12px 13px; }
  .queue-name { margin-top: 1px; color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
  .clear { min-height: 27px; color: var(--muted); background: transparent; font-size: 11px; }
  .clear:hover { color: var(--rose); background: rgba(251, 113, 133, .07); border-color: rgba(251, 113, 133, .28); }
  .counts {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 10px 0;
    border: 1px solid var(--line); border-radius: 7px; background: var(--raised);
  }
  .counts div { min-width: 0; padding: 5px 6px; }
  .counts div + div { border-left: 1px solid var(--line); }
  .counts dt { font-size: 10px; text-transform: capitalize; }
  .counts dd { font: 600 15px/1.35 var(--mono); color: var(--tone); overflow-wrap: anywhere; }
  .workers { display: flex; align-items: center; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
  .workers .pill { text-align: left; font-family: var(--mono); max-width: 100%; }
  .task-list { max-height: 243px; overflow: auto; margin-top: 10px; }
  table { width: 100%; border-collapse: collapse; text-align: left; font-size: 11px; }
  th, td { padding: 5px 4px; vertical-align: top; border-bottom: 1px solid var(--line); }
  th { position: sticky; top: 0; font-weight: 500; background: var(--panel); }
  td { overflow-wrap: anywhere; }
  td .pill { font-size: 10px; padding: 1px 4px; }
  .task-error { min-width: 42px; max-width: 155px; white-space: pre-wrap; color: #fda4af; }
  .queue-note, .empty { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
  .queue-note { margin-top: 8px; }
  .empty { grid-column: 1 / -1; padding: 11px 13px; }
  .chart .empty { padding: 12px 0; }
  .warning { color: var(--rose); }
  .provider { color: #bfc5ff; }
  .provider.kaggle { color: var(--cyan); }
  .hardware { margin: 7px 0 11px; color: var(--muted); overflow-wrap: anywhere; font: 11px/1.5 var(--mono); }
  .metric { margin-top: 7px; }
  .metric-value { margin-bottom: 4px; font-size: 11px; color: var(--muted); }
  .metric-value .mono { color: var(--ink); text-align: right; overflow-wrap: anywhere; }
  .meter-fill { background: var(--fill); border-radius: inherit; }
  .vm-task { margin-top: 11px; padding-top: 8px; border-top: 1px solid var(--line); overflow-wrap: anywhere; }
  .vm-task p { margin-top: 3px; font-size: 11px; }
  @media (max-width: 1050px) {
    .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
  @media (max-width: 760px) {
    .kpis { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .kpi.endpoint { grid-column: 1 / -1; border: 0; border-top: 1px solid var(--line); min-height: 0; }
    .kpi { padding-inline: 9px; }
    .pipeline { grid-template-columns: minmax(0, 1fr) 29px minmax(0, 1fr) 29px minmax(0, 1fr); padding: 8px; }
    .stage { padding: 9px; }
    .stage .spread { flex-wrap: wrap; gap: 5px; }
    .charts { grid-template-columns: minmax(0, 1fr); }
  }
  @media (max-width: 520px) {
    .command { align-items: flex-start; }
    .identity { gap: 5px 7px; }
    .controls { gap: 4px; }
    .controls button { padding-inline: 7px; }
    .controls button svg { display: none; }
    .grid { grid-template-columns: minmax(0, 1fr); }
    .kpi dd { font-size: 14px; }
    .pipeline { grid-template-columns: minmax(0, 1fr); justify-items: stretch; }
    .connector { width: 26px; height: 23px; justify-self: center; transform: rotate(90deg); }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
  }
</style>
</head>
<body>
<header class="topbar">
  <div class="wrap command">
    <div class="identity">
      <div class="brand">
        <svg class="brand-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M2 10h4l2-5 4 10 2-5h4"/>
        </svg>
        <h1>Colab Ultra</h1>
      </div>
      <span class="tag">Execution Monitor</span>
      <span id="project" class="project" hidden></span>
      <div class="live-meta">
        <span id="connection" class="connection" role="status"><i class="live-dot" aria-hidden="true"></i><span id="live-label">Connecting</span></span>
        <time id="last-updated" hidden></time>
      </div>
    </div>
    <div class="controls">
      <button id="toggle" type="button"><span id="toggle-label">Pause</span></button>
      <button id="refresh" type="button">
        <svg viewBox="0 0 16 16" aria-hidden="true" stroke-linecap="round" stroke-linejoin="round">
          <path d="M13 7a5.2 5.2 0 1 0-1.1 4.3M13 3.5V7H9.5"/>
        </svg>
        Refresh
      </button>
    </div>
  </div>
</header>
<main class="wrap">
  <dl class="kpis surface" id="kpis" aria-label="Compute and Redis summary"></dl>
  <section aria-labelledby="flow-heading">
    <h2 id="flow-heading" class="section-head">Pipeline Flow</h2>
    <div id="flow" class="pipeline surface"></div>
  </section>
  <section aria-labelledby="metrics-heading">
    <h2 id="metrics-heading" class="section-head">Live Execution Metrics</h2>
    <div class="charts">
      <article class="chart surface">
        <h3>Queue Throughput / Stage Distribution</h3>
        <div class="legend" aria-label="Chart key">
          <span><i class="key-dot queued"></i>Queued</span><span><i class="key-dot running"></i>Running</span>
          <span><i class="key-dot done"></i>Done</span><span><i class="key-dot failed"></i>Failed</span>
        </div>
        <div id="distribution" class="distribution"></div>
      </article>
      <article class="chart surface">
        <div class="spread">
          <h3>Live Compute Sparklines</h3>
          <span class="sample-note">Last 30 samples</span>
        </div>
        <div id="computes" class="computes scroll" tabindex="0" aria-label="Compute usage history"></div>
      </article>
    </div>
  </section>
  <section aria-labelledby="queues-heading">
    <h2 id="queues-heading" class="section-head">Pipeline Queues &amp; Active Tasks</h2>
    <p id="clear-error" class="section-head warning" role="alert" hidden></p>
    <div id="queues" class="grid"><p class="empty surface">Waiting for queue status…</p></div>
  </section>
  <section aria-labelledby="compute-heading">
    <h2 id="compute-heading" class="section-head">Active Compute Nodes</h2>
    <div id="nodes" class="grid"></div>
  </section>
</main>
<script>
  const $ = id => document.getElementById(id);
  const stages = [
    {key: 'embed', name: 'Embed'}, {key: 'dit', name: 'DiT'}, {key: 'vae', name: 'VAE'},
  ];
  const statuses = ['queued', 'running', 'done', 'failed'];
  const segmentOrder = ['done', 'running', 'queued', 'failed'];
  const historyLimit = 30;
  const vmHistory = new Map();
  const clearingQueues = new Set();
  const clearButtons = new Map();
  let autoRefresh = true;
  let pendingRefresh = null;
  let refreshError = '';
  let queueState = '';

  function el(tag, className = '', text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function svg(tag, attributes) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    return node;
  }

  const present = value => value != null && value !== '';
  const display = value => present(value) ? value : '—';
  const numeric = value => typeof value === 'number' && Number.isFinite(value);
  const unit = (value, suffix, digits = 1) => numeric(value) ? value.toFixed(digits) + suffix : '—';
  const count = (queue, status) => queue?.counts?.[status] ?? 0;
  const total = queue => statuses.reduce((sum, status) => sum + count(queue, status), 0);
  const breakdown = queue => statuses.map(status => count(queue, status) + ' ' + status).join(', ');
  const percent = value => numeric(value) ? Math.min(100, Math.max(0, value)) : null;
  const usage = (used, capacity) => numeric(used) && numeric(capacity) && capacity > 0
    ? percent(100 * used / capacity) : null;

  function statusPill(text, style = text) {
    const tone = [...statuses, 'active', 'error', 'polling'].includes(style) ? ' ' + style : '';
    const pill = el('span', 'pill' + tone);
    pill.append(el('i', 'pill-dot'), el('span', '', display(text)));
    return pill;
  }

  function mapStages(queues) {
    const matches = new Map();
    const remaining = [];
    for (const queue of queues) {
      const parts = queue.queue.toLowerCase().split(/[^a-z0-9]+/);
      const stage = stages.find(({key}) => parts.includes(key));
      if (stage && !matches.has(stage.key)) matches.set(stage.key, queue);
      else remaining.push(queue);
    }
    // Custom queue names can omit the stage. Their configured order identifies it.
    for (const {key} of stages) {
      if (!matches.has(key) && remaining.length) matches.set(key, remaining.shift());
    }
    return matches;
  }

  function stateOf(queue) {
    if (queue?.loading) return 'idle';
    if (queue?.error) return 'error';
    if (count(queue, 'running')) return 'active';
    if (count(queue, 'queued')) return 'queued';
    if (count(queue, 'failed')) return 'error';
    if (count(queue, 'done')) return 'done';
    return 'idle';
  }

  function renderStage(stage, queue, index, state) {
    const node = el('article', 'stage');
    node.dataset.state = state;
    const header = el('div', 'spread');
    const title = el('div');
    title.append(el('span', 'eyebrow', 'Stage ' + (index + 1)), el('h3', 'stage-name', stage.name));
    header.append(title, statusPill(queue?.loading ? 'SYNCING' : state.toUpperCase(), state));
    const sum = total(queue);
    const bar = el('div', 'segments');
    bar.setAttribute('role', 'img');
    bar.setAttribute('aria-label', queue?.loading ? 'Syncing queue' : (queue?.error || breakdown(queue)));
    for (const status of segmentOrder) {
      if (!count(queue, status)) continue;
      const segment = el('span', 'segment ' + status);
      segment.style.width = (100 * count(queue, status) / sum) + '%';
      bar.append(segment);
    }
    const summary = queue?.loading ? 'Syncing…' : (queue?.error ? 'Queue unavailable' : count(queue, 'done') + ' / ' + sum + ' done');
    node.append(header, bar, el('p', 'stage-summary mono', summary));
    return node;
  }

  function connector(active) {
    const arrow = svg('svg', {
      viewBox: '0 0 48 24', class: 'connector' + (active ? ' active' : ''),
      fill: 'none', 'stroke-width': 1.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round',
      'aria-hidden': 'true',
    });
    arrow.append(
      svg('path', {d: 'M1 12H45M39 7l6 5-6 5', stroke: 'currentColor'}),
      svg('path', {d: 'M1 12H43', class: 'signal', 'stroke-dasharray': '3 6'}),
    );
    return arrow;
  }

  function renderDistribution(matches) {
    const max = Math.max(1, ...stages.map(({key}) => total(matches.get(key))));
    $('distribution').replaceChildren(...stages.map(({key, name}) => {
      const queue = matches.get(key);
      const row = el('div', 'distribution-row');
      const labelText = queue?.loading ? 'Syncing…' : (queue?.error || breakdown(queue));
      const bar = svg('svg', {viewBox: '0 0 100 10', preserveAspectRatio: 'none', role: 'img',
        'aria-label': name + ': ' + labelText});
      const title = svg('title', {});
      title.textContent = labelText;
      bar.append(title, svg('rect', {width: '100%', height: 10, fill: '#252d3e'}));
      let x = 0;
      for (const status of statuses) {
        const width = 100 * count(queue, status) / max;
        if (!width) continue;
        bar.append(svg('rect', {x, width, height: 10, class: status, fill: 'var(--tone)'}));
        x += width;
      }
      const rightLabel = queue?.loading ? 'Syncing…' : (queue?.error ? 'Error' : total(queue));
      row.append(el('span', '', name), bar, el('span', 'mono', rightLabel));
      return row;
    }));
  }

  function renderQueue(queue, stageName) {
    const card = el('article', 'card surface');
    const header = el('div', 'spread');
    const title = el('div');
    title.append(el('h3', '', stageName || queue.queue));
    if (stageName && stageName.toLowerCase() !== queue.queue.toLowerCase()) {
      title.append(el('p', 'queue-name mono', queue.queue));
    }
    const clear = el('button', 'clear', 'Clear');
    clear.type = 'button';
    clear.setAttribute('aria-label', 'Clear queue ' + queue.queue);
    clear.disabled = clearingQueues.has(queue.queue);
    clear.onclick = () => clearQueue(queue.queue);
    clearButtons.set(queue.queue, clear);
    header.append(title, clear);
    card.append(header);
    if (queue.loading) {
      card.append(el('p', 'queue-note', 'Syncing Redis queue…'));
      return card;
    }
    if (queue.error) {
      card.append(el('p', 'queue-note warning', queue.error));
      return card;
    }

    const counts = el('dl', 'counts');
    for (const status of statuses) {
      const entry = el('div');
      entry.append(el('dt', '', status), el('dd', status, count(queue, status)));
      counts.append(entry);
    }
    const workers = el('div', 'workers');
    workers.setAttribute('aria-label', 'Worker heartbeats');
    for (const [name, worker] of Object.entries(queue.workers || {})) {
      const detail = [name, worker.status, worker.last_slug, worker.step].filter(present).join(' · ');
      workers.append(statusPill(detail, worker.status));
    }
    card.append(counts, workers.childElementCount ? workers : el('p', 'queue-note', 'No worker heartbeats'));

    const tasks = Object.entries(queue.tasks || {}).sort((a, b) =>
      (b[1].updated_at || 0) - (a[1].updated_at || 0));
    if (!tasks.length) {
      card.append(el('p', 'queue-note', 'No tasks'));
      return card;
    }
    const wrapper = el('div', 'task-list scroll');
    wrapper.tabIndex = 0;
    wrapper.setAttribute('aria-label', 'Tasks for ' + queue.queue);
    const table = el('table');
    const headings = table.createTHead().insertRow();
    for (const label of ['Task', 'Status', 'Worker', 'Step', 'Error']) {
      const heading = el('th', '', label);
      heading.scope = 'col';
      headings.append(heading);
    }
    const body = table.createTBody();
    for (const [slug, task] of tasks) {
      const row = body.insertRow();
      row.append(el('td', 'mono', slug));
      row.insertCell().append(statusPill(task.status));
      row.append(el('td', 'mono', display(task.worker)), el('td', '', display(task.step)),
        el('td', task.error ? 'task-error' : '', display(task.error)));
    }
    wrapper.append(table);
    card.append(wrapper);
    return card;
  }

  async function clearQueue(name) {
    if (clearingQueues.has(name) || !window.confirm('Clear all tasks and worker status for ' + name + '?')) return;
    const note = $('clear-error');
    note.hidden = true;
    clearingQueues.add(name);
    clearButtons.get(name).disabled = true;
    try {
      const response = await fetch('/api/queue/' + encodeURIComponent(name) + '/clear', {method: 'POST'});
      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.error || 'Queue could not be cleared.');
      }
      await pendingRefresh;
      await refresh();
    } catch (error) {
      note.textContent = name + ' · ' + error.message;
      note.hidden = false;
    } finally {
      clearingQueues.delete(name);
      const button = clearButtons.get(name);
      if (button) button.disabled = false;
    }
  }

  function renderExecution(queues) {
    const matches = mapStages(queues);
    const states = stages.map(({key}) => stateOf(matches.get(key)));
    const flow = [];
    stages.forEach((stage, index) => {
      if (index) flow.push(connector(states[index - 1] === 'active' || states[index] === 'active'));
      flow.push(renderStage(stage, matches.get(stage.key), index, states[index]));
    });
    $('flow').replaceChildren(...flow);
    renderDistribution(matches);

    clearButtons.clear();
    const labels = new Map(stages.map(({key, name}) => [matches.get(key), name]));
    const cards = queues.map(queue => renderQueue(queue, labels.get(queue)));
    $('queues').replaceChildren(...(cards.length ? cards : [el('p', 'empty surface', 'No queues configured.')]));
  }

  function kpi(label, value, className = '') {
    const card = el('div', 'kpi ' + className);
    const reading = el('dd');
    reading.append(value instanceof Node ? value : display(value));
    card.append(el('dt', '', label), reading);
    return card;
  }

  function sparkline(label, style, samples) {
    const latest = samples.at(-1);
    const reading = unit(latest, '%', 0);
    const chart = el('div', style);
    const heading = el('p', 'spark-heading');
    heading.append(el('span', '', label), el('strong', '', reading));
    const plot = svg('svg', {viewBox: '0 0 120 26', preserveAspectRatio: 'none',
      class: 'spark', role: 'img', 'aria-label': label + ' utilization history, current ' + reading});
    plot.append(svg('path', {d: 'M0 12H120M0 24H120', fill: 'none', stroke: '#31394b',
      'stroke-width': .7, 'stroke-dasharray': '2 3', 'vector-effect': 'non-scaling-stroke'}));

    const segments = [];
    let segment = [];
    samples.forEach((value, index) => {
      if (value === null) {
        if (segment.length) segments.push(segment);
        segment = [];
        return;
      }
      const offset = historyLimit - samples.length + index;
      segment.push([2 + 116 * offset / (historyLimit - 1), 24 - 21 * value / 100]);
    });
    if (segment.length) segments.push(segment);
    for (const points of segments) {
      const line = 'M' + points.map(point => point.join(' ')).join('L');
      const area = line + 'L' + points.at(-1)[0] + ' 24L' + points[0][0] + ' 24Z';
      plot.append(
        svg('path', {d: area, fill: 'var(--tone)', 'fill-opacity': .14}),
        svg('path', {d: line, fill: 'none', stroke: 'var(--tone)', 'stroke-width': 1.4,
          'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'vector-effect': 'non-scaling-stroke'}),
      );
    }
    if (latest !== null) {
      plot.append(svg('circle', {cx: 118, cy: 24 - 21 * latest / 100, r: 1.7, fill: 'var(--tone)'}));
    }
    chart.append(heading, plot);
    return chart;
  }

  function recordSample(vm) {
    const key = JSON.stringify([vm.provider, vm.endpoint || vm.name]);
    const samples = vmHistory.get(key) || [];
    samples.push({vram: usage(vm.vram_used_mib, vm.vram_total_mib), cpu: percent(vm.cpu_util_pct)});
    if (samples.length > historyLimit) samples.shift();
    vmHistory.set(key, samples);
    return key;
  }

  function renderSparklines(vms, message) {
    const active = new Set();
    const rows = vms.map(vm => {
      const key = recordSample(vm);
      active.add(key);
      const samples = vmHistory.get(key);
      const row = el('div', 'spark-row');
      row.append(el('p', 'spark-name mono', display(vm.name || vm.endpoint)),
        sparkline('VRAM', 'vram', samples.map(sample => sample.vram)),
        sparkline('CPU', 'cpu', samples.map(sample => sample.cpu)));
      return row;
    });
    for (const key of vmHistory.keys()) {
      if (!active.has(key)) vmHistory.delete(key);
    }
    $('computes').replaceChildren(...(rows.length ? rows : [el('p', 'empty', message)]));
  }

  function metric(label, style, reading, value) {
    const row = el('div', 'metric ' + style);
    const heading = el('div', 'metric-value spread');
    heading.append(el('span', '', label), el('span', 'mono', reading));
    const bar = el('div', 'meter');
    if (value !== null) {
      bar.setAttribute('role', 'meter');
      bar.setAttribute('aria-label', label);
      bar.setAttribute('aria-valuemin', 0);
      bar.setAttribute('aria-valuemax', 100);
      bar.setAttribute('aria-valuenow', Math.round(value));
      const fill = el('span', 'meter-fill');
      fill.style.width = value + '%';
      bar.append(fill);
    } else {
      bar.setAttribute('aria-hidden', 'true');
    }
    row.append(heading, bar);
    return row;
  }

  function memory(label, style, used, capacity, scale) {
    const value = usage(used, capacity);
    const reading = value === null ? '—' : (used / scale).toFixed(1) + ' / ' +
      (capacity / scale).toFixed(1) + ' GiB';
    return metric(label, style, reading, value);
  }

  function renderNode(vm) {
    const card = el('article', 'card surface');
    const header = el('div', 'spread');
    const kaggle = vm.provider === 'kaggle';
    header.append(el('h3', '', display(vm.name || vm.endpoint)),
      el('span', 'pill provider' + (kaggle ? ' kaggle' : ''), kaggle ? 'Kaggle Free' : 'Colab'));
    const gpu = display(vm.gpu_name || vm.accelerator);
    const hardware = vm.gpu_count ? [gpu, unit(vm.power_draw_w, ' W', 0), unit(vm.temp_c, ' °C', 0)] : [gpu];
    const task = el('div', 'vm-task');
    const activity = vm.active_task;
    const details = activity ? [display(activity.task), unit(activity.elapsed_sec, ' s elapsed', 0)].join(' · ') : '—';
    task.append(el('span', 'eyebrow', 'Active task'), el('p', 'mono', details));
    card.append(header, el('p', 'hardware', hardware.join(' · ')),
      memory('VRAM', 'vram', vm.vram_used_mib, vm.vram_total_mib, 1024),
      memory('RAM', 'ram', vm.ram_used_gb, vm.ram_total_gb, 1),
      metric('CPU', 'cpu', unit(vm.cpu_util_pct, '%', 0), percent(vm.cpu_util_pct)), task);
    return card;
  }

  function renderTelemetry(telemetry, endpoint) {
    const vms = telemetry.vms || [];
    const active = telemetry.loading ? el('span', 'syncing', 'Syncing…') : telemetry.ok
      ? telemetry.active_vms_count ?? vms.length : '—';
    $('kpis').replaceChildren(
      kpi('Paid CCU', unit(telemetry.paid_balance, ' CU', 2)),
      kpi('Burn Rate', unit(telemetry.burn_rate_hourly, ' CU/h', 2)),
      kpi('Kaggle Free GPU', unit(telemetry.kaggle_free_gpu_hours_left, ' h')),
      kpi('Active VMs', active),
      kpi('Redis', endpoint, 'endpoint'));
    const message = telemetry.loading ? 'Probing remote Colab / Kaggle nodes…' : telemetry.ok ?
      'No active VMs' : 'Compute telemetry unavailable';
    $('nodes').replaceChildren(...(vms.length ? vms.map(renderNode) :
      [el('p', 'empty surface' + (telemetry.error ? ' warning' : ''), message)]));
    renderSparklines(vms, message);
  }

  function update(snapshot) {
    $('project').textContent = snapshot.project;
    $('project').hidden = !snapshot.project;
    const updated = $('last-updated');
    updated.dateTime = snapshot.generated_at;
    updated.textContent = 'Updated ' + new Date(snapshot.generated_at).toLocaleTimeString();
    updated.hidden = false;
    const nextQueues = JSON.stringify(snapshot.queues);
    if (queueState !== nextQueues) {
      renderExecution(snapshot.queues);
      queueState = nextQueues;
    }
    renderTelemetry(snapshot.telemetry, snapshot.redis_endpoint);
  }

  function updateConnection() {
    const state = refreshError ? 'error' : autoRefresh ?
      ($('last-updated').hidden ? 'connecting' : 'live') : 'paused';
    $('connection').dataset.state = state;
    const label = {
      connecting: 'Connecting', live: 'Live', paused: 'Paused', error: refreshError,
    }[state];
    if ($('live-label').textContent !== label) $('live-label').textContent = label;
  }

  function refresh() {
    if (pendingRefresh) return pendingRefresh;
    $('refresh').disabled = true;
    pendingRefresh = (async () => {
      try {
        const response = await fetch('/api/snapshot');
        if (!response.ok) throw new Error('HTTP ' + response.status);
        update(await response.json());
        refreshError = '';
      } catch (error) {
        refreshError = 'Refresh failed · ' + error.message;
      } finally {
        $('refresh').disabled = false;
        pendingRefresh = null;
        updateConnection();
      }
    })();
    return pendingRefresh;
  }

  $('toggle').onclick = () => {
    autoRefresh = !autoRefresh;
    $('toggle-label').textContent = autoRefresh ? 'Pause' : 'Resume';
    updateConnection();
    if (autoRefresh) refresh();
  };
  $('refresh').onclick = refresh;
  setInterval(() => { if (autoRefresh) refresh(); }, 3000);
  renderTelemetry({loading: true});
  refresh();
</script>
</body>
</html>
"""
