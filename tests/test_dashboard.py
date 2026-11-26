import base64
import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from colab_ultra import cli, dashboard


TOKEN = "dashboard@private-token"
ENCODED_TOKEN = "dashboard%40private-token"
QUERY_TOKEN = "dashboard-private-query-token"
REDIS_URL = f"https://{ENCODED_TOKEN}@redis.example.test?access_token={QUERY_TOKEN}"
MASKED_URL = "https://***@redis.example.test"
_SOCKET_CONNECT = socket.socket.connect
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNwCDrwHwAEegJSrNYUngAAAABJRU5ErkJggg=="
)


def _queue_status(name):
    return {
        "queue": name,
        "stream_len": 1,
        "unclaimed": 0,
        "pending": 0,
        "counts": {"failed": 1},
        "all_done": True,
        "workers": {"worker-1": {"status": "running", "last_slug": "slug", "step": "decoding_vae"}},
        "tasks": {"slug": {"status": "failed", "worker": "worker-1", "step": "upload",
                           "output": "/runs/slug.png", "error": f"Backend error for {REDIS_URL} ({TOKEN})"}},
    }


def _assert_private(text):
    for secret in (REDIS_URL, TOKEN, ENCODED_TOKEN, QUERY_TOKEN):
        assert secret not in text


@pytest.fixture
def mocks(monkeypatch):
    def task_queue(_redis_url, name):
        if name == "offline":
            raise ConnectionError(f"Unable to connect to {REDIS_URL}: {TOKEN}")
        return Mock(status=Mock(return_value=_queue_status(name)))

    queues = Mock(side_effect=task_queue)
    ccu = object()
    vms = [{"endpoint": "kaggle-1", "telemetry": {"gpu": {"name": "NVIDIA Tesla T4"}}}]
    inspect = Mock(return_value=(ccu, vms))
    payload = Mock(return_value={
        "ok": True, "paid_balance": 14.25, "burn_rate_hourly": 0.5,
        "kaggle_free_gpu_hours_left": 12.5, "active_vms_count": 1,
        "vms": [{"endpoint": "kaggle-1", "accelerator": "T4x2", "provider": "kaggle"}],
    })
    monkeypatch.setattr(dashboard, "TaskQueue", queues)
    monkeypatch.setattr(dashboard, "inspect_all_vms", inspect)
    monkeypatch.setattr(dashboard, "build_agent_status_payload", payload)
    return SimpleNamespace(queues=queues, inspect=inspect, payload=payload, ccu=ccu, vms=vms)


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "demo" / "runs"
    new_run = root / "new run"
    old_run = root / "old run"
    for run in (new_run, old_run, root / "empty"):
        run.mkdir(parents=True)
    images = {new_run / "scene #2.PNG": 30, new_run / "scene 1.png": 20, old_run / "scene 0.png": 40}
    for path, mtime in images.items():
        path.write_bytes(PNG)
        os.utime(path, (mtime, mtime))
    stats = {
        "vae_sec": 0.2, "embed_sec": 2.3, "dit_sec": 5.6,
        "embed_worker": "embed-1", "dit_worker": "dit-1", "vae_worker": "vae-1",
        "active_attention_backend": "flash_attention", "output_size": [512, 768],
        "peak_vram_bytes": 12345, "seed": 0,
    }
    (new_run / "scene #2.json").write_text(json.dumps({**stats, "other": "ignore me"}), encoding="utf-8")
    (new_run / "scene 1.json").write_text("{malformed", encoding="utf-8")
    (new_run / "notes.txt").write_text("not an image", encoding="utf-8")
    for run, mtime in ((new_run, 15), (old_run, 10)):
        os.utime(run, (mtime, mtime))
    return SimpleNamespace(root=root, stats=stats, new_run=new_run)


def test_build_snapshot(mocks, runs, tmp_path):
    snapshot = dashboard.build_snapshot(REDIS_URL, ["embed", "offline"], "demo",
                                        drive_hub=tmp_path, telemetry_cache=dashboard._TelemetryCache())

    assert set(snapshot) == {"generated_at", "project", "redis_endpoint", "queues", "telemetry", "gallery"}
    assert snapshot["project"] == "demo"
    assert snapshot["redis_endpoint"] == MASKED_URL
    assert snapshot["queues"][0]["workers"] == _queue_status("embed")["workers"]
    assert snapshot["queues"][0]["counts"] == {"failed": 1}
    assert snapshot["queues"][1] == {"queue": "offline", "error": "Redis status failed (ConnectionError)"}
    assert mocks.queues.call_args_list == [call(REDIS_URL, "embed"), call(REDIS_URL, "offline")]
    assert snapshot["telemetry"]["vms"][0]["gpu_name"] == "NVIDIA Tesla T4"
    mocks.payload.assert_called_once_with(mocks.ccu, mocks.vms)

    assert [run["run_name"] for run in snapshot["gallery"]] == ["new run", "old run"]
    first, second = snapshot["gallery"][0]["items"]
    assert first == {
        "slug": "scene #2", "file_name": "scene #2.PNG", "mtime": 30,
        "url": "/api/image/new%20run/scene%20%232.PNG", "stats": runs.stats,
    }
    assert second["slug"] == "scene 1"
    assert second["stats"] == {}
    _assert_private(json.dumps(snapshot))


def test_telemetry_cache_ttl_and_error_fallback(mocks, monkeypatch):
    now = 10.0
    monkeypatch.setattr(dashboard, "time", SimpleNamespace(monotonic=lambda: now))
    cache = dashboard._TelemetryCache()

    def snapshot():
        return dashboard.build_snapshot(REDIS_URL, [], "demo", drive_hub="", telemetry_cache=cache)["telemetry"]

    first = snapshot()
    now = 19.99
    assert snapshot() == first
    mocks.inspect.assert_called_once_with()
    mocks.payload.assert_called_once_with(mocks.ccu, mocks.vms)

    mocks.inspect.side_effect = TimeoutError("telemetry offline")
    now = 20.0
    failure = {"ok": False, "loading": False, "error": "TimeoutError: telemetry offline", "vms": []}
    assert snapshot() == failure
    now = 23.0
    assert snapshot() == failure
    assert mocks.inspect.call_count == 2
    assert mocks.payload.call_count == 1

    mocks.inspect.side_effect = None
    now = 30.0
    assert snapshot() == first
    assert mocks.inspect.call_count == 3
    assert mocks.payload.call_count == 2


@pytest.fixture
def running_server(mocks, runs, tmp_path, monkeypatch):
    @contextmanager
    def serve(queues=("embed stage", "offline")):
        with dashboard.create_dashboard_server("127.0.0.1", 0, redis_url=REDIS_URL,
                                               queues=queues, project="demo", drive_hub=tmp_path) as httpd:
            assert isinstance(httpd, ThreadingHTTPServer)
            reject_offline = socket.socket.connect

            def connect(sock, address):
                if address == httpd.server_address:
                    return _SOCKET_CONNECT(sock, address)
                return reject_offline(sock, address)

            with monkeypatch.context() as patch:
                patch.setattr(socket.socket, "connect", connect)
                thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
                thread.start()
                try:
                    yield httpd
                finally:
                    httpd.shutdown()
                    thread.join(timeout=5)

    return serve


@pytest.fixture
def server(running_server):
    with running_server() as httpd:
        yield httpd


def _request(server, method, path, headers=None, *, timeout=5):
    conn = HTTPConnection(*server.server_address, timeout=timeout)
    try:
        conn.request(method, path, headers=headers or {})
        response = conn.getresponse()
        body = response.read()
        _assert_private(str(response.getheaders()))
        return response.status, response.getheader("Content-Type"), body
    finally:
        conn.close()


def _live_snapshot(server, *, timeout=5):
    status, content_type, body = _request(server, "GET", "/api/snapshot", timeout=timeout)
    assert (status, content_type) == (200, "application/json; charset=utf-8")
    _assert_private(body.decode())
    return json.loads(body)


def _wait_for_telemetry(server, *, ok):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = _live_snapshot(server)
        telemetry = snapshot["telemetry"]
        if not telemetry.get("loading") and telemetry["ok"] is ok:
            return snapshot
        time.sleep(0.01)
    pytest.fail(f"Telemetry did not settle with ok={ok}")


def _wait_for_queues(server, *, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = _live_snapshot(server)
        queues = snapshot["queues"]
        if queues and not any(queue.get("loading") for queue in queues):
            return queues
        time.sleep(0.01)
    pytest.fail("Queues did not finish loading")


def test_live_dashboard_and_snapshot(running_server, mocks):
    mocks.inspect.side_effect = TimeoutError(f"Live error: {REDIS_URL}, {TOKEN}, {QUERY_TOKEN}")
    with running_server() as server:
        status, content_type, body = _request(server, "GET", "/")
        assert (status, content_type) == (200, "text/html; charset=utf-8")
        html = body.decode()
        for heading in ("Colab Ultra", "Execution Monitor", "Pipeline Flow", "Live Execution Metrics",
                        "Queue Throughput / Stage Distribution", "Live Compute Sparklines",
                        "Pipeline Queues &amp; Active Tasks", "Active Compute Nodes"):
            assert heading in html
        for removed in ("gallery", "timing breakdown", "lightbox", '<dialog', 'id="zoom"', 'id="runs"', "<img"):
            assert removed not in html.lower()
        assert "['Task', 'Status', 'Worker', 'Step', 'Error']" in html
        _assert_private(html)

        snapshot = _wait_for_telemetry(server, ok=False)
        queues = _wait_for_queues(server)
        assert snapshot["redis_endpoint"] == MASKED_URL
        assert queues[0]["queue"] == "embed stage"
        assert snapshot["telemetry"]["loading"] is False
        assert snapshot["telemetry"]["error"].startswith("TimeoutError: Live error:")
        assert _live_snapshot(server)["telemetry"] == snapshot["telemetry"]
    mocks.inspect.assert_called_once_with()


def test_live_snapshot_does_not_block_on_slow_telemetry(running_server, mocks):
    started = threading.Event()
    release = threading.Event()

    def inspect():
        started.set()
        release.wait()
        return mocks.ccu, mocks.vms

    mocks.inspect.side_effect = inspect
    queues = ("embed", "dit", "vae")
    with running_server(queues) as server:
        try:
            assert started.wait(timeout=2), "Server did not start warming telemetry before the first request"
            loaded_queues = _wait_for_queues(server)
            before = time.monotonic()
            snapshot = _live_snapshot(server, timeout=0.75)
            assert time.monotonic() - before < 1
            assert snapshot["telemetry"] == {"ok": False, "loading": True, "error": None, "vms": []}
            assert [queue["queue"] for queue in loaded_queues] == list(queues)
            for queue in loaded_queues:
                expected = _queue_status(queue["queue"])
                assert queue["counts"] == expected["counts"]
                assert queue["workers"] == expected["workers"]
                assert queue["tasks"]["slug"]["status"] == "failed"
            assert _live_snapshot(server, timeout=0.75)["telemetry"] == snapshot["telemetry"]
            mocks.inspect.assert_called_once_with()
            mocks.payload.assert_not_called()
        finally:
            release.set()

        telemetry = _wait_for_telemetry(server, ok=True)["telemetry"]
        assert telemetry["active_vms_count"] == len(telemetry["vms"]) == 1
        assert telemetry["vms"][0]["gpu_name"] == "NVIDIA Tesla T4"
        mocks.payload.assert_called_once_with(mocks.ccu, mocks.vms)


def test_live_snapshot_does_not_block_on_slow_queues(running_server, mocks):
    names = ("embed", "dit", "vae")
    started = {name: threading.Event() for name in names}
    release = threading.Event()

    def client(_redis_url, name):
        def status():
            started[name].set()
            assert release.wait(timeout=5)
            return _queue_status(name)
        return Mock(status=status)

    mocks.queues.side_effect = client
    try:
        with running_server(names) as server:
            assert all(event.wait(timeout=1) for event in started.values())
            before = time.monotonic()
            states = _live_snapshot(server, timeout=0.5)["queues"]
            assert time.monotonic() - before < 0.5
            assert states == [{"queue": name, "loading": True, "counts": {}, "workers": {}, "tasks": {}}
                              for name in names]
            assert _live_snapshot(server, timeout=0.5)["queues"] == states
            release.set()
            ready = _wait_for_queues(server)
            assert [state["queue"] for state in ready] == list(names)
            for state in ready:
                assert state["counts"] == {"failed": 1}
                assert state["tasks"]["slug"]["worker"] == "worker-1"
            assert mocks.queues.call_count == len(names)
    finally:
        release.set()


def test_live_snapshot_serves_stale_telemetry_during_refresh(running_server, mocks, monkeypatch):
    now = 10.0
    monkeypatch.setattr(dashboard, "time", SimpleNamespace(monotonic=lambda: now))
    started = threading.Event()
    release = threading.Event()

    def inspect():
        started.set()
        release.wait()
        raise TimeoutError("refresh failed")

    with running_server() as server:
        first = _wait_for_telemetry(server, ok=True)["telemetry"]
        mocks.inspect.side_effect = inspect
        try:
            now = 19.99
            assert _live_snapshot(server)["telemetry"] == first
            mocks.inspect.assert_called_once_with()

            now = 20.0
            assert _live_snapshot(server, timeout=0.75)["telemetry"] == first
            assert started.wait(timeout=2)
            assert _live_snapshot(server, timeout=0.75)["telemetry"] == first
            assert mocks.inspect.call_count == 2
        finally:
            release.set()

        failed = _wait_for_telemetry(server, ok=False)["telemetry"]
        assert failed == {"ok": False, "loading": False, "error": "TimeoutError: refresh failed", "vms": []}
        assert _live_snapshot(server)["telemetry"] == failed
        assert mocks.inspect.call_count == 2


def test_live_image_and_containment(server, runs, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.png").write_bytes(PNG)
    (runs.root / "linked-run").symlink_to(outside, target_is_directory=True)
    (runs.new_run / "linked-file.png").symlink_to(outside / "private.png")
    (runs.new_run / "renamed-text.png").symlink_to(runs.new_run / "notes.txt")
    (outside / "private.json").write_text('{"seed": -1}', encoding="utf-8")
    (runs.new_run / "scene 1.json").unlink()
    (runs.new_run / "scene 1.json").symlink_to(outside / "private.json")

    assert _request(server, "GET", "/api/image/new%20run/scene%20%232.PNG") == (200, "image/png", PNG)
    assert _request(server, "GET", "/api/image/new%20run/missing.png")[0] == 404
    for path in (
        "../outside/private.png", "new%20run/../old%20run/scene%200.png", "%2e%2e/private.png",
        "new%20run/%2e%2e%2fold%20run%2fscene%200.png", "linked-run/private.png", "new%20run/linked-file.png",
        "new%20run/notes.txt", "new%20run/renamed-text.png",
    ):
        assert _request(server, "GET", f"/api/image/{path}")[0] in (400, 404)

    snapshot = json.loads(_request(server, "GET", "/api/snapshot")[2])
    assert all(run["run_name"] != "linked-run" for run in snapshot["gallery"])
    new_run = next(run for run in snapshot["gallery"] if run["run_name"] == "new run")
    assert [item["slug"] for item in new_run["items"]] == ["scene #2", "scene 1"]
    assert new_run["items"][1]["stats"] == {}


def test_live_clear_queue_matches_allowlist_and_redacts_failures(server, mocks):
    _wait_for_queues(server)
    refreshed = threading.Event()

    def status():
        refreshed.set()
        return _queue_status("embed stage")

    queue = Mock(status=status)
    mocks.queues.side_effect = None
    mocks.queues.return_value = queue
    mocks.queues.reset_mock()
    good_path = "/api/queue/embed%20stage/clear"
    assert _request(server, "POST", "/api/queue/unauthorized/clear")[0] == 400
    assert _request(server, "POST", good_path, {"Origin": "https://elsewhere.test"})[0] == 403
    mocks.queues.assert_not_called()
    queue.clear.assert_not_called()

    status_code, _, body = _request(server, "POST", good_path, {"Origin": f"http://127.0.0.1:{server.server_port}"})
    assert status_code == 200
    assert json.loads(body) == {"ok": True}
    queue.clear.assert_called_once_with()
    assert refreshed.wait(timeout=2), "Clear should trigger an immediate background queue poll"

    queue.clear.side_effect = ConnectionError(f"Redis connection failed: {REDIS_URL} {TOKEN}")
    status_code, _, body = _request(server, "POST", good_path)
    assert status_code == 502
    _assert_private(body.decode())


def test_dashboard_cli_parser_and_dispatch(monkeypatch):
    args = cli.build_parser().parse_args(["dashboard", "--port", "9999", "--no-open", "--project", "demo"])
    assert vars(args) == {
        "cmd": "dashboard", "host": "127.0.0.1", "port": 9999, "project": "demo",
        "no_open": True, "queues": None, "redis_url": None,
    }
    command = Mock()
    monkeypatch.setattr(cli, "cmd_dashboard", command)
    cli.main(["dashboard", "--port", "9999", "--no-open", "--project", "demo"])
    command.assert_called_once_with(args)


@pytest.mark.parametrize(("redis_source", "queue_source"), [("flag", "override"), ("env", "file"), ("config", "default")])
def test_dashboard_cli_config(monkeypatch, tmp_path, capsys, redis_source, queue_source):
    monkeypatch.setattr(dashboard, "detect_local_drive_hub", Mock(return_value=tmp_path))
    load_config = Mock(return_value={"redis_url": REDIS_URL if redis_source == "config" else "redis://config.test"})
    monkeypatch.setattr(dashboard, "load_config", load_config)
    monkeypatch.delenv("COLAB_ULTRA_REDIS_URL", raising=False)
    server = MagicMock(server_port=8765)
    server.__enter__.return_value = server
    create_server = Mock(return_value=server)
    monkeypatch.setattr(dashboard, "create_dashboard_server", create_server)
    open_browser = Mock()
    monkeypatch.setattr(dashboard.webbrowser, "open", open_browser)

    argv = ["dashboard"]
    if redis_source == "flag":
        argv += ["--redis-url", REDIS_URL]
    if redis_source != "config":
        monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", REDIS_URL if redis_source == "env" else "redis://env.test")
    if queue_source == "override":
        argv += ["--queues", " sample embed, sample dit,,sample embed ", "--no-open"]
        queues = ["sample embed", "sample dit"]
    elif queue_source == "file":
        path = tmp_path / "h3_pipeline_demo" / "code" / "pipeline_worker.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({f"{stage}_queue": f"custom_{stage}" for stage in ("embed", "dit", "vae")}),
                        encoding="utf-8")
        queues = ["custom_embed", "custom_dit", "custom_vae"]
    else:
        queues = ["h3_pipe_embed", "h3_pipe_dit", "h3_pipe_vae"]

    cli.main(argv)
    create_server.assert_called_once_with(
        "127.0.0.1", 8765, redis_url=REDIS_URL, queues=queues, project="h3_pipeline_demo", drive_hub=tmp_path,
    )
    assert load_config.call_count == (1 if redis_source == "config" else 0)
    if "--no-open" in argv:
        open_browser.assert_not_called()
    else:
        open_browser.assert_called_once_with("http://127.0.0.1:8765/")
    server.serve_forever.assert_called_once_with()
    _assert_private(capsys.readouterr().out)
