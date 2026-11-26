import json
import os
import subprocess
import sys
from collections import deque
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib import request
from urllib.error import HTTPError

import pytest

from colab_ultra import cli, queue_cli, staging
from colab_ultra.remote.queue import ResponseError, TaskQueue, _UpstashRestClient, mask_redis_url
from recipes import redis_mvp_worker as worker


class FakeRedis:
    def __init__(self):
        self.streams = {}
        self.groups = {}
        self.hashes = {}
        self.expirations = {}
        self.now_ms = 0
        self.autoclaim_size = 3
        self.include_lag = True
        self.include_entries_read = True

    def ping(self):
        return True

    def xgroup_create(self, key, group, id, mkstream):
        assert id == "0" and mkstream
        if (key, group) in self.groups:
            raise ResponseError("BUSYGROUP Consumer Group name already exists")
        self.streams.setdefault(key, {})
        self.groups[key, group] = {"read": 0, "pending": {}}

    def xadd(self, key, fields):
        entries = self.streams.setdefault(key, {})
        msg_id = f"{len(entries) + 1}-0"
        entries[msg_id] = fields.copy()
        return msg_id

    def xautoclaim(self, key, group, worker_id, *, min_idle_time, start_id, count):
        assert start_id == "0-0"
        pending = self.groups[key, group]["pending"]
        messages = []
        for msg_id, (_, delivered_at) in pending.items():
            if self.now_ms - delivered_at >= min_idle_time:
                pending[msg_id] = (worker_id, self.now_ms)
                messages.append((msg_id, self.streams[key][msg_id]))
                if len(messages) == count:
                    break
        return ["0-0", messages, []][:self.autoclaim_size]

    def xreadgroup(self, group, worker_id, streams, *, count, block):
        assert len(streams) == 1 and count > 0
        key, cursor = next(iter(streams.items()))
        assert cursor == ">"
        info = self.groups[key, group]
        start = info["read"]
        messages = list(self.streams[key].items())[start:start + count]
        info["read"] += len(messages)
        for msg_id, _ in messages:
            info["pending"][msg_id] = (worker_id, self.now_ms)
        return [(key, messages)] if messages else []

    def xack(self, key, group, msg_id):
        return int(self.groups[key, group]["pending"].pop(msg_id, None) is not None)

    def xlen(self, key):
        return len(self.streams.get(key, ()))

    def xinfo_groups(self, key):
        if key not in self.streams:
            raise ResponseError("no such key")
        return [
            {"name": group, "pending": len(info["pending"]),
             "lag": self.xlen(key) - info["read"] if self.include_lag else None,
             "entries-read": info["read"] if self.include_entries_read else None}
            for (stream, group), info in self.groups.items() if stream == key
        ]

    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hgetall(self, key):
        return self.hashes.get(key, {}).copy()

    def expire(self, key, ttl):
        if key in self.streams or key in self.hashes:
            self.expirations[key] = ttl

    def delete(self, *keys):
        for key in keys:
            self.streams.pop(key, None)
            self.hashes.pop(key, None)
            self.expirations.pop(key, None)
        self.groups = {key: group for key, group in self.groups.items() if key[0] not in keys}


@pytest.fixture
def queue():
    return TaskQueue("", "demo", ttl_sec=60, client=FakeRedis())


def test_enqueue_claim_progress_and_ack(queue):
    assert queue.status()["all_done"]
    task = {"slug": "tile", "prompt": "mountains"}
    assert queue.push([task]) == 1
    status = queue.status()
    assert (status["stream_len"], status["unclaimed"], status["pending"]) == (1, 1, 0)
    assert status["counts"] == {"queued": 1}
    queued = status["tasks"]["tile"]
    assert queued["worker"] == queued["step"] == ""
    assert queued["updated_at"] > 0
    (msg_id, claimed), = queue.claim_batch("A")
    assert claimed == task
    status = queue.status()
    assert (status["unclaimed"], status["pending"], status["all_done"]) == (0, 1, False)
    assert status["counts"] == {"running": 1}
    assert status["tasks"]["tile"]["step"] == "claimed"
    assert status["workers"]["A"]["status"] == "running"
    queue.report_progress("tile", "A", "step 2/4")
    status = queue.status()
    assert status["tasks"]["tile"]["step"] == "step 2/4"
    assert status["workers"]["A"]["last_slug"] == "tile"
    assert status["workers"]["A"]["step"] == "step 2/4"
    queue.ack(msg_id, "tile", "A", output="tile.json")
    status = queue.status()
    assert (status["stream_len"], status["unclaimed"], status["pending"]) == (1, 0, 0)
    assert status["all_done"] is True
    assert status["counts"] == {"done": 1}
    assert status["tasks"]["tile"]["output"] == "tile.json"
    assert queue.client.expirations == {
        queue.stream: 60, queue.state_key: 60, queue.workers_key: 60,
    }
    queue.clear()
    assert queue.status() == {
        "queue": "demo", "stream_len": 0, "unclaimed": 0, "pending": 0,
        "counts": {}, "all_done": True, "workers": {}, "tasks": {},
    }


@pytest.mark.parametrize("task", [{}, {"slug": None}, {"slug": 1}, {"slug": "  "}, "tile"])
def test_invalid_push_does_not_write_partial_batch(queue, task):
    with pytest.raises(ValueError, match="slug"):
        queue.push([{"slug": "valid"}, task])
    assert queue.status()["stream_len"] == 0


def test_push_commits_stream_and_state_in_one_transaction(queue, monkeypatch):
    pipeline = MagicMock()
    batch = pipeline.return_value.__enter__.return_value
    monkeypatch.setattr(queue.client, "pipeline", pipeline, raising=False)
    assert queue.push([{"slug": "tile"}]) == 1
    pipeline.assert_called_once_with(transaction=True)
    assert [call[0] for call in batch.method_calls] == [
        "xadd", "hset", "expire", "expire", "expire", "execute",
    ]


@pytest.mark.parametrize("ack", [True, False])
def test_failure_can_leave_message_pending(queue, ack):
    queue.push([{"slug": "bad"}])
    (msg_id, _), = queue.claim_batch("A")
    queue.fail(msg_id, "bad", "A", "simulation error", ack=ack)
    status = queue.status()
    assert status["counts"] == {"failed": 1}
    assert status["tasks"]["bad"]["error"] == "simulation error"
    assert status["pending"] == int(not ack)
    assert status["all_done"] is ack


@pytest.mark.parametrize("reply_size", [2, 3])
def test_reclaim_after_idle_and_fill_batch_with_new_tasks(queue, reply_size):
    queue.client.autoclaim_size = reply_size
    queue.push([{"slug": "stalled"}])
    stalled, = queue.claim_batch("A")
    queue.client.now_ms = 99
    assert queue.claim_batch("B", reclaim_idle_ms=100) == []
    assert queue.status()["workers"]["B"]["status"] == "polling"
    queue.push([{"slug": "new"}])
    queue.client.now_ms = 100
    reclaimed, fresh = queue.claim_batch("B", reclaim_idle_ms=100)
    assert reclaimed == stalled
    assert fresh[1]["slug"] == "new"
    assert queue.status()["tasks"]["stalled"]["worker"] == "B"
    assert queue.status()["pending"] == 2
    for msg_id, task in (reclaimed, fresh):
        queue.ack(msg_id, task["slug"], "B")
    assert queue.status()["all_done"]


@pytest.mark.parametrize("entries_read", [True, False])
def test_status_when_lag_is_unavailable(queue, entries_read):
    queue.client.include_lag = False
    queue.client.include_entries_read = entries_read
    queue.push([{"slug": "one"}, {"slug": "two"}])
    queue.ensure_group()
    assert queue.status()["unclaimed"] == 2
    (msg_id, _), = queue.claim_batch("A", 1)
    queue.ack(msg_id, "one", "A")
    assert queue.status()["unclaimed"] == 1
    queue.claim_batch("B", 1)
    status = queue.status()
    assert (status["unclaimed"], status["pending"], status["all_done"]) == (0, 1, False)


@pytest.mark.parametrize(("url", "masked"), [
    ("rediss://default:dummy%40token@example.test:6379/0", "rediss://***@example.test:6379/0"),
    ("rediss://dummy-token@example.test", "rediss://***@example.test"),
    ("redis://:dummy-token@[::1]:6379?password=dummy-token", "redis://***@[::1]:6379"),
    ("redis://localhost:6379/0", "redis://localhost:6379/0"),
    ("rediss://default:dummy/token@example.test", "<invalid Redis URL>"),
], ids=["userinfo", "token-only", "ipv6-and-query", "no-userinfo", "invalid-userinfo"])
def test_mask_redis_url(url, masked):
    assert mask_redis_url(url) == masked


@pytest.mark.parametrize(("url", "masked"), [
    ("https://default:dummy-secret@example.upstash.io", "https://***@example.upstash.io"),
    ("http://dummy-token@localhost:8080/ping?password=dummy-token#dummy-token",
     "http://***@localhost:8080"),
    ("https://[::1]:8443?token=dummy-token", "https://[::1]:8443"),
    ("https://default:dummy-secret@/ping", "<invalid Redis URL>"),
    ("https://default:dummy-secret@example.upstash.io:invalid", "<invalid Redis URL>"),
    ("https://example host", "<invalid Redis URL>"),
    ("https://[::1", "<invalid Redis URL>"),
    ("ftp://example.upstash.io", "<invalid Redis URL>"),
], ids=["https", "http", "ipv6", "no-host", "bad-port", "bad-host", "bad-ipv6", "bad-scheme"])
def test_mask_rest_url(url, masked):
    assert mask_redis_url(url) == masked


@pytest.fixture
def rest_http(monkeypatch):
    responses, requests = deque(), []

    def urlopen(req, *, timeout):
        requests.append((req, json.loads(req.data), timeout))
        response = responses.popleft()
        if isinstance(response, HTTPError):
            raise response
        return BytesIO(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr(request, "urlopen", urlopen)
    yield responses, requests
    assert not responses


def test_rest_ping_and_cli_without_redis(rest_http, monkeypatch, capsys):
    responses, requests = rest_http
    responses.extend([{"result": "PONG"}, {"result": "PONG"}])
    monkeypatch.setitem(sys.modules, "redis", None)
    url = "https://default:dummy%40token@example.upstash.io:8443/ping?token=dummy-token"
    assert _UpstashRestClient(url, timeout=2.5).ping() is True
    monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", url)
    cli.main(["queue", "ping"])
    output = capsys.readouterr().out
    assert output.startswith("https://***@example.upstash.io:8443 ") and output.strip().endswith("ms")
    assert "dummy" not in output
    for (req, body, timeout), expected_timeout in zip(requests, (2.5, 15.0), strict=True):
        assert req.full_url == "https://example.upstash.io:8443/"
        assert req.get_method() == "POST"
        assert req.get_header("Authorization") == "Bearer dummy@token"
        assert req.get_header("Content-type") == "application/json"
        assert req.get_header("Accept-encoding") == "identity"
        assert body == ["PING"] and timeout == expected_timeout


def test_rest_requires_a_token():
    with pytest.raises(ValueError, match="token in the URL userinfo"):
        _UpstashRestClient("https://example.upstash.io?token=unused")


def test_rest_push_uses_multi_exec(rest_http):
    responses, requests = rest_http
    responses.append([{"result": value} for value in ("1-0", 1, 1, 1, 0)])
    queue = TaskQueue("http://dummy-token@localhost:8080", "demo", ttl_sec=60)
    assert queue.push([{"slug": "tile", "prompt": "mountains"}]) == 1
    (req, body, _), = requests
    assert req.full_url == "http://localhost:8080/multi-exec"
    assert req.get_header("Authorization") == "Bearer dummy-token"
    assert body[0][:4] == ["XADD", queue.stream, "*", "slug"]
    assert body[0][4:6] == ["tile", "payload"]
    assert json.loads(body[0][6]) == {"slug": "tile", "prompt": "mountains"}
    assert body[1][:3] == ["HSET", queue.state_key, "tile"]
    assert json.loads(body[1][3])["status"] == "queued"
    assert body[2:] == [["EXPIRE", key, "60"] for key in
                        (queue.state_key, queue.stream, queue.workers_key)]


def test_rest_status_decodes_flat_group_and_hash_replies(rest_http):
    responses, requests = rest_http
    queue = TaskQueue("https://dummy-token@example.upstash.io", "demo")
    worker_state = {"status": "running", "updated_at": 1}
    responses.extend([
        {"result": "2"},
        {"result": [["name", "workers", "pending", 1, "lag", None, "entries-read", 1]]},
        {"result": ["tile", json.dumps({"status": "running"})]},
        {"result": {"A": json.dumps(worker_state)}},
    ])
    status = queue.status()
    assert (status["stream_len"], status["unclaimed"], status["pending"]) == (2, 1, 1)
    assert status["tasks"] == {"tile": {"status": "running"}}
    assert status["workers"] == {"A": worker_state}
    assert [body for _, body, _ in requests] == [
        ["XLEN", queue.stream], ["XINFO", "GROUPS", queue.stream],
        ["HGETALL", queue.state_key], ["HGETALL", queue.workers_key],
    ]


def test_rest_decodes_stream_entries_and_polls_locally(rest_http, monkeypatch):
    responses, requests = rest_http
    client = _UpstashRestClient("https://dummy-token@example.upstash.io")
    fields = {"slug": "tile", "payload": '{"slug":"tile"}'}
    flat = [item for pair in fields.items() for item in pair]
    responses.extend([
        {"result": ["2-0", [None, ["1-0", flat], ["0-1", None]], ["0-1"]]},
        {"result": ["0-0", []]},
        {"result": [["stream", [["2-0", flat]]]]},
        {"result": None},
        {"result": []},
        {"result": []},
    ])
    sleeps = []
    monkeypatch.setattr("colab_ultra.remote.queue.time.sleep", sleeps.append)
    assert client.xautoclaim("stream", "group", "A", min_idle_time=100, count=2) == [
        "2-0", [("1-0", fields)], ["0-1"],
    ]
    assert client.xautoclaim("stream", "group", "A", min_idle_time=100) == ["0-0", [], []]
    streams = {"stream": ">"}
    assert client.xreadgroup("group", "A", streams, count=2, block=2000) == [
        ("stream", [("2-0", fields)]),
    ]
    assert client.xreadgroup("group", "A", streams, block=2000) == []
    assert client.xreadgroup("group", "A", streams, block=50) == []
    assert client.xreadgroup("group", "A", streams) == []
    assert sleeps == [1.0, 0.05]
    bodies = [body for _, body, _ in requests]
    assert bodies[0] == ["XAUTOCLAIM", "stream", "group", "A", "100", "0-0", "COUNT", "2"]
    assert bodies[2] == ["XREADGROUP", "GROUP", "group", "A", "COUNT", "2", "STREAMS", "stream", ">"]
    assert all("BLOCK" not in body for body in bodies)


def test_rest_group_errors_and_missing_key(rest_http):
    responses, requests = rest_http
    queue = TaskQueue("https://dummy-token@example.upstash.io", "demo")
    responses.extend([
        {"error": "BUSYGROUP Consumer Group name already exists"},
        {"result": 0},
        HTTPError("https://example.upstash.io/", 400, "Bad Request", None,
                  BytesIO(b'{"error":"ERR no such key"}')),
        {"result": []},
        {"result": None},
        {"result": 0},
        HTTPError("https://example.upstash.io/", 400, "Bad Request", None,
                  BytesIO(b'{"error":"ERR The XINFO GROUPS subcommand requires the key to exist"}')),
        {"result": []},
        {"result": None},
        {"error": "ERR unexpected"},
    ])
    queue.ensure_group()
    assert queue.status()["all_done"]
    upstash_status = queue.status()
    assert (upstash_status["stream_len"], upstash_status["unclaimed"], upstash_status["pending"]) == (0, 0, 0)
    with pytest.raises(ResponseError, match="unexpected"):
        queue.ensure_group()
    assert requests[0][1] == ["XGROUP", "CREATE", queue.stream, "workers", "0", "MKSTREAM"]


def test_rest_pipeline_commands_and_per_item_errors(rest_http):
    responses, requests = rest_http
    client = _UpstashRestClient("https://dummy-token@example.upstash.io")
    responses.extend([
        [{"result": 1}, {"result": "value"}, {"result": 1}, {"result": 1}, {"result": 2}],
        [{"result": 1}, {"error": "ERR wrong type"}],
    ])
    with client.pipeline(transaction=False) as batch:
        batch.hset("state", "tile", "value")
        batch.hget("state", "tile")
        batch.expire("state", 60)
        batch.xack("stream", "group", "1-0")
        batch.delete("stream", "state")
        assert batch.execute() == [1, "value", 1, 1, 2]
    assert requests[0][0].full_url == "https://example.upstash.io/pipeline"
    assert requests[0][1] == [
        ["HSET", "state", "tile", "value"], ["HGET", "state", "tile"], ["EXPIRE", "state", "60"],
        ["XACK", "stream", "group", "1-0"], ["DEL", "stream", "state"],
    ]
    with client.pipeline() as batch:
        batch.delete("state")
        batch.hgetall("stream")
        with pytest.raises(ResponseError, match="wrong type"):
            batch.execute()
    assert requests[1][0].full_url == "https://example.upstash.io/multi-exec"


def test_redis_published_import_without_dependency(tmp_path):
    scripts = staging.sync_remote(tmp_path).parent
    result = subprocess.run(
        [sys.executable, "-S", "-c",
         "from cu_remote.queue import TaskQueue; import sys; assert 'redis' not in sys.modules"],
        cwd=scripts, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def fake_connection(monkeypatch):
    client = FakeRedis()
    connections = []

    def from_url(url, **options):
        connections.append(url)
        assert options == {
            "decode_responses": True, "socket_keepalive": True,
            "health_check_interval": 30, "retry_on_timeout": True,
        }
        return client

    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(from_url=from_url))
    return client, connections


def test_cli_queue_commands(fake_connection, monkeypatch, capsys):
    client, connections = fake_connection
    url = "rediss://default:dummy-token@example.test:6379"
    monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", url)
    cli.main(["queue", "ping"])
    ping = capsys.readouterr().out
    assert ping.startswith("rediss://***@example.test:6379")
    assert ping.strip().endswith(" ms") and "dummy-token" not in ping
    cli.main(["queue", "push", "demo", "--demo", "3"])
    assert "Pushed 3" in capsys.readouterr().out
    payloads = [json.loads(row["payload"]) for row in client.streams["cu:q:demo:stream"].values()]
    assert [task["slug"] for task in payloads] == ["task_01", "task_02", "task_03"]
    assert all(2 <= task["sleep_sec"] <= 4 for task in payloads)
    cli.main(["queue", "--redis-url", url, "status", "demo", "--json"])
    assert json.loads(capsys.readouterr().out)["counts"] == {"queued": 3}
    cli.main(["queue", "status", "demo"])
    summary = capsys.readouterr().out
    assert all(word in summary for word in ("unclaimed=3", "pending=0", "Workers", "Tasks", "task_01"))
    cli.main(["queue", "clear", "demo", "--redis-url", url])
    assert "Cleared queue demo" in capsys.readouterr().out
    assert client.xlen("cu:q:demo:stream") == 0
    assert connections == [url] * 5


def test_cli_prompt_file_and_config_fallback(fake_connection, tmp_path, monkeypatch):
    client, connections = fake_connection
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({
        "out_dir": "out", "run_name": "run",
        "prompts": [{"slug": "tile", "prompt": "mountains", "out_dir": "custom"}],
    }))
    monkeypatch.delenv("COLAB_ULTRA_REDIS_URL", raising=False)
    monkeypatch.setattr(queue_cli, "load_config", lambda: {"redis_url": "redis://localhost"})
    cli.main(["queue", "push", "demo", "--file", str(path)])
    row, = client.streams["cu:q:demo:stream"].values()
    assert json.loads(row["payload"]) == {
        "slug": "tile", "prompt": "mountains", "out_dir": "custom", "run_name": "run",
    }
    assert connections == ["redis://localhost"]


def test_cli_empty_url_and_error_redaction(monkeypatch):
    monkeypatch.delenv("COLAB_ULTRA_REDIS_URL", raising=False)
    monkeypatch.setattr(queue_cli, "load_config", lambda: {})
    with pytest.raises(SystemExit, match="COLAB_ULTRA_REDIS_URL"):
        cli.main(["queue", "ping"])

    def broken_client(url, name):
        raise ValueError(f"Bad connection {url}")

    monkeypatch.setattr(queue_cli, "TaskQueue", broken_client)
    with pytest.raises(SystemExit, match=r"rediss://\*\*\*@example.test") as error:
        cli.main(["queue", "ping", "--redis-url", "rediss://dummy-token@example.test"])
    assert "dummy-token" not in str(error.value)


@pytest.mark.parametrize("mode", ["flag", "env"])
def test_worker_dry_run_without_dependency(tmp_path, mode):
    script = Path(worker.__file__).resolve()
    cmd = [sys.executable, "-S", str(script)]
    env = {**os.environ, "CU_DRY_RUN": "1" if mode == "env" else "0"}
    if mode == "flag":
        cmd.append("--dry-run")
    result = subprocess.run(cmd, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "Dry run" in result.stdout
    assert not (tmp_path / "runs").exists()


def test_worker_reads_project_settings_and_credentials(tmp_path, monkeypatch):
    code = tmp_path / "code"
    code.mkdir()
    (code / "mvp_worker.json").write_text(json.dumps({
        "queue_name": "demo", "batch_size": 3, "reclaim_idle_ms": 400,
        "simulate_crash_slug": "tile",
    }))
    (code / ".redis_url").write_text("redis://file.test\n")
    monkeypatch.chdir(code)
    for key in ("REDIS_URL", "QUEUE_NAME", "WORKER_ID", "BATCH_SIZE",
                "RECLAIM_IDLE_MS", "SIMULATE_CRASH_SLUG"):
        monkeypatch.delenv(f"COLAB_ULTRA_{key}", raising=False)
    monkeypatch.setenv("CU_DRY_RUN", "0")
    monkeypatch.setenv("COLAB_ULTRA_SESSION", "session-A")
    monkeypatch.setenv("COLAB_ULTRA_BATCH_SIZE", "4")
    installed, connections = [], []

    def connect(url, name):
        connections.append((url, name))
        return "fake queue"

    def run(queue, settings, run_dir):
        assert queue == "fake queue"
        assert settings == {
            "queue_name": "demo", "worker_id": "session-A", "batch_size": 4,
            "reclaim_idle_ms": 400, "simulate_crash_slug": "tile",
        }
        assert run_dir == tmp_path / "runs" / "demo"
        return 0

    monkeypatch.setattr(worker, "_install_redis", lambda: installed.append(True))
    monkeypatch.setattr(worker, "_load_queue", lambda: connect)
    monkeypatch.setattr(worker, "_run", run)
    assert worker.main([]) == 0
    monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", "rediss://env.test")
    assert worker.main([]) == 0
    monkeypatch.setenv("COLAB_ULTRA_REDIS_URL", "https://dummy-token@env.test")
    assert worker.main([]) == 0
    assert installed == [True, True]
    assert connections == [
        ("redis://file.test", "demo"), ("rediss://env.test", "demo"),
        ("https://dummy-token@env.test", "demo"),
    ]


def test_worker_crash_reclaim_and_exit_with_other_pending(queue, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(worker, "CRASH_DIR", tmp_path)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    synced = []
    monkeypatch.setattr(worker.os, "sync", lambda: synced.append(True))
    sync_count_at_ack = []
    original_ack = queue.ack

    def check_ack(msg_id, slug, worker_id, *, output):
        assert json.loads(Path(output).read_text())["slug"] == slug
        sync_count_at_ack.append(len(synced))
        original_ack(msg_id, slug, worker_id, output=output)

    monkeypatch.setattr(queue, "ack", check_ack)
    queue.push([{"slug": "other"}, {"slug": "crash", "sleep_sec": 0}])
    queue.claim_batch("other-worker", 1)
    queue.client.now_ms = 1
    settings = {"worker_id": "A", "batch_size": 1, "reclaim_idle_ms": 100,
                "simulate_crash_slug": "crash"}
    run_dir = tmp_path / "runs" / "demo"
    assert worker._run(queue, settings, run_dir) == 1
    assert "[CrashSim]" in capsys.readouterr().out
    assert queue.status()["tasks"]["crash"]["step"] == "step 4/4"
    assert queue.status()["pending"] == 2
    assert not run_dir.exists()

    # Refresh the other consumer's delivery before the crashed task is reclaimed.
    queue.client.now_ms = 100
    queue.claim_batch("other-worker", 1, reclaim_idle_ms=100)
    queue.client.now_ms = 101
    assert worker._run(queue, {**settings, "worker_id": "B"}, run_dir) == 0
    assert capsys.readouterr().out.splitlines() == [
        "[Worker B] claimed 1: crash",
        f"[Worker B] done crash -> {run_dir / 'crash.json'}",
        "[Worker B] queue empty, exiting",
    ]
    assert sync_count_at_ack == [1]
    assert (tmp_path / "crashed_crash").exists()
    status = queue.status()
    assert status["tasks"]["crash"]["status"] == "done"
    assert status["workers"]["B"]["status"] == "stopped"
    assert (status["unclaimed"], status["pending"], status["all_done"]) == (0, 1, False)
