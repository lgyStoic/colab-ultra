"""Redis Streams task queue shared by the workstation and VM."""

import json
import time
from collections import Counter
from contextlib import contextmanager
from urllib import request
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit

_REST_SCHEMES = {"http", "https"}
_REST_PREFIXES = ("https://", "http://")


class ResponseError(Exception):
    pass


def _http_endpoint(parts):
    host = parts.hostname
    if not host or any(char.isspace() or char in "%\\@" for char in host):
        raise ValueError("Invalid Redis REST hostname")
    port = parts.port
    authority = f"[{host}]" if ":" in host else host
    return f"{authority}:{port}" if port is not None else authority


def mask_redis_url(url: str) -> str:
    """Format a Redis endpoint without userinfo or connection parameters."""
    try:
        parts = urlsplit(url)
        if parts.scheme in _REST_SCHEMES:
            authority = _http_endpoint(parts)
            masked = "***@" if "@" in parts.netloc else ""
            return f"{parts.scheme}://{masked}{authority}"
        if parts.scheme in {"redis", "rediss"}:
            if not parts.hostname or (parts.path not in {"", "/"} and not parts.path[1:].isdigit()):
                return "<invalid Redis URL>"
            _ = parts.port  # Validate the port before displaying the authority.
        elif parts.scheme != "unix":
            return "<invalid Redis URL>"
    except ValueError:
        return "<invalid Redis URL>"
    _, separator, host = parts.netloc.rpartition("@")
    authority = f"***@{host}" if separator else parts.netloc
    return f"{parts.scheme}://{authority}{parts.path}"


def _pairs_to_dict(items):
    if isinstance(items, dict):
        return items
    pairs = iter(items)
    return {str(key): value for key, value in zip(pairs, pairs, strict=True)}


def _stream_entries(entries):
    return [(entry[0], _pairs_to_dict(entry[1])) for entry in entries
            if entry is not None and entry[1] is not None]


class _RestCommands:
    """Build REST commands for both immediate and batched execution."""

    def ping(self):
        return self._execute(["PING"], lambda result: result == "PONG")

    def xgroup_create(self, key, group, id="0", mkstream=True):
        command = ["XGROUP", "CREATE", key, group, id]
        if mkstream:
            command.append("MKSTREAM")
        return self._execute(command)

    def xadd(self, key, fields):
        values = [item for key, value in fields.items() for item in (str(key), str(value))]
        return self._execute(["XADD", key, "*", *values])

    def xautoclaim(self, key, group, consumer, *, min_idle_time, start_id="0-0", count=10):
        def decode(result):
            next_start, entries, *deleted = result
            return [next_start, _stream_entries(entries or []), deleted[0] if deleted else []]

        return self._execute([
            "XAUTOCLAIM", key, group, consumer, str(int(min_idle_time)), str(start_id),
            "COUNT", str(int(count)),
        ], decode)

    def xreadgroup(self, group, consumer, streams, *, count=1, block=0):
        def decode(result):
            if not result and block > 0:
                time.sleep(min(block / 1000.0, 1.0))
            return [(key, _stream_entries(entries)) for key, entries in result or []]

        # REST requests are stateless and do not accept BLOCK.
        return self._execute([
            "XREADGROUP", "GROUP", group, consumer, "COUNT", str(int(count)),
            "STREAMS", *streams, *streams.values(),
        ], decode)

    def xack(self, key, group, msg_id):
        return self._execute(["XACK", key, group, msg_id])

    def xlen(self, key):
        return self._execute(["XLEN", key], int)

    def xinfo_groups(self, key):
        return self._execute(["XINFO", "GROUPS", key],
                             lambda result: [_pairs_to_dict(group) for group in result or []])

    def hset(self, key, field, value):
        return self._execute(["HSET", key, str(field), str(value)])

    def hget(self, key, field):
        return self._execute(["HGET", key, str(field)])

    def hgetall(self, key):
        return self._execute(["HGETALL", key], lambda result: _pairs_to_dict(result or []))

    def expire(self, key, ttl):
        return self._execute(["EXPIRE", key, str(int(ttl))])

    def delete(self, *keys):
        return self._execute(["DEL", *keys]) if keys else 0


class _UpstashRestClient(_RestCommands):
    def __init__(self, url: str, *, timeout: float = 15.0):
        parts = urlsplit(url)
        if parts.scheme not in _REST_SCHEMES:
            raise ValueError("Redis REST requires an HTTP or HTTPS URL")
        self._token = unquote(parts.password or parts.username or "")
        if not self._token:
            raise ValueError("Redis REST requires a token in the URL userinfo")
        self.endpoint = f"{parts.scheme}://{_http_endpoint(parts)}"
        self.timeout = timeout

    def _request(self, path: str, body):
        req = request.Request(
            self.endpoint + path, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept-Encoding": "identity"},
        )
        # Send the token only to the configured endpoint, including when it redirects.
        req.add_unredirected_header("Authorization", f"Bearer {self._token}")
        http_error = None
        try:
            response = request.urlopen(req, timeout=self.timeout)
        except HTTPError as exc:
            response = http_error = exc
        with response:
            try:
                data = json.load(response)
            except ValueError:
                if http_error is not None:
                    raise http_error from None
                raise
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict) and item.get("error"):
                raise ResponseError(item["error"])
        if http_error is not None:
            raise http_error
        return [item["result"] for item in items] if isinstance(data, list) else data["result"]

    def _execute(self, command, decode=None):
        result = self._request("/", command)
        return decode(result) if decode is not None else result

    @contextmanager
    def pipeline(self, transaction=True):
        yield _RestPipeline(self, "/multi-exec" if transaction else "/pipeline")


class _RestPipeline(_RestCommands):
    def __init__(self, client, path):
        self._client = client
        self._path = path
        self._commands = []

    def _execute(self, command, decode=None):
        self._commands.append((command, decode))
        return self

    def execute(self):
        if not self._commands:
            return []
        commands, decoders = zip(*self._commands)
        self._commands.clear()
        results = self._client._request(self._path, commands)
        return [decode(result) if decode is not None else result
                for result, decode in zip(results, decoders, strict=True)]


def _response_error(exc, message):
    # Injected clients can provide their own ResponseError without redis-py.
    return message in str(exc) and any(
        cls.__name__ == "ResponseError" for cls in type(exc).__mro__
    )


class TaskQueue:
    def __init__(
        self, redis_url: str, name: str, *, group: str = "workers",
        ttl_sec: int = 7 * 86400, client=None,
    ):
        if client is None and redis_url.startswith(_REST_PREFIXES):
            client = _UpstashRestClient(redis_url)
        elif client is None:
            import redis

            client = redis.from_url(
                redis_url, decode_responses=True, socket_keepalive=True,
                health_check_interval=30, retry_on_timeout=True,
            )
        self.client = client
        self.name = name
        self.group = group
        self.ttl_sec = ttl_sec
        self.stream = f"cu:q:{name}:stream"
        self.state_key = f"cu:q:{name}:state"
        self.workers_key = f"cu:q:{name}:workers"

    @contextmanager
    def _transaction(self):
        # The real Redis client publishes state and stream changes together.
        pipeline = getattr(self.client, "pipeline", None)
        if pipeline is None:
            yield self.client
        else:
            with pipeline(transaction=True) as batch:
                yield batch
                batch.execute()

    def _record(self, client, key, field, **values):
        values["updated_at"] = time.time()
        client.hset(key, field, json.dumps(values))
        client.expire(key, self.ttl_sec)

    def ensure_group(self) -> None:
        try:
            self.client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except Exception as exc:
            if not _response_error(exc, "BUSYGROUP"):
                raise
        else:
            self.client.expire(self.stream, self.ttl_sec)

    def ping(self) -> float:
        start = time.perf_counter()
        self.client.ping()
        return round((time.perf_counter() - start) * 1000, 2)

    def push(self, tasks: list[dict]) -> int:
        entries = []
        for task in tasks:
            slug = task.get("slug") if isinstance(task, dict) else None
            if not isinstance(slug, str) or not slug.strip():
                raise ValueError("Each task must have a non-empty string slug")
            entries.append((slug, json.dumps(task)))
        if entries:
            with self._transaction() as client:
                for slug, payload in entries:
                    client.xadd(self.stream, {"slug": slug, "payload": payload})
                    self._record(client, self.state_key, slug, status="queued", worker="", step="")
                client.expire(self.stream, self.ttl_sec)
                client.expire(self.workers_key, self.ttl_sec)
        return len(entries)

    def claim_batch(
        self, worker_id: str, count: int = 2, *, block_ms: int = 2000,
        reclaim_idle_ms: int = 120_000,
    ) -> list[tuple[str, dict]]:
        if count < 1:
            raise ValueError("count must be positive")
        self.ensure_group()
        self.set_worker_status(worker_id, "polling")
        result = self.client.xautoclaim(
            self.stream, self.group, worker_id, min_idle_time=reclaim_idle_ms,
            start_id="0-0", count=count,
        )
        entries = list(result[1])
        if len(entries) < count:
            streams = self.client.xreadgroup(
                self.group, worker_id, {self.stream: ">"}, count=count - len(entries),
                block=block_ms,
            )
            for _, messages in streams:
                entries.extend(messages)
        claimed = [(msg_id, json.loads(fields["payload"])) for msg_id, fields in entries]
        if claimed:
            with self._transaction() as client:
                for _, task in claimed:
                    self._record(client, self.state_key, task["slug"], status="running",
                                 worker=worker_id, step="claimed")
                self._record(client, self.workers_key, worker_id, status="running")
        return claimed

    def report_progress(self, slug: str, worker_id: str, step: str) -> None:
        with self._transaction() as client:
            self._record(client, self.state_key, slug, status="running", worker=worker_id, step=step)
            self._record(client, self.workers_key, worker_id, status="running",
                         last_slug=slug, step=step)

    def ack(self, msg_id: str, slug: str, worker_id: str, *, output: str = "") -> None:
        with self._transaction() as client:
            client.xack(self.stream, self.group, msg_id)
            self._record(client, self.state_key, slug, status="done", worker=worker_id, output=output)

    def fail(self, msg_id: str, slug: str, worker_id: str, error: str, *, ack: bool = True) -> None:
        with self._transaction() as client:
            self._record(client, self.state_key, slug, status="failed", worker=worker_id, error=error)
            if ack:
                client.xack(self.stream, self.group, msg_id)

    def set_worker_status(self, worker_id: str, status: str) -> None:
        self._record(self.client, self.workers_key, worker_id, status=status)

    def status(self) -> dict:
        stream_len = self.client.xlen(self.stream)
        try:
            groups = self.client.xinfo_groups(self.stream)
        except Exception as exc:
            if not (_response_error(exc, "no such key") or _response_error(exc, "requires the key to exist")):
                raise
            groups = []
            stream_len = 0
        tasks = {slug: json.loads(value) for slug, value in self.client.hgetall(self.state_key).items()}
        workers = {
            worker: json.loads(value)
            for worker, value in self.client.hgetall(self.workers_key).items()
        }
        counts = dict(Counter(task["status"] for task in tasks.values()))
        pending, unclaimed = 0, stream_len
        for group in groups:
            if group["name"] == self.group:
                pending = int(group["pending"])
                lag = group.get("lag")
                if lag is None:
                    entries_read = group.get("entries-read")
                    lag = counts.get("queued", 0) if entries_read is None else stream_len - int(entries_read)
                unclaimed = max(0, int(lag))
                break
        return {
            "queue": self.name, "stream_len": stream_len, "unclaimed": unclaimed,
            "pending": pending, "counts": counts, "all_done": unclaimed == 0 and pending == 0,
            "workers": workers, "tasks": tasks,
        }

    def clear(self) -> None:
        self.client.delete(self.stream, self.state_key, self.workers_key)
