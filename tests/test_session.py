import signal
import tarfile
from http.client import RemoteDisconnected
from types import SimpleNamespace as NS
from unittest.mock import DEFAULT, Mock

import pytest
import requests
import websocket

from colab_ultra import cli, drive, retry, session
from colab_ultra.backends import colab, jupyter, kaggle


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(response=response)


@pytest.mark.parametrize("exc", [ConnectionError(), RemoteDisconnected(), TimeoutError(),
                                  requests.Timeout(), *(http_error(s) for s in (502, 503, 504))])
def test_backoff(monkeypatch, exc):
    sleep = Mock()
    monkeypatch.setattr(retry.time, "sleep", sleep)
    fn = Mock(side_effect=[exc] * 4 + [42])
    assert retry.retry(fn) == 42
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2, 4, 8]


@pytest.mark.parametrize("exc", [ValueError(), http_error(403), http_error(500), requests.exceptions.SSLError()])
def test_permanent_errors(monkeypatch, exc):
    sleep = Mock()
    monkeypatch.setattr(retry.time, "sleep", sleep)
    with pytest.raises(type(exc)):
        retry.retry(Mock(side_effect=exc))
    sleep.assert_not_called()


@pytest.fixture
def kaggle_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle, "KAGGLE_SESSIONS_PATH", tmp_path / "kaggle_sessions.json")
    target = dict(name="target", provider="kaggle", accelerator="T4x2", endpoint="kaggle-123",
                  session_id=123, proxy_url="https://kaggle.invalid")
    other = dict(target, name="other", endpoint="kaggle-456", session_id=456)
    registry = {"target": target, "other": other}
    kaggle._save_kaggle_sessions(registry)
    return registry


@pytest.mark.parametrize("flags", [[], ["--colab"]])
def test_colab_up_removes_same_name_kaggle_record(monkeypatch, kaggle_registry, flags):
    start_colab = Mock()
    start_kaggle = Mock(side_effect=RuntimeError("Kaggle unavailable"))
    monkeypatch.setattr(session.subprocess, "check_call", start_colab)
    monkeypatch.setattr(session, "spin_up_kaggle_session", start_kaggle)
    monkeypatch.setattr(session, "colab_state", Mock(sync_sessions=Mock(return_value=({}, []))))
    monkeypatch.setattr(session, "resolve_colab_bin", lambda: "fake-colab")
    monkeypatch.setattr(session, "load_config", lambda: {})
    monkeypatch.setattr(session, "load_kaggle_token", lambda: "fake-token")
    monkeypatch.setattr(session, "get_kaggle_quota", lambda: {"gpu_remaining_hours": 1})
    monkeypatch.setattr(session, "inspect_all_vms", lambda: (None, []))
    monkeypatch.setattr(session, "_record_starts", Mock())
    monkeypatch.setattr(session, "render_dashboard", Mock())

    args = cli.build_parser().parse_args(["up", "-s", "target", "--gpu", "T4", "--no-drive", *flags])
    session.cmd_up(args)

    assert start_kaggle.call_count == (0 if flags else 1)
    start_colab.assert_called_once_with(["fake-colab", "--auth", "adc", "new", "-s", "target", "--gpu", "T4"],
                                       stdout=None, stderr=None)
    assert kaggle._load_kaggle_sessions() == {"other": kaggle_registry["other"]}


@pytest.mark.parametrize("colab_live", [True, False])
def test_exec_with_unreachable_same_name_kaggle_proxy(monkeypatch, kaggle_registry, colab_live):
    get = Mock(side_effect=requests.exceptions.ReadTimeout("Kaggle proxy timed out"))
    run = Mock(return_value=(0, "", ""))
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(session, "assignments", lambda: [{"name": "target"}] if colab_live else [])
    monkeypatch.setattr(session, "run_detached", run)
    args = cli.build_parser().parse_args(["exec", "-s", "target", "print(1)"])

    if colab_live:
        session.cmd_exec(args)
        assert run.call_args.args[0] is args
        get.assert_not_called()
    else:
        with pytest.raises(SystemExit, match="Cannot verify Kaggle session target"):
            session.cmd_exec(args)
        get.assert_called_once_with("https://kaggle.invalid/api/kernels", timeout=8)
        run.assert_not_called()
    assert kaggle._load_kaggle_sessions() == kaggle_registry


def test_exec_transport_boundary(monkeypatch):
    """Two failed handshakes can retry. A lost acknowledgement cannot resend the job."""
    sock = Mock()
    sock.send.side_effect = ConnectionResetError("send could have reached Jupyter")
    connect = Mock(side_effect=[ConnectionError(), websocket.WebSocketTimeoutException(), sock])
    monkeypatch.setattr(jupyter.websocket, "create_connection", connect)
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    monkeypatch.setattr(session, "_load_kaggle_sessions", lambda: {})
    monkeypatch.setattr(colab, "resolve_proxy", lambda name: ("https://vm.invalid", "mock"))
    monkeypatch.setattr(jupyter, "jupyter_ensure_kernel_id", lambda *a: "kernel")
    with pytest.raises(RuntimeError, match="submission uncertain"):
        session.cmd_exec(NS(code="print(1)", file=None, session="target", restart_kernel=False,
                            project=None, timeout=60, json=False))
    assert connect.call_count == 3
    sock.send.assert_called_once()
    sock.close.assert_called_once()


def test_disconnect_after_send(monkeypatch):
    sock = Mock()
    sock.recv.side_effect = ConnectionResetError()
    connect = Mock(return_value=sock)
    monkeypatch.setattr(jupyter.websocket, "create_connection", connect)
    with pytest.raises(RuntimeError, match="after submission"):
        jupyter.jupyter_exec_ws("https://vm.invalid", "pass", kernel_id="id")
    sock.send.assert_called_once()
    connect.assert_called_once()


def test_resolve_without_selecting_unrelated_vm(monkeypatch):
    named = NS(endpoint="X", runtime_proxy_info=NS(url="u", token="t"))
    sync = Mock(side_effect=[RemoteDisconnected(), ({"a": named}, [named]), ({}, [named])])
    monkeypatch.setattr(colab, "colab_state", NS(sync_sessions=sync))
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    assert colab.resolve_proxy("a") == ("u", "t")
    with pytest.raises(SystemExit, match="colab-ultra status.*colab-ultra up"):
        colab.resolve_proxy("stale")


def test_down_prefers_same_name_colab_assignment(monkeypatch, kaggle_registry):
    vm = dict(name="target", provider="colab", endpoint="colab-123", accelerator="T4")
    teardown = Mock()
    monkeypatch.setattr(session, "assignments", lambda: [vm])
    monkeypatch.setattr(session, "teardown_verified", teardown)
    monkeypatch.setattr(session, "load_config", lambda: {})
    monkeypatch.setattr(session, "detect_local_drive_hub", lambda: None)
    ccu = NS(paid_compute_units_balance=99, consumption_rate_hourly=0)
    monkeypatch.setattr(session, "inspect_all_vms", lambda: (ccu, []))
    for name in ("sync_dashboard_json", "log_breaker_event", "_settle"):
        monkeypatch.setattr(session, name, Mock())

    session.cmd_down(cli.build_parser().parse_args(["down", "-s", "target"]))

    teardown.assert_called_once_with(vm, None, backup=True)


@pytest.mark.parametrize("active_after, stops, success", [([False], 1, True), ([True, False], 2, True), ([True, True], 2, False)])
@pytest.mark.parametrize("flags", [[], ["--no-backup"]])
def test_verified_down(monkeypatch, active_after, stops, success, flags):
    vm = dict(endpoint="X", name="a", provider="colab", accelerator="L4", proxy_url="u", proxy_token="t")
    state = Mock()
    state.client.list_assignments.side_effect = [[NS(endpoint="X")] if flag else [] for flag in active_after]
    state.store.list.return_value = {"a": NS(endpoint="X")}
    monkeypatch.setattr(session, "colab_state", state)
    monkeypatch.setattr(session, "assignments", lambda: [vm])
    monkeypatch.setattr(session, "_load_kaggle_sessions", lambda: {})
    monkeypatch.setattr(session, "inspect_all_vms", lambda: (NS(paid_compute_units_balance=99, consumption_rate_hourly=0), []))
    monkeypatch.setattr(session.os.path, "expanduser", lambda p: "/unused/test/path")
    for name in ("load_config", "detect_local_drive_hub", "sync_dashboard_json", "log_breaker_event"):
        monkeypatch.setattr(session, name, Mock(return_value={}))
    backup = Mock()
    monkeypatch.setattr(session, "backup_vm_outputs_to_drive", backup)
    monkeypatch.setattr(session.time, "sleep", lambda _: None)
    args = cli.build_parser().parse_args(["down", "-s", "a", *flags])
    if success:
        session.cmd_down(args)
        state.store.remove.assert_called_once_with("a")
    else:
        with pytest.raises(SystemExit, match="still active"):
            session.cmd_down(args)
        state.store.remove.assert_not_called()
    assert state.client.unassign.call_count == stops
    if flags:
        backup.assert_not_called()
    else:
        backup.assert_called_once()


def test_down_deadline_includes_discovery(monkeypatch):
    previous = signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(session, "_down", lambda args: signal.raise_signal(signal.SIGALRM))
    with pytest.raises(SystemExit, match="120s"):
        session.cmd_down(None)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


@pytest.mark.parametrize("outcome", [TimeoutError("ws timeout"), ConnectionError("ws reset"), (1, "", "mount err")])
@pytest.mark.parametrize("listing, success", [({"type": "directory", "content": []}, True), ({"type": "file"}, False)])
def test_mount_timeout_checks_remote_listing(monkeypatch, outcome, listing, success):
    monkeypatch.setattr(drive, "mount_drive", Mock())
    monkeypatch.setattr(colab, "resolve_proxy", lambda _: ("https://vm.invalid", "test"))
    ws = Mock(side_effect=outcome) if isinstance(outcome, BaseException) else Mock(return_value=outcome)
    monkeypatch.setattr(drive, "jupyter_exec_ws", ws)
    response = Mock()
    response.json.return_value = listing
    get = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", get)
    if success:
        drive.initialize_drive("colab", "a")
    else:
        with pytest.raises(RuntimeError):
            drive.initialize_drive("colab", "a")
    assert ws.call_args.kwargs["proxy_token"] == "test"
    assert get.call_args.args[0].endswith("/api/contents/content/drive/MyDrive")


@pytest.mark.parametrize("accept", [True, False])
def test_mount_credential_propagation(monkeypatch, accept):
    import json

    client = Mock(colab_domain="https://example.invalid")
    def request(method, url, *, params, headers, timeout):
        assert params["record"] == "false" and url.endswith("/endpoint")
        if method == "GET":
            payload = {"token": "xsrf"}
        else:
            assert headers["x-goog-colab-token"] == "xsrf" and params["dryrun"] == "false"
            payload = {"success": accept}
        return Mock(text=")]}'\n" + json.dumps(payload))

    client.session.request.side_effect = request
    state = Mock(client=client)
    state.get_session.return_value = NS(endpoint="endpoint", url="https://vm.invalid", token="proxy")
    monkeypatch.setattr(drive, "colab_state", state)
    ws = Mock(return_value=(0, "", ""))
    monkeypatch.setattr(drive, "jupyter_exec_ws", ws)
    rt = Mock()
    rt.execute_code.return_value = []
    monkeypatch.setattr(drive, "ColabRuntime", lambda *a: rt)
    drive.mount_drive("a")
    if accept:
        assert "ephemeral=False" in ws.call_args.args[1]
        rt.execute_code.assert_not_called()
    else:
        ws.assert_not_called()
        rt.execute_code.assert_called_once()
        rt.stop.assert_called_once()


def test_pre_submit_retry_limit(monkeypatch):
    op = Mock(side_effect=RemoteDisconnected())
    sleep = Mock()
    monkeypatch.setattr(retry.time, "sleep", sleep)
    with pytest.raises(RemoteDisconnected):
        retry.retry(op)
    assert op.call_count == 5 and sleep.call_count == 4


def test_kaggle_verify_before_forgetting(monkeypatch):
    vm = {"provider": "kaggle", "name": "a", "endpoint": "kaggle-123"}
    stop = Mock()
    forget = Mock()
    monkeypatch.setattr(session.time, "sleep", Mock())
    monkeypatch.setattr(session, "shut_down_kaggle_session", stop)
    monkeypatch.setattr(session, "kaggle_active", Mock(side_effect=[True, False]))
    monkeypatch.setattr(session, "forget_kaggle_session", forget)
    session.teardown_verified(vm, "/drive")
    assert [c.kwargs["flush_outputs"] for c in stop.call_args_list] == [True, False]
    forget.assert_called_once_with(vm)


@pytest.fixture
def kaggle_runs(tmp_path):
    runs = tmp_path / "drive/MyDrive/ColabUltra/training/runs"
    runs.mkdir(parents=True)
    (runs / "metrics.json").write_text("{}")
    with (runs / "weights.bin").open("wb") as handle:
        handle.truncate(26 * 1024 * 1024)
    return runs


@pytest.fixture
def drive_upload_api(monkeypatch):
    def list_files(url, *, headers, params, timeout):
        assert url == "https://www.googleapis.com/drive/v3/files"
        assert headers == {"Authorization": "Bearer test-drive-token"}
        files = [{"id": "folder"}] if "mimeType" in params["q"] else []
        return Mock(json=Mock(return_value={"files": files}))

    def begin_upload(url, *, headers, json, timeout):
        assert url == "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"
        assert headers["Authorization"] == "Bearer test-drive-token"
        return Mock(headers={"Location": f"https://upload.invalid/{json['name']}"})

    upload = Mock(return_value=Mock())
    monkeypatch.setattr(requests, "get", list_files)
    monkeypatch.setattr(requests, "post", begin_upload)
    monkeypatch.setattr(requests, "put", upload)
    return upload


@pytest.mark.parametrize("token, uploaded", [("test-drive-token", True), ("test-drive-token", False), (None, False)])
def test_kaggle_final_sync_prefers_drive_for_both_file_sizes(kaggle_runs, drive_upload_api, tmp_path, token, uploaded):
    if token and not uploaded:
        drive_upload_api.side_effect = http_error(403)
    code = kaggle.build_in_vm_sync_snippet("training", token)
    namespace = {}
    exec(code.replace("/content/", f"{tmp_path}/"), namespace)
    out_tar = tmp_path / "_cu_final_runs.tar.gz"
    out_tar.write_bytes(b"stale archive")

    namespace["_cu_pack_runs"](final=True)

    assert {call.args[0]: int(call.kwargs["headers"]["Content-Length"])
            for call in drive_upload_api.call_args_list} == ({
        "https://upload.invalid/metrics.json": 2,
        "https://upload.invalid/weights.bin": 26 * 1024 * 1024,
    } if token else {})
    if uploaded:
        assert not out_tar.exists()
        namespace["_cu_pack_runs"](final=True)
        assert drive_upload_api.call_count == 2
    else:
        with tarfile.open(out_tar) as archive:
            assert sorted(archive.getnames()) == ["runs/metrics.json", "runs/weights.bin"]


def test_kaggle_final_sync_warns_for_oversized_file_without_token(kaggle_runs, tmp_path, capsys):
    with (kaggle_runs / "oversized.bin").open("wb") as handle:
        handle.truncate(221 * 1024 * 1024)
    namespace = {}
    code = kaggle.build_in_vm_sync_snippet("training", None)
    exec(code.replace("/content/", f"{tmp_path}/"), namespace)

    namespace["_cu_pack_runs"](final=True)

    assert "colab-ultra init --with-drive-scope" in capsys.readouterr().out
    with tarfile.open(tmp_path / "_cu_final_runs.tar.gz") as archive:
        assert sorted(archive.getnames()) == ["runs/metrics.json", "runs/weights.bin"]


@pytest.fixture
def kaggle_flush_calls(monkeypatch):
    calls = NS(kernels=Mock(return_value=[{"execution_state": "idle"}]),
               token=Mock(return_value=None), execute=Mock(return_value=(0, "FLUSHED_TAR: 0\n", "")),
               pull=Mock(wraps=kaggle.pull_live_tarball_to_local, return_value=0))
    monkeypatch.setattr(kaggle, "jupyter_get_kernels", calls.kernels)
    monkeypatch.setattr(kaggle, "get_short_lived_drive_token", calls.token)
    monkeypatch.setattr(kaggle, "jupyter_exec_ws", calls.execute)
    monkeypatch.setattr(kaggle, "pull_live_tarball_to_local", calls.pull)
    return calls


@pytest.mark.parametrize("status", [404, 410, 503, None])
def test_kaggle_flush_distinguishes_gone_proxy_from_request_failure(monkeypatch, kaggle_flush_calls, status):
    monkeypatch.setattr(kaggle, "jupyter_get_kernels", jupyter.jupyter_get_kernels)
    get = Mock(side_effect=requests.ReadTimeout()) if status is None else Mock(return_value=http_error(status).response)
    monkeypatch.setattr(requests, "get", get)

    if status in (404, 410):
        kaggle.flush_kaggle_outputs_before_stop("https://vm.invalid", "/drive")
    else:
        with pytest.raises(kaggle.KaggleFlushError, match="Shutdown was aborted.*--no-backup") as failure:
            kaggle.flush_kaggle_outputs_before_stop("https://vm.invalid", "/drive")
        assert not retry.transient(failure.value)
    assert get.call_args.kwargs["timeout"] == 15.0
    kaggle_flush_calls.execute.assert_not_called()
    kaggle_flush_calls.pull.assert_not_called()


def test_kaggle_flush_refuses_busy_kernel(kaggle_flush_calls):
    kaggle_flush_calls.kernels.return_value.append({"execution_state": "busy"})

    with pytest.raises(kaggle.KaggleFlushError, match="kernel is busy.*--no-backup"):
        kaggle.flush_kaggle_outputs_before_stop("https://vm.invalid", "/drive")

    kaggle_flush_calls.execute.assert_not_called()
    kaggle_flush_calls.pull.assert_not_called()


@pytest.mark.parametrize("outcome", [requests.Timeout(), (124, "FLUSHED_TAR: 1\n", "timeout"), (0, "", "")])
def test_kaggle_flush_refuses_unverified_execution(kaggle_flush_calls, outcome):
    if isinstance(outcome, Exception):
        kaggle_flush_calls.execute.side_effect = outcome
    else:
        kaggle_flush_calls.execute.return_value = outcome

    with pytest.raises(kaggle.KaggleFlushError, match="Shutdown was aborted.*--no-backup"):
        kaggle.flush_kaggle_outputs_before_stop("https://vm.invalid", "/drive")

    kaggle_flush_calls.pull.assert_not_called()


@pytest.mark.parametrize("marker", ["FLUSHED", "FLUSHED_TAR"])
@pytest.mark.parametrize("pulled", [0, 1, 2])
def test_kaggle_flush_checks_tarball_file_count(kaggle_flush_calls, marker, pulled):
    kaggle_flush_calls.execute.return_value = (0, f"{marker}: 2\n", "")
    kaggle_flush_calls.pull.return_value = pulled

    if pulled < 2:
        with pytest.raises(kaggle.KaggleFlushError, match=f"Retrieved {pulled} of 2.*--no-backup"):
            kaggle.flush_kaggle_outputs_before_stop("https://vm.invalid", "/drive")
    else:
        kaggle.flush_kaggle_outputs_before_stop("https://vm.invalid", "/drive")

    kaggle_flush_calls.pull.assert_called_once_with("https://vm.invalid", "/content/_cu_shutdown_flush.tar.gz", "/drive")


def test_kaggle_flush_requires_local_hub_for_pending_archive(kaggle_flush_calls):
    kaggle_flush_calls.execute.return_value = (0, "FLUSHED_TAR: 1\n", "")

    with pytest.raises(kaggle.KaggleFlushError, match="No local Drive hub.*--no-backup"):
        kaggle.flush_kaggle_outputs_before_stop("https://vm.invalid", None)

    kaggle_flush_calls.pull.assert_not_called()


@pytest.mark.parametrize("failed_upload", [False, True])
def test_kaggle_shutdown_script_uploads_to_drive_before_packing(
    monkeypatch, tmp_path, capsys, kaggle_runs, drive_upload_api, kaggle_flush_calls, failed_upload
):
    kaggle_flush_calls.token.return_value = "test-drive-token"
    if failed_upload:
        drive_upload_api.side_effect = [Mock(), http_error(403)]

    def execute(proxy_url, code, **kwargs):
        exec(code.replace("/content/", f"{tmp_path}/"), {})
        return (0, capsys.readouterr().out, "")

    kaggle_flush_calls.execute.side_effect = execute
    remote_tar = tmp_path / "_cu_shutdown_flush.tar.gz"
    local_hub = tmp_path / "local" if failed_upload else None
    if failed_upload:
        get_bytes = Mock(side_effect=lambda *args, **kwargs: remote_tar.read_bytes())
        monkeypatch.setattr(kaggle, "jupyter_get_bytes", get_bytes)
        kaggle_flush_calls.pull.return_value = DEFAULT

    kaggle.flush_kaggle_outputs_before_stop("https://vm.invalid", local_hub)

    assert drive_upload_api.call_count == 2
    if failed_upload:
        get_bytes.assert_called_once_with("https://vm.invalid", "", "/content/_cu_shutdown_flush.tar.gz", timeout=120.0)
        uploaded, fallback = [call.args[0].rsplit("/", 1)[-1] for call in drive_upload_api.call_args_list]
        local_runs = local_hub / "training/runs"
        assert not (local_runs / uploaded).exists()
        assert (local_runs / fallback).read_bytes() == (kaggle_runs / fallback).read_bytes()
    else:
        assert not remote_tar.exists()
        kaggle_flush_calls.pull.assert_not_called()


@pytest.mark.parametrize("backup", [True, False])
def test_kaggle_teardown_protects_outputs_unless_backup_disabled(monkeypatch, backup):
    vm = dict(provider="kaggle", name="training", endpoint="kaggle-123", session_id=123,
              proxy_url="https://vm.invalid")
    failure = kaggle.KaggleFlushError("Backup failed")
    flush = Mock(side_effect=failure)
    stop = Mock(wraps=kaggle.shut_down_kaggle_session)
    cancel = Mock(return_value=Mock(status_code=200))
    active = Mock(return_value=False)
    forget = Mock()
    monkeypatch.setattr(kaggle, "flush_kaggle_outputs_before_stop", flush)
    monkeypatch.setattr(kaggle, "_kaggle_headers", Mock(return_value={}))
    monkeypatch.setattr(requests, "post", cancel)
    monkeypatch.setattr(session, "shut_down_kaggle_session", stop)
    monkeypatch.setattr(session, "kaggle_active", active)
    monkeypatch.setattr(session, "forget_kaggle_session", forget)
    monkeypatch.setattr(session.time, "sleep", Mock())

    if backup:
        with pytest.raises(kaggle.KaggleFlushError) as raised:
            session.teardown_verified(vm, "/drive", backup=True)
        assert raised.value is failure
        flush.assert_called_once_with(vm["proxy_url"], "/drive")
        cancel.assert_not_called()
        active.assert_not_called()
        forget.assert_not_called()
    else:
        session.teardown_verified(vm, "/drive", backup=False)
        flush.assert_not_called()
        cancel.assert_called_once_with(
            f"{kaggle.KAGGLE_API_BASE}/kernels.KernelsApiService/CancelKernelSession",
            headers={}, json={"kernelSessionId": 123}, timeout=15,
        )
        forget.assert_called_once_with(vm)
    stop.assert_called_once_with(vm, local_hub="/drive", flush_outputs=backup)
