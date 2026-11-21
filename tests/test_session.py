import signal
from http.client import RemoteDisconnected
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import requests
import websocket

from colab_ultra import drive, retry, session
from colab_ultra.backends import colab, jupyter


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


@pytest.mark.parametrize("active_after, stops, success", [([False], 1, True), ([True, False], 2, True), ([True, True], 2, False)])
def test_verified_down(monkeypatch, active_after, stops, success):
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
    args = NS(session="a", all=False, json=False)
    if success:
        session.cmd_down(args)
        state.store.remove.assert_called_once_with("a")
    else:
        with pytest.raises(SystemExit, match="still active"):
            session.cmd_down(args)
        state.store.remove.assert_not_called()
    assert state.client.unassign.call_count == stops
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
