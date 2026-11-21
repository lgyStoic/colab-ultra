import base64
import json
import ssl
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import requests
from websocket import WebSocketConnectionClosedException

from colab_ultra.backends import colab, jupyter
from colab_ultra.retry import transient


@pytest.mark.parametrize(
    "error, expected",
    [
        (ssl.SSLEOFError(), True),
        (requests.exceptions.SSLError(ssl.SSLEOFError()), True),
        (requests.exceptions.SSLError(ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")), True),
        (requests.exceptions.SSLError("proxy failed with SSLEOFError"), True),
        (WebSocketConnectionClosedException(), True),
        (requests.exceptions.SSLError(), False),
        (requests.exceptions.SSLError(ssl.SSLCertVerificationError("cert failed")), False),
        (ssl.SSLCertVerificationError("SSLEOFError"), False),
    ],
)
def test_transient_tls_and_websocket_errors(error, expected):
    assert transient(error) is expected


def test_tls_exception_chain_with_cycle_and_certificate_error():
    outer = requests.exceptions.SSLError()
    cause = RuntimeError()
    context = RuntimeError()
    outer.__cause__ = cause
    cause.__context__ = context
    context.reason = ssl.SSLEOFError()
    context.args = (outer,)
    assert transient(outer) is True

    context.reason.__context__ = ssl.SSLCertVerificationError("cert failed")
    assert transient(outer) is False


LAUNCHED = (0, '{"launched": true, "pid": 42}\n', "")
CLEANUP = (0, "", "")


def poll_result(offset, output, rc=None):
    return 0, json.dumps({
        "offset": offset + len(output),
        "b64": base64.b64encode(output).decode("ascii"),
        "done": rc is not None,
        "rc": rc,
    }), ""


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"HTTP {status}", response=response)


@pytest.fixture
def detached(monkeypatch):
    args = SimpleNamespace(session="test", timeout=60, restart_kernel=False)
    proxy = Mock(return_value=("https://vm.invalid", "token"))
    kernel = Mock(return_value="kernel")
    execute = Mock()
    sleep = Mock()
    monkeypatch.setattr(colab, "resolve_proxy", proxy)
    monkeypatch.setattr(jupyter, "jupyter_ensure_kernel_id", kernel)
    monkeypatch.setattr(colab, "jupyter_exec_ws", execute)
    monkeypatch.setattr(colab.time, "sleep", sleep)
    return SimpleNamespace(args=args, proxy=proxy, kernel=kernel, execute=execute, sleep=sleep)


def test_detached_recovers_from_consecutive_poll_errors(detached, capsys):
    d = detached
    d.proxy.side_effect = [
        ("https://vm.invalid", "token-0"),
        ("https://vm.invalid", "token-1"),
        requests.ConnectionError("refresh temporarily unavailable"),
        ("https://vm.invalid", "token-2"),
        ("https://vm.invalid", "token-3"),
        ("https://vm.invalid", "token-4"),
    ]
    d.kernel.side_effect = [f"kernel-{i}" for i in range(6)]
    d.execute.side_effect = [
        LAUNCHED,
        poll_result(0, b"first\n"),
        requests.exceptions.SSLError(ssl.SSLEOFError("unexpected EOF")),
        WebSocketConnectionClosedException(),
        RuntimeError("Jupyter closed after submission; check the job output"),
        RuntimeError("Jupyter connection lost after submission"),
        (124, "", ""),
        poll_result(6, b"last\n", rc=17),
        CLEANUP,
    ]

    assert colab.run_detached(d.args, "pass") == (17, "first\nlast\n", "")
    assert capsys.readouterr().out == "first\nlast\n"
    assert d.proxy.call_args_list == [call("test")] * 6
    assert [c.args[1] for c in d.kernel.call_args_list] == [
        "token-0", "token-1", "token-1", "token-2", "token-3", "token-4",
    ]
    assert [c.kwargs["kernel_id"] for c in d.execute.call_args_list] == [
        "kernel-0", "kernel-0", "kernel-0", "kernel-1", "kernel-2",
        "kernel-3", "kernel-4", "kernel-5", "kernel-5",
    ]
    assert all("_off = 6\n" in c.args[1] for c in d.execute.call_args_list[2:-1])
    assert d.sleep.call_args_list == [call(n) for n in [1, 1, 2, 4, 8, 15]]


def test_detached_reuses_kernel_after_restart(detached, monkeypatch):
    d = detached
    d.args.restart_kernel = True
    restart = Mock()
    lifecycle = Mock()
    lifecycle.attach_mock(restart, "restart")
    lifecycle.attach_mock(d.kernel, "kernel")
    monkeypatch.setattr(colab, "restart_vm_kernel", restart)
    d.execute.side_effect = [
        LAUNCHED, poll_result(0, b"one\n"), poll_result(4, b"two\n", rc=0), CLEANUP,
    ]

    assert colab.run_detached(d.args, "pass") == (0, "one\ntwo\n", "")
    assert lifecycle.mock_calls == [
        call.restart("https://vm.invalid", "token"),
        call.kernel("https://vm.invalid", "token"),
    ]
    assert [c.kwargs["kernel_id"] for c in d.execute.call_args_list] == ["kernel"] * 4
    d.proxy.assert_called_once_with("test")


@pytest.mark.parametrize("status", [404, 410])
def test_detached_stops_when_poll_loses_session(detached, status):
    d = detached
    d.execute.side_effect = [LAUNCHED, http_error(status)]

    with pytest.raises(SystemExit, match="Active Colab session not found: test"):
        colab.run_detached(d.args, "pass")

    assert d.execute.call_count == 2
    d.proxy.assert_called_once_with("test")
    d.sleep.assert_not_called()


@pytest.mark.parametrize("error", [http_error(404), http_error(410), RuntimeError("No active Colab session")])
def test_detached_reports_lost_session_during_kernel_discovery(detached, error):
    d = detached
    d.kernel.side_effect = error

    with pytest.raises(SystemExit, match="Active Colab session not found: test"):
        colab.run_detached(d.args, "pass")

    d.kernel.assert_called_once()
    d.execute.assert_not_called()
    d.sleep.assert_not_called()


def test_detached_stops_when_proxy_refresh_loses_session(detached):
    d = detached
    lost = SystemExit(colab.lost_session_message("test"))
    d.proxy.side_effect = [("https://vm.invalid", "token"), lost]
    d.execute.side_effect = [LAUNCHED, WebSocketConnectionClosedException()]

    with pytest.raises(SystemExit) as error:
        colab.run_detached(d.args, "pass")

    assert error.value is lost
    assert d.execute.call_count == 2
    d.sleep.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        requests.exceptions.SSLError("certificate verify failed"),
        requests.exceptions.SSLError(),
        ssl.SSLCertVerificationError("cert failed"),
        ValueError("bad response"),
        http_error(400),
        http_error(403),
    ],
)
@pytest.mark.parametrize("wrapped", [False, True], ids=["direct", "after-submission"])
def test_detached_stops_on_permanent_poll_errors(detached, error, wrapped):
    d = detached
    if wrapped:
        disconnect = RuntimeError("Jupyter connection lost after submission")
        disconnect.__cause__ = error
        error = disconnect
    d.execute.side_effect = [LAUNCHED, error, CLEANUP]

    rc, out, stderr = colab.run_detached(d.args, "pass")

    assert (rc, out) == (1, "")
    assert "Unable to read" in stderr
    assert [c.kwargs["timeout"] for c in d.execute.call_args_list] == [30.0, 25.0, 10.0]
    d.proxy.assert_called_once_with("test")
    d.kernel.assert_called_once()
    d.sleep.assert_not_called()


def test_detached_stops_after_eight_consecutive_poll_errors(detached):
    d = detached
    d.execute.side_effect = [LAUNCHED, *([(124, "", "")] * 8), CLEANUP]

    rc, out, stderr = colab.run_detached(d.args, "pass")

    assert (rc, out) == (1, "")
    assert "poll exited with 124" in stderr
    assert [c.kwargs["timeout"] for c in d.execute.call_args_list] == [30.0] + [25.0] * 8 + [10.0]
    assert d.sleep.call_args_list == [call(n) for n in [1, 2, 4, 8, 15, 15, 15]]


@pytest.mark.parametrize("kernels, kernel_id", [([{"id": "kernel-1"}], "kernel-1"), ([], None)])
def test_remote_probe_uses_jupyter_websocket(monkeypatch, kernels, kernel_id):
    from colab_ultra import telemetry

    fetch_kernels = Mock(return_value=kernels)
    execute = Mock()
    monkeypatch.setattr(telemetry, "fetch_vm_kernels", fetch_kernels)
    monkeypatch.setattr(telemetry, "jupyter_exec_ws", execute)

    telemetry.start_remote_probe_via_runtime("https://vm.invalid", "token")

    assert not hasattr(telemetry, "ColabRuntime")
    fetch_kernels.assert_called_once_with("https://vm.invalid", "token")
    code = execute.call_args.args[1]
    assert "/content/.colab_probe.py" in code
    assert "--once" in code
    execute.assert_called_once_with(
        "https://vm.invalid", code, proxy_token="token", kernel_id=kernel_id, timeout=12
    )


def test_telemetry_retries_tls_eof_without_redeploying(monkeypatch):
    from colab_ultra import retry, telemetry

    data = {"unix_time": 100}
    response = Mock(status_code=200)
    response.json.return_value = {"content": json.dumps(data)}
    get = Mock(side_effect=[ssl.SSLEOFError("unexpected EOF"), response])
    upload = Mock()
    start_probe = Mock()
    sleep = Mock()
    monkeypatch.setattr(telemetry.requests, "get", get)
    monkeypatch.setattr(telemetry, "time", SimpleNamespace(time=lambda: 101))
    monkeypatch.setattr(telemetry, "os", SimpleNamespace(path=SimpleNamespace(exists=lambda _: True)))
    monkeypatch.setattr(telemetry, "upload_file_to_vm", upload)
    monkeypatch.setattr(telemetry, "start_remote_probe_via_runtime", start_probe)
    monkeypatch.setattr(retry.time, "sleep", sleep)

    assert telemetry.fetch_vm_telemetry("https://vm.invalid", "token") == data

    read = call(
        "https://vm.invalid/api/contents/content/.colab_telemetry.json",
        params={"content": "1", "authuser": "0", "colab-runtime-proxy-token": "token"},
        headers={"Authorization": "Bearer token", "X-Colab-Runtime-Proxy-Token": "token"},
        timeout=15,
    )
    assert get.call_args_list == [read, read]
    sleep.assert_called_once_with(1)
    upload.assert_not_called()
    start_probe.assert_not_called()


@pytest.mark.parametrize("error", [None, TimeoutError("backup timed out")])
def test_backup_outputs_uses_jupyter_websocket(monkeypatch, error):
    execute = Mock(return_value=(0, "", ""), side_effect=error)
    monkeypatch.setattr(colab, "jupyter_exec_ws", execute)

    assert colab.backup_vm_outputs_to_drive("https://vm.invalid", "token") is None

    code = execute.call_args.args[1]
    assert "shutil.copytree('/content/outputs', dst, dirs_exist_ok=True)" in code
    execute.assert_called_once_with("https://vm.invalid", code, proxy_token="token", timeout=6)


def test_patch_kernel_client_stop_is_safe_and_idempotent(monkeypatch):
    from jupyter_kernel_client import KernelClient

    from colab_ultra.compat.colab_cli_patch import patch_kernel_client

    stop = KernelClient.stop
    monkeypatch.setattr(KernelClient, "stop", getattr(stop, "__wrapped__", stop))
    patch_kernel_client()
    patched_stop = KernelClient.stop
    patch_kernel_client()
    assert KernelClient.stop is patched_stop

    client = KernelClient.__new__(KernelClient)
    assert not hasattr(client, "_manager")
    assert client.stop() is None

    manager = Mock(has_kernel=True)
    client._manager = manager
    client.stop(True, shutdown_now=False, timeout=7)

    manager.client.stop_channels.assert_called_once_with()
    manager.shutdown_kernel.assert_called_once_with(now=False, timeout=7)
