import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open

import pytest
import requests
from rich.console import Console

from colab_ultra import drive, telemetry
from colab_ultra.backends import colab, kaggle
from colab_ultra.config import BASE_DIR
from workflows import init_20tb_drive


OAUTH_TOKEN = "test-drive-oauth-token"
DRIVE_LIMIT = 21990232555520
DRIVE_USAGE = 810191481905
CAPACITY = {"total_tb": 20.0, "used_tb": 0.74, "free_tb": 19.26}
QUOTA = {
    "total_bytes": DRIVE_LIMIT,
    "used_bytes": DRIVE_USAGE,
    "free_bytes": DRIVE_LIMIT - DRIVE_USAGE,
    **CAPACITY,
}


@pytest.fixture
def drive_api(monkeypatch):
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(drive, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(drive, "_drive_quota_cache", {"expires_at": 0.0, "quota": None})
    token = Mock(return_value=OAUTH_TOKEN)
    monkeypatch.setattr(kaggle, "get_short_lived_drive_token", token)
    response = Mock()
    response.json.return_value = {
        "storageQuota": {"limit": str(DRIVE_LIMIT), "usage": str(DRIVE_USAGE)},
    }
    get = Mock(return_value=response)
    monkeypatch.setattr(drive.requests, "get", get)
    return SimpleNamespace(clock=clock, token=token, response=response, get=get)


def test_drive_storage_quota_cache(drive_api):
    assert drive.get_drive_storage_quota() == QUOTA
    drive_api.get.assert_called_once_with(
        "https://www.googleapis.com/drive/v3/about?fields=storageQuota",
        headers={"Authorization": f"Bearer {OAUTH_TOKEN}"},
        timeout=10,
    )
    drive_api.clock.return_value = 159.99
    assert drive.get_drive_storage_quota() == QUOTA
    drive_api.token.assert_called_once_with()
    assert drive_api.get.call_count == 1

    drive_api.clock.return_value = 160.0
    assert drive.get_drive_storage_quota() == QUOTA
    assert drive_api.get.call_count == 2

    drive_api.response.json.return_value["storageQuota"]["usage"] = str(DRIVE_LIMIT + 1)
    assert drive.get_drive_storage_quota(force_refresh=True)["free_bytes"] == 0
    assert drive.get_drive_storage_quota()["free_tb"] == 0
    assert drive_api.get.call_count == 3


def test_drive_storage_quota_without_token(drive_api):
    drive_api.token.return_value = None
    assert drive.get_drive_storage_quota() is None
    drive_api.get.assert_not_called()


def test_drive_storage_quota_http_error(drive_api, capsys, caplog):
    drive_api.response.raise_for_status.side_effect = requests.HTTPError(OAUTH_TOKEN)
    assert drive.get_drive_storage_quota() is None
    captured = capsys.readouterr()
    assert OAUTH_TOKEN not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("storage_quota", [{}, {"limit": "0"}, {"limit": "invalid"}])
def test_drive_storage_quota_without_valid_limit(drive_api, storage_quota):
    drive_api.response.json.return_value = {"storageQuota": storage_quota}
    assert drive.get_drive_storage_quota() is None


@pytest.mark.parametrize("has_quota", [True, False])
def test_initialize_drive_sends_only_numeric_capacity(monkeypatch, drive_api, has_quota):
    if not has_quota:
        drive_api.token.return_value = None
    mount = Mock()
    monkeypatch.setattr(drive, "mount_drive", mount)
    monkeypatch.setattr(colab, "resolve_proxy", lambda _: ("https://vm.example.test", "proxy-token"))
    execute = Mock(return_value=(0, "", ""))
    monkeypatch.setattr(drive, "jupyter_exec_ws", execute)

    drive.initialize_drive("colab", "test-vm")

    mount.assert_called_once_with("test-vm")
    script = Path(BASE_DIR, "workflows", "init_20tb_drive.py").read_text(encoding="utf-8")
    expected = f"_CU_DRIVE_QUOTA = {CAPACITY!r}\n{script}" if has_quota else script
    execute.assert_called_once_with(
        "https://vm.example.test", expected, proxy_token="proxy-token", timeout=60, stream_stdout=True,
    )
    assert OAUTH_TOKEN not in execute.call_args.args[1]


@pytest.mark.parametrize("has_quota", [True, False])
def test_setup_drive_hub_reports_capacity(monkeypatch, capsys, has_quota):
    workflow = init_20tb_drive
    statvfs = Mock(return_value=SimpleNamespace(f_blocks=230, f_bavail=180, f_frsize=1024 ** 3))
    monkeypatch.setattr(workflow, "os", SimpleNamespace(
        statvfs=statvfs,
        path=SimpleNamespace(join=workflow.os.path.join, isdir=lambda p: p == workflow.MY_DRIVE,
                             exists=lambda _: True),
        makedirs=Mock(),
        environ={},
        fsync=Mock(),
        remove=Mock(),
    ))
    monkeypatch.setattr(workflow, "open", mock_open(), raising=False)
    if has_quota:
        monkeypatch.setattr(workflow, "_CU_DRIVE_QUOTA", CAPACITY, raising=False)
    else:
        monkeypatch.delattr(workflow, "_CU_DRIVE_QUOTA", raising=False)

    assert workflow.setup_drive_hub()
    output = capsys.readouterr().out
    if has_quota:
        assert "📦 云盘总容量: 20.00 TB | 已用: 0.74 TB | 剩余可用: 19.26 TB" in output
        statvfs.assert_not_called()
    else:
        assert ("📦 云盘本地缓存区: 0.22 TB | 已用: 0.05 TB | 剩余可用: 0.18 TB"
                "（未获取云端配额，此为 Colab 本地挂载缓存分区）") in output
        assert "云盘总容量" not in output
        statvfs.assert_called_once_with(workflow.DRIVE_MOUNT)


@pytest.mark.parametrize("mounted", [True, False])
def test_inspect_all_vms_drive_capacity(monkeypatch, tmp_path, mounted):
    ccu = SimpleNamespace(paid_compute_units_balance=20.0, consumption_rate_hourly=0.0)
    assignments = [
        SimpleNamespace(
            endpoint=f"colab-{i}",
            accelerator=SimpleNamespace(value="L4"),
            machine_shape=SimpleNamespace(name="STANDARD"),
            runtime_proxy_info=SimpleNamespace(url=f"https://vm{i}.example.test", token="proxy-token"),
        )
        for i in range(2)
    ]
    monkeypatch.setattr(telemetry, "colab_state", SimpleNamespace(
        client=SimpleNamespace(get_consumption_user_info=lambda: ccu),
        sync_sessions=lambda: ({}, assignments),
    ))
    monkeypatch.setattr(telemetry, "fetch_vm_kernels", lambda *_: [])
    monkeypatch.setattr(telemetry, "fetch_vm_telemetry", lambda *_args, **_kwargs: {
        "drive": {"mounted": mounted, "total_tb": 0.23, "used_tb": 0.05, "free_tb": 0.18},
    })
    kaggle_drive = {"mounted": True, "mode": "kaggle_synced", "free_gb": 68.0}
    kaggle_vm = {
        "name": "kaggle-test", "provider": "kaggle", "endpoint": "kaggle-0",
        "accelerator": "T4x2", "shape": "STANDARD", "kernel_state": "idle",
        "telemetry": {"drive": kaggle_drive.copy()},
    }
    monkeypatch.setattr(telemetry, "list_active_kaggle_vms", lambda **_: [kaggle_vm])
    get_quota = Mock(return_value=QUOTA)
    monkeypatch.setattr(telemetry, "get_drive_storage_quota", get_quota)

    assert_ccu, vms = telemetry.inspect_all_vms()

    assert assert_ccu is ccu
    assert vms[2]["telemetry"]["drive"] == kaggle_drive
    if not mounted:
        get_quota.assert_not_called()
        assert vms[0]["telemetry"]["drive"]["free_tb"] == 0.18
        return
    get_quota.assert_called_once_with()
    for vm in vms[:2]:
        assert vm["telemetry"]["drive"] == {"mounted": True, **CAPACITY}

    monkeypatch.setattr(telemetry, "detect_local_drive_hub", lambda: None)
    monkeypatch.setattr(telemetry, "get_kaggle_quota", lambda: {})
    agent_payload = telemetry.build_agent_status_payload(ccu, vms, cfg={})
    assert agent_payload["vms"][0]["drive_free_tb"] == 19.26
    assert agent_payload["vms"][2]["drive_mode"] == "kaggle_synced"

    stats_path = tmp_path / "stats.json"
    cfg = {"local_stats_out": str(stats_path)}
    monkeypatch.setattr(telemetry, "load_config", lambda: cfg)
    output = StringIO()
    monkeypatch.setattr(telemetry, "console", Console(file=output, width=140))
    telemetry.render_dashboard(ccu, vms)
    assert "19.3T空" in output.getvalue()
    assert "68G空" in output.getvalue()
    dashboard_payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert dashboard_payload["vms"][0]["drive_free_tb"] == 19.26
