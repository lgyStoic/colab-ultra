import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console

from colab_ultra import cli, session, telemetry
from colab_ultra.backends import kaggle


def choose(gpu_raw="T4", **overrides):
    options = {
        "force_colab": False,
        "force_kaggle": False,
        "is_cpu_or_tpu": False,
        "high_mem": False,
        "workers": 1,
        "cfg": {},
        "token_present": True,
        "quota": {"gpu_remaining_hours": 0.5, "gpu_reserved_seconds": 0.0},
    }
    return session.choose_kaggle_for_t4(gpu_raw, **(options | overrides))


@pytest.mark.parametrize("hours, reserved, expected", [(0.5, 0, True), (0.49, 0, False), (8, 1, False)])
def test_live_quota(hours, reserved, expected):
    quota = {"gpu_remaining_hours": hours, "gpu_reserved_seconds": reserved}
    assert choose(" t4 ", quota=quota) == (expected, None)


def test_api_error_uses_available_cache_even_with_invalid_refresh_time():
    fetched_at = "2026-11-20T10:00:00+00:00"
    cache = {
        "gpu_remaining_hours": 0.5,
        "gpu_reserved_seconds": 0,
        "refresh_time": "invalid",
        "fetched_at": fetched_at,
    }
    use_kaggle, message = choose(quota={"error": "API timeout", "cache": cache})
    assert use_kaggle
    assert "本地缓存" in message and fetched_at in message


@pytest.mark.parametrize("refresh_time", ["2026-11-21T10:00:00Z", "2026-11-21T18:00:00+08:00"])
@pytest.mark.parametrize("hours, reserved", [(0, 0), (5, 60)])
@pytest.mark.parametrize("now_tz", [None, timezone.utc])
def test_api_error_uses_unavailable_cache_only_after_refresh(refresh_time, hours, reserved, now_tz):
    quota = {
        "error": "API timeout",
        "cache": {
            "gpu_remaining_hours": hours,
            "gpu_reserved_seconds": reserved,
            "refresh_time": refresh_time,
        },
    }
    assert choose(quota=quota, now=datetime(2026, 11, 21, 9, 59, tzinfo=now_tz)) == (
        False, "Kaggle 额度接口失败（API timeout），改用 Colab T4",
    )
    use_kaggle, message = choose(quota=quota, now=datetime(2026, 11, 21, 10, 1, tzinfo=now_tz))
    assert use_kaggle
    assert "本地缓存（未知）" in message


@pytest.mark.parametrize("cache", [None, [], {}, {"gpu_remaining_hours": "bad", "refresh_time": 123}])
def test_api_error_without_usable_cache(cache):
    assert choose(quota={"error": "API timeout", "cache": cache}) == (
        False, "Kaggle 额度接口失败（API timeout），改用 Colab T4",
    )


def test_force_colab_warns_when_live_quota_is_available():
    use_kaggle, message = choose(" t4 ", force_colab=True, high_mem=True, workers=2)
    assert not use_kaggle
    assert "0.5 小时" in message and "1.19 CU/h" in message and "--colab" in message


@pytest.mark.parametrize("gpu_raw, overrides", [
    ("T4", {"token_present": False}),
    ("T4", {"quota": None}),
    ("T4", {"quota": {"error": "offline", "cache": {"gpu_remaining_hours": 8}}}),
    ("T4", {"quota": {"gpu_remaining_hours": 0.49}}),
    ("T4", {"quota": {"gpu_remaining_hours": 8, "gpu_reserved_seconds": 10}}),
    ("T4x2", {}),
    ("L4", {}),
])
def test_force_colab_without_available_live_t4_quota_is_silent(gpu_raw, overrides):
    assert choose(gpu_raw, force_colab=True, **overrides) == (False, None)


@pytest.mark.parametrize("overrides", [
    {"high_mem": True},
    {"workers": 2},
    {"cfg": {"prefer_kaggle_for_t4": False}},
    {"token_present": False},
    {"quota": None},
])
def test_t4_without_routing_requirements_uses_colab(overrides):
    assert choose(**overrides) == (False, None)


def test_explicit_kaggle_overrides_automatic_t4_requirements():
    assert choose("L4", force_kaggle=True, force_colab=True, token_present=False, quota=None) == (True, None)
    assert choose("dual-T4", token_present=False, quota=None) == (True, None)


def test_cpu_or_tpu_ignores_explicit_kaggle():
    assert choose("T4x2", is_cpu_or_tpu=True, force_kaggle=True) == (False, None)


@pytest.fixture
def up_calls(monkeypatch):
    calls = {
        "load_config": Mock(return_value={}),
        "load_kaggle_token": Mock(return_value="fake-token"),
        "get_kaggle_quota": Mock(return_value={
            "configured": True,
            "gpu_total_hours": 30,
            "gpu_remaining_hours": 4.5,
            "gpu_reserved_seconds": 0,
        }),
        "_spin_up_single_session": Mock(),
        "spin_up_kaggle_session": Mock(),
    }
    for name, fake in calls.items():
        monkeypatch.setattr(session, name, fake)
    monkeypatch.setattr(session, "resolve_colab_bin", lambda: "fake-colab")
    monkeypatch.setattr(session, "colab_state", Mock(sync_sessions=Mock(return_value=({}, []))))
    ccu = SimpleNamespace(paid_compute_units_balance=20, consumption_rate_hourly=1)
    monkeypatch.setattr(session, "inspect_all_vms", lambda: (ccu, []))
    monkeypatch.setattr(session, "_record_starts", Mock())
    monkeypatch.setattr(session, "sync_dashboard_json", Mock())
    monkeypatch.setattr(telemetry, "sync_dashboard_json", Mock())
    monkeypatch.setattr(telemetry, "load_config", lambda: {})
    monkeypatch.setattr(telemetry, "detect_local_drive_hub", lambda: None)
    monkeypatch.setattr(telemetry, "get_kaggle_quota", Mock(return_value={
        "configured": True,
        "gpu_total_hours": 30,
        "gpu_remaining_hours": 4.5,
        "gpu_reserved_seconds": 0,
    }))
    return calls


@pytest.mark.parametrize("flags, provider", [
    (["--cpu"], "colab"),
    (["--cpu", "--colab"], "colab"),
    (["--gpu", "CPU"], "colab"),
    (["--gpu", "CPU", "--json"], "colab"),
    (["--tpu", "v5e1"], "colab"),
    (["--gpu", "L4"], "colab"),
    (["--gpu", "A100", "--colab"], "colab"),
    (["--high-mem"], "colab"),
    (["--workers", "2"], "colab"),
    (["--kaggle"], "kaggle"),
    (["--gpu", "T4x2"], "kaggle"),
    (["--gpu", "T4x2", "--colab"], "colab"),
])
def test_up_skips_quota_for_other_routes(up_calls, flags, provider):
    cli.main(["up", *flags])
    up_calls["get_kaggle_quota"].assert_not_called()
    up_calls["load_kaggle_token"].assert_not_called()
    assert up_calls["_spin_up_single_session"].called == (provider == "colab")
    assert up_calls["spin_up_kaggle_session"].called == (provider == "kaggle")


@pytest.mark.parametrize("missing_token", [True, False])
def test_up_skips_quota_when_token_is_missing_or_preference_is_disabled(up_calls, missing_token):
    if missing_token:
        up_calls["load_kaggle_token"].return_value = None
    else:
        up_calls["load_config"].return_value = {"prefer_kaggle_for_t4": False}
    cli.main(["up"])
    up_calls["get_kaggle_quota"].assert_not_called()
    up_calls["_spin_up_single_session"].assert_called_once()


@pytest.mark.parametrize("json_mode", [False, True])
def test_up_prints_yellow_colab_warning_to_the_correct_stream(monkeypatch, up_calls, capsys, json_mode):
    def color_console(**kwargs):
        return Console(
            force_terminal=True, color_system="standard", no_color=False,
            soft_wrap=True, highlight=False, **kwargs,
        )

    monkeypatch.setattr(session, "console", color_console())
    monkeypatch.setattr(session, "Console", color_console)
    up_calls["load_config"].return_value = {"prefer_kaggle_for_t4": False}
    flags = ["--json"] if json_mode else []
    cli.main(["up", "--colab", "--high-mem", "--workers", "2", *flags])
    up_calls["get_kaggle_quota"].assert_called_once_with()
    up_calls["spin_up_kaggle_session"].assert_not_called()
    assert up_calls["_spin_up_single_session"].call_count == 2
    captured = capsys.readouterr()
    warning_output = captured.err if json_mode else captured.out
    assert "\x1b[33m⚠️ Kaggle 免费额度剩 4.5 小时" in warning_output
    assert "1.19 CU/h" in warning_output
    if json_mode:
        payload = json.loads(captured.out)
        assert payload["started_sessions"] == ["ultra-1", "ultra-2"]
        assert payload["kaggle_free_gpu_hours_left"] == 4.5
    else:
        assert captured.err == ""


@pytest.mark.parametrize("use_cache", [False, True])
def test_up_falls_back_to_colab_if_automatic_kaggle_start_fails(up_calls, capsys, use_cache):
    quota = up_calls["get_kaggle_quota"]
    if use_cache:
        quota.return_value = {"error": "API timeout", "cache": quota.return_value}
    up_calls["spin_up_kaggle_session"].side_effect = RuntimeError("Kaggle busy")
    cli.main(["up"])
    quota.assert_called_once_with()
    assert up_calls["spin_up_kaggle_session"].call_args.kwargs["gpu_choice"] == "T4x2"
    up_calls["_spin_up_single_session"].assert_called_once()
    captured = capsys.readouterr().out
    assert "自动回退至 Colab T4" in captured
    assert ("本地缓存" in captured) == use_cache


@pytest.fixture
def quota_api(monkeypatch, tmp_path):
    response = Mock()
    response.json.return_value = {
        "gpuQuota": {"totalTimeAllowed": "108000s", "timeUsed": "1800s", "timeReserved": "120s"},
        "tpuQuota": {"totalTimeAllowed": "72000s", "timeUsed": "3600s"},
        "quotaRefreshTime": "2026-11-22T00:00:00Z",
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(kaggle.requests, "post", post)
    monkeypatch.setattr(kaggle, "load_kaggle_token", Mock(return_value="fake-token"))
    monkeypatch.setattr(kaggle, "KAGGLE_QUOTA_CACHE_PATH", tmp_path / "default-cache.json")
    return post


EXPECTED_QUOTA = {
    "configured": True,
    "gpu_total_hours": 30.0,
    "gpu_used_hours": 0.5,
    "gpu_remaining_hours": 29.5,
    "gpu_reserved_seconds": 120.0,
    "tpu_total_hours": 20.0,
    "tpu_remaining_hours": 19.0,
    "refresh_time": "2026-11-22T00:00:00Z",
}


def test_successful_quota_query_writes_cache(tmp_path, quota_api):
    cache_path = tmp_path / "quota.json"
    cache_path.write_text('{"outdated": true}', encoding="utf-8")
    before = datetime.now(timezone.utc)
    quota = kaggle.get_kaggle_quota(timeout=1.5, cache_path=cache_path)
    after = datetime.now(timezone.utc)

    assert quota == EXPECTED_QUOTA
    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    fetched_at = datetime.fromisoformat(saved.pop("fetched_at"))
    assert fetched_at.utcoffset() == timedelta(0)
    assert before <= fetched_at <= after
    assert saved == quota
    assert quota_api.call_args.kwargs["timeout"] == 1.5


def test_cache_write_failure_is_silent(monkeypatch, tmp_path, quota_api, capsys):
    save = Mock(side_effect=OSError("read-only cache directory"))
    monkeypatch.setattr(kaggle, "save_config", save)
    cache_path = str(tmp_path / "quota.json")
    assert kaggle.get_kaggle_quota(cache_path=cache_path) == EXPECTED_QUOTA
    save.assert_called_once()
    assert save.call_args.args[1] == cache_path
    assert capsys.readouterr() == ("", "")


def test_failed_quota_query_reads_saved_default_cache(quota_api):
    assert kaggle.get_kaggle_quota() == EXPECTED_QUOTA
    cache_path = kaggle.KAGGLE_QUOTA_CACHE_PATH
    saved = json.loads(cache_path.read_text(encoding="utf-8"))

    quota_api.side_effect = kaggle.requests.Timeout("network timeout")
    assert kaggle.get_kaggle_quota(timeout=1) == {
        "configured": True, "error": "network timeout", "cache": saved,
    }


@pytest.mark.parametrize("contents", [None, "invalid JSON", "[]", "null"])
def test_failed_quota_query_handles_missing_or_invalid_cache(tmp_path, quota_api, contents):
    cache_path = tmp_path / "quota.json"
    if contents is not None:
        cache_path.write_text(contents, encoding="utf-8")
    quota_api.side_effect = kaggle.requests.Timeout("network timeout")
    assert kaggle.get_kaggle_quota(cache_path=cache_path) == {
        "configured": True, "error": "network timeout", "cache": None,
    }
