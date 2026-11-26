"""Offline tests use an isolated home and reject outbound sockets."""
import socket
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path, request):
    def reject(*args, **kwargs):
        pytest.fail("Unexpected network access in offline test")
    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setenv("HOME", str(tmp_path))
    if Path(__file__).with_name("h3_infer") in request.path.parents:
        return

    from colab_ultra.backends import kaggle

    monkeypatch.setattr(kaggle, "DRIVE_ADC_PATH", str(tmp_path / "drive_adc.json"))
    monkeypatch.setattr(kaggle, "DEFAULT_GCLOUD_ADC", str(tmp_path / "application_default_credentials.json"))
    monkeypatch.setattr(kaggle, "_DRIVE_TOKEN_STATUS", {"ok": False, "token": None, "expires_at": 0.0, "checked_at": 0.0})
