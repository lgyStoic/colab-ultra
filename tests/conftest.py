"""Offline tests use an isolated home and reject outbound sockets."""
import socket

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def reject(*args, **kwargs):
        pytest.fail("Unexpected network access in offline test")
    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setenv("HOME", str(tmp_path))
