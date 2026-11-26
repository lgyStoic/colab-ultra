from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from colab_ultra import cli, ssh_cli


def test_build_ssh_argv_variants():
    base = ssh_cli.build_ssh_argv(
        SimpleNamespace(session="l4", proxy_mode=False, identity=None, command=[]),
        "/usr/local/bin/colab",
    )
    assert base == ["/usr/local/bin/colab", "--auth", "adc", "ssh", "-s", "l4"]

    with_sep = ssh_cli.build_ssh_argv(
        SimpleNamespace(
            session="l4",
            proxy_mode=False,
            identity="~/.ssh/id_ed25519",
            command=["--", "nvidia-smi"],
        ),
        "/usr/local/bin/colab",
    )
    assert with_sep == [
        "/usr/local/bin/colab", "--auth", "adc", "ssh", "-s", "l4",
        "-i", "~/.ssh/id_ed25519", "--", "nvidia-smi",
    ]

    without_sep = ssh_cli.build_ssh_argv(
        SimpleNamespace(
            session="l4",
            proxy_mode=True,
            identity=None,
            command=["df", "-h", "/content"],
        ),
        "/usr/local/bin/colab",
    )
    assert without_sep == [
        "/usr/local/bin/colab", "--auth", "adc", "ssh", "-s", "l4",
        "--proxy-mode", "--", "df", "-h", "/content",
    ]


def test_cmd_ssh_rejects_kaggle_session(monkeypatch):
    monkeypatch.setattr(ssh_cli, "check_compatibility", Mock())
    monkeypatch.setattr(ssh_cli, "_load_kaggle_sessions", lambda: {"t4": {"endpoint": "kaggle-1"}})
    run_mock = Mock()
    monkeypatch.setattr(ssh_cli.subprocess, "run", run_mock)

    args = cli.build_parser().parse_args(["ssh", "-s", "t4", "--", "nvidia-smi"])
    with pytest.raises(SystemExit) as exc:
        ssh_cli.cmd_ssh(args)

    assert "Kaggle" in str(exc.value)
    run_mock.assert_not_called()


def test_cmd_ssh_forwards_command_and_exit_code(monkeypatch):
    monkeypatch.setattr(ssh_cli, "check_compatibility", Mock())
    monkeypatch.setattr(ssh_cli, "_load_kaggle_sessions", lambda: {})
    monkeypatch.setattr(ssh_cli, "_find_colab_bin", lambda: "/opt/bin/colab")
    run_mock = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(ssh_cli.subprocess, "run", run_mock)

    args = cli.build_parser().parse_args(["ssh", "-s", "l4", "--", "nvidia-smi"])
    ssh_cli.cmd_ssh(args)
    run_mock.assert_called_once_with(
        ["/opt/bin/colab", "--auth", "adc", "ssh", "-s", "l4", "--", "nvidia-smi"]
    )

    run_mock.return_value = SimpleNamespace(returncode=7)
    with pytest.raises(SystemExit) as exc:
        ssh_cli.cmd_ssh(args)
    assert exc.value.code == 7
