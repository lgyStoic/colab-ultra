import argparse
import json

from colab_ultra import cli, config
from colab_ultra.compat.colab_cli_patch import check_compatibility


# Taken from the pre-refactor argparse help, including short aliases.
OPTIONS = {
    "init": "--auth --with-drive-scope --alert-minutes --stop-minutes --vps --webhook",
    "status": "--json", "watch": "--once --interval --alert-minutes --stop-minutes --gpu-threshold",
    "up": "-s --session --gpu --kaggle --colab --cpu --tpu --high-mem --std-mem --workers --no-drive --json",
    "exec": "-s --session -p --project -f --file --timeout --restart-kernel --json",
    "pull": "-p --project --dest --filename -s --session --json",
    "down": "-s --session --all --json", "test-alert": "", "sync-remote": "",
    "models": "", "logs": "-s --session -n --lines --json",
}


def test_help_parity():
    parser = cli.build_parser()
    commands = next(a.choices for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert set(commands) == OPTIONS.keys()
    for name, options in OPTIONS.items():
        assert set(commands[name]._option_string_actions) == {"-h", "--help", *options.split()}
    assert parser.parse_args(["pull", "org/repo", "-p", "tiles"]).source == "org/repo"
    assert parser.parse_args(["exec", "print(1)"]).code == "print(1)"
    for action in ("ls", "verify", "rm"):
        args = parser.parse_args(["models", action, *(["demo/v1"] if action == "rm" else [])])
        assert (args.cmd, args.action) == ("models", action)
    logs = parser.parse_args(["logs"])
    assert (logs.session, logs.lines, logs.json) == ("ultra", 100, False)


def test_config(tmp_path):
    path = tmp_path / "new" / "config.json"
    cfg = config.load_config(path)
    assert cfg == config.DEFAULT_CONFIG
    assert json.loads(path.read_text()) == cfg
    assert path.stat().st_mode & 0o777 == 0o600
    config.save_config({"poll_interval_sec": 3, "custom": 1}, path)
    assert config.load_config(path) == {**cfg, "poll_interval_sec": 3, "custom": 1}


def test_installed_interfaces():
    check_compatibility()


def test_init_deploys_shared_runtime(tmp_path, monkeypatch):
    from unittest.mock import Mock

    hub = tmp_path / "ColabUltra"
    hub.mkdir()
    monkeypatch.setattr(cli, "DEFAULT_GCLOUD_ADC", str(tmp_path / "adc"))
    (tmp_path / "adc").touch()
    monkeypatch.setattr(cli, "load_config", lambda: config.DEFAULT_CONFIG.copy())
    monkeypatch.setattr(cli, "save_config", Mock())
    monkeypatch.setattr(cli, "get_kaggle_quota", lambda: {})
    monkeypatch.setattr(cli, "inspect_all_vms", lambda: (None, []))
    monkeypatch.setattr(cli, "render_dashboard", Mock())
    cli.main(["init"])
    assert (hub / "scripts/cu_remote/stage.py").is_file()
