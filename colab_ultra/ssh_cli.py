"""Thin passthrough to `google-colab-cli`'s `colab ssh` command."""
import argparse
import os
import shutil
import subprocess
import sys

from colab_ultra.backends.kaggle import _load_kaggle_sessions
from colab_ultra.compat.colab_cli_patch import check_compatibility


def add_ssh_parser(subparsers):
    """Register the `colab-ultra ssh` subcommand."""
    ssh = subparsers.add_parser("ssh", help="通过 Colab SSH 连接或执行单条诊断命令（仅 Colab，不支持 Kaggle）")
    ssh.add_argument("-s", "--session", default="ultra", help="会话名称")
    ssh.add_argument("-i", "--identity", default=None, help="SSH 密钥路径（.pub 或私钥，Colab 仅支持 ed25519/ecdsa）")
    ssh.add_argument("--proxy-mode", action="store_true", help="OpenSSH ProxyCommand 模式（供 ~/.ssh/config 桥接）")
    ssh.add_argument("command", nargs=argparse.REMAINDER, help="要在远端执行并立即退出的命令（建议写成 `-- CMD`）")
    return ssh


def _find_colab_bin() -> str:
    """Locate the `colab` executable from the current Python environment or PATH."""
    sibling = os.path.join(os.path.dirname(sys.executable), "colab")
    for candidate in (sibling, shutil.which("colab"), os.path.expanduser("~/.local/bin/colab")):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise SystemExit("`colab` executable not found. Install google-colab-cli>=0.8.0 via uv tool install.")


def build_ssh_argv(args, colab_bin: str) -> list[str]:
    """Build the underlying `colab --auth adc ssh` argument list."""
    argv = [colab_bin, "--auth", "adc", "ssh", "-s", args.session]
    if getattr(args, "proxy_mode", False):
        argv.append("--proxy-mode")
    if getattr(args, "identity", None):
        argv.extend(["-i", args.identity])
    remote_cmd = list(getattr(args, "command", None) or [])
    if remote_cmd and remote_cmd[0] == "--":
        remote_cmd = remote_cmd[1:]
    if remote_cmd:
        argv.append("--")
        argv.extend(remote_cmd)
    return argv


def cmd_ssh(args):
    """Forward `colab-ultra ssh` to `colab ssh` for Colab sessions."""
    check_compatibility()
    if args.session in _load_kaggle_sessions():
        raise SystemExit(
            f"Session '{args.session}' is a Kaggle instance; SSH is only supported on Google Colab sessions. "
            f"Use `colab-ultra logs -s {args.session}` or `colab-ultra status --json` instead."
        )
    proc = subprocess.run(build_ssh_argv(args, _find_colab_bin()))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
