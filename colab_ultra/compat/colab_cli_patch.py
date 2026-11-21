"""Validate the upstream interfaces used for ADC and Drive mounting."""
from importlib.metadata import version
from inspect import signature


def check_compatibility():
    """Report upstream drift before the first remote operation."""
    from colab_cli.auth import AuthProvider
    from colab_cli.runtime import ColabRuntime
    from colab_cli.common import state

    installed = version("google-colab-cli")
    if not installed.startswith("0.7."):
        raise RuntimeError(f"Review google-colab-cli {installed} compatibility (tested with 0.7.5)")
    required = {"code", "allow_stdin", "timeout"}
    if not required.issubset(signature(ColabRuntime.execute_code).parameters):
        raise RuntimeError(f"google-colab-cli {installed}: ColabRuntime.execute_code signature changed")
    if not hasattr(AuthProvider, "ADC") or not callable(getattr(state, "sync_sessions", None)):
        raise RuntimeError(f"google-colab-cli {installed}: ADC or session API changed")
