"""Validate the upstream interfaces used for ADC and Drive mounting."""
from functools import wraps
from importlib.metadata import version
from inspect import signature


def patch_kernel_client():
    """Allow stopping a kernel client before its manager is initialized."""
    try:
        from jupyter_kernel_client import KernelClient
    except ImportError:
        return

    stop = KernelClient.stop
    if getattr(stop, "_cu_safe_stop_patched", False):
        return

    @wraps(stop)
    def safe_stop(self, *args, **kwargs):
        if getattr(self, "_manager", None) is None:
            return
        return stop(self, *args, **kwargs)

    safe_stop._cu_safe_stop_patched = True
    KernelClient.stop = safe_stop


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
    patch_kernel_client()
