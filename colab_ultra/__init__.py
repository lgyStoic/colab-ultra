"""Control Colab and Kaggle sessions from a local workstation."""
import os
import subprocess

__version__ = "0.1.0"


def get_version_string() -> str:
    """Return the package version, appending the short git commit hash when available."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
    except Exception:
        sha = ""
    return f"colab-ultra {__version__} ({sha})" if sha else f"colab-ultra {__version__}"
