"""Workstation configuration and paths."""
import json
import os
import shutil
import tempfile
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

PROBE_LOCAL_PATH = os.path.join(BASE_DIR, "colab_ultra", "telemetry_probe.py")
VPS_SENTINEL_LOCAL_PATH = os.path.join(BASE_DIR, "colab_ultra", "vps_sentinel.py")
CONFIG_PATH = os.path.expanduser(
    os.environ.get("COLAB_ULTRA_CONFIG", "~/.config/colab-ultra/config.json")
)
LOG_DIR = os.path.join(os.path.dirname(CONFIG_PATH), "logs")

DEFAULT_CONFIG = {
    "poll_interval_sec": 15,
    "gpu_min_util_pct": 5.0,
    "cpu_min_util_pct": 55.0,
    "cpu_only_min_util_pct": 8.0,
    "alert_after_minutes": 5.0,
    "stop_after_minutes": 10.0,
    "enable_macos_notification": True,
    "enable_voice_alert": True,
    "webhook_url": "",  # Optional Bark / Feishu / Telegram / Custom webhook URL
    "auto_backup_to_drive_before_stop": True,
    "drive_hub_root": "/content/drive/MyDrive/ColabUltra",
    "kaggle_carrier_slug": "garryling/colab-ultra-t4x2-probe",
    "prefer_kaggle_for_t4": True,
    "vps_host": "",
    "vps_ssh_key": "~/.ssh/id_ed25519",
    "vps_remote_stats_path": "/tmp/colab_stats.json",
    "local_stats_out": "",
}


CONFIG_DIR = os.path.dirname(CONFIG_PATH) or "."
KAGGLE_SESSIONS_PATH = os.path.join(CONFIG_DIR, "kaggle_sessions.json")
DRIVE_ADC_PATH = os.path.join(CONFIG_DIR, "drive_adc.json")
DEFAULT_GCLOUD_ADC = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")


def resolve_colab_bin() -> str:
    """Find the installed upstream CLI."""
    return shutil.which("colab") or os.path.expanduser("~/.local/bin/colab")


def save_config(cfg: dict, path=None):
    """Publish a complete, private configuration file atomically."""
    path = Path(path or CONFIG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as f:
        temp = Path(f.name)
        try:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.flush()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def load_config(path=None) -> dict:
    """Merge user settings over defaults, creating a private file on first use."""
    path = Path(path or CONFIG_PATH)
    cfg = DEFAULT_CONFIG.copy()
    if path.exists():
        try:
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, TypeError, OSError) as exc:
            print(f"Unable to read {path}, using defaults: {exc}", file=sys.stderr)
    else:
        save_config(cfg, path)
    return cfg
