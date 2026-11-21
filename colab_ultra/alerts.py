import json
import os
import subprocess
import sys
from datetime import datetime

import requests
from rich.console import Console
from rich.panel import Panel

from colab_ultra.config import LOG_DIR

console = Console()


def log_breaker_event(event_type: str, details: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "breaker_events.jsonl")
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_type,
        **details,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def send_alert(title: str, message: str, cfg: dict, voice_text: str = None):
    """Send multi-channel alert: Terminal + macOS Notification + Voice + Webhook."""
    console.print(Panel(f"[bold yellow]{message}[/bold yellow]", title=f"🚨 {title}", border_style="yellow"))
    log_breaker_event("ALERT", {"title": title, "message": message})

    # 1. macOS Native Notification
    if cfg.get("enable_macos_notification", True) and sys.platform == "darwin":
        try:
            safe_title = title.replace('"', '\\"')
            safe_msg = message.replace('"', '\\"').replace("\n", " ")
            script = f'display notification "{safe_msg}" with title "{safe_title}" sound name "Glass"'
            subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    # 2. macOS Voice Alert (TTS)
    if cfg.get("enable_voice_alert", True) and voice_text and sys.platform == "darwin":
        try:
            subprocess.Popen(["say", voice_text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    # 3. Webhook (Bark / Feishu / Custom)
    webhook_url = cfg.get("webhook_url", "").strip()
    if webhook_url:
        try:
            requests.post(
                webhook_url,
                json={
                    "title": title,
                    "text": message,
                    "msg_type": "text",
                    "content": {"text": f"{title}\n{message}"},
                },
                timeout=5,
            )
        except Exception:
            pass
