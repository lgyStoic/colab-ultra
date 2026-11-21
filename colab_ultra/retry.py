"""Bounded backoff for operations that are safe to repeat."""
import time
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError

import requests
from websocket import WebSocketTimeoutException


def transient(exc):
    """Recognize transport failures and temporary HTTP gateway errors."""
    if isinstance(exc, requests.exceptions.SSLError):
        return False
    if isinstance(exc, HTTPError):
        return exc.code in (502, 503, 504)
    status = getattr(getattr(exc, "response", None), "status_code", None)
    status = status or getattr(exc, "status_code", None)
    if status is not None:
        return status in (502, 503, 504)
    if isinstance(exc, URLError) and isinstance(exc.reason, Exception):
        return transient(exc.reason)
    return isinstance(exc, (ConnectionError, RemoteDisconnected, TimeoutError,
                            requests.ConnectionError, requests.Timeout, WebSocketTimeoutException))


def retry(operation, *, attempts=5):
    """Run a safe operation up to five times with exponential backoff."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if not transient(exc) or attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
