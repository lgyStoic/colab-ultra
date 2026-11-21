"""Bounded backoff for operations that are safe to repeat."""
import ssl
import time
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError

import requests
from websocket import WebSocketConnectionClosedException, WebSocketTimeoutException


def _is_ssl_eof(exc):
    """Find a TLS EOF while giving certificate verification failures precedence."""
    pending = [exc]
    seen = set()
    found_eof = False
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, ssl.SSLCertVerificationError):
            return False
        if isinstance(item, ssl.SSLEOFError):
            found_eof = True
        if not found_eof:
            try:
                message = str(item)
            except RecursionError:
                message = ""
            found_eof = "UNEXPECTED_EOF_WHILE_READING" in message or "SSLEOFError" in message
        links = (item.__cause__, item.__context__, getattr(item, "reason", None), *item.args)
        pending.extend(link for link in links if isinstance(link, BaseException))
    return found_eof


def transient(exc):
    """Recognize transport failures and temporary HTTP gateway errors."""
    if isinstance(exc, requests.exceptions.SSLError):
        return _is_ssl_eof(exc)
    if isinstance(exc, ssl.SSLError) and not _is_ssl_eof(exc):
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
                            requests.ConnectionError, requests.Timeout, ssl.SSLEOFError,
                            WebSocketConnectionClosedException, WebSocketTimeoutException))


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
