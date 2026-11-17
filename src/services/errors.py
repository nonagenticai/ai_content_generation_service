"""Error classification for retry logic."""

from __future__ import annotations

import asyncio


def is_retriable_error(exc: BaseException) -> bool:
    """Determine if an error is transient and worth retrying.

    Returns True for timeouts, rate limits, and server errors.
    Returns False for client errors, validation errors, etc.
    """
    from src.services.exceptions import ProviderError

    if isinstance(exc, ProviderError):
        return exc.retriable

    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return True

    try:
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in (429, 500, 502, 503, 504)
        # Transport-level failures (ConnectError, ReadError, RemoteProtocolError,
        # ...) are transient by definition. They are NOT OSError subclasses
        # (ConnectError -> NetworkError -> TransportError -> RequestError ->
        # HTTPError -> Exception), so they escape the providers' ``except
        # (OSError, RuntimeError, ValueError)`` and land here; classifying them
        # non-retriable burned the provider's whole retry budget on one TCP reset
        # and fell through to a different PAID provider. Checked AFTER
        # HTTPStatusError, which is a sibling under HTTPError, not a RequestError.
        if isinstance(exc, httpx.RequestError):
            return True
    except ImportError:
        pass

    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "timeout" in msg or "rate limit" in msg:
            return True

    return False
