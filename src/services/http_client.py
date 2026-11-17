"""Shared httpx client manager with connection pooling."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_clients: dict[str, httpx.AsyncClient] = {}


def get_client(
    name: str,
    *,
    base_url: str = "",
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    event_hooks: dict[str, list] | None = None,
) -> httpx.AsyncClient:
    """Get or create a named httpx.AsyncClient with connection pooling.

    Args:
        name: Unique client name (e.g. "runway", "downloads").
        base_url: Optional base URL for the client.
        timeout: Request timeout in seconds.
        headers: Default headers for all requests.
        event_hooks: Optional httpx event hooks (e.g. a request hook that
            stamps per-request identity from ContextVars). Forwarded to httpx
            verbatim; hooks fire at send-time so a cached/shared client stays
            correct under concurrency. Applied only on first creation of a name.

    Returns:
        A reusable httpx.AsyncClient instance.
    """
    if name not in _clients:
        _clients[name] = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers=headers or {},
            http2=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            event_hooks=event_hooks or {},
        )
        logger.debug("Created shared httpx client: %s", name)
    return _clients[name]


async def close_all() -> None:
    """Close all managed httpx clients. Call during app shutdown."""
    for name, client in _clients.items():
        try:
            await client.aclose()
            logger.debug("Closed httpx client: %s", name)
        except OSError:
            logger.warning("Error closing httpx client %s", name, exc_info=True)
    _clients.clear()
