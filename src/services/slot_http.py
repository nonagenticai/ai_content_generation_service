"""SSRF-guarded HTTP clients for tenant provider-slot instances (P5-2).

A slot's ``base_url`` is tenant input that receives the tenant's credential and
our request bodies. The request-time check (``_validate_provider_slot_base_url``)
resolves the host ONCE, at enqueue; the job may run minutes later (or after a
reaper resume) and httpx would resolve the host AGAIN at connect time. A host
that answered a public IP at enqueue and a private one at connect (DNS
rebinding) would therefore reach the estate.

:class:`SlotSSRFGuardTransport` closes that window on EVERY request the client
sends, redirect hops included (httpx follows redirects above the transport, so
each hop is a fresh ``handle_async_request``): it resolves + validates the host
with :func:`w2v_common.net.validate_public_url` and then connects to the
validated IP literally (``sni_hostname`` + the original ``Host`` header keep TLS
certificate verification and virtual hosting on the hostname). There is no
second resolution for an attacker to race.

w2v_common (0.89.0) ships the validator and the one-shot ``safe_request`` helpers
but no transport a long-lived ``httpx.AsyncClient`` can be built on, so the
transport lives here and delegates every decision to the shared validator.
"""

from __future__ import annotations

import asyncio

import httpx
from w2v_common.net import SSRFError, validate_public_url

from src.services.exceptions import ProviderError, format_error


class SlotHostRefusedError(ProviderError):
    """A slot request refused by the connect-time SSRF guard. Never retried.

    Deliberately a ``ProviderError`` (``retriable=False``), NOT the raw
    ``SSRFError``: that is a ``ValueError``, which
    ``VideoGenerationProvider.generate`` flattens into a routine
    ``success=False`` result -- indistinguishable from a provider fault. The
    ``SSRFError`` stays reachable as ``__cause__`` for the slot classifier.
    """

    def __init__(self, cause: SSRFError) -> None:
        super().__init__(
            f"slot provider host refused: {format_error(cause)}", retriable=False, cause=cause
        )


class SlotSSRFGuardTransport(httpx.AsyncBaseTransport):
    """Validate + IP-pin every outgoing request; refuse non-public destinations.

    Raises :class:`SlotHostRefusedError` (cause: ``SSRFError``) before any
    connection is opened when the host is estate-internal by name, resolves to a
    private / reserved / loopback / link-local / CGNAT address, or does not
    resolve at all (fail-closed).

    Note: the inner pool keys connections by the pinned IP. A slot client talks
    to the single host of its ``base_url``, so two hostnames sharing one IP (and
    so one pooled TLS connection) is not a case this client meets.
    """

    def __init__(
        self, inner: httpx.AsyncBaseTransport | None = None, *, http2: bool = False
    ) -> None:
        self.inner: httpx.AsyncBaseTransport = inner or httpx.AsyncHTTPTransport(http2=http2)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        try:
            # getaddrinfo blocks; never on the event loop.
            pinned_ip = await asyncio.to_thread(validate_public_url, str(url))
        except SSRFError as exc:
            raise SlotHostRefusedError(exc) from exc
        if pinned_ip is None:  # pragma: no cover -- no allow_hosts are passed
            return await self.inner.handle_async_request(request)
        hostname = url.host
        # A fresh request for the inner transport: the client keeps the original
        # (hostname) request as ``response.request`` and for redirect resolution.
        pinned = httpx.Request(
            request.method,
            url.copy_with(host=pinned_ip),
            headers=request.headers,  # already carries Host: <hostname>
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": hostname},
        )
        return await self.inner.handle_async_request(pinned)

    async def aclose(self) -> None:
        await self.inner.aclose()


def slot_http_client(
    *, base_url: str, timeout: float | httpx.Timeout, headers: dict[str, str]
) -> httpx.AsyncClient:
    """A per-job client whose every request passes :class:`SlotSSRFGuardTransport`.

    Passing ``transport`` also disables httpx's env-proxy mounts, so no
    ``HTTP(S)_PROXY`` setting can route a slot request around the guard.
    """
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        headers=headers,
        transport=SlotSSRFGuardTransport(http2=True),
    )
