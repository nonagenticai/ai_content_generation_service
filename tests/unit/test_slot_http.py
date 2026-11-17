"""Connect-time SSRF guard for tenant provider-slot HTTP clients (DNS rebinding).

The REAL ``w2v_common.net.validate_public_url`` runs in every test; only DNS
(``socket.getaddrinfo``) is faked, per hostname, delegating everything else.
"""

from __future__ import annotations

import socket
from typing import Any, cast

import httpx
import pytest
from w2v_common.net import SSRFError

from src.services.provider_slots import (
    SlotJob,
    build_image_provider,
    build_video_provider,
    is_permanent_slot_error,
)
from src.services.slot_http import SlotHostRefusedError, SlotSSRFGuardTransport, slot_http_client

_PUBLIC_IP = "93.184.216.34"
_SECRET = "sk-tenant-0123456789abcdef"  # pragma: allowlist secret


@pytest.fixture
def dns(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """``{hostname: ip}`` answered by a faked getaddrinfo; other names resolve normally."""
    table: dict[str, str] = {}
    real = socket.getaddrinfo

    def _fake(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host in table:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], 0))]
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _fake)
    return table


class _Recorder(httpx.AsyncBaseTransport):
    def __init__(self, respond: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond or httpx.Response(200, json={"data": {"task_id": "t-1"}})


def _guarded(client: httpx.AsyncClient, inner: httpx.AsyncBaseTransport) -> None:
    transport = client._transport
    assert isinstance(transport, SlotSSRFGuardTransport)
    transport.inner = inner


@pytest.mark.parametrize("private_ip", ["10.0.0.5", "127.0.0.1", "169.254.169.254", "100.64.0.1"])
async def test_host_resolving_private_at_connect_time_is_refused(dns, private_ip) -> None:
    dns["rebind.tenant.example"] = private_ip
    inner = _Recorder()
    async with slot_http_client(
        base_url="https://rebind.tenant.example/v1", timeout=5.0, headers={}
    ) as client:
        _guarded(client, inner)
        with pytest.raises(SlotHostRefusedError) as caught:
            await client.post("/videos/text2video", json={})
    assert isinstance(caught.value.__cause__, SSRFError)
    assert caught.value.retriable is False
    assert inner.requests == [], "nothing may be sent to a private address"


async def test_public_host_is_pinned_to_the_validated_ip(dns) -> None:
    dns["api.tenant.example"] = _PUBLIC_IP
    inner = _Recorder()
    async with slot_http_client(
        base_url="https://api.tenant.example/v1", timeout=5.0, headers={"X-K": "v"}
    ) as client:
        _guarded(client, inner)
        response = await client.get("/status", params={"q": "1"})

    assert response.status_code == 200
    (sent,) = inner.requests
    assert sent.url.host == _PUBLIC_IP
    assert sent.url.path == "/v1/status" and sent.url.query == b"q=1"
    assert sent.headers["host"] == "api.tenant.example"
    assert sent.extensions["sni_hostname"] == "api.tenant.example"
    assert sent.headers["x-k"] == "v"
    # The client still sees the hostname request (redirect resolution, errors).
    assert response.request.url.host == "api.tenant.example"


async def test_redirect_hop_to_a_private_host_is_refused(dns) -> None:
    dns["api.tenant.example"] = _PUBLIC_IP
    dns["inner.tenant.example"] = "10.1.2.3"
    redirect = httpx.Response(302, headers={"location": "https://inner.tenant.example/x"})
    inner = _Recorder(redirect)
    async with httpx.AsyncClient(
        transport=SlotSSRFGuardTransport(inner), follow_redirects=True
    ) as client:
        with pytest.raises(SlotHostRefusedError):
            await client.get("https://api.tenant.example/start")
    assert [r.headers["host"] for r in inner.requests] == ["api.tenant.example"]


async def test_slot_built_video_provider_refuses_a_rebound_host(dns) -> None:
    """End to end through the per-job provider factory: submit never leaves."""
    dns["kling.tenant.example"] = "192.168.1.10"
    provider = build_video_provider(
        SlotJob(
            slot="video_gen",
            kind="kling",
            label="video_gen/kling (connection c v1)",
            base_url="https://kling.tenant.example/v1",
            model=None,
            secret=_SECRET,
        )
    )
    try:
        with pytest.raises(SlotHostRefusedError) as caught:
            await provider.submit("a clip", duration_s=5)
    finally:
        await provider.aclose()
    assert is_permanent_slot_error(caught.value)


async def test_slot_built_image_provider_uses_the_guarded_client() -> None:
    provider = build_image_provider(
        SlotJob(
            slot="image",
            kind="openai",
            label="image/openai (connection c v1)",
            base_url=None,
            model=None,
            secret=_SECRET,
        )
    )
    try:
        assert isinstance(cast(Any, provider)._client._transport, SlotSSRFGuardTransport)
    finally:
        await provider.aclose()


async def test_platform_provider_keeps_the_shared_pool() -> None:
    """No slot -> no guard transport (platform hosts are operator config)."""
    from src.services.providers.kling import KlingProvider

    provider = KlingProvider()
    assert not isinstance(provider._client._transport, SlotSSRFGuardTransport)
