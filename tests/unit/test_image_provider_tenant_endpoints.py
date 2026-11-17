"""Tenant self-hosted image slot kinds: ``nonagentic_endpoint`` + ``openai_compatible``.

The provider HTTP API is faked with ``httpx.MockTransport`` INSIDE the real
connect-time SSRF guard (DNS validation stubbed to a public IP), so every request
still goes through the guard + IP-pin path a production slot request takes.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

import src.services.slot_http as slot_http
from src.services.image_providers.tenant_endpoints import (
    TENANT_IMAGE_CONNECT_TIMEOUT_S,
    TENANT_IMAGE_TIMEOUT_S,
    NonagenticEndpointImageProvider,
    OpenAICompatibleImageProvider,
)
from src.services.provider_slots import (
    ImageProviderSlot,
    ProviderSlotError,
    SlotJob,
    build_image_provider,
    image_provider_name,
    is_permanent_slot_error,
)
from src.services.slot_http import SlotSSRFGuardTransport

SECRET = "tenant-gpu-secret-xyz"  # pragma: allowlist secret
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9VE0cJQAAAAASUVORK5CYII="
)
PNG_B64 = base64.b64encode(PNG).decode()
_PINNED_IP = "93.184.216.34"
_REF = {"connection_id": "4f1c2d3e-0000-4000-8000-000000000001", "version": 3}


def _job(
    kind: str, *, base_url: str | None = "https://gpu.tenant.example/v1", **kw: Any
) -> SlotJob:
    return SlotJob(
        slot="image",
        kind=kind,
        label=f"image/{kind} (connection x v3)",
        base_url=base_url,
        model=kw.get("model"),
        secret=kw.get("secret", SECRET),
    )


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(slot_http, "validate_public_url", lambda url, **_: _PINNED_IP)


def _install(provider: Any, handler: Callable[[httpx.Request], httpx.Response]) -> list:
    seen: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    transport = provider._client._transport
    assert isinstance(transport, SlotSSRFGuardTransport)
    transport.inner = httpx.MockTransport(_record)
    return seen


# ---------------------------------------------------------------------------
# nonagentic_endpoint
# ---------------------------------------------------------------------------


async def test_nonagentic_endpoint_posts_platform_protocol_with_bearer_secret() -> None:
    provider = build_image_provider(_job("nonagentic_endpoint", model="flux-dev"))
    assert isinstance(provider, NonagenticEndpointImageProvider)
    seen = _install(
        provider,
        lambda _r: httpx.Response(200, json={"image_base64": PNG_B64, "width": 512, "height": 512}),
    )
    try:
        result = await provider.generate(
            prompt="a lighthouse",
            negative_prompt="blur",
            size="512x512",
            extra_params={"steps": 4, "seed": 7, "ignored": 1},
        )
    finally:
        await provider.aclose()

    assert result.success and result.image_data == PNG
    assert (result.width, result.height, result.mime_type) == (512, 512, "image/png")
    (req,) = seen
    assert req.method == "POST" and req.url.path == "/v1/api/image/generate"
    assert req.headers["host"] == "gpu.tenant.example"
    assert req.headers["authorization"] == f"Bearer {SECRET}"
    assert json.loads(req.content) == {
        "prompt": "a lighthouse",
        "negative_prompt": "blur",
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": 7,
        "model": "flux-dev",
    }
    timeout = req.extensions["timeout"]
    assert timeout["read"] == TENANT_IMAGE_TIMEOUT_S
    assert timeout["connect"] == TENANT_IMAGE_CONNECT_TIMEOUT_S


async def test_nonagentic_endpoint_without_secret_sends_no_authorization() -> None:
    provider = build_image_provider(_job("nonagentic_endpoint", secret=""))
    seen = _install(provider, lambda _r: httpx.Response(200, json={"image_base64": PNG_B64}))
    try:
        result = await provider.generate(prompt="p")
    finally:
        await provider.aclose()
    assert result.success
    assert "authorization" not in seen[0].headers


async def test_nonagentic_endpoint_401_raises_and_is_permanent_never_leaking_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    job = _job("nonagentic_endpoint")
    provider = build_image_provider(job)
    _install(provider, lambda _r: httpx.Response(401, json={"detail": "bad key"}))
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await provider.generate(prompt="p")
    finally:
        await provider.aclose()
    assert is_permanent_slot_error(exc_info.value)
    message = job.failure_message(exc_info.value)
    assert "HTTP 401" in message and SECRET not in message
    assert SECRET not in caplog.text


async def test_nonagentic_endpoint_malformed_body_is_a_failed_result() -> None:
    provider = build_image_provider(_job("nonagentic_endpoint"))
    _install(provider, lambda _r: httpx.Response(200, json={"nope": True}))
    try:
        result = await provider.generate(prompt="p")
    finally:
        await provider.aclose()
    assert not result.success and "malformed" in (result.error or "")


# ---------------------------------------------------------------------------
# openai_compatible
# ---------------------------------------------------------------------------


async def test_openai_compatible_b64_json_posts_images_generations() -> None:
    provider = build_image_provider(_job("openai_compatible", model="sdxl-turbo"))
    assert isinstance(provider, OpenAICompatibleImageProvider)
    seen = _install(
        provider, lambda _r: httpx.Response(200, json={"data": [{"b64_json": PNG_B64}]})
    )
    try:
        result = await provider.generate(prompt="a fox", size="1024x768")
    finally:
        await provider.aclose()

    assert result.success and result.image_data == PNG
    assert (result.width, result.height) == (1024, 768)
    (req,) = seen
    assert req.url.path == "/v1/images/generations"
    assert req.headers["authorization"] == f"Bearer {SECRET}"
    body = json.loads(req.content)
    assert body == {"prompt": "a fox", "n": 1, "model": "sdxl-turbo", "size": "1024x768"}
    assert req.extensions["timeout"]["read"] == TENANT_IMAGE_TIMEOUT_S


async def test_openai_compatible_without_model_omits_it() -> None:
    provider = build_image_provider(_job("openai_compatible"))
    seen = _install(
        provider, lambda _r: httpx.Response(200, json={"data": [{"b64_json": PNG_B64}]})
    )
    try:
        await provider.generate(prompt="p")
    finally:
        await provider.aclose()
    assert "model" not in json.loads(seen[0].content)


async def test_openai_compatible_url_response_is_fetched_via_safe_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.services.image_providers.tenant_endpoints as mod

    fetched: list[str] = []

    async def _safe_get(_client: Any, url: str) -> httpx.Response:
        fetched.append(url)
        return httpx.Response(200, content=PNG, request=httpx.Request("GET", url))

    monkeypatch.setattr(mod, "safe_get", _safe_get)
    provider = build_image_provider(_job("openai_compatible"))
    _install(
        provider,
        lambda _r: httpx.Response(
            200, json={"data": [{"url": "https://cdn.tenant.example/a.png"}]}
        ),
    )
    try:
        result = await provider.generate(prompt="p")
    finally:
        await provider.aclose()
    assert result.success and result.image_data == PNG
    assert result.image_url == "https://cdn.tenant.example/a.png"
    assert fetched == ["https://cdn.tenant.example/a.png"]


async def test_openai_compatible_empty_data_is_a_failed_result() -> None:
    provider = build_image_provider(_job("openai_compatible"))
    _install(provider, lambda _r: httpx.Response(200, json={"data": []}))
    try:
        result = await provider.generate(prompt="p")
    finally:
        await provider.aclose()
    assert not result.success


# ---------------------------------------------------------------------------
# Dispatch: unknown kind + missing base_url fail loud
# ---------------------------------------------------------------------------


def test_unknown_kind_fails_loud_and_never_returns_a_platform_provider() -> None:
    with pytest.raises(ProviderSlotError, match="unknown image slot kind"):
        build_image_provider(_job("midjourney"))
    with pytest.raises(ProviderSlotError, match="unknown image slot kind"):
        image_provider_name("midjourney")


@pytest.mark.parametrize("kind", ["nonagentic_endpoint", "openai_compatible"])
def test_self_hosted_kind_without_base_url_fails_loud(kind: str) -> None:
    with pytest.raises(ProviderSlotError, match="base_url is required"):
        build_image_provider(_job(kind, base_url=None))
    with pytest.raises(ValidationError, match="requires base_url"):
        ImageProviderSlot(slot="image", kind=kind, base_url=None, ref=_REF)


def test_provider_names_for_the_row() -> None:
    assert image_provider_name("nonagentic_endpoint") == "tenant_nonagentic_endpoint"
    assert image_provider_name("openai_compatible") == "tenant_openai_compatible_images"
    assert image_provider_name("openai") == "openai_images"


def test_request_slot_accepts_self_hosted_kinds_and_hides_secret() -> None:
    slot = ImageProviderSlot(
        slot="image",
        kind="nonagentic_endpoint",
        base_url="https://gpu.tenant.example",
        secret=SECRET,
        ref=_REF,
    )
    assert SECRET not in repr(slot)
    assert "secret" not in slot.to_persisted()
