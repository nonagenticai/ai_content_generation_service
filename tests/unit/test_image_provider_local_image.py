"""Unit tests for LocalImageProvider (mflux/FLUX backend)."""

from __future__ import annotations

import asyncio
import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.config import LocalImageSettings
from src.services.image_providers.local_image import LocalImageProvider, _clamp_dim, _parse_size

_MINIMAL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9VE0cJQAAAAASUVORK5CYII="
)
_MINIMAL_PNG_B64 = base64.b64encode(_MINIMAL_PNG_BYTES).decode()


def _make_settings(**overrides: Any) -> LocalImageSettings:
    # busy_retry_base_delay=0.0 keeps the retry loop instantaneous in tests
    # (real backoff is exercised in the retry-timing test via monkeypatched sleep).
    params: dict[str, Any] = {
        "enabled": True,
        "backend_url": "http://image-gen.test:9204",
        "timeout": 60.0,
        "busy_retry_base_delay": 0.0,
    }
    params.update(overrides)
    return LocalImageSettings(**params)


def _ok_response(body: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = body
    response.raise_for_status = MagicMock()
    return response


def _busy_response(status: int, *, retry_after: str | None = None) -> MagicMock:
    """A response whose raise_for_status raises an HTTPStatusError for `status`."""
    response = MagicMock(status_code=status)
    response.headers = {} if retry_after is None else {"Retry-After": retry_after}
    response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("busy", request=MagicMock(), response=response)
    )
    return response


class TestDimensionHelpers:
    def test_clamp_dim_rounds_to_multiple_of_64(self) -> None:
        assert _clamp_dim(1000) == 960  # 1000 // 64 * 64 = 15 * 64 = 960
        assert _clamp_dim(1024) == 1024
        assert _clamp_dim(1) == 256  # clamped up
        assert _clamp_dim(5000) == 2048  # clamped down

    def test_parse_size_happy(self) -> None:
        assert _parse_size("1024x768") == (1024, 768)

    def test_parse_size_clamps(self) -> None:
        # 100 → 256, 5000 → 2048
        assert _parse_size("100x5000") == (256, 2048)

    def test_parse_size_none(self) -> None:
        assert _parse_size(None) == (None, None)
        assert _parse_size("garbage") == (None, None)
        assert _parse_size("abcxdef") == (None, None)


class TestGenerate:
    @patch("src.services.image_providers.local_image.get_client")
    async def test_happy_path(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.post = AsyncMock(
            return_value=_ok_response(
                {
                    "image_base64": _MINIMAL_PNG_B64,
                    "width": 1024,
                    "height": 1024,
                    "seed": 42,
                    "steps": 4,
                    "inference_time_ms": 1234.5,
                    "model_repo": "schnell",
                }
            )
        )
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings())
        result = await provider.generate(prompt="a cat")

        assert result.success is True
        assert result.image_data == _MINIMAL_PNG_BYTES
        assert result.mime_type == "image/png"
        assert result.width == 1024
        assert result.height == 1024
        assert result.error is None

        client.post.assert_awaited_once()
        (path,) = client.post.call_args.args
        assert path == "/api/image/generate"
        payload = client.post.call_args.kwargs["json"]
        assert payload == {"prompt": "a cat"}  # no negative_prompt, no size, no extras

    @patch("src.services.image_providers.local_image.get_client")
    async def test_size_parsed_into_payload(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.post = AsyncMock(
            return_value=_ok_response(
                {"image_base64": _MINIMAL_PNG_B64, "width": 768, "height": 512}
            )
        )
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings())
        await provider.generate(prompt="x", size="768x512")

        payload = client.post.call_args.kwargs["json"]
        assert payload["width"] == 768
        assert payload["height"] == 512

    @patch("src.services.image_providers.local_image.get_client")
    async def test_extra_params_forwarded(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.post = AsyncMock(return_value=_ok_response({"image_base64": _MINIMAL_PNG_B64}))
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings())
        await provider.generate(
            prompt="x",
            extra_params={"steps": 20, "guidance": 7.5, "seed": 42, "ignored": "no"},
        )

        payload = client.post.call_args.kwargs["json"]
        assert payload["steps"] == 20
        assert payload["guidance"] == 7.5
        assert payload["seed"] == 42
        assert "ignored" not in payload  # only steps/guidance/seed are whitelisted

    @patch("src.services.image_providers.local_image.get_client")
    async def test_negative_prompt_forwarded(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.post = AsyncMock(return_value=_ok_response({"image_base64": _MINIMAL_PNG_B64}))
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings())
        await provider.generate(prompt="x", negative_prompt="blurry")

        payload = client.post.call_args.kwargs["json"]
        assert payload["negative_prompt"] == "blurry"

    @patch("src.services.image_providers.local_image.get_client")
    async def test_5xx_returns_error_result(self, mock_get_client: MagicMock) -> None:
        response = MagicMock(status_code=503)
        response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("boom", request=MagicMock(), response=response)
        )
        client = AsyncMock()
        client.post = AsyncMock(return_value=response)
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings())
        result = await provider.generate(prompt="x")

        assert result.success is False
        assert result.error is not None
        assert "local_image" in result.error
        assert "503" in result.error

    @patch("src.services.image_providers.local_image.get_client")
    async def test_timeout_returns_error_result(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings())
        result = await provider.generate(prompt="x")

        assert result.success is False
        assert result.error is not None
        assert "local_image" in result.error
        assert "transport error" in result.error

    @patch("src.services.image_providers.local_image.get_client")
    async def test_connect_error_returns_error_result(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings())
        result = await provider.generate(prompt="x")

        assert result.success is False
        assert result.error is not None
        assert "transport error" in result.error

    @patch("src.services.image_providers.local_image.get_client")
    async def test_malformed_response_missing_b64(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.post = AsyncMock(return_value=_ok_response({"width": 1024}))
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings())
        result = await provider.generate(prompt="x")

        assert result.success is False
        assert result.error is not None
        assert "malformed response" in result.error


class TestBusyRetryAndConcurrency:
    """503/429 retry-with-backoff and the single-flight in-pod semaphore.

    These guard the live saturation regression: a concept-image batch now
    dispatches per-child concurrent tasks, but the mflux backend is
    single-flight and 503s while busy. The provider serializes children
    within a pod (semaphore) and retries transient busy responses.
    """

    @patch("src.services.image_providers.local_image.get_client")
    async def test_retries_503_then_succeeds(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.post = AsyncMock(
            side_effect=[
                _busy_response(503),
                _ok_response({"image_base64": _MINIMAL_PNG_B64, "width": 1024, "height": 1024}),
            ]
        )
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings(busy_retry_attempts=5))
        result = await provider.generate(prompt="a cat")

        assert result.success is True
        assert result.image_data == _MINIMAL_PNG_BYTES
        assert client.post.await_count == 2  # one 503, one success

    @patch("src.services.image_providers.local_image.get_client")
    async def test_429_is_retried(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.post = AsyncMock(
            side_effect=[_busy_response(429), _ok_response({"image_base64": _MINIMAL_PNG_B64})]
        )
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings(busy_retry_attempts=3))
        result = await provider.generate(prompt="x")

        assert result.success is True
        assert client.post.await_count == 2

    @patch("src.services.image_providers.local_image.get_client")
    async def test_503_on_all_attempts_returns_failure(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.post = AsyncMock(return_value=_busy_response(503))
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings(busy_retry_attempts=4))
        result = await provider.generate(prompt="x")

        assert result.success is False
        assert result.error == "local_image: backend returned 503"
        # initial attempt + 4 retries = 5 backend calls
        assert client.post.await_count == 5

    @patch("src.services.image_providers.local_image.get_client")
    async def test_non_busy_status_fails_immediately(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.post = AsyncMock(return_value=_busy_response(500))
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings(busy_retry_attempts=5))
        result = await provider.generate(prompt="x")

        assert result.success is False
        assert result.error == "local_image: backend returned 500"
        assert client.post.await_count == 1  # 500 is not retried

    @patch("src.services.image_providers.local_image.get_client")
    async def test_retry_after_header_is_honoured(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.post = AsyncMock(
            side_effect=[
                _busy_response(503, retry_after="7"),
                _ok_response({"image_base64": _MINIMAL_PNG_B64}),
            ]
        )
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings(busy_retry_attempts=3))
        with patch(
            "src.services.image_providers.local_image.asyncio.sleep",
            new=AsyncMock(),
        ) as mock_sleep:
            result = await provider.generate(prompt="x")

        assert result.success is True
        mock_sleep.assert_awaited_once_with(7.0)

    @patch("src.services.image_providers.local_image.get_client")
    async def test_semaphore_serializes_concurrent_calls(self, mock_get_client: MagicMock) -> None:
        """With max_concurrency=1, two concurrent generate() calls never overlap
        the backend request — max observed in-flight must be exactly 1."""
        in_flight = 0
        max_in_flight = 0

        async def _post(*_args: object, **_kwargs: object) -> MagicMock:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            # yield control so a second (unsemaphored) call could interleave
            await asyncio.sleep(0)
            in_flight -= 1
            return _ok_response({"image_base64": _MINIMAL_PNG_B64})

        client = AsyncMock()
        client.post = _post
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings(max_concurrency=1))
        results = await asyncio.gather(
            provider.generate(prompt="a"),
            provider.generate(prompt="b"),
        )

        assert all(r.success for r in results)
        assert max_in_flight == 1


class TestHealthCheck:
    @patch("src.services.image_providers.local_image.get_client")
    async def test_health_check_model_loaded(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.get = AsyncMock(return_value=_ok_response({"status": "ok", "model_loaded": True}))
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings())
        assert await provider.health_check() is True

    @patch("src.services.image_providers.local_image.get_client")
    async def test_health_check_model_not_loaded(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.get = AsyncMock(
            return_value=_ok_response({"status": "degraded", "model_loaded": False})
        )
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings())
        assert await provider.health_check() is False

    @patch("src.services.image_providers.local_image.get_client")
    async def test_health_check_5xx(self, mock_get_client: MagicMock) -> None:
        response = MagicMock()
        response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("boom", request=MagicMock(), response=MagicMock())
        )
        client = AsyncMock()
        client.get = AsyncMock(return_value=response)
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings())
        assert await provider.health_check() is False

    @patch("src.services.image_providers.local_image.get_client")
    async def test_health_check_transport_error(self, mock_get_client: MagicMock) -> None:
        client = AsyncMock()
        client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_get_client.return_value = client

        provider = LocalImageProvider(_make_settings())
        assert await provider.health_check() is False


class TestMetadata:
    def test_name(self) -> None:
        assert LocalImageProvider(_make_settings()).name == "local_image"

    def test_estimate_cost_zero(self) -> None:
        provider = LocalImageProvider(_make_settings())
        assert provider.estimate_cost() == 0.0
        assert provider.estimate_cost(size="2048x2048", quality="hd") == 0.0

    def test_get_models(self) -> None:
        assert LocalImageProvider(_make_settings()).get_models() == ["schnell", "dev"]

    def test_get_styles_empty(self) -> None:
        assert LocalImageProvider(_make_settings()).get_styles() == []


class TestTenantIdentityForwarding:
    """The mflux GPU call must carry the request's tenant for cost metering."""

    def test_client_registers_identity_hook(self) -> None:
        from w2v_common.auth.identity_context import identity_event_hook

        provider = LocalImageProvider(_make_settings())
        request_hooks = provider._client._event_hooks.get("request", [])
        assert identity_event_hook in request_hooks

    async def test_outbound_request_carries_bound_tenant(self) -> None:
        """End-to-end: a bound tenant ContextVar lands as X-Tenant-Id on the wire."""
        from w2v_common.auth.identity_context import (
            clear_identity_context,
            identity_event_hook,
            set_identity_context,
        )

        captured: dict[str, str] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["x_tenant_id"] = request.headers.get("X-Tenant-Id", "")
            captured["x_subject_id"] = request.headers.get("X-Subject-Id", "")
            return httpx.Response(200, json={"ok": True})

        # Build a client configured exactly as the provider does, but with a
        # MockTransport so we can inspect the request the event hook produced.
        client = httpx.AsyncClient(
            base_url="http://image-gen.test:9204",
            event_hooks={"request": [identity_event_hook]},
            transport=httpx.MockTransport(_handler),
        )
        try:
            set_identity_context("bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb", "lic-1", "subj-1")
            await client.post("/api/image/generate", json={"prompt": "x"})
            assert captured["x_tenant_id"] == "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
            assert captured["x_subject_id"] == "subj-1"
        finally:
            clear_identity_context()
            await client.aclose()

    async def test_outbound_request_with_no_tenant_is_refused(self) -> None:
        """An unbound outbound hop must FAIL, not go out unattributed.

        REWRITTEN. This used to assert ``captured["x_tenant_id"] == ""`` --
        i.e. the GPU call left the pod with no tenant on it and was simply
        never billed. That is the outbound twin of the nil-UUID substitution:
        real GPU work, silently unattributed. w2v_common 0.67.0 fails closed
        (``MissingTenantError``) and the request never reaches the wire, so the
        assertion is now the refusal PLUS the proof that nothing was sent.
        """
        from w2v_common.auth.identity_context import (
            clear_identity_context,
            identity_event_hook,
        )
        from w2v_common.tenant import MissingTenantError

        captured: dict[str, str] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["x_tenant_id"] = request.headers.get("X-Tenant-Id", "")
            return httpx.Response(200, json={"ok": True})

        clear_identity_context()
        client = httpx.AsyncClient(
            base_url="http://image-gen.test:9204",
            event_hooks={"request": [identity_event_hook]},
            transport=httpx.MockTransport(_handler),
        )
        try:
            with pytest.raises(MissingTenantError):
                await client.post("/api/image/generate", json={"prompt": "x"})
            assert captured == {}, "an unattributed request reached the wire"
        finally:
            await client.aclose()


class TestGatewayDelegation:
    """With a service_token set, the provider authenticates to the modality-gateway
    as a delegating service account: X-Tenant-Id carries the SA Bearer token and
    identity_event_hook adds X-Effective-Tenant-Id from the bound tenant."""

    @patch("src.services.image_providers.local_image.get_client")
    def test_service_token_becomes_bearer_default_header(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = AsyncMock()
        settings = LocalImageSettings(
            enabled=True,
            backend_url="http://modality-gateway.modality-gateway:8890",
            service_token="sa-token-abc",
        )
        LocalImageProvider(settings)
        headers = mock_get_client.call_args.kwargs["headers"]
        assert headers == {"X-Tenant-Id": "Bearer sa-token-abc"}

    @patch("src.services.image_providers.local_image.get_client")
    def test_no_service_token_sends_no_default_headers(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = AsyncMock()
        LocalImageProvider(_make_settings())  # default settings: service_token=""
        assert mock_get_client.call_args.kwargs["headers"] is None

    async def test_delegation_shape_on_the_wire(self, monkeypatch) -> None:
        """SA Bearer X-Tenant-Id is preserved; the customer tenant rides in
        X-Effective-Tenant-Id — exactly what the fail-closed gateway needs."""
        from w2v_common.auth.identity_context import (
            clear_identity_context,
            identity_event_hook,
            set_identity_context,
        )
        from w2v_common.tenant import TenantSource

        captured: dict[str, str] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["x_tenant_id"] = request.headers.get("X-Tenant-Id", "")
            captured["x_effective_tenant_id"] = request.headers.get("X-Effective-Tenant-Id", "")
            return httpx.Response(200, json={"ok": True})

        # 0.67.0 gates X-Effective-Tenant-Id on the DESTINATION being a
        # license-introspecting, metered peer. The modality-gateway is one, but
        # it is not in w2v_common's built-in default set, so the deployment
        # names it via W2V_METERED_DESTINATIONS.
        monkeypatch.setenv(
            "W2V_METERED_DESTINATIONS",
            "modality-gateway.modality-gateway",
        )
        # Constructed exactly as the provider does when service_token is set.
        client = httpx.AsyncClient(
            base_url="http://modality-gateway.modality-gateway:8890",
            headers={"X-Tenant-Id": "Bearer sa-token-abc"},
            event_hooks={"request": [identity_event_hook]},
            transport=httpx.MockTransport(_handler),
        )
        try:
            set_identity_context(
                "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb",
                "lic-1",
                "subj-1",
                tenant_source=TenantSource.LICENSE_TOKEN,
            )
            await client.post("/api/image/generate", json={"prompt": "x"})
            # Bearer SA token authenticates the caller; the bare-UUID customer
            # tenant does NOT clobber it and instead rides the delegation side-channel.
            assert captured["x_tenant_id"] == "Bearer sa-token-abc"
            assert captured["x_effective_tenant_id"] == "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
        finally:
            clear_identity_context()
            await client.aclose()


@pytest.fixture(autouse=True)
def _reset_http_client_cache():
    """Avoid leaking the patched client between tests via the module-level dict."""
    from src.services import http_client

    yield
    http_client._clients.clear()
