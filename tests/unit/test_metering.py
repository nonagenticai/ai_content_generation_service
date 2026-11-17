"""Unit tests for the ai_content_generation cost-metering shim.

Emit/field-marshalling/stream-resolution logic now lives in
``w2v_common.metering`` (tested there). These tests cover the thin service
shim: the routing identity constants, the re-exported helpers, and that an
emit carries this service's ``service_key`` / ``cost_source`` with the
EMITTER_GUIDE field shape.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.services.metering import (
    COST_SOURCE,
    SERVICE_KEY,
    MissingTenantError,
    emit_cost_event,
    init_metering_redis,
    require_tenant,
    resolve_stream,
    schedule_cost_event,
)

# The retired sentinel, spelled out so the test can prove it is REJECTED.
NIL_TENANT = "00000000-0000-0000-0000-000000000000"


class TestRoutingIdentity:
    """Service-specific constants the shim owns."""

    def test_cost_source_and_service_key(self) -> None:
        assert SERVICE_KEY == "w2v-ai-content"  # pragma: allowlist secret -- routing slug
        assert COST_SOURCE == "w2v-ai-content"

    def test_no_tenant_sentinel_is_reachable_from_this_package(self) -> None:
        """Product-owner ruling: a tenant_id may NEVER hold a sentinel.

        ``DEFAULT_TENANT_ID`` / ``UNATTRIBUTED_TENANT_ID`` used to be exported
        from this shim and re-exported from ``src.auth.middleware``. Both are
        gone; nothing in this package may hand out a substituted tenant.
        (w2v_common 0.67.0 deletes both names upstream too.)
        """
        import src.auth.middleware as mw
        import src.services.metering as metering

        for module in (metering, mw):
            for name in ("DEFAULT_TENANT_ID", "UNATTRIBUTED_TENANT_ID"):
                assert not hasattr(module, name), f"{module.__name__}.{name} is back"
                assert name not in getattr(module, "__all__", ())

    def test_helpers_are_reexported(self) -> None:
        # asgi.py / dependencies.py import these from the shim.
        assert callable(emit_cost_event)
        assert callable(schedule_cost_event)
        assert callable(init_metering_redis)


class TestEmitCostEvent:
    """One XADD per generation carrying this service's routing identity."""

    async def test_xadd_called_with_required_fields(self) -> None:
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(return_value=b"1234567890-0")

        await emit_cost_event(
            redis_mock,
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id="aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
            license_id="lic-1",
            subject_id="sub-1",
            app_id="app-1",
            provider="sora",
            model="sora",
            input_tokens=10,
            output_tokens=0,
            latency_ms=42_000,
            cost_usd=0.20,
            usage_extra={"unit": "video_seconds", "duration_s": 10.0},
        )

        redis_mock.xadd.assert_awaited_once()
        call_args = redis_mock.xadd.call_args
        # Stream is resolved centrally (METERING_STREAM env, else canonical).
        assert call_args.args[0] == resolve_stream()
        fields = call_args.args[1]
        assert fields["tenant_id"] == "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
        assert fields["license_id"] == "lic-1"
        assert fields["subject_id"] == "sub-1"
        assert fields["app_id"] == "app-1"
        assert fields["service_key"] == SERVICE_KEY == "w2v-ai-content"  # pragma: allowlist secret
        assert fields["cost_source"] == COST_SOURCE == "w2v-ai-content"
        assert fields["provider"] == "sora"
        assert fields["model"] == "sora"
        assert fields["served_model"] == "sora"
        assert fields["input_tokens"] == "10"
        assert fields["output_tokens"] == "0"
        assert fields["latency_ms"] == "42000"
        assert fields["cost_usd"] == "0.20000000"
        usage_extra = json.loads(fields["usage_extra"])
        assert usage_extra == {"duration_s": 10.0, "unit": "video_seconds"}
        assert call_args.kwargs["maxlen"] == 100_000
        assert call_args.kwargs["approximate"] is True

    async def test_xadd_failure_is_swallowed(self) -> None:
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(side_effect=ConnectionError("redis down"))

        # Must not raise.
        await emit_cost_event(
            redis_mock,
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id="aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
            license_id="l",
            subject_id="s",
            app_id="a",
            provider="dall-e-3",
            model="dall-e-3",
            input_tokens=1,
            output_tokens=0,
            latency_ms=2000,
        )

    @pytest.mark.parametrize(
        "absent_tenant",
        [
            "",
            "   ",
            None,
            NIL_TENANT,
            # Every other spelling of the SAME nil uuid. A string-equality
            # check against the canonical form lets all four of these through.
            "00000000000000000000000000000000",
            "{00000000-0000-0000-0000-000000000000}",
            "urn:uuid:00000000-0000-0000-0000-000000000000",
            "  00000000-0000-0000-0000-000000000000  ",
        ],
    )
    async def test_absent_tenant_raises_and_emits_nothing(self, absent_tenant: str | None) -> None:
        """REWRITTEN (was: asserts the falsy tenant is coerced to the nil sentinel).

        The ruling reversed that contract. An absent tenant is an ERROR, and the
        nil UUID is itself an absent tenant -- billing buckets it as unattributed
        and NEVER CHARGES IT, so a substituted event is real usage gone unbilled.
        Both halves matter: the guard raises AND no cost event is written.
        """
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(return_value=b"id")

        with pytest.raises(MissingTenantError):
            await emit_cost_event(
                redis_mock,
                service_key=SERVICE_KEY,
                cost_source=COST_SOURCE,
                tenant_id=require_tenant(absent_tenant),
                license_id="l",
                subject_id="s",
                app_id="",
                provider="sora",
                model="sora",
                input_tokens=10,
                output_tokens=0,
                latency_ms=1000,
            )

        redis_mock.xadd.assert_not_awaited()

    def test_require_tenant_passes_a_real_tenant_through(self) -> None:
        assert require_tenant("  0195f48b-1234-7000-8000-000000000001  ") == (
            "0195f48b-1234-7000-8000-000000000001"
        )
        import uuid as _uuid

        real = _uuid.UUID("0195f48b-1234-7000-8000-000000000001")
        assert require_tenant(real) == str(real)

    async def test_real_tenant_passed_through_unchanged(self) -> None:
        """A real tenant id must never be coerced to the sentinel."""
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(return_value=b"id")

        await emit_cost_event(
            redis_mock,
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id="0195f48b-1234-7000-8000-000000000001",
            license_id="l",
            subject_id="s",
            app_id="",
            provider="sora",
            model="sora",
            input_tokens=10,
            output_tokens=0,
            latency_ms=1000,
        )

        fields = redis_mock.xadd.call_args.args[1]
        assert fields["tenant_id"] == "0195f48b-1234-7000-8000-000000000001"

    async def test_image_emission_megapixels_encoding(self) -> None:
        """Image generation encodes width*height/1M as input_tokens."""
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(return_value=b"id")

        await emit_cost_event(
            redis_mock,
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id="aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
            license_id="l",
            subject_id="s",
            app_id="",
            provider="dall-e-3",
            model="dall-e-3",
            input_tokens=1,  # 1024 * 1024 // 1_000_000
            output_tokens=0,
            latency_ms=4500,
            cost_usd=0.04,
            usage_extra={
                "unit": "megapixels",
                "size": "1024x1024",
                "quality": "standard",
                "width": 1024,
                "height": 1024,
            },
        )

        fields = redis_mock.xadd.call_args.args[1]
        assert fields["input_tokens"] == "1"
        usage_extra = json.loads(fields["usage_extra"])
        assert usage_extra["unit"] == "megapixels"
        assert usage_extra["size"] == "1024x1024"


class TestInitMeteringRedis:
    """Init returns None when the URL is empty; never raises on connect failure."""

    async def test_empty_url_returns_none(self) -> None:
        client = await init_metering_redis("")
        assert client is None

    async def test_connect_failure_returns_none(self) -> None:
        client = await init_metering_redis("redis://127.0.0.1:1/0")
        assert client is None
