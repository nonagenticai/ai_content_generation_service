"""Verify both generation services delegate metering to the GC-safe scheduler.

Emission moved to ``w2v_common.metering.schedule_cost_event``, which retains a
strong reference to the in-flight task until the XADD lands (the event loop only
holds a weak ref, so the old bare ``create_task`` pattern could drop the billing
event mid-flight). These tests assert the service hands off to that scheduler
with the correct routing identity / encoding, and no-ops when metering is off.

Covers both VideoGenerationService._emit_cost_event and
ImageGenService._emit_cost_event.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestVideoGenerationServiceMeteringDelegation:
    """VideoGenerationService delegates to schedule_cost_event."""

    def _make_service(self):
        from src.services.providers.base import VideoProviderRegistry
        from src.services.video_generation_service import VideoGenerationService

        return VideoGenerationService(
            storage=MagicMock(),
            registry=VideoProviderRegistry(),
            default_provider="fake",
            metering_redis=AsyncMock(),
        )

    def test_emit_cost_event_delegates_with_routing_identity(self, monkeypatch) -> None:
        from src.services import metering

        svc = self._make_service()
        captured: dict[str, Any] = {}

        def _spy(_redis: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        # The helper imports schedule_cost_event inside the method; patch the
        # symbol on the shim module so the local import sees our spy.
        monkeypatch.setattr(metering, "schedule_cost_event", _spy)

        svc._emit_cost_event(
            job_id="job-v1",
            tenant_id="t",
            provider="runway",
            duration_s=5.0,
            estimated_cost=0.10,
            elapsed_s=4.2,
            provider_request_id="req-1",
            file_size_bytes=1024,
        )

        assert captured["service_key"] == metering.SERVICE_KEY
        assert captured["cost_source"] == metering.COST_SOURCE
        assert captured["tenant_id"] == "t"
        assert captured["provider"] == "runway"
        assert captured["model"] == "runway"
        assert captured["app_id"] == "w2v"
        assert captured["funding"] == "platform_external"
        assert captured["event_key"] == metering.job_event_key("job-v1", "video")
        assert captured["input_tokens"] == 5  # int(duration_s), video_seconds
        assert captured["latency_ms"] == 4200
        assert captured["cost_usd"] == 0.10
        assert captured["provider_request_id"] == "req-1"
        assert captured["usage_extra"]["unit"] == "video_seconds"
        assert captured["usage_extra"]["file_size_bytes"] == 1024

    def test_emit_cost_event_noop_when_metering_disabled(self, monkeypatch) -> None:
        from src.services import metering
        from src.services.providers.base import VideoProviderRegistry
        from src.services.video_generation_service import VideoGenerationService

        called = False

        def _spy(*_a: Any, **_k: Any) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(metering, "schedule_cost_event", _spy)

        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=VideoProviderRegistry(),
            default_provider="fake",
            metering_redis=None,
        )
        svc._emit_cost_event(
            job_id="job-v1",
            tenant_id="t",
            provider="runway",
            duration_s=5.0,
            estimated_cost=0.10,
            elapsed_s=4.2,
            provider_request_id="req-1",
            file_size_bytes=1024,
        )
        assert called is False


class TestImageGenServiceMeteringDelegation:
    """ImageGenService delegates to schedule_cost_event."""

    def _make_service(self):
        from src.services.image_providers.base import ImageProviderRegistry
        from src.services.imagegen_service import ImageGenService

        return ImageGenService(
            storage=MagicMock(),
            registry=ImageProviderRegistry(),
            default_provider="fake",
            metering_redis=AsyncMock(),
        )

    def test_emit_cost_event_delegates_with_routing_identity(self, monkeypatch) -> None:
        from src.services import metering

        svc = self._make_service()
        captured: dict[str, Any] = {}

        def _spy(_redis: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        monkeypatch.setattr(metering, "schedule_cost_event", _spy)

        job = SimpleNamespace(
            tenant_id="t",
            job_id="job-i1",
            provider="openai_images",
            size="1024x1024",
            quality="standard",
        )
        result = SimpleNamespace(width=1024, height=1024)

        svc._emit_cost_event(
            job=cast(Any, job),
            result=cast(Any, result),
            estimated_cost=0.04,
            latency_ms=4500,
        )

        assert captured["service_key"] == metering.SERVICE_KEY
        assert captured["cost_source"] == metering.COST_SOURCE
        assert captured["tenant_id"] == "t"
        assert captured["provider"] == "openai_images"
        assert captured["model"] == "openai_images"
        assert captured["app_id"] == "w2v"
        assert captured["funding"] == "platform_external"
        assert captured["event_key"] == metering.job_event_key("job-i1", "image")
        assert captured["input_tokens"] == 1  # max(1, 1024*1024 // 1_000_000)
        assert captured["latency_ms"] == 4500
        assert captured["cost_usd"] == 0.04
        assert captured["usage_extra"]["unit"] == "megapixels"
        assert captured["usage_extra"]["size"] == "1024x1024"

    @pytest.mark.parametrize(
        "absent_tenant",
        ["", "   ", None, "00000000-0000-0000-0000-000000000000"],
    )
    def test_falsy_job_tenant_raises_and_emits_nothing(self, monkeypatch, absent_tenant) -> None:
        """REWRITTEN (was: asserts a falsy job tenant emits ``DEFAULT_TENANT_ID``).

        Product-owner ruling: no substituted tenant. ``image_gen_jobs.tenant_id``
        is ``uuid NOT NULL``, so a job that reached a completed render always has
        a real one -- this is a tripwire, not an expected path. Emitting an
        unattributed cost event is worse than failing: billing never charges it.
        """
        from src.services import metering

        svc = self._make_service()
        emitted: list[Any] = []

        def _spy(_redis: Any, **kwargs: Any) -> None:
            emitted.append(kwargs)

        monkeypatch.setattr(metering, "schedule_cost_event", _spy)

        job = SimpleNamespace(
            tenant_id=absent_tenant,
            job_id="job-i1",
            provider="openai_images",
            size=None,
            quality=None,
        )
        result = SimpleNamespace(width=1024, height=1024)
        with pytest.raises(metering.MissingTenantError):
            svc._emit_cost_event(
                job=cast(Any, job),
                result=cast(Any, result),
                estimated_cost=0.04,
                latency_ms=4500,
            )
        assert emitted == []

    def test_emit_cost_event_noop_when_metering_disabled(self, monkeypatch) -> None:
        from src.services import metering
        from src.services.image_providers.base import ImageProviderRegistry
        from src.services.imagegen_service import ImageGenService

        called = False

        def _spy(*_a: Any, **_k: Any) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(metering, "schedule_cost_event", _spy)

        svc = ImageGenService(
            storage=MagicMock(),
            registry=ImageProviderRegistry(),
            default_provider="fake",
            metering_redis=None,
        )
        job = SimpleNamespace(
            tenant_id="t",
            job_id="job-i1",
            provider="openai_images",
            size="1024x1024",
            quality="standard",
        )
        result = SimpleNamespace(width=1024, height=1024)
        svc._emit_cost_event(
            job=cast(Any, job),
            result=cast(Any, result),
            estimated_cost=0.04,
            latency_ms=4500,
        )
        assert called is False
