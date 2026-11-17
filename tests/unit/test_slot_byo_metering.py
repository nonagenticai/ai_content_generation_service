"""Tenant provider-slot jobs are metered as tenant-funded BYO, platform jobs are not.

A slot job runs on the TENANT's own key: the platform paid nothing upstream, so
its cost event must never carry a platform price. billing-service derives BYO
from ``usage_extra.byo == "true"`` (modality_gateway ``byo_cost`` convention, and
the ``byo`` rollup dimension), so that key stays. The owner-decided
``funding="tenant_byo"`` is emitted TOP-LEVEL via w2v_common 0.94's ``funding=``
kwarg -- never in ``usage_extra``. Platform jobs assert no funding (billing
derives it).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from w2v_common.metering import emit_cost_event

from src.services import metering
from src.services.provider_slots import SlotJob, open_slot_job

_CONNECTION = "0195f48b-5555-7000-8000-00000000c0de"
_SECRET = "sk-tenant-0123456789abcdef"  # pragma: allowlist secret


def _slot_job(slot: str = "video_gen", kind: str = "kling", version: int = 4) -> SlotJob:
    return SlotJob(
        slot=slot,
        kind=kind,
        label=f"{slot}/{kind} (connection {_CONNECTION} v{version})",
        base_url=None,
        model=None,
        secret=_SECRET,
        connection_id=_CONNECTION,
        version=version,
    )


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        metering, "schedule_cost_event", lambda _redis, **kwargs: events.append(kwargs)
    )
    return events


def _video_service():
    from src.services.providers.base import VideoProviderRegistry
    from src.services.video_generation_service import VideoGenerationService

    return VideoGenerationService(
        storage=MagicMock(),
        registry=VideoProviderRegistry(),
        default_provider="fake",
        metering_redis=AsyncMock(),
    )


def _image_service():
    from src.services.image_providers.base import ImageProviderRegistry
    from src.services.imagegen_service import ImageGenService

    return ImageGenService(
        storage=MagicMock(),
        registry=ImageProviderRegistry(),
        default_provider="fake",
        metering_redis=AsyncMock(),
    )


def _emit_video(slot_job: SlotJob | None) -> None:
    _video_service()._emit_cost_event(
        job_id="job-v1",
        tenant_id="t",
        provider="kling",
        duration_s=5.0,
        estimated_cost=0.35,
        elapsed_s=1.0,
        provider_request_id="req-1",
        file_size_bytes=10,
        slot_job=slot_job,
    )


def _emit_image(slot_job: SlotJob | None) -> None:
    _image_service()._emit_cost_event(
        job=cast(
            Any,
            SimpleNamespace(
                tenant_id="t", job_id="job-i1", provider="openai_images", size=None, quality=None
            ),
        ),
        result=cast(Any, SimpleNamespace(width=1024, height=1024)),
        estimated_cost=0.04,
        latency_ms=10,
        slot_job=slot_job,
    )


def _assert_byo(event: dict[str, Any], *, slot: str, kind: str, version: int) -> None:
    assert event["cost_usd"] == 0.0
    extra = event["usage_extra"]
    assert extra["byo"] == "true"
    assert "owner_provided" not in extra  # tenant key, not an owner-managed key
    assert event["funding"] == "tenant_byo"
    assert "funding" not in extra  # top-level only, never duplicated in usage_extra
    assert (extra["slot"], extra["slot_kind"]) == (slot, kind)
    assert extra["connection_id"] == _CONNECTION
    assert extra["version"] == version
    assert _SECRET not in repr(event)


def test_video_slot_job_is_metered_as_tenant_byo(captured) -> None:
    _emit_video(_slot_job())
    assert len(captured) == 1
    _assert_byo(captured[0], slot="video_gen", kind="kling", version=4)
    # The unit encoding billing reads is untouched.
    assert captured[0]["input_tokens"] == 5
    assert captured[0]["usage_extra"]["unit"] == "video_seconds"


def test_video_platform_job_metering_is_unchanged(captured) -> None:
    _emit_video(None)
    assert captured[0]["cost_usd"] == pytest.approx(0.35)
    # platform job on an external vendor: funding stated, never left to billing
    assert captured[0]["funding"] == "platform_external"
    assert not {"byo", "funding", "owner_provided", "slot"} & set(captured[0]["usage_extra"])


def test_image_slot_job_is_metered_as_tenant_byo(captured) -> None:
    _emit_image(_slot_job(slot="image", kind="openai", version=1))
    assert len(captured) == 1
    _assert_byo(captured[0], slot="image", kind="openai", version=1)
    assert captured[0]["usage_extra"]["unit"] == "megapixels"


def test_image_platform_job_metering_is_unchanged(captured) -> None:
    _emit_image(None)
    assert captured[0]["cost_usd"] == pytest.approx(0.04)
    # platform job on an external vendor: funding stated, never left to billing
    assert captured[0]["funding"] == "platform_external"
    assert not {"byo", "funding", "owner_provided", "slot"} & set(captured[0]["usage_extra"])


def test_open_slot_job_carries_the_connection_ref_into_metering() -> None:
    """The ref persisted on the row (not the request) is what the reaper path meters."""
    job = open_slot_job(
        "job-1",
        {
            "slot": "video_gen",
            "kind": "kling",
            "ref": {"connection_id": _CONNECTION, "version": 7},
            "has_secret": False,
        },
        None,
    )
    assert job.byo_usage_extra()["connection_id"] == _CONNECTION
    assert job.byo_usage_extra()["version"] == 7


@pytest.mark.parametrize("emit", [_emit_video, _emit_image])
async def test_funding_reaches_the_stream_top_level(emit, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the REAL w2v_common emitter: ``funding`` is a top-level stream field
    (billing ``cost_flusher`` reads ``c.get("funding")``); ``usage_extra`` keeps
    ``byo`` but never ``funding``; a platform event carries no ``funding`` at all."""
    scheduled: list[dict[str, Any]] = []
    monkeypatch.setattr(metering, "schedule_cost_event", lambda _r, **kw: scheduled.append(kw))
    emit(_slot_job())
    emit(None)

    redis = AsyncMock()
    for kwargs in scheduled:  # the real emitter rejects the unit-test "t" tenant
        await cast(Any, emit_cost_event)(redis, **{**kwargs, "tenant_id": _CONNECTION})
    byo, platform = (call.args[1] for call in redis.xadd.await_args_list)

    assert byo["funding"] == "tenant_byo"
    assert json.loads(byo["usage_extra"])["byo"] == "true"
    assert "funding" not in json.loads(byo["usage_extra"])
    assert platform["funding"] == "platform_external"
    assert "byo" not in json.loads(platform["usage_extra"])


@pytest.mark.parametrize("secret", ["k", "ab", "xyz"])
def test_scrub_redacts_a_secret_of_any_length(secret: str) -> None:
    """A short secret is still a secret: the old ``len >= 4`` floor leaked it."""
    job = SlotJob(
        slot="image", kind="openai", label="image/openai", base_url=None, model=None, secret=secret
    )
    assert secret not in job.scrub(f"provider said: bad key {secret}!")
    assert "[redacted]" in job.scrub(f"bad key {secret}")


def test_scrub_without_a_secret_is_a_noop() -> None:
    job = SlotJob(
        slot="image", kind="openai", label="image/openai", base_url=None, model=None, secret=""
    )
    assert job.scrub("text") == "text"
