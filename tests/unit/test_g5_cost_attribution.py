"""G5: every ai-content cost event carries full, explicit attribution.

Both emitters (image + video) must state ``app_id`` (the w2v app key), a
``funding`` that is never left for billing to derive, and a DETERMINISTIC
``event_key`` derived from the job id, so a re-emit after a restart dedupes at
billing. These go through the REAL w2v_common emitter into fakeredis -- the
fields asserted are the ones billing's ``cost_flusher`` reads off the stream.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import fakeredis.aioredis
import pytest
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context
from w2v_common.metering import MissingAttributionError, drain_pending, resolve_stream

from src.services import metering

_TENANT = "0195f48b-1111-7000-8000-0000000000aa"
_LICENCE = "0195f48b-2222-7000-8000-0000000000bb"
_SUBJECT = "0195f48b-3333-7000-8000-0000000000cc"


@pytest.fixture
def redis() -> fakeredis.aioredis.FakeRedis:
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture(autouse=True)
def _identity() -> Any:
    set_identity_context(_TENANT, _LICENCE, _SUBJECT)
    yield
    clear_identity_context()


def _video(redis: Any) -> Any:
    from src.services.providers.base import VideoProviderRegistry
    from src.services.video_generation_service import VideoGenerationService

    return VideoGenerationService(
        storage=MagicMock(),
        registry=VideoProviderRegistry(),
        default_provider="stub",
        metering_redis=redis,
    )


def _image(redis: Any) -> Any:
    from src.services.image_providers.base import ImageProviderRegistry
    from src.services.imagegen_service import ImageGenService

    return ImageGenService(
        storage=MagicMock(),
        registry=ImageProviderRegistry(),
        default_provider="stub",
        metering_redis=redis,
    )


def _emit_video(redis: Any, job_id: str, provider: str = "sora") -> None:
    _video(redis)._emit_cost_event(
        job_id=job_id,
        tenant_id=_TENANT,
        provider=provider,
        duration_s=4.0,
        estimated_cost=0.2,
        elapsed_s=3.0,
        provider_request_id="req-1",
        file_size_bytes=10,
    )


def _emit_image(redis: Any, job_id: str, provider: str = "local_image") -> None:
    job = SimpleNamespace(
        tenant_id=_TENANT, job_id=job_id, provider=provider, size="1024x1024", quality=""
    )
    _image(redis)._emit_cost_event(
        job=cast(Any, job),
        result=cast(Any, SimpleNamespace(width=1024, height=1024)),
        estimated_cost=0.01,
        latency_ms=10,
    )


async def _events(redis: fakeredis.aioredis.FakeRedis) -> list[dict[str, str]]:
    await drain_pending()
    entries = cast(list[tuple[str, dict[str, str]]], await redis.xrange(resolve_stream()))
    return [fields for _id, fields in entries]


@pytest.mark.parametrize(
    ("emit", "provider", "funding", "kind"),
    [
        (_emit_video, "sora", "platform_external", "video"),
        (_emit_video, "wan", "local_gpu", "video"),
        (_emit_image, "openai_images", "platform_external", "image"),
        (_emit_image, "local_image", "local_gpu", "image"),
    ],
)
async def test_event_carries_full_attribution(
    redis: fakeredis.aioredis.FakeRedis, emit: Any, provider: str, funding: str, kind: str
) -> None:
    emit(redis, "job-a", provider)
    [event] = await _events(redis)
    assert event["tenant_id"] == _TENANT
    assert event["license_id"] == _LICENCE
    assert event["subject_id"] == _SUBJECT
    assert event["app_id"] == "w2v"
    assert event["funding"] == funding
    assert event["event_key"] == metering.job_event_key("job-a", kind)
    assert event["project_id"]  # tenant-level fallback at minimum, never absent


@pytest.mark.parametrize("emit", [_emit_video, _emit_image])
async def test_event_key_is_deterministic_per_job(
    redis: fakeredis.aioredis.FakeRedis, emit: Any
) -> None:
    emit(redis, "job-a")
    emit(redis, "job-a")  # a re-emit of the same job (restart / redelivery)
    emit(redis, "job-b")
    keys = [e["event_key"] for e in await _events(redis)]
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]


def test_image_and_video_keys_never_collide() -> None:
    assert metering.job_event_key("j", "image") != metering.job_event_key("j", "video")


@pytest.mark.parametrize("emit", [_emit_video, _emit_image])
async def test_undeclared_provider_fails_loud_and_emits_nothing(
    redis: fakeredis.aioredis.FakeRedis, emit: Any
) -> None:
    with pytest.raises(metering.UnknownFundingProviderError):
        emit(redis, "job-a", "brand-new-vendor")
    assert await _events(redis) == []


def test_every_registered_provider_declares_funding() -> None:
    """A provider class added without a funding entry would raise at emit time."""
    import importlib
    import pkgutil

    from src.services import image_providers, providers
    from src.services.image_providers.base import ImageGenProvider
    from src.services.providers.base import VideoGenerationProvider

    for pkg in (providers, image_providers):
        for mod in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{pkg.__name__}.{mod.name}")
    names = {
        cls.name
        for base in (VideoGenerationProvider, ImageGenProvider)
        for cls in base.__subclasses__()
        if cls.__module__.startswith("src.") and isinstance(getattr(cls, "name", None), str)
    }
    assert len(names) >= 10
    for name in names:
        assert metering.platform_funding(name) in {"local_gpu", "platform_external"}


@pytest.mark.parametrize("emit", [_emit_video, _emit_image])
async def test_strict_mode_full_attribution_emits(
    redis: fakeredis.aioredis.FakeRedis, emit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    emit(redis, "job-a")
    assert len(await _events(redis)) == 1


@pytest.mark.parametrize("emit", [_emit_video, _emit_image])
async def test_strict_mode_refuses_a_partial_event(
    redis: fakeredis.aioredis.FakeRedis, emit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    set_identity_context(_TENANT, None, _SUBJECT)  # no licence on the request
    with pytest.raises(MissingAttributionError, match="license_id"):
        emit(redis, f"job-{uuid.uuid4()}")
    assert await _events(redis) == []
