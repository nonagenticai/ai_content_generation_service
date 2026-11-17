"""Two spend-control defects on the admission path.

D2 — the daily spend cap failed OPEN. ``check_quota`` caught every Redis fault
and returned ``allowed=True``, so for the whole duration of an outage every
paid GPU generation was admitted, silently: nothing errored, nothing counted
it, and ``record_generation_*`` dropped the spend as well, so the counters
under-reported afterwards too. The replacement is BOUNDED fail-open — one
retry, then a per-pod ``degraded_spend_budget`` that refuses admission once
exhausted — and, crucially, an ERROR log plus
``ai_content_gen_quota_degraded_total`` on every degraded decision, because an
unmeasurable window cannot be alerted on or reconciled against the bill.

D3 — admission was blind to the cost of what it admitted. The video routes
called ``check_quota(tenant_id)`` with no ``estimated_cost``, which defaults to
``0.0``, so the spend gate read ``daily_spend + 0.0 > max``: it could only ever
trip on budget ALREADY spent, never on the job being let through. A tenant one
cent under its cap was admitted for an arbitrarily expensive generation.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Response
from redis.exceptions import ConnectionError as RedisConnectionError

from src.services.providers.base import VideoGenerationProvider
from src.services.quota_service import _DEGRADED_MIN_CHARGE, QuotaService

# ---------------------------------------------------------------------------
# D2: the degraded posture
# ---------------------------------------------------------------------------


def _broken_redis() -> Any:
    redis = AsyncMock()
    redis.get.side_effect = RedisConnectionError("quota store unreachable")
    return redis


def _healthy_redis(generations: int = 0, spend: float = 0.0, concurrent: int = 0) -> Any:
    redis = AsyncMock()

    async def fake_get(key: str) -> Any:
        if "daily" in key:
            return {"generations": generations, "spend": spend}
        return concurrent

    redis.get.side_effect = fake_get
    return redis


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    """Skip the real 0.2s reconnect wait; the retry itself still happens."""
    monkeypatch.setattr("src.services.quota_service._DEGRADED_RETRY_DELAY_S", 0.0)


async def test_a_redis_blip_is_absorbed_by_the_retry_not_the_degraded_path():
    """First read fails, second succeeds -> a NORMAL decision, nothing degraded."""
    redis = AsyncMock()
    calls = {"n": 0}

    async def flaky_get(key: str) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RedisConnectionError("dropped connection")
        if "daily" in key:
            return {"generations": 0, "spend": 49.0}
        return 0

    redis.get.side_effect = flaky_get
    svc = QuotaService(redis_client=redis, max_daily_spend=50.0)

    with patch("src.services.quota_service.record_quota_degraded") as counter:
        result = await svc.check_quota("tenant-1", estimated_cost=5.0)

    # It got a real answer, and that answer enforced the real cap.
    assert result.allowed is False
    assert result.reason is not None
    assert "spend limit" in result.reason.lower()
    counter.assert_not_called()


async def test_redis_outage_does_not_silently_admit(caplog):
    """A degraded admission is loud: ERROR log + counter. Never silent."""
    svc = QuotaService(redis_client=_broken_redis(), degraded_spend_budget=25.0)

    with (
        caplog.at_level(logging.ERROR, logger="src.services.quota_service"),
        patch("src.services.quota_service.record_quota_degraded") as counter,
    ):
        result = await svc.check_quota("tenant-1", estimated_cost=2.0)

    assert result.allowed is True  # bounded fail-open, not fail-closed
    counter.assert_any_call(phase="check", outcome="admitted")
    messages = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert "DEGRADED" in messages
    assert "UNMETERED" in messages


async def test_degraded_admissions_are_bounded_by_the_pod_spend_budget(caplog):
    """Once the budget is spent the outage stops admitting: REFUSED, not open."""
    svc = QuotaService(redis_client=_broken_redis(), degraded_spend_budget=5.0)

    with patch("src.services.quota_service.record_quota_degraded"):
        first = await svc.check_quota("tenant-1", estimated_cost=3.0)
        second = await svc.check_quota("tenant-1", estimated_cost=1.0)

    assert first.allowed is True
    assert second.allowed is True  # 4.0 of 5.0 consumed

    with (
        caplog.at_level(logging.ERROR, logger="src.services.quota_service"),
        patch("src.services.quota_service.record_quota_degraded") as counter,
    ):
        third = await svc.check_quota("tenant-1", estimated_cost=3.0)

    assert third.allowed is False, "the fail-open window is unbounded again"
    assert third.reason is not None
    assert "degraded spend budget" in third.reason.lower()
    counter.assert_any_call(phase="check", outcome="rejected")
    assert "REFUSING" in " ".join(r.getMessage() for r in caplog.records)


async def test_an_unpriced_request_still_consumes_degraded_budget():
    """estimated_cost=0.0 must not be free forever during an outage."""
    budget = 3 * _DEGRADED_MIN_CHARGE
    svc = QuotaService(redis_client=_broken_redis(), degraded_spend_budget=budget)

    with patch("src.services.quota_service.record_quota_degraded"):
        outcomes = [(await svc.check_quota("t", estimated_cost=0.0)).allowed for _ in range(4)]

    assert outcomes == [True, True, True, False]


async def test_a_zero_budget_makes_the_check_fail_closed():
    """The operator-selectable end of the dial still exists."""
    svc = QuotaService(redis_client=_broken_redis(), degraded_spend_budget=0.0)

    with patch("src.services.quota_service.record_quota_degraded"):
        result = await svc.check_quota("tenant-1", estimated_cost=1.0)

    assert result.allowed is False


async def test_recovery_closes_the_window_and_restores_the_real_cap():
    """After Redis comes back the degraded budget resets and the cap applies."""
    svc = QuotaService(redis_client=_broken_redis(), degraded_spend_budget=1.0)
    with patch("src.services.quota_service.record_quota_degraded"):
        assert (await svc.check_quota("t", estimated_cost=1.0)).allowed is True
        assert (await svc.check_quota("t", estimated_cost=1.0)).allowed is False

    svc._redis = _healthy_redis(generations=0, spend=0.0)
    healthy = await svc.check_quota("t", estimated_cost=1.0)
    assert healthy.allowed is True
    assert svc._degraded_since is None
    assert svc._degraded_spend == 0.0


async def test_a_dropped_completion_is_reported_not_swallowed(caplog):
    """Spend that Redis never recorded is real money; it must surface."""
    redis = AsyncMock()
    redis.eval_script.side_effect = RedisConnectionError("quota store unreachable")
    svc = QuotaService(redis_client=redis)

    with (
        caplog.at_level(logging.ERROR, logger="src.services.quota_service"),
        patch("src.services.quota_service.record_quota_degraded") as counter,
    ):
        await svc.record_generation_complete("tenant-1", "job-1", cost=4.25)

    counter.assert_any_call(phase="complete", outcome="dropped")
    assert "DROPPED" in " ".join(r.getMessage() for r in caplog.records)
    # ...and the pod is now in the degraded window, so the NEXT admission is bounded.
    assert svc._degraded_since is not None


# ---------------------------------------------------------------------------
# D3: admission prices what it admits
# ---------------------------------------------------------------------------


class _PricedProvider(VideoGenerationProvider):
    """A real provider subclass -- the registry is typed, and no ty ignores here."""

    name = "priced"

    async def submit(self, prompt: str, **kwargs: Any) -> str:
        raise AssertionError("admission must reject before any provider call")

    async def check_status(self, task_id: str) -> dict:
        raise AssertionError("admission must reject before any provider call")

    async def download(self, task_id: str, video_url: str) -> bytes:
        raise AssertionError("admission must reject before any provider call")

    def estimate_cost(self, duration_s: float) -> float:
        return 2.0 * duration_s


def _video_service(default_provider: str = "priced") -> Any:
    from src.services.providers.base import VideoProviderRegistry
    from src.services.video_generation_service import VideoGenerationService

    registry = VideoProviderRegistry()
    registry.register(_PricedProvider())
    return VideoGenerationService(
        storage=AsyncMock(), registry=registry, default_provider=default_provider
    )


def _settings_with_default_duration(duration: float) -> Any:
    settings = MagicMock()
    settings.generation.default_duration_s = duration
    return settings


def test_estimate_request_cost_prices_an_uncreated_request():
    svc = _video_service()
    assert svc.estimate_request_cost("priced", 3.0) == 6.0
    # No explicit provider -> resolves the same default create_job would.
    assert svc.estimate_request_cost(None, 3.0) == 6.0


def test_estimate_request_cost_falls_back_to_the_configured_duration():
    svc = _video_service()
    with patch("src.config.get_settings", return_value=_settings_with_default_duration(5.0)):
        assert svc.estimate_request_cost("priced", None) == 10.0


def test_an_unpriceable_provider_stays_at_zero():
    svc = _video_service()
    assert svc.estimate_request_cost("does-not-exist", 3.0) == 0.0


async def test_a_job_that_would_exceed_the_cap_is_rejected_at_admission():
    """The whole point of D3: the cap must see the cost of the NEW job.

    daily_spend is 45 of 50 — under the cap, so the pre-fix ``daily_spend +
    0.0 > max`` admitted anything at all. The job itself costs 20.
    """
    svc = _video_service()
    estimated = svc.estimate_request_cost("priced", 10.0)
    assert estimated == 20.0

    quota = QuotaService(redis_client=_healthy_redis(spend=45.0), max_daily_spend=50.0)

    blind = await quota.check_quota("tenant-1")
    assert blind.allowed is True, "control: with no cost the check cannot trip"

    priced = await quota.check_quota("tenant-1", estimated_cost=estimated)
    assert priced.allowed is False
    assert priced.reason is not None
    assert "spend limit" in priced.reason.lower()


# ---------------------------------------------------------------------------
# D3, at the ROUTES — the defect was the omitted argument at the call sites,
# so the property has to be observed through the route, not the quota service.
# ---------------------------------------------------------------------------

TENANT = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"


def _request_for(svc: Any, quota: QuotaService) -> Any:
    request = MagicMock()
    request.headers = {}  # no Idempotency-Key
    request.state.tenant_id = TENANT
    request.app.state.video_generation_service = svc
    request.app.state.quota_service = quota
    request.app.state.provider_registry = svc._registry
    return request


def _near_cap_quota() -> QuotaService:
    """45 of 50 spent: UNDER the cap, so only the new job's cost can trip it."""
    return QuotaService(redis_client=_healthy_redis(spend=45.0), max_daily_spend=50.0)


async def test_generate_route_rejects_a_job_whose_cost_exceeds_the_cap():
    from fastapi import HTTPException

    from src.api.routes.generation import generate_video
    from src.api.schemas import GenerateRequest

    svc = _video_service()
    body = GenerateRequest(prompt="an expensive clip", provider="priced", duration_s=10.0)

    with pytest.raises(HTTPException) as excinfo:
        await generate_video(body, _request_for(svc, _near_cap_quota()), Response(), db=AsyncMock())

    assert excinfo.value.status_code == 429
    assert "spend limit" in str(excinfo.value.detail).lower()


async def test_batch_route_prices_the_whole_batch_not_zero():
    from fastapi import HTTPException

    from src.api.routes.generation import generate_batch
    from src.api.schemas import GenerateBatchRequest, GenerateRequest

    svc = _video_service()
    # Each clip is 2.0 * 1.0 = $2; three of them is $6 against $5 of headroom.
    body = GenerateBatchRequest(
        items=[
            GenerateRequest(prompt=f"clip {i}", provider="priced", duration_s=1.0) for i in range(3)
        ],
        budget_limit=50.0,
    )

    with pytest.raises(HTTPException) as excinfo:
        await generate_batch(body, _request_for(svc, _near_cap_quota()), db=AsyncMock())

    assert excinfo.value.status_code == 429
    assert "spend limit" in str(excinfo.value.detail).lower()


async def test_concept_image_route_prices_the_image_it_admits():
    from fastapi import HTTPException

    from src.api.content_routes import create_concept_image
    from src.api.content_schemas import ConceptImageRequest
    from src.services.image_providers.base import ImageGenProvider, ImageProviderRegistry
    from src.services.imagegen_service import ImageGenService

    class _PricedImageProvider(ImageGenProvider):
        name = "priced_image"

        async def generate(self, **kwargs: Any) -> Any:
            raise AssertionError("admission must reject before any provider call")

        def estimate_cost(self, *, size: str | None = None, quality: str | None = None) -> float:
            del size, quality
            return 20.0

    registry = ImageProviderRegistry()
    registry.register(_PricedImageProvider())
    svc = ImageGenService(storage=AsyncMock(), registry=registry, default_provider="priced_image")

    request = MagicMock()
    request.headers = {}  # no Idempotency-Key
    request.state.tenant_id = TENANT
    request.app.state.imagegen_service = svc
    request.app.state.quota_service = _near_cap_quota()
    request.app.state.image_provider_registry = registry

    body = ConceptImageRequest(prompt="an expensive image", provider="priced_image")

    with pytest.raises(HTTPException) as excinfo:
        await create_concept_image(body, request, Response(), db=AsyncMock())

    assert excinfo.value.status_code == 429
    assert "spend limit" in str(excinfo.value.detail).lower()


async def test_batch_clip_admission_prices_the_persisted_clip():
    """The fourth call site: ``_admit_batch_clip``, priced from the stored row."""
    from src.api.routes.generation import _admit_batch_clip

    svc = _video_service()
    job = MagicMock()
    job.provider = "priced"
    job.duration_s = 10.0  # -> $20 against $5 of headroom

    tracker = MagicMock()
    tracker.record_clip_failed = AsyncMock()

    with (
        patch("src.api.routes.generation.JobManager") as mock_jm,
        patch("src.services.quota_service.record_quota_degraded"),
    ):
        mock_jm.get_job = AsyncMock(return_value=job)
        mock_jm.mark_failed = AsyncMock()
        admitted = await _admit_batch_clip(
            AsyncMock(), svc, "job-1", "batch-1", tracker, TENANT, _near_cap_quota()
        )

    assert admitted is False
    tracker.record_clip_failed.assert_awaited_once_with("batch-1")
