"""A tenant-funded (provider-slot) job is outside the PLATFORM spend cap.

A slot job runs on the tenant's own provider key, so the platform pays nothing.
Counting it against the platform daily spend cap (or the degraded spend budget)
blocked a tenant on its own key whenever the platform budget was exhausted. The
generation-count and concurrency limits still apply: they protect this
service's capacity, not the platform's wallet.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from src.services.quota_service import QuotaService


def _redis(*, generations: int = 0, spend: float = 0.0, concurrent: int = 0) -> Any:
    redis = AsyncMock()

    async def fake_get(key: str) -> Any:
        if "daily" in key:
            return {"generations": generations, "spend": spend}
        return concurrent

    redis.get.side_effect = fake_get
    return redis


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.services.quota_service._DEGRADED_RETRY_DELAY_S", 0.0)


async def test_tenant_funded_job_passes_an_exhausted_spend_cap() -> None:
    svc = QuotaService(redis_client=_redis(spend=55.0), max_daily_spend=50.0)

    platform = await svc.check_quota("t", estimated_cost=1.0)
    byo = await svc.check_quota("t", estimated_cost=1.0, tenant_funded=True)

    assert platform.allowed is False and "spend limit" in (platform.reason or "").lower()
    assert byo.allowed is True


async def test_tenant_funded_job_still_honours_the_generation_limit() -> None:
    svc = QuotaService(redis_client=_redis(generations=10), max_daily_generations=10)
    result = await svc.check_quota("t", tenant_funded=True)
    assert result.allowed is False
    assert "generation limit" in (result.reason or "").lower()


async def test_tenant_funded_job_never_touches_the_degraded_spend_budget() -> None:
    redis = AsyncMock()
    redis.get.side_effect = RedisConnectionError("down")
    svc = QuotaService(redis_client=redis, degraded_spend_budget=0.0)

    byo = await svc.check_quota("t", estimated_cost=5.0, tenant_funded=True)
    platform = await svc.check_quota("t", estimated_cost=5.0)

    assert byo.allowed is True
    assert svc._degraded_spend == 0.0
    # Control: the zero budget still fails a platform job closed.
    assert platform.allowed is False
