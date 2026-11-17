"""Unit tests for quota service (Phase 5)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from w2v_common.cache.redis_client import RedisClient
from w2v_common.testing import inject_fakeredis

from src.services.quota_service import QuotaService


@pytest.fixture
def redis_backed() -> RedisClient:
    """A real (fakeredis) client — the record_* paths run server-side Lua now.

    An AsyncMock cannot express what these tests care about (the value that
    ends up in Redis), and asserting ``redis.set.call_count >= 1`` is what let
    a non-atomic counter ship in the first place: it pins the implementation
    and says nothing about the number the tenant is billed against.
    """
    client = RedisClient(redis_url="redis://localhost:6379/0")
    inject_fakeredis(client)
    return client


def _make_redis(daily_data=None, concurrent=None):
    """Create a mock Redis client."""
    redis = AsyncMock()

    async def fake_get(key):
        if "daily" in key and daily_data is not None:
            return daily_data
        if "concurrent" in key and concurrent is not None:
            return concurrent
        return None

    redis.get.side_effect = fake_get
    redis.set = AsyncMock(return_value=True)
    # The concurrency gauge is a ZSET read (prune + ZCARD) through a Lua script,
    # not a GET — a mock that only answers ``get`` would make every check_quota
    # here raise instead of exercising the limit logic.
    redis.eval_script = AsyncMock(return_value=concurrent or 0)
    return redis


async def test_allow_within_limits():
    redis = _make_redis(daily_data={"generations": 5, "spend": 10.0}, concurrent=2)
    svc = QuotaService(redis_client=redis, max_daily_generations=100, max_daily_spend=50.0)
    result = await svc.check_quota("tenant-1")
    assert result.allowed is True
    assert result.remaining_generations == 95
    assert result.remaining_spend == 40.0


async def test_exceed_daily_generations():
    redis = _make_redis(daily_data={"generations": 100, "spend": 10.0}, concurrent=0)
    svc = QuotaService(redis_client=redis, max_daily_generations=100)
    result = await svc.check_quota("tenant-1")
    assert result.allowed is False
    assert result.reason is not None
    assert "generation limit" in result.reason.lower()
    assert result.remaining_generations == 0


async def test_exceed_daily_spend():
    redis = _make_redis(daily_data={"generations": 5, "spend": 48.0}, concurrent=0)
    svc = QuotaService(redis_client=redis, max_daily_spend=50.0)
    result = await svc.check_quota("tenant-1", estimated_cost=5.0)
    assert result.allowed is False
    assert result.reason is not None
    assert "spend limit" in result.reason.lower()


async def test_exceed_concurrent_jobs():
    redis = _make_redis(daily_data={"generations": 5, "spend": 10.0}, concurrent=10)
    svc = QuotaService(redis_client=redis, max_concurrent_jobs=10)
    result = await svc.check_quota("tenant-1")
    assert result.allowed is False
    assert result.reason is not None
    assert "concurrent" in result.reason.lower()


async def test_no_redis_allows():
    svc = QuotaService(redis_client=None)
    result = await svc.check_quota("tenant-1")
    assert result.allowed is True


async def test_redis_error_admits_within_the_degraded_budget():
    """Fail-open is now BOUNDED, not unconditional.

    The first admission during an outage still goes through — a Redis blip must
    not take paid generation down — but it is charged against the pod's
    ``degraded_spend_budget`` and logged at ERROR with a counter. Exhausting
    that budget REFUSES admission; see
    ``test_quota_degraded_and_admission_cost.py`` for that half.
    """
    redis = AsyncMock()
    redis.get.side_effect = OSError("connection error")
    svc = QuotaService(redis_client=redis)
    result = await svc.check_quota("tenant-1")
    assert result.allowed is True


async def test_record_generation_start(redis_backed: RedisClient):
    """A start bumps the generation count and takes a concurrency slot."""
    svc = QuotaService(redis_client=redis_backed)
    for i in range(2):
        await svc.record_generation_start("tenant-1", f"prior-{i}")

    await svc.record_generation_start("tenant-1", "job-1")

    status = await svc.get_quota_status("tenant-1")
    assert status["daily_generations_used"] == 3
    assert status["concurrent_jobs"] == 3


async def test_repeat_start_for_the_same_job_takes_only_one_slot(redis_backed: RedisClient):
    """A reclaim re-dispatching the SAME job must not take a second slot.

    The reaper re-runs a job that already holds a slot. Under the old INCR gauge
    that was a second unit the job could only ever release once.
    """
    svc = QuotaService(redis_client=redis_backed)
    await svc.record_generation_start("tenant-1", "job-1")
    await svc.record_generation_start("tenant-1", "job-1")

    status = await svc.get_quota_status("tenant-1")
    assert status["concurrent_jobs"] == 1


async def test_record_generation_complete(redis_backed: RedisClient):
    """A completion accumulates spend and releases a concurrency slot."""
    svc = QuotaService(redis_client=redis_backed)
    await svc.record_generation_start("tenant-1", "job-1")

    await svc.record_generation_complete("tenant-1", "job-1", cost=0.5)

    status = await svc.get_quota_status("tenant-1")
    assert status["daily_generations_used"] == 1
    assert status["daily_spend_used"] == pytest.approx(0.5)
    assert status["concurrent_jobs"] == 0


async def test_get_quota_status():
    redis = _make_redis(daily_data={"generations": 5, "spend": 10.0}, concurrent=3)
    svc = QuotaService(redis_client=redis, max_daily_generations=100, max_daily_spend=50.0)
    status = await svc.get_quota_status("tenant-1")
    assert status["daily_generations_used"] == 5
    assert status["daily_spend_used"] == 10.0
    assert status["concurrent_jobs"] == 3
    assert status["daily_generations_limit"] == 100
    assert status["daily_spend_limit"] == 50.0


async def test_decrement_concurrent(redis_backed: RedisClient):
    """A zero-cost completion still releases exactly one concurrency slot."""
    svc = QuotaService(redis_client=redis_backed)
    for i in range(5):
        await svc.record_generation_start("tenant-1", f"job-{i}")

    await svc.record_generation_complete("tenant-1", "job-0", cost=0.0)

    status = await svc.get_quota_status("tenant-1")
    assert status["concurrent_jobs"] == 4


async def test_a_repeat_release_frees_nothing_extra(redis_backed: RedisClient):
    """The reaper double-release must not hand the tenant a free slot.

    ``ReaperService`` re-dispatches a reclaimed job straight into the same
    background function whose ``finally`` releases, so a job can be released
    twice while its siblings are still running. Under the old DECR gauge the
    second release stole a slot from one of those siblings.
    """
    svc = QuotaService(redis_client=redis_backed)
    for i in range(3):
        await svc.record_generation_start("tenant-1", f"job-{i}")

    await svc.record_generation_complete("tenant-1", "job-0", cost=0.0)
    await svc.record_generation_complete("tenant-1", "job-0", cost=0.0)

    status = await svc.get_quota_status("tenant-1")
    assert status["concurrent_jobs"] == 2


async def test_a_release_for_a_job_that_holds_nothing_frees_nothing(redis_backed: RedisClient):
    """A stray completion must never free somebody else's slot."""
    svc = QuotaService(redis_client=redis_backed)
    await svc.record_generation_start("tenant-1", "job-live")

    await svc.record_generation_complete("tenant-1", "job-that-never-started", cost=0.0)

    status = await svc.get_quota_status("tenant-1")
    assert status["concurrent_jobs"] == 1


async def test_concurrent_gauge_never_goes_negative(redis_backed: RedisClient):
    """A completion with no matching start must not hand out free concurrency."""
    svc = QuotaService(redis_client=redis_backed)

    await svc.record_generation_complete("tenant-1", "job-1", cost=0.0)

    status = await svc.get_quota_status("tenant-1")
    assert status["concurrent_jobs"] == 0


async def test_spend_is_recorded_even_without_a_prior_start(redis_backed: RedisClient):
    """Spend must never be dropped because the daily key had expired."""
    svc = QuotaService(redis_client=redis_backed)

    await svc.record_generation_complete("tenant-1", "job-1", cost=1.25)

    status = await svc.get_quota_status("tenant-1")
    assert status["daily_spend_used"] == pytest.approx(1.25)
