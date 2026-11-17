"""Concurrency contract for :mod:`src.services.quota_service`.

These are *lost-update* tests, not serial ones. The service is deployed at 2
replicas, so every quota counter is mutated by concurrent callers. A serial
"increment twice, assert 2" test passes against a plainly-racy
get-mutate-set implementation and therefore proves nothing.

To make the race observable in-process, ``_with_network_latency`` wraps the
fakeredis transport (``execute_command``) so *every* Redis command yields one
event-loop turn before it runs — the in-process stand-in for a real network
round trip to Redis. Under that transport:

* a get -> mutate-in-Python -> set implementation interleaves and loses
  updates (that is exactly what two replicas do to each other);
* a server-side atomic (Lua ``EVAL`` / ``INCR``) is a *single* command, so it
  cannot be interleaved no matter how slow the transport is.

The latency wrapper is applied identically to both, so the test discriminates
on atomicity alone, not on how many round trips a method makes.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from w2v_common.cache.redis_client import RedisClient
from w2v_common.testing import inject_fakeredis

from src.services.quota_service import QuotaService

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# Enough concurrency that a lost update is certain, not probabilistic.
CONCURRENCY = 50


def _with_network_latency(client: RedisClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every Redis command cost one event-loop turn.

    fakeredis answers synchronously, so without this an ``await`` on it never
    suspends and ``asyncio.gather`` degenerates into serial execution — hiding
    the very race we are testing for.
    """
    raw = client._ensure_connected()
    original: Callable[..., Awaitable[Any]] = raw.execute_command

    async def slow_execute_command(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0)  # a round trip to Redis
        return await original(*args, **kwargs)

    monkeypatch.setattr(raw, "execute_command", slow_execute_command)


@pytest.fixture
def racy_redis(monkeypatch: pytest.MonkeyPatch) -> RedisClient:
    """fakeredis-backed RedisClient whose transport suspends on every command."""
    client = RedisClient(redis_url="redis://localhost:6379/0")
    inject_fakeredis(client)
    _with_network_latency(client, monkeypatch)
    return client


@pytest.fixture
def quota(racy_redis: RedisClient) -> QuotaService:
    # Limits raised out of the way: these tests measure counting, not blocking.
    return QuotaService(
        redis_client=racy_redis,
        max_daily_generations=1_000_000,
        max_daily_spend=1_000_000.0,
        max_concurrent_jobs=1_000_000,
    )


async def _daily(racy_redis: RedisClient, quota: QuotaService, tenant: str) -> dict:
    value = await racy_redis.get(quota._daily_key(tenant))
    assert isinstance(value, dict), f"daily key is not a dict: {value!r}"
    return value


async def _concurrent(racy_redis: RedisClient, quota: QuotaService, tenant: str) -> int:
    """Live holders of ``tenant``'s concurrency slots.

    Read straight off the ZSET rather than through ``_read_concurrent`` so the
    assertions see the STORED state, not a value the read path could have
    repaired on its way past.
    """
    raw = racy_redis._ensure_connected()
    return int(await raw.zcard(quota._concurrent_key(tenant)))


async def _holders(racy_redis: RedisClient, quota: QuotaService, tenant: str) -> list[str]:
    raw = racy_redis._ensure_connected()
    members = await raw.zrange(quota._concurrent_key(tenant), 0, -1)
    return [m.decode() if isinstance(m, bytes) else str(m) for m in members]


@pytest.mark.asyncio
async def test_concurrent_starts_do_not_lose_daily_generations(
    quota: QuotaService, racy_redis: RedisClient
) -> None:
    """N concurrent starts must leave the daily generation count at exactly N.

    Against the get/set implementation this reports a count far below N: every
    lost update is a generation the tenant is not billed or throttled for.
    """
    await asyncio.gather(
        *(quota.record_generation_start("tenant-a", f"job-{i}") for i in range(CONCURRENCY))
    )

    daily = await _daily(racy_redis, quota, "tenant-a")
    assert daily["generations"] == CONCURRENCY, (
        f"lost {CONCURRENCY - daily['generations']} of {CONCURRENCY} generation increments"
    )


@pytest.mark.asyncio
async def test_concurrent_starts_do_not_lose_concurrent_slots(
    quota: QuotaService, racy_redis: RedisClient
) -> None:
    """N concurrent starts must leave the concurrent-job gauge at exactly N."""
    await asyncio.gather(
        *(quota.record_generation_start("tenant-b", f"job-{i}") for i in range(CONCURRENCY))
    )

    concurrent = await _concurrent(racy_redis, quota, "tenant-b")
    assert concurrent == CONCURRENCY, (
        f"concurrent gauge undercounts: {concurrent} instead of {CONCURRENCY}"
    )


@pytest.mark.asyncio
async def test_concurrent_completions_do_not_lose_spend(
    quota: QuotaService, racy_redis: RedisClient
) -> None:
    """N concurrent completions must accumulate every dollar of spend.

    Lost spend updates are the money defect: the tenant blows past the daily
    spend cap because most of what they spent was never recorded.
    """
    await quota.record_generation_start("tenant-c", "job-c")
    per_call_cost = 0.10

    await asyncio.gather(
        *(
            quota.record_generation_complete("tenant-c", f"job-{i}", cost=per_call_cost)
            for i in range(CONCURRENCY)
        )
    )

    daily = await _daily(racy_redis, quota, "tenant-c")
    assert daily["spend"] == pytest.approx(per_call_cost * CONCURRENCY), (
        f"recorded ${daily['spend']:.2f} of ${per_call_cost * CONCURRENCY:.2f} actually spent"
    )


@pytest.mark.asyncio
async def test_concurrent_completions_do_not_lose_concurrent_decrements(
    quota: QuotaService, racy_redis: RedisClient
) -> None:
    """Every start must be released by exactly one completion.

    A lost decrement leaks a concurrency slot forever (the key only clears on
    TTL), which eventually wedges the tenant at "concurrent job limit
    exceeded" with nothing actually running.
    """
    await asyncio.gather(
        *(quota.record_generation_start("tenant-d", f"job-{i}") for i in range(CONCURRENCY))
    )
    await asyncio.gather(
        *(
            quota.record_generation_complete("tenant-d", f"job-{i}", cost=0.0)
            for i in range(CONCURRENCY)
        )
    )

    concurrent = await _concurrent(racy_redis, quota, "tenant-d")
    assert concurrent == 0, f"leaked {concurrent} concurrency slots"


@pytest.mark.asyncio
async def test_interleaved_starts_and_completions_stay_consistent(
    quota: QuotaService, racy_redis: RedisClient
) -> None:
    """Starts and completions racing each other must not corrupt either field."""
    cost = 0.25
    await asyncio.gather(
        *(quota.record_generation_start("tenant-e", f"job-{i}") for i in range(CONCURRENCY)),
        *(
            quota.record_generation_complete("tenant-e", f"job-{i}", cost=cost)
            for i in range(CONCURRENCY)
        ),
    )

    daily = await _daily(racy_redis, quota, "tenant-e")
    assert daily["generations"] == CONCURRENCY
    assert daily["spend"] == pytest.approx(cost * CONCURRENCY)


# ---------------------------------------------------------------------------
# TTL semantics — an atomic rewrite that drops the TTL leaks keys forever,
# which would be a worse bug than the race it fixes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daily_key_keeps_its_48h_ttl(quota: QuotaService, racy_redis: RedisClient) -> None:
    raw = racy_redis._ensure_connected()
    await quota.record_generation_start("tenant-ttl", "job-1")

    ttl = await raw.ttl(quota._daily_key("tenant-ttl"))
    assert 172_000 < ttl <= 172_800, f"daily key TTL is {ttl}, expected ~172800s (48h)"


@pytest.mark.asyncio
async def test_concurrent_key_keeps_its_1h_ttl(
    quota: QuotaService, racy_redis: RedisClient
) -> None:
    raw = racy_redis._ensure_connected()
    await quota.record_generation_start("tenant-ttl2", "job-1")

    ttl = await raw.ttl(quota._concurrent_key("tenant-ttl2"))
    # The KEY outlives its longest-lived member by a minute, so an in-flight
    # holder can never have the key expire out from under it.
    assert 3_460 < ttl <= 3_660, f"concurrent key TTL is {ttl}, expected ~3660s (1h + 60s)"


@pytest.mark.asyncio
async def test_completion_refreshes_daily_ttl(quota: QuotaService, racy_redis: RedisClient) -> None:
    raw = racy_redis._ensure_connected()
    await quota.record_generation_start("tenant-ttl3", "job-1")
    await quota.record_generation_complete("tenant-ttl3", "job-1", cost=1.0)

    ttl = await raw.ttl(quota._daily_key("tenant-ttl3"))
    assert 172_000 < ttl <= 172_800, f"daily key TTL is {ttl} after completion"


# ---------------------------------------------------------------------------
# Wire-format compatibility: the service is live, so keys written by the
# previous implementation must keep working across the rolling upgrade.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reads_and_updates_a_key_written_by_the_previous_format(
    quota: QuotaService, racy_redis: RedisClient
) -> None:
    """A pre-existing ``{"generations": n, "spend": f}`` key must be preserved."""
    key = quota._daily_key("tenant-legacy")
    await racy_redis.set(key, {"generations": 7, "spend": 3.5}, ttl=172_800)

    await quota.record_generation_start("tenant-legacy", "job-1")
    await quota.record_generation_complete("tenant-legacy", "job-1", cost=0.5)

    daily = await _daily(racy_redis, quota, "tenant-legacy")
    assert daily["generations"] == 8
    assert daily["spend"] == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_check_quota_reads_what_record_wrote(
    quota: QuotaService, racy_redis: RedisClient
) -> None:
    """The read path must still understand the format the write path produces."""
    small = QuotaService(
        redis_client=racy_redis,
        max_daily_generations=10,
        max_daily_spend=10.0,
        max_concurrent_jobs=10,
    )
    for i in range(3):
        await small.record_generation_start("tenant-read", f"job-{i}")
    await small.record_generation_complete("tenant-read", "job-0", cost=2.0)

    result = await small.check_quota("tenant-read")
    assert result.allowed is True
    assert result.remaining_generations == 7
    assert result.remaining_spend == pytest.approx(8.0)
    assert result.remaining_concurrent == 8

    status = await small.get_quota_status("tenant-read")
    assert status["daily_generations_used"] == 3
    assert status["daily_spend_used"] == pytest.approx(2.0)
    assert status["concurrent_jobs"] == 2


@pytest.mark.asyncio
async def test_no_redis_is_still_a_no_op(racy_redis: RedisClient) -> None:
    """The degraded (no Redis) path must stay non-throwing."""
    service = QuotaService(redis_client=None)
    await service.record_generation_start("t", "job-1")
    await service.record_generation_complete("t", "job-1", cost=1.0)
    assert (await service.check_quota("t")).allowed is True
