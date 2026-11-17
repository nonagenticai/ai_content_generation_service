"""The concurrency gauge must be able to reclaim a slot nobody released.

This is the regression suite for a defect whose whole danger was that the code
*said* it could not happen. Three call-site comments justified their error
handling with "the slot leaks only until its 1h TTL", and the module header
called self-healing "part of the stored contract". It was not true: the gauge
was ``INCR``/``DECR`` on a plain key and BOTH scripts re-issued ``EXPIRE`` on
that key, so for any tenant generating at least once an hour the TTL was pushed
forward forever and a unit leaked by a SIGKILLed pod was never reclaimed.

Production shape: a tenant drifts to a permanent 429 "Concurrent job limit
exceeded" with nothing actually running, gradually and then totally, emitting
ZERO error lines — the leak has no failure path of its own.

``test_the_key_ttl_alone_cannot_reclaim_a_leaked_slot`` is the load-bearing one:
it asserts the *reason* the old design failed (traffic refreshes the key TTL) is
no longer what reclaim depends on.
"""

from __future__ import annotations

import pytest
from w2v_common.cache.redis_client import RedisClient
from w2v_common.testing import inject_fakeredis

from src.services import quota_service as quota_module
from src.services.quota_service import QuotaService

TENANT = "tenant-leak"


@pytest.fixture
def redis_backed() -> RedisClient:
    client = RedisClient(redis_url="redis://localhost:6379/0")
    inject_fakeredis(client)
    return client


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    """A movable clock for the module under test.

    The gauge reclaims by DEADLINE, so the only way to observe reclaim is to
    move time past one. Patching the module's ``time`` is what lets a unit test
    reach an hour ahead without sleeping.
    """

    class _Clock:
        now = 1_000_000.0

        def advance(self, seconds: float) -> None:
            self.now += seconds

    c = _Clock()
    monkeypatch.setattr(quota_module.time, "time", lambda: c.now)
    return c


async def test_a_slot_whose_holder_never_released_is_reclaimed(
    redis_backed: RedisClient, clock
) -> None:
    """The core property: a start with no matching completion frees itself."""
    svc = QuotaService(redis_client=redis_backed, max_concurrent_jobs=2)
    await svc.record_generation_start(TENANT, "job-killed")

    # The holder is gone (pod SIGKILLed between start and completion). Nothing
    # will ever call record_generation_complete for it.
    assert (await svc.get_quota_status(TENANT))["concurrent_jobs"] == 1

    clock.advance(quota_module._CONCURRENT_TTL_SECONDS + 1)

    assert (await svc.get_quota_status(TENANT))["concurrent_jobs"] == 0


async def test_a_leaked_slot_does_not_permanently_throttle_the_tenant(
    redis_backed: RedisClient, clock
) -> None:
    """The observable symptom: admission recovers on its own.

    A gauge that self-heals in ZCARD but not in the admission decision would
    still wedge the tenant, so assert on ``check_quota``, not on the count.
    """
    svc = QuotaService(redis_client=redis_backed, max_concurrent_jobs=1)
    await svc.record_generation_start(TENANT, "job-killed")
    assert (await svc.check_quota(TENANT)).allowed is False

    clock.advance(quota_module._CONCURRENT_TTL_SECONDS + 1)

    result = await svc.check_quota(TENANT)
    assert result.allowed is True, (
        "a leaked slot still throttles the tenant after its deadline — the gauge "
        "cannot reclaim, which is the 9h-wedge shape"
    )


async def test_the_key_ttl_alone_cannot_reclaim_a_leaked_slot(
    redis_backed: RedisClient, clock
) -> None:
    """Reclaim must NOT depend on the key's TTL, because traffic refreshes it.

    This is the exact invariant the old implementation assumed and violated: it
    leaned on a 1h key TTL while re-issuing EXPIRE on every start and every
    completion. Here a busy tenant keeps the key alive for the whole window and
    the leaked holder is STILL reclaimed, because reclaim is per-member.
    """
    svc = QuotaService(redis_client=redis_backed, max_concurrent_jobs=10)
    await svc.record_generation_start(TENANT, "job-killed")

    # Sustained traffic: a start+complete every few minutes for well over the
    # slot TTL. Every one of these refreshes the KEY's expiry, so the key is
    # never allowed to lapse — under the old design that is precisely what made
    # the leak permanent.
    step = quota_module._CONCURRENT_TTL_SECONDS // 10
    for i in range(12):
        clock.advance(step)
        await svc.record_generation_start(TENANT, f"healthy-{i}")
        await svc.record_generation_complete(TENANT, f"healthy-{i}", cost=0.0)

    # One genuinely in-flight job, started last, so the key is non-empty and the
    # assertion below discriminates "the leaked member was pruned" from "the
    # whole key vanished" (an empty zset is deleted by Redis).
    await svc.record_generation_start(TENANT, "job-still-running")

    raw = redis_backed._ensure_connected()
    assert await raw.exists(svc._concurrent_key(TENANT)), "precondition: key must be live"

    holders = await raw.zrange(svc._concurrent_key(TENANT), 0, -1)
    holders = {h.decode() if isinstance(h, bytes) else str(h) for h in holders}
    assert holders == {"job-still-running"}, (
        f"expected only the live holder, got {sorted(holders)} — the leaked "
        "holder survived past its deadline because the key TTL kept being "
        "refreshed, i.e. reclaim is not per-member"
    )
    assert (await svc.get_quota_status(TENANT))["concurrent_jobs"] == 1


async def test_a_live_holder_is_not_reclaimed_early(redis_backed: RedisClient, clock) -> None:
    """The gauge must not free a slot before its deadline (that would uncap spend)."""
    svc = QuotaService(redis_client=redis_backed, max_concurrent_jobs=5)
    await svc.record_generation_start(TENANT, "job-running")

    clock.advance(quota_module._CONCURRENT_TTL_SECONDS - 60)

    assert (await svc.get_quota_status(TENANT))["concurrent_jobs"] == 1


async def test_the_gauge_key_is_not_the_old_counter_key(redis_backed: RedisClient) -> None:
    """A rolling upgrade must not run ZSET commands against the old STRING key.

    The old counter lives at ``…:quota:concurrent:{tenant}``. If the new ZSET
    reused that name, every new pod would get ``WRONGTYPE`` from any tenant with
    a live old key — i.e. the whole fleet failing into degraded mode on deploy.
    """
    svc = QuotaService(redis_client=redis_backed, max_concurrent_jobs=5)
    legacy_key = f"content_generation:quota:concurrent:{TENANT}"

    # Seed the legacy counter exactly as a pre-upgrade pod would have left it,
    # THEN drive the new code paths. Asserting on the key name alone would pass
    # against a build that still collides; this exercises the actual failure.
    await redis_backed.set(legacy_key, 3, ttl=3600)

    await svc.record_generation_start(TENANT, "job-1")
    status = await svc.get_quota_status(TENANT)

    assert status["concurrent_jobs"] == 1
    assert svc._concurrent_key(TENANT) != legacy_key
