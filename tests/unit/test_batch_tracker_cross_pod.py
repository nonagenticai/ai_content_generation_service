"""BatchTracker cross-pod atomic counters + missing-state reload fallback."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, cast

from w2v_common.cache.redis_client import RedisClient
from w2v_common.testing import inject_fakeredis

from src.services.batch_tracker import BatchTracker


class _YieldingRedis:
    """fakeredis proxy where every async command SUSPENDS the event loop first.

    ⚠️ Load-bearing for the cross-pod race test, not decoration. ``fakeredis``
    resolves a command without ever yielding, so ``asyncio.gather`` over two
    "pods" that only await fakeredis runs them one-at-a-time to completion and
    NO interleaving is possible — a read-modify-write counter would look atomic
    and ``test_two_pods_concurrent_completions_sum_with_no_lost_update`` would
    pass against the very lost-update bug it exists to catch. Verified by
    mutation on 2026-09-04: replacing the ``HINCRBY``/``HINCRBYFLOAT`` pair in
    ``BatchTracker._incr_counters`` with ``HGETALL`` -> ``+1`` -> ``HSET`` left
    all 3 tests GREEN. A real Redis command is a network round-trip and always
    suspends; this models that.
    """

    def __init__(self, fake: object) -> None:
        self._fake = fake

    def __getattr__(self, name: str) -> object:
        attr = getattr(self._fake, name)
        if not callable(attr):
            return attr

        # ⚠️ redis-py's command methods are PLAIN functions returning an
        # awaitable, not coroutine functions — an ``inspect.iscoroutinefunction``
        # guard here matches NOTHING and silently restores the non-yielding
        # behaviour this class exists to remove (observed while writing it).
        async def _call(*args: object, **kwargs: object) -> object:
            await asyncio.sleep(0)
            result = attr(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        return _call


def _shared_redis() -> RedisClient:
    client = RedisClient(redis_url="redis://localhost:6379/0")
    fake = inject_fakeredis(client)
    cast(Any, client)._client = _YieldingRedis(fake)
    return client


async def test_two_pods_concurrent_completions_sum_with_no_lost_update():
    """Two trackers over one Redis record concurrently; every clip is counted."""
    redis = _shared_redis()
    job_ids = [f"j{i}" for i in range(20)]

    # Pod A creates the batch (persists the snapshot both pods share).
    pod_a = BatchTracker(redis_client=redis)
    await pod_a.create_batch("b1", job_ids, budget_limit=100.0)

    # Pod B never created the batch — it must still update the shared counters
    # (the previously-missing _load_from_redis fallback).
    pod_b = BatchTracker(redis_client=redis)

    # 10 completions on each pod, interleaved concurrently.
    async def complete(tracker: BatchTracker, n: int) -> None:
        for _ in range(n):
            await tracker.record_clip_completed("b1", cost=0.5)

    await asyncio.gather(complete(pod_a, 10), complete(pod_b, 10))

    # Authoritative Redis counters summed all 20 with no lost update.
    state = await pod_a.get_batch("b1")
    assert state is not None
    assert state.clips_completed == 20
    assert abs(state.total_cost - 10.0) < 1e-6
    assert state.status == "completed"

    # A third pod that only ever read the batch sees the same authoritative count.
    pod_c = BatchTracker(redis_client=redis)
    state_c = await pod_c.get_batch("b1")
    assert state_c is not None
    assert state_c.clips_completed == 20


async def test_mixed_counters_sum_across_pods():
    redis = _shared_redis()
    pod_a = BatchTracker(redis_client=redis)
    await pod_a.create_batch("b2", ["j1", "j2", "j3", "j4"], budget_limit=10.0)
    pod_b = BatchTracker(redis_client=redis)

    await pod_a.record_clip_completed("b2", cost=1.0)
    await pod_b.record_clip_failed("b2")
    await pod_b.record_clip_skipped("b2")
    await pod_a.record_clip_completed("b2", cost=2.0)

    state = await pod_b.get_batch("b2")
    assert state is not None
    assert state.clips_completed == 2
    assert state.clips_failed == 1
    assert state.clips_skipped == 1
    assert abs(state.total_cost - 3.0) < 1e-6
    # 4 of 4 accounted for → terminal (partial: some failed/skipped).
    assert state.status == "partial"


async def test_in_memory_fallback_without_redis():
    """No Redis → single-replica in-memory counters still work."""
    tracker = BatchTracker()
    await tracker.create_batch("b3", ["j1", "j2"])
    await tracker.record_clip_completed("b3", cost=0.25)
    await tracker.record_clip_failed("b3")
    state = await tracker.get_batch("b3")
    assert state is not None
    assert state.clips_completed == 1
    assert state.clips_failed == 1
    assert abs(state.total_cost - 0.25) < 1e-6
    assert state.status == "partial"
