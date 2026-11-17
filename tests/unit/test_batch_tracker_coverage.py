"""Tests for batch_tracker.py to cover Redis persistence and serialization.

Covers: to_dict, from_dict, _redis_key, _persist, _load_from_redis,
get_batch with Redis fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from src.services.batch_tracker import BatchState, BatchTracker

# ============================================================================
# BatchState serialization
# ============================================================================


class TestBatchStateSerialization:
    def test_to_dict(self) -> None:
        """to_dict returns a serializable dict."""
        state = BatchState(
            batch_id="b1",
            job_ids=["j1", "j2"],
            clips_submitted=2,
            total_cost=0.10,
            budget_limit=5.0,
        )
        d = state.to_dict()
        assert d["batch_id"] == "b1"
        assert d["job_ids"] == ["j1", "j2"]
        assert d["clips_submitted"] == 2
        assert d["total_cost"] == 0.10

    def test_from_dict(self) -> None:
        """from_dict roundtrips with to_dict."""
        state = BatchState(
            batch_id="b1",
            job_ids=["j1"],
            clips_submitted=1,
            budget_limit=10.0,
            total_cost=0.5,
            clips_completed=1,
        )
        d = state.to_dict()
        restored = BatchState.from_dict(d)
        assert restored.batch_id == "b1"
        assert restored.job_ids == ["j1"]
        assert restored.clips_completed == 1
        assert restored.total_cost == 0.5


# ============================================================================
# BatchTracker Redis methods
# ============================================================================


class TestBatchTrackerRedis:
    def test_redis_key(self) -> None:
        """_redis_key formats correctly."""
        tracker = BatchTracker()
        assert tracker._redis_key("batch-abc") == "content_generation:batch:batch-abc"

    async def test_persist_with_redis(self) -> None:
        """_persist calls redis.set."""
        redis = AsyncMock()
        redis.set = AsyncMock()

        tracker = BatchTracker(redis_client=redis)
        state = BatchState(batch_id="b1", job_ids=["j1"])
        await tracker._persist(state)

        redis.set.assert_awaited_once()
        call_args = redis.set.call_args
        assert call_args[0][0] == "content_generation:batch:b1"

    async def test_persist_without_redis(self) -> None:
        """_persist is a no-op without Redis."""
        tracker = BatchTracker(redis_client=None)
        state = BatchState(batch_id="b1", job_ids=["j1"])
        # Should not raise
        await tracker._persist(state)

    async def test_persist_redis_exception(self) -> None:
        """_persist handles Redis errors gracefully."""
        redis = AsyncMock()
        redis.set = AsyncMock(side_effect=OSError("redis down"))

        tracker = BatchTracker(redis_client=redis)
        state = BatchState(batch_id="b1", job_ids=["j1"])
        # Should not raise
        await tracker._persist(state)

    async def test_load_from_redis_success(self) -> None:
        """_load_from_redis restores state from Redis dict."""
        redis = AsyncMock()
        redis.get = AsyncMock(
            return_value={
                "batch_id": "b1",
                "job_ids": ["j1"],
                "status": "processing",
                "clips_submitted": 1,
                "total_cost": 0.0,
                "budget_limit": 5.0,
                "clips_completed": 0,
                "clips_failed": 0,
                "clips_skipped": 0,
                "created_at": 1700000000.0,
            }
        )

        tracker = BatchTracker(redis_client=redis)
        state = await tracker._load_from_redis("b1")

        assert state is not None
        assert state.batch_id == "b1"
        assert state.job_ids == ["j1"]

    async def test_load_from_redis_none(self) -> None:
        """_load_from_redis returns None when Redis has no data."""
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)

        tracker = BatchTracker(redis_client=redis)
        state = await tracker._load_from_redis("nonexistent")

        assert state is None

    async def test_load_from_redis_without_client(self) -> None:
        """_load_from_redis returns None when no Redis client."""
        tracker = BatchTracker(redis_client=None)
        state = await tracker._load_from_redis("b1")
        assert state is None

    async def test_load_from_redis_exception(self) -> None:
        """_load_from_redis handles Redis errors gracefully."""
        redis = AsyncMock()
        redis.get = AsyncMock(side_effect=OSError("redis down"))

        tracker = BatchTracker(redis_client=redis)
        state = await tracker._load_from_redis("b1")

        assert state is None

    async def test_load_from_redis_non_dict_data(self) -> None:
        """_load_from_redis ignores non-dict data."""
        redis = AsyncMock()
        redis.get = AsyncMock(return_value="not a dict")

        tracker = BatchTracker(redis_client=redis)
        state = await tracker._load_from_redis("b1")

        assert state is None

    async def test_get_batch_falls_back_to_redis(self) -> None:
        """get_batch loads from Redis when not in memory."""
        redis = AsyncMock()
        redis.get = AsyncMock(
            return_value={
                "batch_id": "redis-batch",
                "job_ids": ["j1"],
                "status": "completed",
                "clips_submitted": 1,
                "total_cost": 0.05,
                "budget_limit": 5.0,
                "clips_completed": 1,
                "clips_failed": 0,
                "clips_skipped": 0,
                "created_at": 1700000000.0,
            }
        )

        tracker = BatchTracker(redis_client=redis)
        state = await tracker.get_batch("redis-batch")

        assert state is not None
        assert state.batch_id == "redis-batch"

    async def test_create_batch_persists_to_redis(self) -> None:
        """create_batch persists state to Redis."""
        redis = AsyncMock()
        redis.set = AsyncMock()

        tracker = BatchTracker(redis_client=redis)
        state = await tracker.create_batch("b1", ["j1", "j2"], budget_limit=10.0)

        assert state.batch_id == "b1"
        redis.set.assert_awaited_once()

    async def test_record_clip_completed_persists(self) -> None:
        """record_clip_completed persists updated state to Redis."""
        redis = AsyncMock()
        redis.set = AsyncMock()

        tracker = BatchTracker(redis_client=redis)
        await tracker.create_batch("b1", ["j1"])
        redis.set.reset_mock()

        await tracker.record_clip_completed("b1", cost=0.05)

        redis.set.assert_awaited_once()

    async def test_record_clip_failed_persists(self) -> None:
        """record_clip_failed persists updated state to Redis."""
        redis = AsyncMock()
        redis.set = AsyncMock()

        tracker = BatchTracker(redis_client=redis)
        await tracker.create_batch("b1", ["j1"])
        redis.set.reset_mock()

        await tracker.record_clip_failed("b1")

        redis.set.assert_awaited_once()

    async def test_record_clip_skipped_persists(self) -> None:
        """record_clip_skipped persists updated state to Redis."""
        redis = AsyncMock()
        redis.set = AsyncMock()

        tracker = BatchTracker(redis_client=redis)
        await tracker.create_batch("b1", ["j1"])
        redis.set.reset_mock()

        await tracker.record_clip_skipped("b1")

        redis.set.assert_awaited_once()
