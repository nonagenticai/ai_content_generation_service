"""Unit tests for BatchTracker."""

from __future__ import annotations

import asyncio

from src.services.batch_tracker import BatchState, BatchTracker


async def test_create_batch():
    tracker = BatchTracker()
    state = await tracker.create_batch("batch-1", ["j1", "j2", "j3"], budget_limit=10.0)
    assert state.batch_id == "batch-1"
    assert state.job_ids == ["j1", "j2", "j3"]
    assert state.clips_submitted == 3
    assert state.budget_limit == 10.0
    assert state.total_cost == 0.0
    assert state.status == "pending"
    assert state.clips_completed == 0
    assert state.clips_failed == 0
    assert state.clips_skipped == 0


async def test_get_batch():
    tracker = BatchTracker()
    original = await tracker.create_batch("b1", ["j1"])
    state = await tracker.get_batch("b1")
    assert state is original
    assert state.batch_id == "b1"
    assert state.job_ids == ["j1"]


async def test_get_batch_not_found():
    tracker = BatchTracker()
    assert await tracker.get_batch("nonexistent") is None


async def test_record_clip_completed():
    tracker = BatchTracker()
    await tracker.create_batch("b1", ["j1", "j2"])
    await tracker.record_clip_completed("b1", cost=0.05)
    state = await tracker.get_batch("b1")
    assert state is not None
    assert state.clips_completed == 1
    assert state.total_cost == 0.05
    assert state.status == "processing"


async def test_record_clip_completed_accumulates_cost():
    """Multiple completions accumulate cost correctly."""
    tracker = BatchTracker()
    await tracker.create_batch("b1", ["j1", "j2", "j3"])
    await tracker.record_clip_completed("b1", cost=0.05)
    await tracker.record_clip_completed("b1", cost=0.10)
    state = await tracker.get_batch("b1")
    assert state is not None
    assert state.clips_completed == 2
    assert abs(state.total_cost - 0.15) < 0.001


async def test_record_clip_failed():
    tracker = BatchTracker()
    await tracker.create_batch("b1", ["j1", "j2"])
    await tracker.record_clip_failed("b1")
    state = await tracker.get_batch("b1")
    assert state is not None
    assert state.clips_failed == 1
    assert state.status == "processing"


async def test_record_clip_skipped():
    tracker = BatchTracker()
    await tracker.create_batch("b1", ["j1", "j2"])
    await tracker.record_clip_skipped("b1")
    state = await tracker.get_batch("b1")
    assert state is not None
    assert state.clips_skipped == 1
    assert state.status == "processing"


async def test_computed_status_pending():
    state = BatchState(batch_id="b1", job_ids=["j1", "j2"])
    assert state.computed_status == "pending"
    assert state.clips_completed == 0
    assert state.clips_failed == 0
    assert state.clips_skipped == 0


async def test_computed_status_processing():
    state = BatchState(batch_id="b1", job_ids=["j1", "j2"])
    state.clips_completed = 1
    assert state.computed_status == "processing"


async def test_computed_status_completed():
    state = BatchState(batch_id="b1", job_ids=["j1", "j2"])
    state.clips_completed = 2
    assert state.computed_status == "completed"


async def test_computed_status_failed():
    state = BatchState(batch_id="b1", job_ids=["j1", "j2"])
    state.clips_failed = 2
    assert state.computed_status == "failed"


async def test_computed_status_partial():
    state = BatchState(batch_id="b1", job_ids=["j1", "j2"])
    state.clips_completed = 1
    state.clips_failed = 1
    assert state.computed_status == "partial"


async def test_computed_status_partial_with_skipped():
    """Mixed completed + skipped results in 'partial'."""
    state = BatchState(batch_id="b1", job_ids=["j1", "j2"])
    state.clips_completed = 1
    state.clips_skipped = 1
    assert state.computed_status == "partial"


async def test_status_transitions_through_lifecycle():
    """Track status through a full batch lifecycle: pending -> processing -> completed."""
    tracker = BatchTracker()
    state = await tracker.create_batch("b1", ["j1", "j2"])
    assert state.status == "pending"

    await tracker.record_clip_completed("b1", cost=0.05)
    state = await tracker.get_batch("b1")
    assert state is not None
    assert state.status == "processing"

    await tracker.record_clip_completed("b1", cost=0.03)
    state = await tracker.get_batch("b1")
    assert state is not None
    assert state.status == "completed"
    assert abs(state.total_cost - 0.08) < 0.001


async def test_cleanup_expired():
    tracker = BatchTracker(ttl=0.0)  # Immediate expiry
    await tracker.create_batch("b1", ["j1"])
    await asyncio.sleep(0.01)
    removed = await tracker.cleanup_expired()
    assert removed == 1
    assert await tracker.get_batch("b1") is None


async def test_cleanup_keeps_fresh():
    tracker = BatchTracker(ttl=3600.0)
    await tracker.create_batch("b1", ["j1"])
    removed = await tracker.cleanup_expired()
    assert removed == 0
    state = await tracker.get_batch("b1")
    assert state is not None
    assert state.batch_id == "b1"


async def test_record_on_nonexistent_batch_is_noop():
    """Recording on a non-existent batch does not raise or create state."""
    tracker = BatchTracker()
    await tracker.record_clip_completed("missing", cost=1.0)
    await tracker.record_clip_failed("missing")
    await tracker.record_clip_skipped("missing")
    assert await tracker.get_batch("missing") is None
