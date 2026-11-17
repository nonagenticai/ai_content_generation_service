"""Integration test: batch submit -> poll -> result flow."""

from __future__ import annotations

from src.services.batch_tracker import BatchTracker


async def test_batch_full_lifecycle():
    """Create a batch, record completions, verify final state."""
    tracker = BatchTracker()
    batch = await tracker.create_batch("b1", ["j1", "j2", "j3"], budget_limit=5.0)

    assert batch.computed_status == "pending"

    # Simulate processing
    await tracker.record_clip_completed("b1", cost=0.05)
    batch = await tracker.get_batch("b1")
    assert batch is not None
    assert batch.computed_status == "processing"

    await tracker.record_clip_completed("b1", cost=0.03)
    batch = await tracker.get_batch("b1")
    assert batch is not None
    assert batch.computed_status == "processing"

    await tracker.record_clip_completed("b1", cost=0.04)
    batch = await tracker.get_batch("b1")
    assert batch is not None
    assert batch.computed_status == "completed"
    assert batch.total_cost == 0.12
    assert batch.clips_completed == 3


async def test_batch_with_failures():
    """Batch with mixed results -> partial status."""
    tracker = BatchTracker()
    await tracker.create_batch("b2", ["j1", "j2", "j3"], budget_limit=5.0)

    await tracker.record_clip_completed("b2", cost=0.05)
    await tracker.record_clip_failed("b2")
    await tracker.record_clip_skipped("b2")

    batch = await tracker.get_batch("b2")
    assert batch is not None
    assert batch.computed_status == "partial"
    assert batch.clips_completed == 1
    assert batch.clips_failed == 1
    assert batch.clips_skipped == 1
    assert batch.total_cost == 0.05


async def test_batch_all_failed():
    """Batch with all failures -> failed status."""
    tracker = BatchTracker()
    await tracker.create_batch("b3", ["j1", "j2"], budget_limit=5.0)

    await tracker.record_clip_failed("b3")
    await tracker.record_clip_failed("b3")

    batch = await tracker.get_batch("b3")
    assert batch is not None
    assert batch.computed_status == "failed"
    assert batch.clips_failed == 2
