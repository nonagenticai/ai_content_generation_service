"""Regression: a locally-cancelled batch clip must be ACCOUNTED for.

``_process_batch_job_bg``'s ``except asyncio.CancelledError`` arm logged and
returned, bumping NO tracker counter. ``BatchState.computed_status`` sums
``clips_completed + clips_failed + clips_skipped``, so a cancelled child left
``done < total`` FOREVER:

    computed_status                 -> "processing", permanently
    GET /batches/{id}/result        -> 409 "Batch still processing"
    ...for 2h, until the Redis TTL  -> 404 "Batch not found"

The clip was never counted anywhere. Every OTHER exit path from that worker
accounts itself (``record_clip_completed`` / ``record_clip_failed`` /
``record_clip_skipped``, plus the catch-all ``except Exception``) -- this was the
one hole. The sibling concept-image lane self-heals through
``ImageBatchManager.reap_stale_batches``, which re-derives its counters from the
child rows; the video lane's Redis-backed tracker has no such reclamation, so the
accounting has to happen on the cancel path itself.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.generation import _process_batch_job_bg
from src.services.batch_tracker import BatchState, BatchTracker

TENANT = "0199aaaa-bbbb-7ccc-8ddd-eeeeffff0001"
BATCH = "batch-cancel"


class TestComputedStatusCountsCancelledAsTerminal:
    def test_a_cancelled_clip_no_longer_pins_the_batch_in_processing(self) -> None:
        state = BatchState(batch_id=BATCH, job_ids=["a", "b"], clips_completed=1)
        # Before the fix there was nowhere to put the cancelled clip, so this
        # batch was stuck at "processing" for the whole TTL.
        assert state.computed_status == "processing"
        state.clips_cancelled = 1
        assert state.computed_status == "partial"

    def test_an_all_cancelled_batch_reports_cancelled_not_failed(self) -> None:
        """A user-cancelled batch is a different claim from a failed one."""
        state = BatchState(batch_id=BATCH, job_ids=["a", "b"], clips_cancelled=2)
        assert state.computed_status == "cancelled"

    def test_the_result_endpoint_would_no_longer_409(self) -> None:
        """``get_batch_result`` 409s exactly on {"pending","processing"}."""
        state = BatchState(batch_id=BATCH, job_ids=["a"], clips_cancelled=1)
        assert state.computed_status not in ("pending", "processing")

    def test_an_old_snapshot_without_the_field_still_deserialises(self) -> None:
        """Forward rollout: a snapshot written by the previous release must load."""
        legacy = {
            "batch_id": BATCH,
            "job_ids": ["a"],
            "status": "processing",
            "tenant_id": TENANT,
            "clips_submitted": 1,
            "total_cost": 0.0,
            "budget_limit": 5.0,
            "clips_completed": 0,
            "clips_failed": 0,
            "clips_skipped": 0,
            "created_at": 0.0,
        }
        state = BatchState.from_dict(legacy)
        assert state.clips_cancelled == 0


class TestTrackerRecordsTheCancellation:
    @pytest.mark.asyncio
    async def test_record_clip_cancelled_drives_the_batch_terminal(self) -> None:
        """No Redis: the pod-local increment path."""
        tracker = BatchTracker()
        await tracker.create_batch(BATCH, ["job-1"])
        created = await tracker.get_batch(BATCH)
        assert created is not None
        assert created.computed_status == "pending"

        await tracker.record_clip_cancelled(BATCH)

        batch = await tracker.get_batch(BATCH)
        assert batch is not None
        assert batch.clips_cancelled == 1
        assert batch.computed_status == "cancelled"


class TestTheWorkerAccountsItsOwnCancellation:
    @staticmethod
    def _service() -> MagicMock:
        service = MagicMock()
        service.process_job = AsyncMock(side_effect=asyncio.CancelledError())
        service.unregister_task = MagicMock()
        return service

    @pytest.mark.asyncio
    async def test_cancelled_child_bumps_the_cancelled_counter(self) -> None:
        tracker = MagicMock()
        tracker.get_batch = AsyncMock(return_value=None)
        tracker.record_clip_completed = AsyncMock()
        tracker.record_clip_failed = AsyncMock()
        tracker.record_clip_skipped = AsyncMock()
        tracker.record_clip_cancelled = AsyncMock()

        with (
            patch("src.api.routes.generation.get_db_session") as session,
            patch("src.api.routes.generation._admit_batch_clip", AsyncMock(return_value=True)),
        ):
            session.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            session.return_value.__aexit__ = AsyncMock(return_value=False)

            await _process_batch_job_bg(
                self._service(), "job-1", TENANT, BATCH, tracker, tenant_source=None
            )

        tracker.record_clip_cancelled.assert_awaited_once_with(BATCH)
        # A cancellation is NOT a failure and NOT a completion -- it must not be
        # laundered into either bucket.
        tracker.record_clip_failed.assert_not_awaited()
        tracker.record_clip_completed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_end_to_end_a_cancelled_child_unsticks_a_real_batch(self) -> None:
        """The observable consequence, against the REAL tracker."""
        tracker = BatchTracker()
        await tracker.create_batch(BATCH, ["job-1"])

        with (
            patch("src.api.routes.generation.get_db_session") as session,
            patch("src.api.routes.generation._admit_batch_clip", AsyncMock(return_value=True)),
        ):
            session.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            session.return_value.__aexit__ = AsyncMock(return_value=False)

            await _process_batch_job_bg(
                self._service(), "job-1", TENANT, BATCH, tracker, tenant_source=None
            )

        batch = await tracker.get_batch(BATCH)
        assert batch is not None
        # Previously: clips_* all 0 -> "pending"/"processing" -> /result 409s
        # until the 2h TTL, then 404s.
        assert batch.clips_cancelled == 1
        assert batch.computed_status not in ("pending", "processing")
