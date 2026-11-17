"""Regression: the VIDEO batch endpoint must take a quota slot before releasing one.

``POST /api/content/videos/generate/batch`` (``generation.generate_batch``) never
called ``check_quota`` and never called ``record_generation_start`` — yet every
child it spawns runs ``_process_batch_job_bg``, whose ``finally`` unconditionally
calls ``record_generation_complete``. One DECR per clip against zero INCRs.

That is exactly the invariant the sibling concept-image batch already documents in
``_handle_batch_job_exception``: "releasing here too would double-decrement the
concurrency gauge (and it fired even on paths where no start had been recorded,
handing the tenant free concurrency)". The image batch records its start inside
``_check_quota_for_batch_job`` and gates the release on ``quota_started``; the
video batch does neither.

Two observable consequences, one test each:

1. The per-tenant concurrency gauge is driven DOWN by the batch, cancelling slots
   held by that tenant's *other* in-flight generations. In production the Lua in
   ``_RECORD_COMPLETE_LUA`` clamps at 0, so the effect is not a negative number —
   it is that the gauge is reset to 0 and the concurrency cap stops binding.
2. Daily generation counts (``record_generation_start`` bumps ``generations`` in
   the daily key) never move for batch clips, so ``max_daily_generations`` can be
   bypassed entirely by routing work through ``/generate/batch``.

The existing ``tests/unit/test_quota_slot_release.py::TestVideoBatchJobSlotRelease``
does not catch this: it calls ``record_generation_start`` by hand before invoking
the child, asserting a balance the production route never establishes.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.generation import generate_batch
from src.api.schemas import GenerateBatchRequest, GenerateRequest
from tests.unit.test_quota_slot_release import TENANT, _CountingQuotaService

BATCH_SIZE = 3
OTHER_JOBS_IN_FLIGHT = 2


def _make_request(*, quota: _CountingQuotaService, tracker: MagicMock) -> MagicMock:
    """A stand-in Request exposing exactly what the route's ``_get_*`` helpers read."""
    request = MagicMock()
    request.state.tenant_id = TENANT

    service = MagicMock()
    service.create_job = AsyncMock(side_effect=[f"vid-{i}" for i in range(BATCH_SIZE)])
    service.process_job = AsyncMock(return_value=0.25)
    service.register_task = MagicMock()
    service.unregister_task = MagicMock()

    registry = MagicMock()
    registry.get = MagicMock(return_value=None)
    registry.provider_names = []

    request.app.state.video_generation_service = service
    request.app.state.batch_tracker = tracker
    request.app.state.quota_service = quota
    request.app.state.provider_registry = registry
    return request


def _priced_clip_row() -> MagicMock:
    """The persisted batch-clip row ``_admit_batch_clip`` prices its quota check from."""
    job = MagicMock()
    job.provider = "mock"
    job.duration_s = 5.0
    job.status = "completed"
    return job


def _make_tracker() -> MagicMock:
    tracker = MagicMock()
    tracker.create_batch = AsyncMock()
    tracker.get_batch = AsyncMock(return_value=None)
    tracker.record_clip_completed = AsyncMock()
    tracker.record_clip_failed = AsyncMock()
    tracker.record_clip_skipped = AsyncMock()
    return tracker


async def _passthrough_lease(*, job_id, lease_seconds, renew, work, heartbeat_enabled) -> None:
    """Run the child's work without the heartbeat task (no lease renewal in a unit test)."""
    del job_id, lease_seconds, renew, heartbeat_enabled
    await work


async def _run_batch(quota: _CountingQuotaService) -> MagicMock:
    """Drive the real route, then drain every child task it spawned."""
    tracker = _make_tracker()
    request = _make_request(quota=quota, tracker=tracker)
    body = GenerateBatchRequest(
        items=[GenerateRequest(prompt=f"clip {i}") for i in range(BATCH_SIZE)],
        budget_limit=5.0,
    )
    db = AsyncMock()
    db.commit = AsyncMock()

    with (
        patch("src.api.routes.generation.run_with_lease", _passthrough_lease),
        patch(
            "src.api.routes.generation.JobManager.register_dispatch",
            new_callable=AsyncMock,
        ),
        patch("src.api.routes.generation.get_db_session") as mock_ctx,
        # ``_admit_batch_clip`` reads the persisted row to price its quota check.
        patch(
            "src.api.routes.generation.JobManager.get_job",
            new_callable=AsyncMock,
            return_value=_priced_clip_row(),
        ),
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        response = await generate_batch(body, request, db)
        service = request.app.state.video_generation_service
        tasks = [call.args[1] for call in service.register_task.call_args_list]
        assert len(tasks) == BATCH_SIZE, "precondition: one child task per batch item"
        await asyncio.gather(*tasks)

    assert len(response.job_ids) == BATCH_SIZE
    return request


class TestVideoBatchQuotaAccounting:
    async def test_batch_does_not_steal_other_jobs_concurrency_slots(self) -> None:
        """A finished batch must leave the tenant's gauge exactly where it found it."""
        quota = _CountingQuotaService(max_concurrent=10)
        # PRECONDITION: two of this tenant's single generations are already in
        # flight, each holding a slot they took via record_generation_start().
        for i in range(OTHER_JOBS_IN_FLIGHT):
            await quota.record_generation_start(TENANT, f"other-{i}")
        assert quota.concurrent == OTHER_JOBS_IN_FLIGHT
        assert quota.starts == OTHER_JOBS_IN_FLIGHT

        await _run_batch(quota)

        # Every release the batch performed must be matched by a start it took.
        assert quota.concurrent == OTHER_JOBS_IN_FLIGHT, (
            f"batch of {BATCH_SIZE} released {quota.completes} slots against "
            f"{quota.starts - OTHER_JOBS_IN_FLIGHT} starts — the other "
            f"{OTHER_JOBS_IN_FLIGHT} in-flight jobs lost their slots"
        )

    async def test_batch_clips_count_toward_the_daily_generation_quota(self) -> None:
        """Each batch clip is a generation and must bump the daily counter."""
        quota = _CountingQuotaService(max_concurrent=10)
        assert quota.starts == 0  # precondition

        await _run_batch(quota)

        assert quota.starts == BATCH_SIZE, (
            f"{BATCH_SIZE} clips generated but only {quota.starts} counted against "
            "the tenant's daily generation quota"
        )


class TestVideoBatchSizeBound:
    async def test_batch_item_count_is_bounded(self) -> None:
        """An unbounded ``items`` list is one request that creates N rows + N tasks."""
        with pytest.raises(ValueError):
            GenerateBatchRequest(items=[GenerateRequest(prompt="x") for _ in range(11)])
