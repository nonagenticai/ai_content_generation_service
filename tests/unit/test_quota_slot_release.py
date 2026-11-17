"""Regression tests: the tenant concurrency slot must be released on EVERY exit path.

``record_generation_start`` increments a per-tenant Redis concurrency gauge with a
1h TTL. The matching ``record_generation_complete`` used to sit inside the ``try``
block of each background processor, so any exception — or an ``asyncio.CancelledError``
— skipped it and leaked a slot for an hour. Enough leaks and ``check_quota`` throttles
the tenant to zero.

These tests deliberately do NOT use ``AsyncMock`` for the quota service: an
``AsyncMock`` assertion passes even when production calls a different method name,
and a mock has no state, so it cannot show that a leaked slot actually blocks the
NEXT request. ``_CountingQuotaService`` below keeps a real integer gauge and the
assertions are on the observable effect — "is a second job for this tenant
admitted?" — not on call bookkeeping.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.content_routes import (
    _process_concept_image_batch_job_bg,
    _process_concept_image_job_bg,
)
from src.api.routes.generation import _process_batch_job_bg, _process_job_bg
from src.services.quota_service import QuotaCheckResult, QuotaService

# w2v_common >= 0.67.0 parses every tenant id as a uuid.UUID and raises on a
# slug, so the old "tenant-slot" label died at set_identity_context before the
# slot-release assertion could run.
TENANT = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
# The slot is named by the job that holds it, so a test must seed the SAME id the
# background task will release — that pairing is the property under test.
VIDEO_JOB_ID = "vid-1"
IMAGE_JOB_ID = "img-job-1"


class _CountingQuotaService(QuotaService):
    """In-memory stand-in for QuotaService with a REAL concurrency gauge.

    Subclasses the real service (rather than duck-typing or AsyncMock) so the
    overridden method NAMES and signatures are checked against production: a
    rename in QuotaService breaks these tests instead of silently passing.
    ``max_concurrent=1`` so a single leaked slot is immediately observable —
    ``check_quota`` starts refusing.
    """

    def __init__(self, max_concurrent: int = 1) -> None:
        super().__init__(redis_client=None, max_concurrent_jobs=max_concurrent)
        self.starts = 0
        self.completes = 0
        self.costs: list[float] = []
        self._max_concurrent = max_concurrent
        self.complete_raises: BaseException | None = None
        self.held: set[str] = set()

    async def check_quota(
        self, tenant_id: str, estimated_cost: float = 0.0, *, tenant_funded: bool = False
    ) -> QuotaCheckResult:
        del tenant_id, estimated_cost, tenant_funded
        if self.concurrent >= self._max_concurrent:
            return QuotaCheckResult(
                allowed=False,
                reason=f"Concurrent job limit exceeded ({self._max_concurrent})",
                remaining_concurrent=0,
            )
        return QuotaCheckResult(
            allowed=True, remaining_concurrent=self._max_concurrent - self.concurrent
        )

    async def record_generation_start(self, tenant_id: str, job_id: str) -> None:
        del tenant_id
        self.starts += 1
        self.held.add(job_id)

    async def record_generation_complete(
        self, tenant_id: str, job_id: str, cost: float = 0.0
    ) -> None:
        del tenant_id
        if self.complete_raises is not None:
            raise self.complete_raises
        self.completes += 1
        self.costs.append(cost)
        # Mirrors the real ZREM: releasing names a job, so a release for a job
        # that holds nothing frees nothing. A counter-shaped fake would let a
        # stray release hand the tenant free concurrency and hide the defect
        # this suite exists to pin.
        self.held.discard(job_id)

    @property
    def concurrent(self) -> int:
        return len(self.held)

    async def admits_another(self) -> bool:
        """Observable effect: would a NEXT request for this tenant be admitted?"""
        return (await self.check_quota(TENANT)).allowed


@dataclass
class _FakeImageJob:
    job_id: str = "img-job-1"
    asset_id: str = "asset-1"
    status: str = "completed"
    provider: str | None = "openai_images"
    prompt: str | None = "a hero image"
    image_path: str | None = "/tmp/image.png"
    image_url: str | None = None
    minio_key: str | None = None
    mime_type: str | None = "image/png"
    width: int | None = 1024
    height: int | None = 768
    file_size_bytes: int | None = 2048
    cost: float | None = 0.04
    size: str | None = "1024x1024"
    quality: str | None = "standard"
    usage_context: dict | None = None
    target_usage: str | None = "redesign_concept"
    error_message: str | None = None
    provider_job_id: str | None = None
    callback_url: str | None = None
    created_at: datetime | None = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    completed_at: datetime | None = datetime(2024, 1, 15, 12, 0, 3, tzinfo=UTC)


@dataclass
class _FakeImageBatch:
    batch_id: str = "batch-1"
    total_cost: float = 0.0


def _db_session_patch(module: str):
    """Patch ``<module>.get_db_session`` so no real Postgres session is opened."""
    return patch(f"{module}.get_db_session")


def _priced_clip_row() -> MagicMock:
    """The persisted batch-clip row ``_admit_batch_clip`` prices its quota check from."""
    job = MagicMock()
    job.provider = "mock"
    job.duration_s = 5.0
    return job


def _bind_db(mock_ctx: MagicMock) -> AsyncMock:
    db = AsyncMock()
    db.commit = AsyncMock()
    mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
    mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
    return db


# ---------------------------------------------------------------------------
# Video single-job path: src.api.routes.generation._process_job_bg
# ---------------------------------------------------------------------------


class TestVideoJobSlotRelease:
    def _service(self, side_effect=None, cost: float = 0.42) -> MagicMock:
        service = MagicMock()
        service.process_job = AsyncMock(
            side_effect=side_effect, return_value=None if side_effect else cost
        )
        service.unregister_task = MagicMock()
        return service

    async def test_runtime_error_releases_the_slot(self) -> None:
        quota = _CountingQuotaService(max_concurrent=1)
        await quota.record_generation_start(TENANT, VIDEO_JOB_ID)  # what the route does
        assert not await quota.admits_another()

        service = self._service(side_effect=RuntimeError("provider exploded"))
        with (
            _db_session_patch("src.api.routes.generation") as mock_ctx,
            patch("src.api.routes.generation.JobManager.mark_failed", new_callable=AsyncMock),
        ):
            _bind_db(mock_ctx)
            await _process_job_bg(service, "vid-1", TENANT, quota_service=quota)

        assert quota.concurrent == 0
        assert quota.completes == 1
        # The observable effect: the tenant is not throttled by the failed job.
        assert await quota.admits_another()

    async def test_cancelled_error_releases_the_slot(self) -> None:
        quota = _CountingQuotaService(max_concurrent=1)
        await quota.record_generation_start(TENANT, VIDEO_JOB_ID)

        service = self._service(side_effect=asyncio.CancelledError())
        with _db_session_patch("src.api.routes.generation") as mock_ctx:
            _bind_db(mock_ctx)
            await _process_job_bg(service, "vid-1", TENANT, quota_service=quota)

        assert quota.concurrent == 0
        assert await quota.admits_another()

    async def test_happy_path_releases_exactly_once_with_cost(self) -> None:
        quota = _CountingQuotaService(max_concurrent=1)
        await quota.record_generation_start(TENANT, VIDEO_JOB_ID)

        service = self._service(cost=0.42)
        with _db_session_patch("src.api.routes.generation") as mock_ctx:
            _bind_db(mock_ctx)
            await _process_job_bg(service, "vid-1", TENANT, quota_service=quota)

        # Exactly ONE release: a double release would drive the gauge negative
        # and hand the tenant free concurrency.
        assert quota.completes == 1
        assert quota.concurrent == 0
        assert quota.costs == [0.42]

    async def test_release_failure_does_not_escape_the_finally(self) -> None:
        """A broken quota store must not turn a completed job into a raising task."""
        quota = _CountingQuotaService()
        await quota.record_generation_start(TENANT, VIDEO_JOB_ID)
        quota.complete_raises = RuntimeError("redis down")

        service = self._service(cost=0.42)
        with _db_session_patch("src.api.routes.generation") as mock_ctx:
            _bind_db(mock_ctx)
            await _process_job_bg(service, "vid-1", TENANT, quota_service=quota)

    async def test_cancellation_during_release_still_propagates(self) -> None:
        """The best-effort guard catches Exception, never BaseException."""
        quota = _CountingQuotaService()
        await quota.record_generation_start(TENANT, VIDEO_JOB_ID)
        quota.complete_raises = asyncio.CancelledError()

        service = self._service(cost=0.42)
        with (
            _db_session_patch("src.api.routes.generation") as mock_ctx,
            pytest.raises(asyncio.CancelledError),
        ):
            _bind_db(mock_ctx)
            await _process_job_bg(service, "vid-1", TENANT, quota_service=quota)


# ---------------------------------------------------------------------------
# Video batch-child path: src.api.routes.generation._process_batch_job_bg
# ---------------------------------------------------------------------------


class TestVideoBatchJobSlotRelease:
    """The video batch child records its OWN start, like the concept-image child.

    The route cannot do it: batch children are also spawned by the reaper on
    reclaim, which has no route. So the child takes the slot and the ``finally``
    release is gated on whether it did — see ``_admit_batch_clip``.
    """

    def _tracker(self) -> MagicMock:
        tracker = MagicMock()
        tracker.get_batch = AsyncMock(return_value=None)
        tracker.record_clip_completed = AsyncMock()
        tracker.record_clip_failed = AsyncMock()
        tracker.record_clip_skipped = AsyncMock()
        return tracker

    async def _run(self, service: MagicMock, quota: _CountingQuotaService) -> MagicMock:
        tracker = self._tracker()
        with (
            _db_session_patch("src.api.routes.generation") as mock_ctx,
            patch("src.api.routes.generation.JobManager.mark_failed", new_callable=AsyncMock),
            # ``_admit_batch_clip`` now reads the persisted row to PRICE the clip
            # it is admitting (the quota check used to pass estimated_cost=0.0,
            # so the spend cap could never trip on the job being let through).
            patch(
                "src.api.routes.generation.JobManager.get_job",
                new_callable=AsyncMock,
                return_value=_priced_clip_row(),
            ),
        ):
            _bind_db(mock_ctx)
            await _process_batch_job_bg(
                service, "vid-1", TENANT, "batch-1", tracker, quota_service=quota
            )
        return tracker

    async def test_runtime_error_releases_the_slot(self) -> None:
        quota = _CountingQuotaService(max_concurrent=1)

        service = MagicMock()
        service.process_job = AsyncMock(side_effect=RuntimeError("boom"))
        service.unregister_task = MagicMock()

        await self._run(service, quota)

        assert quota.starts == 1
        assert quota.completes == 1
        assert quota.concurrent == 0
        assert await quota.admits_another()

    async def test_cancelled_error_releases_the_slot(self) -> None:
        quota = _CountingQuotaService(max_concurrent=1)

        service = MagicMock()
        service.process_job = AsyncMock(side_effect=asyncio.CancelledError())
        service.unregister_task = MagicMock()

        await self._run(service, quota)

        assert quota.starts == 1
        assert quota.completes == 1
        assert quota.concurrent == 0
        assert await quota.admits_another()

    async def test_happy_path_releases_exactly_once(self) -> None:
        quota = _CountingQuotaService(max_concurrent=1)

        service = MagicMock()
        service.process_job = AsyncMock(return_value=0.25)
        service.unregister_task = MagicMock()

        await self._run(service, quota)

        assert quota.starts == 1
        assert quota.completes == 1
        assert quota.concurrent == 0
        assert quota.costs == [0.25]

    async def test_quota_rejected_child_does_not_release_a_slot_it_never_took(self) -> None:
        """Under-flow guard, mirroring the concept-image batch child."""
        quota = _CountingQuotaService(max_concurrent=1)
        quota.held.add("some-other-job")  # some OTHER job holds the only slot

        service = MagicMock()
        service.process_job = AsyncMock(return_value=0.25)
        service.unregister_task = MagicMock()

        tracker = await self._run(service, quota)

        assert quota.starts == 0
        assert quota.completes == 0
        assert quota.concurrent == 1  # the other job's slot is untouched
        service.process_job.assert_not_awaited()  # no paid generation happened
        tracker.record_clip_failed.assert_awaited_once_with("batch-1")


# ---------------------------------------------------------------------------
# Concept-image single-job path
# ---------------------------------------------------------------------------


class TestConceptImageJobSlotRelease:
    def _service(self, side_effect=None, cost: float = 0.04) -> MagicMock:
        service = MagicMock()
        service.process_job = AsyncMock(
            side_effect=side_effect, return_value=None if side_effect else cost
        )
        service._storage = MagicMock()
        service.unregister_task = MagicMock()
        return service

    async def _run(self, service: MagicMock, quota: _CountingQuotaService) -> None:
        with (
            _db_session_patch("src.api.content_routes") as mock_ctx,
            patch(
                "src.api.content_routes.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=_FakeImageJob(),
            ),
            patch("src.api.content_routes.ImageJobManager.mark_failed", new_callable=AsyncMock),
            patch(
                "src.api.content_routes._deliver_concept_image_callback",
                new_callable=AsyncMock,
            ),
        ):
            _bind_db(mock_ctx)
            await _process_concept_image_job_bg(
                service, "img-job-1", TENANT, budget_limit=5.0, quota_service=quota
            )

    async def test_runtime_error_releases_the_slot(self) -> None:
        quota = _CountingQuotaService(max_concurrent=1)
        await quota.record_generation_start(TENANT, IMAGE_JOB_ID)
        assert not await quota.admits_another()

        await self._run(self._service(side_effect=RuntimeError("boom")), quota)

        assert quota.concurrent == 0
        assert await quota.admits_another()

    async def test_cancelled_error_releases_the_slot(self) -> None:
        quota = _CountingQuotaService(max_concurrent=1)
        await quota.record_generation_start(TENANT, IMAGE_JOB_ID)

        await self._run(self._service(side_effect=asyncio.CancelledError()), quota)

        assert quota.concurrent == 0
        assert await quota.admits_another()

    async def test_happy_path_releases_exactly_once(self) -> None:
        quota = _CountingQuotaService(max_concurrent=1)
        await quota.record_generation_start(TENANT, IMAGE_JOB_ID)

        await self._run(self._service(cost=0.04), quota)

        assert quota.completes == 1
        assert quota.concurrent == 0
        assert quota.costs == [0.04]


# ---------------------------------------------------------------------------
# Concept-image batch-child path (start is recorded INSIDE the bg task)
# ---------------------------------------------------------------------------


class TestConceptImageBatchJobSlotRelease:
    def _service(self, side_effect=None, cost: float = 0.04) -> MagicMock:
        service = MagicMock()
        service.process_job = AsyncMock(
            side_effect=side_effect, return_value=None if side_effect else cost
        )
        service.estimate_cost = MagicMock(return_value=cost)
        service._storage = MagicMock()
        service.unregister_task = MagicMock()
        return service

    async def _run(
        self,
        service: MagicMock,
        quota: _CountingQuotaService,
        *,
        job: _FakeImageJob | None = None,
    ) -> None:
        with (
            _db_session_patch("src.api.content_routes") as mock_ctx,
            patch(
                "src.api.content_routes.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=_FakeImageJob() if job is None else job,
            ),
            patch("src.api.content_routes.ImageJobManager.mark_failed", new_callable=AsyncMock),
            patch(
                "src.api.content_routes.ImageBatchManager.get_batch",
                new_callable=AsyncMock,
                return_value=_FakeImageBatch(),
            ),
            patch(
                "src.api.content_routes.ImageBatchManager.record_job_completed",
                new_callable=AsyncMock,
            ),
            patch(
                "src.api.content_routes.ImageBatchManager.record_job_failed",
                new_callable=AsyncMock,
            ),
            patch(
                "src.api.content_routes._deliver_concept_image_callback",
                new_callable=AsyncMock,
            ),
        ):
            _bind_db(mock_ctx)
            await _process_concept_image_batch_job_bg(
                service,
                "img-job-1",
                TENANT,
                "batch-1",
                budget_limit=5.0,
                quota_service=quota,
            )

    async def test_runtime_error_releases_the_slot_exactly_once(self) -> None:
        # This child records its OWN start (inside _check_quota_for_batch_job).
        quota = _CountingQuotaService(max_concurrent=1)

        await self._run(self._service(side_effect=RuntimeError("boom")), quota)

        assert quota.starts == 1
        # Exactly one release: _handle_batch_job_exception must NOT release too.
        assert quota.completes == 1
        assert quota.concurrent == 0
        assert await quota.admits_another()

    async def test_cancelled_error_releases_the_slot(self) -> None:
        quota = _CountingQuotaService(max_concurrent=1)

        await self._run(self._service(side_effect=asyncio.CancelledError()), quota)

        assert quota.starts == 1
        assert quota.completes == 1
        assert quota.concurrent == 0
        assert await quota.admits_another()

    async def test_happy_path_releases_exactly_once(self) -> None:
        quota = _CountingQuotaService(max_concurrent=1)

        await self._run(self._service(cost=0.04), quota)

        assert quota.starts == 1
        assert quota.completes == 1
        assert quota.concurrent == 0

    async def test_quota_rejected_child_does_not_release_a_slot_it_never_took(self) -> None:
        """Under-flow guard: no start recorded => no release."""
        quota = _CountingQuotaService(max_concurrent=1)
        quota.held.add("some-other-job")  # some OTHER job holds the only slot

        await self._run(self._service(cost=0.04), quota)

        assert quota.starts == 0
        assert quota.completes == 0
        # The other job's slot is untouched.
        assert quota.concurrent == 1
