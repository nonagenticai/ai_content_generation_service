"""Unit tests for ImageBatchManager and _compute_batch_status."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.database.models import ImageGenBatch
from src.services.image_batch_manager import ImageBatchManager, _compute_batch_status
from tests.conftest import TENANT_A


def _make_batch(
    *,
    batch_id: str = "batch-1",
    tenant_id: str = TENANT_A,
    status: str = "pending",
    budget_limit: float = 5.0,
    total_cost: float = 0.0,
    job_count: int = 3,
    completed_jobs: int = 0,
    failed_jobs: int = 0,
) -> ImageGenBatch:
    """Create a detached ImageGenBatch for testing (no session required)."""
    return ImageGenBatch(
        batch_id=batch_id,
        tenant_id=tenant_id,
        status=status,
        budget_limit=budget_limit,
        total_cost=total_cost,
        job_count=job_count,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
    )


class TestComputeBatchStatus:
    def test_pending_when_no_done(self) -> None:
        batch = _make_batch(completed_jobs=0, failed_jobs=0, job_count=3)
        assert _compute_batch_status(batch) == "pending"

    def test_processing_when_partial_done(self) -> None:
        batch = _make_batch(completed_jobs=1, failed_jobs=0, job_count=3)
        assert _compute_batch_status(batch) == "processing"

    def test_completed_when_all_done(self) -> None:
        batch = _make_batch(completed_jobs=3, failed_jobs=0, job_count=3)
        assert _compute_batch_status(batch) == "completed"

    def test_failed_when_all_failed(self) -> None:
        batch = _make_batch(completed_jobs=0, failed_jobs=3, job_count=3)
        assert _compute_batch_status(batch) == "failed"

    def test_partial_when_mixed(self) -> None:
        batch = _make_batch(completed_jobs=2, failed_jobs=1, job_count=3)
        assert _compute_batch_status(batch) == "partial"


class TestCreateBatch:
    async def test_create_batch(self) -> None:
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        batch = await ImageBatchManager.create_batch(
            db,
            tenant_id=TENANT_A,
            budget_limit=10.0,
            job_count=5,
        )

        db.add.assert_called_once()
        db.flush.assert_called_once()
        assert batch.tenant_id == TENANT_A
        assert batch.budget_limit == 10.0
        assert batch.job_count == 5
        assert batch.status == "pending"


class TestGetBatch:
    async def test_get_batch_found(self) -> None:
        fake_batch = _make_batch()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = fake_batch

        db = AsyncMock()
        db.execute.return_value = result_mock

        batch = await ImageBatchManager.get_batch(db, "batch-1")
        assert batch is not None
        assert batch.batch_id == "batch-1"

    async def test_get_batch_not_found(self) -> None:
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None

        db = AsyncMock()
        db.execute.return_value = result_mock

        batch = await ImageBatchManager.get_batch(db, "missing")
        assert batch is None


class TestGetBatchJobs:
    async def test_get_batch_jobs(self) -> None:
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [MagicMock(), MagicMock()]
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock

        db = AsyncMock()
        db.execute.return_value = result_mock

        jobs = await ImageBatchManager.get_batch_jobs(db, "batch-1")
        assert len(jobs) == 2


class TestMarkProcessing:
    async def test_mark_processing(self) -> None:
        fake_batch = _make_batch()
        with patch(
            "src.services.image_batch_manager.ImageBatchManager.get_batch",
            new_callable=AsyncMock,
            return_value=fake_batch,
        ):
            db = AsyncMock()
            await ImageBatchManager.mark_processing(db, "batch-1")

        assert fake_batch.status == "processing"

    async def test_mark_processing_missing_batch(self) -> None:
        with patch(
            "src.services.image_batch_manager.ImageBatchManager.get_batch",
            new_callable=AsyncMock,
            return_value=None,
        ):
            db = AsyncMock()
            # Should not raise
            await ImageBatchManager.mark_processing(db, "missing")


class TestRecordJobCompleted:
    """Atomic ``UPDATE ... RETURNING`` counter bumps (real Postgres — no mock).

    The counter increment happens in SQL (``completed_jobs = completed_jobs + 1``)
    so concurrent cross-pod completions cannot lose a count; it cannot be mocked.
    """

    async def test_record_job_completed_increments(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=2
        )
        await ImageBatchManager.record_job_completed(db_session, batch.batch_id, cost=0.05)

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.completed_jobs == 1
        assert fetched.total_cost == 0.05
        assert fetched.status == "processing"

    async def test_record_job_completed_sets_completed_at_when_done(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=2
        )
        await ImageBatchManager.record_job_completed(db_session, batch.batch_id, cost=0.05)
        await ImageBatchManager.record_job_completed(db_session, batch.batch_id, cost=0.04)

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.completed_jobs == 2
        assert fetched.status == "completed"
        assert fetched.completed_at is not None

    async def test_record_job_completed_missing_batch(self, db_session) -> None:
        # Should not raise (no matching row → no-op).
        await ImageBatchManager.record_job_completed(db_session, "missing")


class TestRecordJobFailed:
    async def test_record_job_failed_increments(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=2
        )
        await ImageBatchManager.record_job_failed(db_session, batch.batch_id)

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.failed_jobs == 1
        assert fetched.status == "processing"

    async def test_record_job_failed_sets_completed_at_when_all_failed(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=2
        )
        await ImageBatchManager.record_job_failed(db_session, batch.batch_id)
        await ImageBatchManager.record_job_failed(db_session, batch.batch_id)

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.failed_jobs == 2
        assert fetched.status == "failed"
        assert fetched.completed_at is not None

    async def test_record_job_failed_partial(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=2
        )
        await ImageBatchManager.record_job_completed(db_session, batch.batch_id, cost=0.05)
        await ImageBatchManager.record_job_failed(db_session, batch.batch_id)

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.status == "partial"
        assert fetched.completed_at is not None

    async def test_record_job_failed_missing_batch(self, db_session) -> None:
        # Should not raise (no matching row → no-op).
        await ImageBatchManager.record_job_failed(db_session, "missing")
