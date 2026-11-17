"""Persistence helpers for concept-image batch jobs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ImageGenBatch, ImageGenJob


@dataclass(frozen=True, slots=True)
class ReconciledBatch:
    """One batch the reaper closed out, with the counters it derived from children."""

    batch_id: str
    status: str
    completed_jobs: int
    failed_jobs: int


def _status_from_counts(*, completed: int, failed: int, total: int) -> str:
    done = completed + failed
    if done <= 0:
        return "pending"
    if done < total:
        return "processing"
    if completed == total:
        return "completed"
    if failed == total:
        return "failed"
    return "partial"


def _compute_batch_status(batch: ImageGenBatch) -> str:
    return _status_from_counts(
        completed=batch.completed_jobs, failed=batch.failed_jobs, total=batch.job_count
    )


class ImageBatchManager:
    """CRUD helpers for persisted concept-image batches."""

    @staticmethod
    async def create_batch(
        db: AsyncSession,
        *,
        tenant_id: str,
        budget_limit: float,
        job_count: int,
    ) -> ImageGenBatch:
        batch = ImageGenBatch(
            batch_id=str(uuid.uuid7()),
            tenant_id=tenant_id,
            budget_limit=budget_limit,
            job_count=job_count,
            status="pending",
        )
        db.add(batch)
        await db.flush()
        return batch

    @staticmethod
    async def get_batch(
        db: AsyncSession, batch_id: str, tenant_id: str | None = None
    ) -> ImageGenBatch | None:
        """Fetch one batch by ID, optionally scoped to a tenant.

        ``tenant_id`` is passed by every request-serving caller so a foreign
        batch reads as absent; background workers span tenants and pass
        nothing. See ``JobManager.get_job``.
        """
        query = select(ImageGenBatch).where(ImageGenBatch.batch_id == batch_id)
        if tenant_id is not None:
            query = query.where(ImageGenBatch.tenant_id == tenant_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_batch_jobs(
        db: AsyncSession, batch_id: str, tenant_id: str | None = None
    ) -> list[ImageGenJob]:
        """List a batch's child jobs, optionally scoped to a tenant.

        Scoped even though the parent batch is already checked: a child row
        carrying another tenant's ``tenant_id`` must not ride along in the
        results just because it shares a ``batch_id``.
        """
        query = select(ImageGenJob).where(ImageGenJob.batch_id == batch_id)
        if tenant_id is not None:
            query = query.where(ImageGenJob.tenant_id == tenant_id)
        result = await db.execute(
            query.order_by(ImageGenJob.created_at.asc(), ImageGenJob.id.asc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def mark_processing(db: AsyncSession, batch_id: str) -> None:
        batch = await ImageBatchManager.get_batch(db, batch_id)
        if batch is None:
            return
        batch.status = "processing"
        batch.updated_at = datetime.now(UTC)

    # Counter updates are a single atomic ``UPDATE ... RETURNING``. The old ORM
    # read-modify-write (``batch.completed_jobs += 1``) was only safe while a
    # batch's children ran sequentially in one task. Now that children fan out
    # durably (one leased job each, reclaimable cross-pod), two children can
    # finish concurrently on different pods — a Python-side increment would lose
    # a count (last-writer-wins). Doing ``completed_jobs = completed_jobs + 1``
    # in SQL takes the row lock, so concurrent bumps serialise with no lost
    # update, and the terminal status/``completed_at`` are derived from the
    # post-increment values in the SAME statement (no second-write race).

    # The atomic increment + status derivation. ``:delta_completed``/``:delta_failed``
    # is exactly one of the two counters; the other is 0. Status is computed from the
    # freshly-incremented counters so it can never disagree with them.
    _RECORD_SQL = text(
        """
        UPDATE imagegen_batches
        SET completed_jobs = completed_jobs + :delta_completed,
            failed_jobs = failed_jobs + :delta_failed,
            total_cost = total_cost + :cost,
            updated_at = now(),
            status = CASE
                WHEN (completed_jobs + :delta_completed)
                     + (failed_jobs + :delta_failed) < job_count THEN 'processing'
                WHEN (completed_jobs + :delta_completed) = job_count THEN 'completed'
                WHEN (failed_jobs + :delta_failed) = job_count THEN 'failed'
                ELSE 'partial'
            END,
            completed_at = CASE
                WHEN (completed_jobs + :delta_completed)
                     + (failed_jobs + :delta_failed) >= job_count THEN now()
                ELSE completed_at
            END
        WHERE batch_id = :batch_id
        RETURNING completed_jobs, failed_jobs, job_count, total_cost, status
        """
    )

    @staticmethod
    async def _record(
        db: AsyncSession,
        batch_id: str,
        *,
        delta_completed: int,
        delta_failed: int,
        cost: float,
    ) -> None:
        result = await db.execute(
            ImageBatchManager._RECORD_SQL,
            {
                "batch_id": batch_id,
                "delta_completed": delta_completed,
                "delta_failed": delta_failed,
                "cost": cost,
            },
        )
        row = result.first()
        if row is None:
            # Unknown batch — no-op (parity with the old missing-batch guard).
            return
        # The raw UPDATE bypasses the ORM identity map, so any cached ImageGenBatch
        # instance still holds stale counters. Expunge so a subsequent ORM read
        # (get_batch) loads a fresh, fully-populated instance with the counts just
        # written (rather than returning the stale identity-mapped copy, or an
        # expired one that would trigger a sync lazy-load under async).
        db.expunge_all()

    @staticmethod
    async def record_job_completed(db: AsyncSession, batch_id: str, *, cost: float = 0.0) -> None:
        await ImageBatchManager._record(db, batch_id, delta_completed=1, delta_failed=0, cost=cost)

    @staticmethod
    async def record_job_failed(db: AsyncSession, batch_id: str) -> None:
        await ImageBatchManager._record(db, batch_id, delta_completed=0, delta_failed=1, cost=0.0)

    # ------------------------------------------------------------------
    # Batch reaper
    # ------------------------------------------------------------------

    _RECONCILE_SQL = text(
        "SELECT batch_id, status, completed_jobs, failed_jobs "
        "FROM reconcile_stale_image_batches(:stale_seconds, :limit)"
    )

    @staticmethod
    async def reap_stale_batches(
        db: AsyncSession, *, stale_seconds: int, limit: int = 50
    ) -> list[ReconciledBatch]:
        """Close out batches whose children are ALL terminal but whose row is not.

        Delegates to the SECURITY DEFINER ``reconcile_stale_image_batches``
        resolver: the reaper spans tenants with no ``app.current_tenant`` bound,
        so under RLS enforce a plain UPDATE here would fail closed to 0 rows —
        the batch would stay immortal and the sweep would report success.

        Counters are RE-DERIVED from the child rows, never incremented, because
        this path exists precisely for the cases where an increment was lost.
        """
        result = await db.execute(
            ImageBatchManager._RECONCILE_SQL,
            {"stale_seconds": stale_seconds, "limit": limit},
        )
        rows = [
            ReconciledBatch(
                batch_id=row.batch_id,
                status=row.status,
                completed_jobs=row.completed_jobs,
                failed_jobs=row.failed_jobs,
            )
            for row in result.all()
        ]
        await db.commit()
        # The raw UPDATE bypassed the identity map — see ``_record``.
        db.expunge_all()
        return rows

    @staticmethod
    async def delete_batches(db: AsyncSession, batch_ids: list[str]) -> int:
        """Delete batch rows by batch_id. Returns count deleted.

        Callers must have already removed (or confirmed the absence of) the
        batch's children: ``imagegen_jobs.batch_id`` is a real FK, so a parent
        deleted while a child survives raises IntegrityError. The
        ``expired_imagegen_batches`` resolver enforces that with a ``NOT EXISTS``
        so the guarantee does not rest on sweep ordering.
        """
        if not batch_ids:
            return 0
        from sqlalchemy import delete

        result = cast(
            CursorResult[Any],
            await db.execute(delete(ImageGenBatch).where(ImageGenBatch.batch_id.in_(batch_ids))),
        )
        return result.rowcount or 0
