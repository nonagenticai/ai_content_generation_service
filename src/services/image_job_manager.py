"""Job lifecycle management for AI image generation jobs."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import CursorResult, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ImageGenJob
from src.services.idempotency import insert_job_once
from src.services.job_manager import ClaimedJob, JobStateConflictError

logger = logging.getLogger(__name__)


class ImageJobManager:
    """Static CRUD helpers for image generation jobs."""

    @staticmethod
    async def _invalidate_cache(job_id: str) -> None:
        try:
            from src.cache.redis_client import get_redis_client

            redis = get_redis_client()
            if redis:
                await redis.delete(f"content:image:{job_id}:status")
        except OSError:
            logger.debug("Failed to clean up Redis status key for job %s", job_id)

    @staticmethod
    async def create_job(
        db: AsyncSession,
        *,
        tenant_id: str,
        provider: str,
        prompt: str,
        batch_id: str | None = None,
        negative_prompt: str | None = None,
        style: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        reference_image_urls: list[str] | None = None,
        usage_context: dict | None = None,
        target_usage: str | None = None,
        extra_params: dict | None = None,
        callback_url: str | None = None,
        idempotency_key: str | None = None,
        idempotency_fingerprint: str | None = None,
    ) -> ImageGenJob:
        """Create an image job row.

        With ``idempotency_key`` the insert is ``ON CONFLICT DO NOTHING`` against
        the per-tenant key index and raises :class:`IdempotentReplayError`
        carrying the existing job when the key already names a live one.
        """
        values: dict[str, object] = {
            "job_id": str(uuid.uuid7()),
            "asset_id": str(uuid.uuid7()),
            "batch_id": batch_id,
            "tenant_id": tenant_id,
            "provider": provider,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "style": style,
            "size": size,
            "quality": quality,
            "reference_image_urls": reference_image_urls,
            "usage_context": usage_context,
            "target_usage": target_usage,
            "extra_params": extra_params,
            "callback_url": callback_url,
            "status": "pending",
        }
        if idempotency_key is not None:
            values["idempotency_key"] = idempotency_key
            values["idempotency_fingerprint"] = idempotency_fingerprint
            job = await insert_job_once(
                db, ImageGenJob, values, tenant_id=tenant_id, key=idempotency_key
            )
            return cast(ImageGenJob, job)
        job = ImageGenJob(**values)
        db.add(job)
        await db.flush()
        return job

    @staticmethod
    async def get_job(
        db: AsyncSession, job_id: str, tenant_id: str | None = None
    ) -> ImageGenJob | None:
        """Fetch one image job by ID, optionally scoped to a tenant.

        ``tenant_id`` is passed by every request-serving caller so a foreign
        job reads as absent; background workers legitimately span tenants and
        pass nothing. See ``JobManager.get_job``.
        """
        query = select(ImageGenJob).where(ImageGenJob.job_id == job_id)
        if tenant_id is not None:
            query = query.where(ImageGenJob.tenant_id == tenant_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_job_by_asset_id(
        db: AsyncSession, asset_id: str, tenant_id: str | None = None
    ) -> ImageGenJob | None:
        """Fetch one image job by ASSET id — the same lookup on a different key.

        Scoping rules are identical to :meth:`get_job`; the asset id is just as
        enumerable as the job id, so it needs the same predicate.
        """
        query = select(ImageGenJob).where(ImageGenJob.asset_id == asset_id)
        if tenant_id is not None:
            query = query.where(ImageGenJob.tenant_id == tenant_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_jobs_for_batch(db: AsyncSession, batch_id: str) -> list[ImageGenJob]:
        result = await db.execute(
            select(ImageGenJob)
            .where(ImageGenJob.batch_id == batch_id)
            .order_by(ImageGenJob.created_at.asc(), ImageGenJob.id.asc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def mark_processing(
        db: AsyncSession, job_id: str, provider_job_id: str | None = None
    ) -> None:
        """Atomically transition a pending image job to processing.

        Accepts an optional provider_job_id in the same UPDATE — unlike the
        video JobManager, image generation is synchronous per-request so we
        only transition from pending once per job.

        Raises :class:`JobStateConflictError` if the job is not currently pending.
        """
        values: dict[str, object] = {
            "status": "processing",
            "updated_at": datetime.now(UTC),
        }
        if provider_job_id is not None:
            values["provider_job_id"] = provider_job_id

        result = cast(
            CursorResult[Any],
            await db.execute(
                update(ImageGenJob)
                .where(ImageGenJob.job_id == job_id, ImageGenJob.status == "pending")
                .values(**values)
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Image job {job_id!r} is not pending — cannot mark_processing"
            )
        await ImageJobManager._invalidate_cache(job_id)

    @staticmethod
    async def mark_completed(
        db: AsyncSession,
        job_id: str,
        *,
        image_path: str | None = None,
        image_url: str | None = None,
        mime_type: str | None = None,
        width: int | None = None,
        height: int | None = None,
        file_size_bytes: int | None = None,
        cost: float | None = None,
        minio_key: str | None = None,
    ) -> None:
        """Atomically transition a processing image job to completed."""
        now = datetime.now(UTC)
        values: dict[str, object] = {
            "status": "completed",
            "image_path": image_path,
            "image_url": image_url,
            "mime_type": mime_type,
            "width": width,
            "height": height,
            "file_size_bytes": file_size_bytes,
            "cost": cost,
            "updated_at": now,
            "completed_at": now,
        }
        if minio_key is not None:
            values["minio_key"] = minio_key

        result = cast(
            CursorResult[Any],
            await db.execute(
                update(ImageGenJob)
                .where(
                    ImageGenJob.job_id == job_id,
                    ImageGenJob.status == "processing",
                )
                .values(**values)
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Image job {job_id!r} is not processing — cannot mark_completed"
            )
        await ImageJobManager._invalidate_cache(job_id)

    @staticmethod
    async def mark_failed(db: AsyncSession, job_id: str, error: str) -> None:
        """Atomically transition a non-terminal image job to failed."""
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(ImageGenJob)
                .where(
                    ImageGenJob.job_id == job_id,
                    ImageGenJob.status.in_(["pending", "processing"]),
                )
                .values(
                    status="failed",
                    error_message=error[:4000],
                    updated_at=now,
                    completed_at=now,
                )
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Image job {job_id!r} is already terminal — cannot mark_failed"
            )
        await ImageJobManager._invalidate_cache(job_id)

    @staticmethod
    async def mark_cancelled(db: AsyncSession, job_id: str) -> None:
        """Atomically transition a non-terminal image job to cancelled."""
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(ImageGenJob)
                .where(
                    ImageGenJob.job_id == job_id,
                    ImageGenJob.status.in_(["pending", "processing"]),
                )
                .values(status="cancelled", updated_at=now, completed_at=now)
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Image job {job_id!r} is already terminal — cannot mark_cancelled"
            )
        await ImageJobManager._invalidate_cache(job_id)

    # ------------------------------------------------------------------
    # Durable dispatch: claim-lease queue (multi-replica reclaim)
    # ------------------------------------------------------------------

    @staticmethod
    async def register_dispatch(
        db: AsyncSession,
        job_id: str,
        *,
        options: dict,
        worker_id: str,
        lease_seconds: int,
    ) -> None:
        """Mark a freshly-created ``pending`` image job as queue-owned + lease it.

        Also stamps ``started_at`` (the wall-clock lifetime anchor) — this IS the
        first claim. ``COALESCE`` keeps the first stamp so a re-dispatch never
        resets the clock. See ``JobManager.register_dispatch``.
        """
        await db.execute(
            update(ImageGenJob)
            .where(ImageGenJob.job_id == job_id)
            .values(
                job_options=options,
                claimed_by=worker_id,
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=lease_seconds),
                started_at=func.coalesce(ImageGenJob.started_at, func.now()),
            )
        )

    @staticmethod
    async def attach_provider_slot(
        db: AsyncSession,
        job_id: str,
        *,
        provider_slot: dict,
        provider_secret_sealed: bytes | None,
    ) -> None:
        """Persist a tenant provider slot on a freshly-created job (P5-2).

        ``provider_slot`` is the NON-secret metadata; the secret arrives already
        SEALED for this ``job_id`` (``provider_slots.seal_slot``). The caller
        commits with the rest of the enqueue. A DB trigger purges the sealed
        secret when the job turns terminal.
        """
        await db.execute(
            update(ImageGenJob)
            .where(ImageGenJob.job_id == job_id)
            .values(provider_slot=provider_slot, provider_secret_sealed=provider_secret_sealed)
        )

    @staticmethod
    async def renew_lease(db: AsyncSession, job_id: str, lease_seconds: int) -> None:
        """Extend a non-terminal image job's lease (heartbeat). No-op once terminal."""
        await db.execute(
            update(ImageGenJob)
            .where(
                ImageGenJob.job_id == job_id,
                ImageGenJob.status.in_(["pending", "processing"]),
            )
            .values(lease_expires_at=datetime.now(UTC) + timedelta(seconds=lease_seconds))
        )

    @staticmethod
    async def release_claims(db: AsyncSession, claimed_by: str) -> int:
        """Expire this pod's in-flight image-job leases at shutdown AND refund.

        The attempt this claim consumed is refunded, exactly as in
        ``JobManager.release_claims`` — see that docstring for the full rationale.
        A graceful handoff is not a crash, so it must not spend one of the job's
        lives; a real crash never reaches this path and still trips the counter.

        Idempotent: only rows whose lease this pod still HOLDS
        (``lease_expires_at > now()``) are touched, so a repeated release refunds
        nothing, and ``GREATEST(..., 0)`` keeps the counter non-negative.

        ⚠️ ``lease_expires_at`` is stamped with ``func.now()`` — the SERVER clock —
        for the reason given in ``JobManager.release_claims``: the reclaim predicate
        is ``lease_expires_at < now()`` evaluated by Postgres, so a client-clock
        value makes the just-released lease invisible to the sibling reaper whenever
        the app clock leads the DB's. Measured 39/40 on a dev box; invisible in CI,
        where the runner and its postgres container share one kernel clock.
        """
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(ImageGenJob)
                .where(
                    ImageGenJob.claimed_by == claimed_by,
                    ImageGenJob.status.in_(["pending", "processing"]),
                    ImageGenJob.lease_expires_at.is_not(None),
                    ImageGenJob.lease_expires_at > func.now(),
                )
                .values(
                    lease_expires_at=func.now(),
                    attempts=func.greatest(func.coalesce(ImageGenJob.attempts, 1) - 1, 0),
                )
            ),
        )
        return result.rowcount or 0

    @staticmethod
    async def reclaim_orphaned(
        db: AsyncSession,
        *,
        worker_id: str,
        lease_seconds: int,
        max_attempts: int,
        max_lifetime_seconds: int = 0,
    ) -> ClaimedJob | None:
        """Atomically reclaim one orphaned image job, or return None.

        Delegates to the SECURITY DEFINER ``reclaim_orphaned_image_job`` resolver.
        ``max_lifetime_seconds`` (0 = disabled) is the wall-clock backstop that
        makes the attempt refund safe — see ``JobManager.reclaim_orphaned``.
        """
        result = await db.execute(
            text(
                "SELECT job_id, tenant_id, job_options "
                "FROM reclaim_orphaned_image_job(:worker, :lease, :max_attempts, :max_lifetime)"
            ),
            {
                "worker": worker_id,
                "lease": lease_seconds,
                "max_attempts": max_attempts,
                "max_lifetime": max_lifetime_seconds,
            },
        )
        row = result.first()
        await db.commit()
        db.expunge_all()
        if row is None:
            return None
        claimed = ClaimedJob(
            job_id=row.job_id, tenant_id=row.tenant_id, options=dict(row.job_options or {})
        )
        logger.info("Reclaimed orphaned image job %s (worker=%s)", claimed.job_id, worker_id)
        return claimed

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    #: Statuses a retention delete must NEVER touch, enforced in the DELETE itself.
    NON_TERMINAL_STATUSES = ("pending", "processing")

    @staticmethod
    async def delete_terminal_jobs(db: AsyncSession, job_ids: list[str]) -> int:
        """Delete TERMINAL image jobs by job_id. Returns count deleted.

        ⛔ The ``status NOT IN (pending, processing)`` predicate is load-bearing and
        belongs HERE, in the statement, not in the caller's reasoning.

        Every retention caller enumerates its rows in one session and deletes them
        in another (the cleanup sweep reads through a SECURITY DEFINER resolver with
        no tenant bound, then deletes per-tenant). Under READ COMMITTED, with no
        ``FOR UPDATE`` anywhere, the two are a genuine check-then-act window: a row
        that was terminal at enumeration is merely *believed* terminal at delete.

        Today nothing can revive a terminal child inside that window — the only
        INSERT of a ``pending`` row creates its batch in the same request,
        ``mark_processing`` requires ``pending``, and ``reclaim_orphaned_image_job``
        only picks rows already ``pending``/``processing``. But that is an UNSTATED
        invariant spread over three call sites, and the cost of a later refactor
        breaking it is deleting live work and its storage. So the guarantee is
        enforced by the statement: a row that is running is not deletable, whatever
        the caller believed when it read it.

        Tenant-bound: the caller binds ``app.current_tenant`` for the row's own
        tenant before calling, so RLS scopes the DELETE. The cross-tenant SELECT
        that finds the rows is the SECURITY DEFINER resolver's job, never this one's.
        """
        if not job_ids:
            return 0
        from sqlalchemy import delete

        result = cast(
            CursorResult[Any],
            await db.execute(
                delete(ImageGenJob).where(
                    ImageGenJob.job_id.in_(job_ids),
                    ImageGenJob.status.not_in(ImageJobManager.NON_TERMINAL_STATUSES),
                )
            ),
        )
        return result.rowcount or 0
