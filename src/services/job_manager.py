"""Job lifecycle management for AI video generation jobs.

Historically this module exposed ``JobManager`` as a namespace of
``@staticmethod`` callables. That shape made it impossible to inject
collaborators (DB session factory, Redis client, metrics) and forced
hundreds of test sites to monkeypatch internals.

This module now exposes ``JobManager`` as a real instance class with
constructor-injected dependencies, while preserving the legacy
static-method interface for backward compatibility:

* Production code wires a ``JobManager(session_factory=..., redis_client=...)``
  instance during ``asgi.lifespan`` and pulls it via
  ``Depends(get_job_manager)``.
* Legacy callers (``JobManager.create_job(db, ...)``) continue to work
  identically because every public method is still exposed as a
  ``@staticmethod`` that operates on the explicitly-passed ``db``
  argument. Existing test patches against
  ``src.services.job_manager.JobManager.<method>`` and against the
  re-exported symbols in route modules continue to work unchanged.

The structural shift is real: instance methods now exist as a thin
delegation layer that can use ``self.redis_client`` / ``self.session_factory``
where appropriate (currently ``_invalidate_cache``). Future work can
migrate hot paths to the instance interface without breaking the legacy
shim.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import CursorResult, Row, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.database.models import VideoGenerationJob, VideoGenerationUsage
from src.services.idempotency import insert_job_once

if TYPE_CHECKING:
    from src.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)

_SORT_ALLOWLIST = {"created_at", "completed_at", "status", "provider", "cost"}


class JobStateConflictError(RuntimeError):
    """Raised when an atomic job-state transition loses a race.

    A caller should treat this as "another worker already transitioned
    this job" — log and bail cleanly, do not retry.
    """


@dataclass(frozen=True)
class ClaimedJob:
    """A job reclaimed from the durable queue by the reaper."""

    job_id: str
    tenant_id: str
    options: dict


class JobManager:
    """Video generation job CRUD and usage tracking.

    Two equivalent calling conventions are supported:

    1. **Instance** (preferred for new code, supports DI):
       ``jm = JobManager(session_factory=..., redis_client=...)``
       ``await jm.create_job(db, tenant_id=..., ...)``

    2. **Legacy static** (preserved for back-compat with existing
       call sites and the large number of unit tests that patch
       ``JobManager.<method>``):
       ``await JobManager.create_job(db, tenant_id=..., ...)``

    Both forms route to the same SQL. The only behavioral difference
    is that ``_invalidate_cache``, when called via an instance whose
    ``redis_client`` was constructor-injected, uses the injected client
    directly instead of reaching back through ``get_redis_client()``.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        redis_client: RedisClient | None = None,
    ) -> None:
        """Build a JobManager with optionally-injected collaborators.

        Args:
            session_factory: Async sessionmaker for callers that want
                ``JobManager`` to own its own session lifetime. Optional;
                callers that pass ``db`` explicitly to each method do
                not need this.
            redis_client: Redis client used for cache invalidation. If
                ``None``, ``_invalidate_cache`` falls back to the
                module-level ``get_redis_client()`` lookup so legacy
                callers behave identically.
        """
        self._session_factory = session_factory
        self._redis_client = redis_client

    # ------------------------------------------------------------------
    # Public properties (for testing and observability)
    # ------------------------------------------------------------------

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession] | None:
        """Return the injected session factory, if any."""
        return self._session_factory

    @property
    def redis_client(self) -> RedisClient | None:
        """Return the injected Redis client, if any."""
        return self._redis_client

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    async def invalidate_cache(self, job_id: str) -> None:
        """Instance-bound cache invalidation using the injected client.

        Falls back to the legacy ``get_redis_client()`` lookup if no
        client was injected, so behavior is identical to the static
        path when ``redis_client=None``.
        """
        from src.observability.metrics import record_cache_invalidate_failure

        try:
            redis = self._redis_client
            if redis is None:
                from src.cache.redis_client import get_redis_client

                redis = get_redis_client()
            if redis:
                await redis.delete(f"content_generation:job:{job_id}:status")
        except OSError as exc:
            logger.warning(
                "Cache invalidation failed for job %s (network): %s",
                job_id,
                exc,
            )
            record_cache_invalidate_failure(reason="network")
        except RuntimeError as exc:
            logger.warning(
                "Cache invalidation failed for job %s (runtime): %s",
                job_id,
                exc,
            )
            record_cache_invalidate_failure(reason="runtime")

    @staticmethod
    async def _invalidate_cache(job_id: str) -> None:
        """Legacy static cache invalidation.

        Best-effort: cache invalidation failures must not break the
        atomic state transition that just succeeded. Failures are logged
        and counted via the
        ``ai_content_gen_cache_invalidate_failures_total`` Prometheus
        counter so silent degradation is observable.
        """
        from src.observability.metrics import record_cache_invalidate_failure

        try:
            from src.cache.redis_client import get_redis_client

            redis = get_redis_client()
            if redis:
                await redis.delete(f"content_generation:job:{job_id}:status")
        except OSError as exc:
            logger.warning(
                "Cache invalidation failed for job %s (network): %s",
                job_id,
                exc,
            )
            record_cache_invalidate_failure(reason="network")
        except RuntimeError as exc:
            logger.warning(
                "Cache invalidation failed for job %s (runtime): %s",
                job_id,
                exc,
            )
            record_cache_invalidate_failure(reason="runtime")

    # ------------------------------------------------------------------
    # Job CRUD (static, behavior-preserving)
    # ------------------------------------------------------------------

    @staticmethod
    async def create_job(
        db: AsyncSession,
        *,
        tenant_id: str,
        provider: str,
        prompt: str,
        negative_prompt: str | None = None,
        image_url: str | None = None,
        duration_s: float | None = None,
        width: int | None = None,
        height: int | None = None,
        style: str | None = None,
        seed: int | None = None,
        extra_params: dict | None = None,
        fallback_providers: list[str] | None = None,
        callback_url: str | None = None,
        idempotency_key: str | None = None,
        idempotency_fingerprint: str | None = None,
    ) -> VideoGenerationJob:
        """Create a new video generation job record.

        With ``idempotency_key`` the insert is ``ON CONFLICT DO NOTHING`` against
        the per-tenant key index and raises ``IdempotentReplayError`` carrying the
        existing job when the key already names a live one.

        Returns:
            The newly created VideoGenerationJob ORM instance.
        """
        logger.debug("Creating job: tenant=%s provider=%s", tenant_id, provider)
        values: dict[str, object] = {
            "job_id": str(uuid.uuid7()),
            "tenant_id": tenant_id,
            "provider": provider,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_url": image_url,
            "duration_s": duration_s,
            "width": width,
            "height": height,
            "style": style,
            "seed": seed,
            "extra_params": extra_params,
            "fallback_providers": fallback_providers,
            "callback_url": callback_url,
            "status": "pending",
        }
        if idempotency_key is not None:
            values["idempotency_key"] = idempotency_key
            values["idempotency_fingerprint"] = idempotency_fingerprint
            created = await insert_job_once(
                db, VideoGenerationJob, values, tenant_id=tenant_id, key=idempotency_key
            )
            return cast(VideoGenerationJob, created)
        job = VideoGenerationJob(**values)
        db.add(job)
        await db.flush()
        return job

    @staticmethod
    async def get_job(
        db: AsyncSession, job_id: str, tenant_id: str | None = None
    ) -> VideoGenerationJob | None:
        """Fetch a single job by ID, optionally scoped to a tenant.

        Args:
            db: Async database session.
            job_id: Job identifier.
            tenant_id: When given, the row must ALSO belong to this tenant —
                a foreign job reads as absent. Every request-serving caller
                passes it; background workers (the reaper, callback delivery,
                the generation tasks) legitimately operate across tenants and
                pass nothing.

        Returns:
            VideoGenerationJob if found (and owned, when scoped), None otherwise.
        """
        query = select(VideoGenerationJob).where(VideoGenerationJob.job_id == job_id)
        if tenant_id is not None:
            query = query.where(VideoGenerationJob.tenant_id == tenant_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def mark_processing(db: AsyncSession, job_id: str) -> None:
        """Atomically transition a pending job to processing.

        Raises :class:`JobStateConflictError` if the job is not currently pending.
        """
        logger.debug("Marking job processing: job_id=%s", job_id)
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(VideoGenerationJob)
                .where(
                    VideoGenerationJob.job_id == job_id,
                    VideoGenerationJob.status == "pending",
                )
                .values(status="processing", updated_at=datetime.now(UTC))
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Video generation job {job_id!r} is not pending — cannot mark_processing"
            )
        await JobManager._invalidate_cache(job_id)

    @staticmethod
    async def set_provider_job_id(db: AsyncSession, job_id: str, provider_job_id: str) -> None:
        """Attach the external provider's task ID to a processing job.

        Atomic UPDATE scoped to ``status == "processing"`` so a late-arriving
        callback cannot overwrite a terminal state.

        Raises :class:`JobStateConflictError` if the job is not currently processing.
        """
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(VideoGenerationJob)
                .where(
                    VideoGenerationJob.job_id == job_id,
                    VideoGenerationJob.status == "processing",
                )
                .values(provider_job_id=provider_job_id, updated_at=datetime.now(UTC))
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Video generation job {job_id!r} is not processing — cannot set provider_job_id"
            )

    @staticmethod
    async def mark_completed(
        db: AsyncSession,
        job_id: str,
        *,
        video_path: str | None = None,
        video_url: str | None = None,
        file_size_bytes: int | None = None,
        actual_duration_s: float | None = None,
        generation_time_seconds: float | None = None,
        cost: float | None = None,
        minio_key: str | None = None,
        degradation: str | None = None,
    ) -> None:
        """Atomically transition a processing job to completed.

        Args:
            degradation: Marker for a completion that is not what the pipeline
                promised -- e.g. a clip delivered WITHOUT normalization. It is
                written unconditionally (None included) so a completed row's
                ``degradation`` is always the truth about THIS completion and
                never a leftover from an earlier attempt.

        Raises :class:`JobStateConflictError` if the job is not currently processing.
        """
        logger.debug("Marking job completed: job_id=%s cost=%s", job_id, cost)
        now = datetime.now(UTC)
        values: dict[str, object] = {
            "status": "completed",
            "video_path": video_path,
            "video_url": video_url,
            "file_size_bytes": file_size_bytes,
            "actual_duration_s": actual_duration_s,
            "generation_time_seconds": generation_time_seconds,
            "cost": cost,
            "updated_at": now,
            "completed_at": now,
            "degradation": degradation,
        }
        if minio_key is not None:
            values["minio_key"] = minio_key

        result = cast(
            CursorResult[Any],
            await db.execute(
                update(VideoGenerationJob)
                .where(
                    VideoGenerationJob.job_id == job_id,
                    VideoGenerationJob.status == "processing",
                )
                .values(**values)
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Video generation job {job_id!r} is not processing — cannot mark_completed"
            )
        await JobManager._invalidate_cache(job_id)

    @staticmethod
    async def mark_failed(db: AsyncSession, job_id: str, error: str) -> None:
        """Atomically transition a non-terminal job to failed.

        Raises :class:`JobStateConflictError` if the job is already terminal.
        """
        logger.debug("Marking job failed: job_id=%s error=%s", job_id, error[:200])
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(VideoGenerationJob)
                .where(
                    VideoGenerationJob.job_id == job_id,
                    VideoGenerationJob.status.in_(["pending", "processing"]),
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
                f"Video generation job {job_id!r} is already terminal — cannot mark_failed"
            )
        await JobManager._invalidate_cache(job_id)

    @staticmethod
    async def mark_cancelled(db: AsyncSession, job_id: str) -> None:
        """Atomically transition a non-terminal job to cancelled.

        Raises :class:`JobStateConflictError` if the job is already terminal.
        """
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(VideoGenerationJob)
                .where(
                    VideoGenerationJob.job_id == job_id,
                    VideoGenerationJob.status.in_(["pending", "processing"]),
                )
                .values(status="cancelled", updated_at=now, completed_at=now)
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Video generation job {job_id!r} is already terminal — cannot mark_cancelled"
            )
        await JobManager._invalidate_cache(job_id)

    @staticmethod
    async def mark_retrying(db: AsyncSession, job_id: str, retry_count: int, error: str) -> None:
        """Atomically reset a processing job to pending for retry.

        Only transitions when the job is currently ``processing`` — prevents
        a late-arriving retry from resurrecting a cancelled or completed job.

        Raises :class:`JobStateConflictError` if the job is not processing.
        """
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(VideoGenerationJob)
                .where(
                    VideoGenerationJob.job_id == job_id,
                    VideoGenerationJob.status == "processing",
                )
                .values(
                    status="pending",
                    retry_count=retry_count,
                    error_message=error[:4000],
                    updated_at=datetime.now(UTC),
                )
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Video generation job {job_id!r} is not processing — cannot mark_retrying"
            )
        await JobManager._invalidate_cache(job_id)

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
        """Mark a freshly-created ``pending`` job as queue-owned + acquire a lease.

        Persists the dispatch ``options`` bundle (so any pod can reconstruct the
        work) and stamps ``claimed_by`` + ``lease_expires_at``. If the owning pod
        dies before the job finishes, the lease lapses and the reaper reclaims it.

        This IS the job's first claim (the request path dispatches the work
        immediately), so it also stamps ``started_at`` — the anchor for the
        wall-clock lifetime bound. ``COALESCE`` keeps the first stamp: a re-dispatch
        must never reset the clock, or a churning job would be immortal.
        """
        await db.execute(
            update(VideoGenerationJob)
            .where(VideoGenerationJob.job_id == job_id)
            .values(
                job_options=options,
                claimed_by=worker_id,
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=lease_seconds),
                started_at=func.coalesce(VideoGenerationJob.started_at, func.now()),
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
            update(VideoGenerationJob)
            .where(VideoGenerationJob.job_id == job_id)
            .values(provider_slot=provider_slot, provider_secret_sealed=provider_secret_sealed)
        )

    @staticmethod
    async def renew_lease(db: AsyncSession, job_id: str, lease_seconds: int) -> None:
        """Extend a non-terminal job's lease (heartbeat). No-op once terminal."""
        await db.execute(
            update(VideoGenerationJob)
            .where(
                VideoGenerationJob.job_id == job_id,
                VideoGenerationJob.status.in_(["pending", "processing"]),
            )
            .values(lease_expires_at=datetime.now(UTC) + timedelta(seconds=lease_seconds))
        )

    @staticmethod
    async def release_claims(db: AsyncSession, claimed_by: str) -> int:
        """Expire this pod's in-flight leases at shutdown AND refund the attempt.

        preStop-drain-friendly: a graceful shutdown can't finish a long
        generation, so expiring the lease lets a sibling reclaim it on the next
        reaper sweep instead of waiting out the full lease TTL. Returns the count.

        **The attempt is refunded.** ``reclaim_orphaned_video_job`` charges
        ``attempts += 1`` on every reclaim and hard-fails the job once ``attempts
        >= max_attempts``. That counter exists to stop a job whose payload keeps
        *killing* its worker from cycling forever — it is a CRASH counter. A
        graceful handoff is not a crash: the claim is surrendered without a
        verdict, so the job never got its fair try and must not be charged for it.

        Without the refund, every rollout / eviction / node drain that interrupts
        an in-flight generation silently burns one of the job's lives, and four of
        them permanently fail a *paid provider video generation* — a silently dead
        customer video, on a job that never once misbehaved. That is the bug.

        A genuine crash (OOMKill / SIGKILL) never reaches this code path — the
        lease simply lapses unrefunded — so real poison detection is preserved.

        Idempotent and safe under contention:

        * ``lease_expires_at > now()`` — only rows whose lease this pod still
          actually HOLDS. A second release is a no-op (the lease is already
          expired), so the attempt can never be refunded twice.
        * ``claimed_by == claimed_by`` + non-terminal status — never touches a job
          another worker has since re-claimed, nor one that already completed.
        * ``GREATEST(..., 0)`` — the counter can never go negative.

        ⚠️ ``lease_expires_at`` is stamped with ``func.now()`` — the SERVER clock —
        and that is load-bearing, not style. The reclaim predicate this release is
        meant to unblock is ``lease_expires_at < now()`` (``rls_resolvers.py``,
        migrations 009/010), evaluated by Postgres. Writing ``datetime.now(UTC)``
        stamped the APP's clock into a value the DB then compares against its own,
        so the same statement mixed two clocks — note the ``> func.now()`` in the
        WHERE directly above. Whenever the app clock leads the DB's, the lease this
        release just "expired" is still in the database's future and the sibling
        reaper CANNOT see the row: the drain handoff silently degrades to waiting
        out the full lease instead of handing over immediately.

        Measured 2026-08-28 on a Docker-Desktop dev box: a client-stamped
        "expired" lease was invisible to the predicate in **39 of 40** trials.
        CI never caught it because a Linux runner's postgres container shares the
        host kernel clock, so the skew there is zero.
        """
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(VideoGenerationJob)
                .where(
                    VideoGenerationJob.claimed_by == claimed_by,
                    VideoGenerationJob.status.in_(["pending", "processing"]),
                    VideoGenerationJob.lease_expires_at.is_not(None),
                    VideoGenerationJob.lease_expires_at > func.now(),
                )
                .values(
                    lease_expires_at=func.now(),
                    attempts=func.greatest(func.coalesce(VideoGenerationJob.attempts, 1) - 1, 0),
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
        """Atomically reclaim one orphaned video job, or return None.

        Delegates to the SECURITY DEFINER ``reclaim_orphaned_video_job`` resolver
        (``FOR UPDATE SKIP LOCKED``, poison-cap, reset-to-pending + fresh lease).

        ``max_lifetime_seconds`` (0 = disabled) is the wall-clock backstop that
        makes the attempt refund safe: a job still unfinished that long after its
        FIRST claim is failed instead of re-claimed, so a genuinely crash-looping
        job cannot cycle — and re-spend on a paid provider — forever.
        """
        result = await db.execute(
            text(
                "SELECT job_id, tenant_id, job_options "
                "FROM reclaim_orphaned_video_job(:worker, :lease, :max_attempts, :max_lifetime)"
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
        logger.info("Reclaimed orphaned video job %s (worker=%s)", claimed.job_id, worker_id)
        return claimed

    @staticmethod
    async def list_jobs(
        db: AsyncSession,
        tenant_id: str,
        *,
        status: str | None = None,
        provider: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[VideoGenerationJob], int]:
        """List jobs with filtering and pagination. Returns (jobs, total_count)."""
        if sort_by not in _SORT_ALLOWLIST:
            sort_by = "created_at"

        query = select(VideoGenerationJob).where(VideoGenerationJob.tenant_id == tenant_id)
        count_query = (
            select(func.count())
            .select_from(VideoGenerationJob)
            .where(VideoGenerationJob.tenant_id == tenant_id)
        )

        if status:
            query = query.where(VideoGenerationJob.status == status)
            count_query = count_query.where(VideoGenerationJob.status == status)
        if provider:
            query = query.where(VideoGenerationJob.provider == provider)
            count_query = count_query.where(VideoGenerationJob.provider == provider)
        if created_after:
            query = query.where(VideoGenerationJob.created_at >= created_after)
            count_query = count_query.where(VideoGenerationJob.created_at >= created_after)
        if created_before:
            query = query.where(VideoGenerationJob.created_at <= created_before)
            count_query = count_query.where(VideoGenerationJob.created_at <= created_before)

        sort_col = getattr(VideoGenerationJob, sort_by, VideoGenerationJob.created_at)
        if sort_order == "asc":
            query = query.order_by(sort_col.asc())
        else:
            query = query.order_by(sort_col.desc())

        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        result = await db.execute(query)
        jobs = list(result.scalars().all())
        return jobs, total

    @staticmethod
    async def get_usage(
        db: AsyncSession,
        tenant_id: str,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> tuple[list[VideoGenerationUsage], list[Row[str, float | None, int]]]:
        """Get usage data. Returns (usage_rows, provider_breakdown).

        provider_breakdown is list of (provider, total_cost, count).
        """
        usage_query = select(VideoGenerationUsage).where(
            VideoGenerationUsage.tenant_id == tenant_id
        )
        if date_from:
            usage_query = usage_query.where(VideoGenerationUsage.date >= date_from)
        if date_to:
            usage_query = usage_query.where(VideoGenerationUsage.date <= date_to)
        usage_query = usage_query.order_by(VideoGenerationUsage.date.asc())

        usage_result = await db.execute(usage_query)
        usage_rows = list(usage_result.scalars().all())

        # Provider cost breakdown
        provider_query = (
            select(
                VideoGenerationJob.provider,
                func.coalesce(func.sum(VideoGenerationJob.cost), 0.0),
                func.count(),
            )
            .where(VideoGenerationJob.tenant_id == tenant_id)
            .where(VideoGenerationJob.status == "completed")
        )
        if date_from:
            provider_query = provider_query.where(VideoGenerationJob.created_at >= date_from)
        if date_to:
            provider_query = provider_query.where(VideoGenerationJob.created_at <= date_to)
        provider_query = provider_query.group_by(VideoGenerationJob.provider)

        provider_result = await db.execute(provider_query)
        provider_breakdown = list(provider_result.all())

        return usage_rows, provider_breakdown

    @staticmethod
    async def find_expired_jobs(
        db: AsyncSession, status: str, before: datetime, limit: int = 100
    ) -> list[VideoGenerationJob]:
        """Find expired jobs for cleanup."""
        result = await db.execute(
            select(VideoGenerationJob)
            .where(VideoGenerationJob.status == status)
            .where(VideoGenerationJob.completed_at <= before)
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def delete_jobs(db: AsyncSession, job_ids: list[str]) -> int:
        """Delete jobs by job_id. Returns count deleted."""
        if not job_ids:
            return 0
        from sqlalchemy import delete

        result = cast(
            CursorResult[Any],
            await db.execute(
                delete(VideoGenerationJob).where(VideoGenerationJob.job_id.in_(job_ids))
            ),
        )
        return result.rowcount or 0

    @staticmethod
    async def record_usage(
        db: AsyncSession,
        *,
        tenant_id: str,
        generations_completed: int = 0,
        is_error: bool = False,
        duration_generated: float = 0.0,
        file_size: int = 0,
    ) -> None:
        """Upsert daily usage counters for a tenant."""
        logger.debug(
            "Recording usage: tenant=%s completed=%d error=%s",
            tenant_id,
            generations_completed,
            is_error,
        )
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        stmt = (
            pg_insert(VideoGenerationUsage)
            .values(
                tenant_id=tenant_id,
                date=today,
                generations_started=1,
                generations_completed=generations_completed,
                generations_failed=1 if is_error else 0,
                total_duration_generated_s=duration_generated,
                total_file_size_bytes=file_size,
            )
            .on_conflict_do_update(
                constraint="uq_video_generation_usage_tenant_date",
                set_={
                    "generations_started": text("video_generation_usage.generations_started + 1"),
                    "generations_completed": text(
                        f"video_generation_usage.generations_completed + {generations_completed}"
                    ),
                    "generations_failed": text(
                        f"video_generation_usage.generations_failed + {1 if is_error else 0}"
                    ),
                    "total_duration_generated_s": text(
                        f"video_generation_usage.total_duration_generated_s + {duration_generated}"
                    ),
                    "total_file_size_bytes": text(
                        f"video_generation_usage.total_file_size_bytes + {file_size}"
                    ),
                },
            )
        )
        await db.execute(stmt)


# ---------------------------------------------------------------------------
# Module-level singleton accessor + FastAPI dependency
# ---------------------------------------------------------------------------

_default_job_manager: JobManager | None = None


def set_default_job_manager(jm: JobManager | None) -> None:
    """Install the process-wide default ``JobManager`` instance.

    Called from ``asgi.lifespan`` after collaborators are wired.
    Tests typically leave this unset and rely on ``app.state.job_manager``
    or direct patching of ``JobManager`` static methods.
    """
    global _default_job_manager
    _default_job_manager = jm


def get_default_job_manager() -> JobManager:
    """Return the process-wide default ``JobManager``.

    Lazily constructs an unconfigured ``JobManager()`` if one was not
    installed, so tests that import this helper without going through
    lifespan still get a working object. Cache invalidation will fall
    back to ``get_redis_client()`` in that case, identical to legacy
    behavior.
    """
    global _default_job_manager
    if _default_job_manager is None:
        _default_job_manager = JobManager()
    return _default_job_manager
