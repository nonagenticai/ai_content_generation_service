"""Background job cleanup service for expired jobs and files."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class CleanupService:
    """Periodically removes expired jobs, local files, and MinIO objects.

    Runs a background loop that wakes up at ``interval_seconds`` and
    deletes jobs whose ``completed_at`` exceeds the configured TTL.

    Attributes:
        _interval: Seconds between cleanup cycles.
        _completed_ttl_days: Days before completed jobs expire.
        _failed_ttl_days: Days before failed jobs expire.
        _cancelled_ttl_days: Days before cancelled jobs expire.
        _storage: Storage service for file deletion.
        _session_factory: Async session factory for database access.
        _task: Background asyncio task.
    """

    def __init__(
        self,
        interval_seconds: int = 3600,
        completed_ttl_days: int = 30,
        failed_ttl_days: int = 7,
        cancelled_ttl_days: int = 7,
        storage: StorageService | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._interval = interval_seconds
        self._completed_ttl_days = completed_ttl_days
        self._failed_ttl_days = failed_ttl_days
        self._cancelled_ttl_days = cancelled_ttl_days
        self._storage = storage
        self._session_factory = session_factory
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the cleanup background loop."""
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Cleanup service started (interval=%ds, completed_ttl=%dd, failed_ttl=%dd, "
            "cancelled_ttl=%dd)",
            self._interval,
            self._completed_ttl_days,
            self._failed_ttl_days,
            self._cancelled_ttl_days,
        )

    async def stop(self) -> None:
        """Stop the cleanup background loop."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Cleanup service stopped")

    async def _loop(self) -> None:
        # Work FIRST, sleep after. A sleep-first loop never completes a single
        # cycle in a pod that restarts more often than the interval, which is the
        # normal case during a rollout -- so the sweep silently never runs.
        while True:
            try:
                await self.run_cleanup()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Cleanup cycle failed")
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break

    async def _delete_job_storage(self, local_path: str | None, minio_key: str | None) -> None:
        """Free a job's artifacts. This sweep is their ONLY deletion path.

        Nothing else in the service ever deletes a generated object, and the
        cluster's whole-bucket lifecycle rule is hand-applied and unversioned —
        so a status the sweep does not visit leaks its MinIO object permanently.
        """
        storage = self._storage
        if storage is None:
            return
        if local_path:
            try:
                await storage.delete_local(local_path)
            except OSError as e:
                logger.warning("Failed to clean up file: %s", e)
        if minio_key:
            try:
                await storage.delete_from_minio(minio_key)
            except OSError as e:
                logger.warning("Failed to clean up file: %s", e)

    async def _cleanup_status(
        self,
        factory: async_sessionmaker[AsyncSession],
        status: str,
        before: datetime,
        ttl_days: int,
        *,
        kind: str = "video",
    ) -> int:
        """Delete expired jobs of a status (cross-tenant).

        Phase D RLS: the expired-job scan is cross-tenant (no tenant context at boot),
        so it runs through the ``expired_video_generation_jobs`` /
        ``expired_imagegen_jobs`` SECURITY DEFINER read resolvers (owner bypasses
        ENABLE). Each job's storage is deleted, then the DB row is removed in its own
        tenant-bound session via the existing ``delete_jobs``.

        ``kind`` picks the table. The two halves are identical apart from the
        resolver name, the local-path column, and the manager — deliberately one
        code path, so a status added to one side cannot silently skip the other.
        """
        from sqlalchemy import text
        from w2v_common.auth.identity_context import set_identity_context

        from src.services.image_job_manager import ImageJobManager
        from src.services.job_manager import JobManager

        if kind == "video":
            sql = (
                "SELECT job_id, tenant_id, video_path AS local_path, minio_key "
                "FROM expired_video_generation_jobs(:s, :b, :lim)"
            )
            delete_jobs = JobManager.delete_jobs
        else:
            sql = (
                "SELECT job_id, tenant_id, image_path AS local_path, minio_key "
                "FROM expired_imagegen_jobs(:s, :b, :lim)"
            )
            delete_jobs = ImageJobManager.delete_terminal_jobs

        try:
            async with factory() as db:
                rows = (await db.execute(text(sql), {"s": status, "b": before, "lim": 100})).all()
            if not rows:
                return 0
            deleted = 0
            for r in rows:
                await self._delete_job_storage(r.local_path, r.minio_key)
                # The resolvers return ``tenant_id uuid``
                # since migration 011, so ``r.tenant_id`` is a ``uuid.UUID``.
                # set_config() has no uuid overload -- passing one raises
                # ``function set_config(unknown, uuid, boolean) does not exist``
                # and kills this sweep on every run under TENANT_RLS_MODE=enforce.
                # No tenant_source, DELIBERATELY. This reaper has no inbound
                # request, so there is no provenance to carry, and it rebinds a
                # DIFFERENT tenant on every row. Preserving an inherited value
                # here (the fix applied to the request-dispatched _bg tasks under
                # api/) would vouch one tenant's provenance for another's row --
                # exactly the escalation WS-4 exists to stop. None means any
                # metered delegation from this loop fails closed, which is right:
                # a reaper deletes storage and rows, it does not spend a tenant's
                # credentials.
                set_identity_context(str(r.tenant_id), None, None)
                async with factory() as db2:
                    await delete_jobs(db2, [r.job_id])
                    await db2.commit()
                deleted += 1
            logger.info(
                "Cleaned up %d %s %s jobs older than %d days", deleted, status, kind, ttl_days
            )
            return deleted
        except Exception:
            logger.exception("Cleanup failed for kind=%s status=%s", kind, status)
            return 0

    async def _cleanup_image_batches(
        self,
        factory: async_sessionmaker[AsyncSession],
        status: str,
        before: datetime,
        ttl_days: int,
    ) -> int:
        """Retire an expired image batch AND its children as one unit (cross-tenant).

        The batch owns the retention decision for everything under it. A batched
        child is exempt from the per-status job sweep (``expired_imagegen_jobs``
        selects ``batch_id IS NULL``), because ``get_concept_image_batch`` renders
        ``results`` from surviving child rows while reporting ``job_count`` /
        ``completed_jobs`` from the batch's stored counters: a child that aged out
        before its parent would make the endpoint return a truncated list beside
        counters claiming the original totals, with nothing to tell a caller the
        difference between a reaped row and a lost one. Retiring the unit as a whole
        is what makes that state unreachable rather than merely unlikely.

        So the children's storage is freed, their rows deleted, and the parent
        deleted, all in ONE transaction (``db2``) — the endpoint therefore only ever
        sees the batch whole or absent. That also satisfies the FK without help: the
        children go first, in the same statement batch as the parent.

        ⚠️ The ELIGIBILITY read is a SEPARATE, already-closed session (``db``), so
        the resolver's ``NOT EXISTS pending/processing`` guard is NOT part of that
        transaction. Under READ COMMITTED with no ``FOR UPDATE``, a child could in
        principle become non-terminal between the two. Two independent things stop
        that from deleting live work, and only the second is structural:

        * the resolver guard keeps a batch with in-flight children out of the sweep
          in the first place — necessary, but check-then-act; and
        * ``delete_terminal_jobs`` re-asserts ``status NOT IN (pending, processing)``
          IN THE DELETE, so a row that turned live inside the window is not
          deletable however stale this loop's belief about it is.

        The batch row itself then fails its FK (the surviving child still references
        it), the transaction rolls back, and the next cycle re-evaluates. Losing a
        cycle is the correct outcome; deleting a running job is not.

        Returns rows deleted, children included, so a caller's total reflects the
        work actually done.
        """
        from sqlalchemy import text
        from w2v_common.auth.identity_context import set_identity_context

        from src.services.image_batch_manager import ImageBatchManager
        from src.services.image_job_manager import ImageJobManager

        try:
            async with factory() as db:
                rows = (
                    await db.execute(
                        text(
                            "SELECT batch_id, tenant_id FROM expired_imagegen_batches(:s, :b, :lim)"
                        ),
                        {"s": status, "b": before, "lim": 100},
                    )
                ).all()
            if not rows:
                return 0
            deleted = 0
            for r in rows:
                # Per-row tenant rebind, same reasoning as ``_cleanup_status``.
                set_identity_context(str(r.tenant_id), None, None)
                async with factory() as db2:
                    children = await ImageJobManager.get_jobs_for_batch(db2, r.batch_id)
                    for child in children:
                        await self._delete_job_storage(child.image_path, child.minio_key)
                    children_deleted = await ImageJobManager.delete_terminal_jobs(
                        db2, [child.job_id for child in children]
                    )
                    # If a child survived the terminal guard the parent's FK fails
                    # here, the transaction rolls back, and the next cycle retries.
                    await ImageBatchManager.delete_batches(db2, [r.batch_id])
                    await db2.commit()
                deleted += 1 + children_deleted
            logger.info(
                "Cleaned up %d rows for %s image batches older than %d days",
                deleted,
                status,
                ttl_days,
            )
            return deleted
        except Exception:
            logger.exception("Cleanup failed for image batches status=%s", status)
            return 0

    async def run_cleanup(self) -> int:
        """Run one cleanup cycle. Returns total jobs deleted."""
        session_factory = self._session_factory
        if session_factory is None:
            return 0
        factory: async_sessionmaker[AsyncSession] = session_factory
        now = datetime.now(UTC)
        total_deleted = 0
        # ``cancelled`` is a TERMINAL status written by ``mark_cancelled`` on both
        # managers (it stamps ``completed_at``, so the resolvers see it), and it was
        # missing from this list. That was not only a row-retention gap: this sweep
        # is the only caller of ``_delete_job_storage``, so every cancelled job's
        # MinIO object was stranded permanently. Its TTL matches ``failed`` by
        # default -- a cancelled job has no deliverable to retain, and a week is
        # enough to look at one after the fact -- but it is its own knob, because a
        # cancel is a user action and an operator may want to keep the evidence
        # longer than a crash.
        for status, ttl_days in [
            ("completed", self._completed_ttl_days),
            ("failed", self._failed_ttl_days),
            ("cancelled", self._cancelled_ttl_days),
        ]:
            before = now - timedelta(days=ttl_days)
            for kind in ("video", "image"):
                total_deleted += await self._cleanup_status(
                    factory, status, before, ttl_days, kind=kind
                )
        # Batches retire as a UNIT, on their OWN status TTL, taking their children
        # with them -- see ``_cleanup_image_batches``. ``partial`` shares the
        # ``completed`` TTL: it contains real delivered images.
        for status, ttl_days in [
            ("completed", self._completed_ttl_days),
            ("partial", self._completed_ttl_days),
            ("failed", self._failed_ttl_days),
            ("cancelled", self._cancelled_ttl_days),
        ]:
            total_deleted += await self._cleanup_image_batches(
                factory, status, now - timedelta(days=ttl_days), ttl_days
            )
        return total_deleted
