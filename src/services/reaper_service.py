"""Per-pod reaper for orphaned durable-dispatch jobs.

A fire-and-forget ``asyncio.create_task`` dies with its pod, orphaning the job
(stuck ``processing``, or ``pending`` if the pod died right after enqueue). This
service runs a periodic loop that reclaims such jobs across ALL tenants via the
SECURITY DEFINER ``reclaim_orphaned_*`` functions (``FOR UPDATE SKIP LOCKED`` so
concurrent reapers never grab the same job) and re-dispatches them on THIS pod —
under a fresh lease + heartbeat, so any replica can resume a crashed pod's work.

The reclaim is lease-gated: only jobs whose ``lease_expires_at`` has lapsed are
touched, so a job actively heartbeating on a live pod is never reclaimed. Jobs
past ``max_attempts`` are failed as poison inside the reclaim function.

⛔ TENANT PROVENANCE COMES FROM THE ROW, NEVER FROM THIS LOOP'S CONTEXT.

The ``_process_*_bg`` entrypoints below are shared with the request path, where
``asyncio.create_task`` copies the caller's provenance into the task and the
re-seed simply preserves it. None of that holds here: this loop runs on a
different pod, walks rows for MANY tenants, and has no inbound request. Its
ambient ``TenantSource`` is either unset — every metered hop of the re-dispatch
then fails closed with *"provenance is 'unrecorded', which is not vouched"*
against a modality_gateway URL — or, worse, the PREVIOUS row's, which would vouch
tenant A's proof for tenant B's spend. ``set_identity_context`` drops an inherited
source on a tenant change precisely so that the second cannot happen, which leaves
nothing ambient to fall back on, by design.

So each dispatch reads ``tenant_source`` off the PERSISTED job options — written
at enqueue while the request context was live — via
``durable_dispatch.restore_tenant_source``, and passes it explicitly. A row from
before that key existed restores as ``None`` and keeps today's fail-closed
behaviour; nothing is guessed.

At shutdown ``release_claims`` expires this pod's leases AND **refunds the attempt**
each claim consumed: a rollout / eviction is not a crash, and without the refund
every deploy burned one of a healthy job's lives (four deploys permanently failed a
paid provider generation). The refund is bounded by ``max_lifetime_seconds`` — a
wall-clock ceiling measured from a job's first claim — so a genuinely crash-looping
job still dies instead of re-claiming, and re-spending, forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

from src.database.session import get_db_session
from src.services.durable_dispatch import (
    KIND_IMAGE_BATCH,
    KIND_VIDEO_BATCH,
    pod_id,
    restore_tenant_source,
    run_with_lease,
)
from src.services.image_batch_manager import ImageBatchManager, ReconciledBatch
from src.services.image_job_manager import ImageJobManager
from src.services.job_manager import ClaimedJob, JobManager

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Bound per-sweep reclaims so one pod cannot monopolise a large backlog and so
# the loop yields between sweeps.
_MAX_RECLAIMS_PER_SWEEP = 50


class ReaperService:
    """Reclaims and re-dispatches orphaned video/image jobs on this pod."""

    def __init__(
        self,
        app: FastAPI,
        *,
        interval_s: float,
        lease_seconds: int,
        max_attempts: int,
        max_lifetime_seconds: int = 0,
        batch_stale_seconds: int = 300,
        heartbeat_enabled: bool = True,
    ) -> None:
        self._app = app
        self._interval = interval_s
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._max_lifetime_seconds = max_lifetime_seconds
        self._batch_stale_seconds = batch_stale_seconds
        self._heartbeat_enabled = heartbeat_enabled
        self._pod_id = pod_id()
        self._task: asyncio.Task[None] | None = None
        # Re-dispatched jobs run as background tasks; hold refs so they are not
        # GC'd mid-flight (RUF006) and can be drained at shutdown.
        self._inflight: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Reaper started (pod=%s, interval=%.0fs, lease=%ds, max_attempts=%d, max_lifetime=%ds)",
            self._pod_id,
            self._interval,
            self._lease_seconds,
            self._max_attempts,
            self._max_lifetime_seconds,
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        # Drain any re-dispatched jobs still running so their terminal writes land.
        for task in list(self._inflight):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        # Cancel this pod's ORIGINAL request-dispatched generations BEFORE releasing
        # their leases: otherwise a sibling reaper reclaims + re-dispatches a job this
        # pod is still generating → double provider spend during the drain window.
        # Cancelling hands the job cleanly to exactly one sibling.
        for svc_attr in ("video_generation_service", "imagegen_service"):
            svc = getattr(self._app.state, svc_attr, None)
            active = getattr(svc, "_active_tasks", None)
            if not active:
                continue
            for task in list(active.values()):
                task.cancel()
            for task in list(active.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        # Fast reclaim: expire this pod's in-flight leases so a sibling picks the
        # work up on its next sweep instead of waiting the full lease TTL.
        try:
            async with get_db_session() as db:
                released = await JobManager.release_claims(db, self._pod_id)
                released += await ImageJobManager.release_claims(db, self._pod_id)
                await db.commit()
            if released:
                logger.info("Released %d in-flight lease(s) at shutdown", released)
        except Exception:  # best-effort; leases expire on their own anyway
            logger.warning("Lease release at shutdown failed", exc_info=True)
        logger.info("Reaper stopped")

    async def _loop(self) -> None:
        # Work FIRST, sleep after. A sleep-first loop never completes a single
        # cycle in a pod that restarts more often than the interval, which is the
        # normal case during a rollout -- so the sweep silently never runs.
        while True:
            try:
                await self.reap_once()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Reaper sweep failed")
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break

    async def reap_once(self) -> int:
        """Run one reclaim sweep across both job tables. Returns jobs reclaimed."""
        reclaimed = 0
        reclaimed += await self._drain(JobManager.reclaim_orphaned, self._dispatch_video)
        reclaimed += await self._drain(ImageJobManager.reclaim_orphaned, self._dispatch_image)
        # The batch AGGREGATE row has its own terminal condition and is reaped
        # separately: reclaiming children can never close a parent whose counter
        # bumps were lost with the pod that owned them.
        await self.reap_batches_once()
        return reclaimed

    async def reap_batches_once(self) -> list[ReconciledBatch]:
        """Close out ``imagegen_batches`` rows whose children are all terminal.

        Isolated from the job sweep on purpose: a failure here (a missing
        resolver on a not-yet-migrated DB, say) must not stop jobs from being
        reclaimed, which is the money-relevant half of the loop.
        """
        try:
            async with get_db_session() as db:
                reconciled = await ImageBatchManager.reap_stale_batches(
                    db, stale_seconds=self._batch_stale_seconds, limit=_MAX_RECLAIMS_PER_SWEEP
                )
        except Exception:
            logger.exception("Batch reconcile sweep failed")
            return []
        for batch in reconciled:
            logger.info(
                "Reaped stale image batch %s -> %s (completed=%d, failed=%d)",
                batch.batch_id,
                batch.status,
                batch.completed_jobs,
                batch.failed_jobs,
            )
        return reconciled

    async def _drain(self, reclaim: Any, dispatch: Any) -> int:
        count = 0
        for _ in range(_MAX_RECLAIMS_PER_SWEEP):
            async with get_db_session() as db:
                claimed = await reclaim(
                    db,
                    worker_id=self._pod_id,
                    lease_seconds=self._lease_seconds,
                    max_attempts=self._max_attempts,
                    max_lifetime_seconds=self._max_lifetime_seconds,
                )
            if claimed is None:
                break
            self._spawn(dispatch(claimed))
            count += 1
        return count

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _dispatch_video(self, claimed: ClaimedJob) -> None:
        from src.api.routes.generation import _process_batch_job_bg, _process_job_bg

        app = self._app
        service = app.state.video_generation_service
        quota = getattr(app.state, "quota_service", None)
        opts = claimed.options
        budget = float(opts.get("budget_limit", 5.0))
        # Provenance comes from the ROW, never from this loop's ambient context —
        # see the module docstring.
        tenant_source = restore_tenant_source(opts.get("tenant_source"))
        if opts.get("kind") == KIND_VIDEO_BATCH:
            tracker = app.state.batch_tracker
            work = _process_batch_job_bg(
                service,
                claimed.job_id,
                claimed.tenant_id,
                opts.get("batch_id", ""),
                tracker,
                budget,
                quota_service=quota,
                tenant_source=tenant_source,
            )
        else:
            work = _process_job_bg(
                service,
                claimed.job_id,
                claimed.tenant_id,
                budget_limit=budget,
                callback_url=opts.get("callback_url"),
                quota_service=quota,
                tenant_source=tenant_source,
            )
        await run_with_lease(
            job_id=claimed.job_id,
            lease_seconds=self._lease_seconds,
            renew=JobManager.renew_lease,
            work=work,
            heartbeat_enabled=self._heartbeat_enabled,
        )

    async def _dispatch_image(self, claimed: ClaimedJob) -> None:
        from src.api.content_routes import (
            _process_concept_image_batch_job_bg,
            _process_concept_image_job_bg,
        )

        app = self._app
        service = app.state.imagegen_service
        quota = getattr(app.state, "quota_service", None)
        opts = claimed.options
        budget = float(opts.get("budget_limit", 5.0))
        # Provenance comes from the ROW, never from this loop's ambient context —
        # see the module docstring.
        tenant_source = restore_tenant_source(opts.get("tenant_source"))
        if opts.get("kind") == KIND_IMAGE_BATCH:
            work = _process_concept_image_batch_job_bg(
                service,
                claimed.job_id,
                claimed.tenant_id,
                opts.get("batch_id", ""),
                budget_limit=budget,
                quota_service=quota,
                tenant_source=tenant_source,
            )
        else:
            work = _process_concept_image_job_bg(
                service,
                claimed.job_id,
                claimed.tenant_id,
                budget_limit=budget,
                quota_service=quota,
                tenant_source=tenant_source,
            )
        await run_with_lease(
            job_id=claimed.job_id,
            lease_seconds=self._lease_seconds,
            renew=ImageJobManager.renew_lease,
            work=work,
            heartbeat_enabled=self._heartbeat_enabled,
        )
