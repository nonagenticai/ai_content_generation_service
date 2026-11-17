"""Generation endpoints: POST /generate, POST /generate/batch."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from src.services.batch_tracker import BatchTracker
    from src.services.quota_service import QuotaService

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.auth.identity_context import (
    KEEP_TENANT_SOURCE,
    # `_Keep` is the private TYPE of the KEEP_TENANT_SOURCE sentinel, imported for the
    # `tenant_source` annotation only — never constructed, never called. Deliberate, not
    # a slip: identity_context declares no `__all__` (verified at w2v-common 0.67.25), so
    # `_Keep` is exactly as supported here as its neighbours, and there is no public alias
    # for the sentinel's type. The one real exposure is an upstream RENAME, which lands as
    # an ImportError at boot — loud, and never as silently wrong provenance.
    _Keep,
    set_identity_context,
)
from w2v_common.tenant import TenantSource

from src.api.routes.dependencies import (
    _get_batch_tracker,
    _get_quota_service,
    _get_tenant_id,
    _get_video_generation_service,
    _validate_generate_request,
    effective_video_slot,
    provider_slots_enforced_or_503,
    reject_slot_on_batch,
    reject_tenant_slot_on_batch,
    require_slot_keyring_or_503,
    resolve_run_video_slot,
)
from src.api.schemas import (
    AsyncJobResponse,
    BatchJobResponse,
    GenerateBatchRequest,
    GenerateRequest,
)
from src.config import get_settings
from src.database.models import VideoGenerationJob
from src.database.session import get_db, get_db_session
from src.services.durable_dispatch import (
    build_video_batch_options,
    build_video_single_options,
    pod_id,
    run_with_lease,
)
from src.services.idempotency import (
    REPLAYED_HEADER,
    IdempotentReplayError,
    ensure_same_request,
    find_live_job,
    read_idempotency_key,
    request_fingerprint,
)
from src.services.job_manager import JobManager, JobStateConflictError
from src.services.provider_slots import seal_slot, video_provider_name
from src.services.video_generation_service import VideoGenerationService

logger = logging.getLogger(__name__)

router = APIRouter()


def _replayed_video_submission(
    job: VideoGenerationJob, fingerprint: str, response: Response
) -> AsyncJobResponse:
    """The ORIGINAL job for a repeated ``Idempotency-Key`` -- no new work started."""
    ensure_same_request(job, fingerprint)
    response.headers[REPLAYED_HEADER] = "true"
    logger.info("Generate Idempotency-Key replay -> existing job %s", job.job_id)
    return AsyncJobResponse(
        job_id=job.job_id,
        status=job.status,
        poll_url=f"/api/content/videos/jobs/{job.job_id}",
    )


@router.post(
    "/generate",
    operation_id="start_video_generation",
    responses={
        422: {"description": "Validation error, or Idempotency-Key reused with a different body"},
        429: {"description": "Quota exceeded"},
    },
)
async def generate_video(
    body: GenerateRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AsyncJobResponse:
    """Generate an AI video clip asynchronously.

    Honours ``Idempotency-Key``: a repeat with the same key (same tenant, same
    body) returns the original job -- same id, its CURRENT status, header
    ``Idempotent-Replayed: true`` -- and starts no second paid render. A failed
    or cancelled job releases its key, so a retry after one gets a fresh job.

    Args:
        body: Generation parameters including prompt, provider, and options.
        request: FastAPI request with tenant context.
        response: Outgoing response (carries ``Idempotent-Replayed`` on a replay).
        db: Async database session.

    Returns:
        AsyncJobResponse with job_id and poll URL.

    Raises:
        HTTPException: 422 for validation errors, 429 for quota exceeded.
    """
    tenant_id = _get_tenant_id(request)
    logger.info(
        "Generate request received: tenant=%s provider=%s prompt_len=%d",
        tenant_id,
        body.provider,
        len(body.prompt),
    )
    enforce_slots = provider_slots_enforced_or_503()
    if not enforce_slots:
        _validate_generate_request(request, body)
    idempotency_key = read_idempotency_key(request)
    # Fingerprint of the body AS SENT (secret masked), in both modes: a retry with
    # the same Idempotency-Key replays exactly as before, whatever the switch says.
    fingerprint = request_fingerprint(body)
    if idempotency_key is not None:
        existing = await find_live_job(
            db, VideoGenerationJob, tenant_id=tenant_id, key=idempotency_key
        )
        if existing is not None:
            return _replayed_video_submission(
                cast(VideoGenerationJob, existing), fingerprint, response
            )
    # enforce: the RUN's video_gen slot from the orchestrator, body ``provider_slot``
    # IGNORED; off: the body's. After the replay check, so a replay never resolves.
    slot = await effective_video_slot(request, body, enforce=enforce_slots)
    if slot is not None:
        require_slot_keyring_or_503()
        logger.info("Generate request uses tenant provider slot %s", slot.label())

    service = _get_video_generation_service(request)

    # Quota check
    quota_service = _get_quota_service(request)
    if quota_service is not None:
        # Price the clip we are about to admit. Omitting estimated_cost made the
        # spend cap read ``daily_spend + 0.0 > max``, i.e. blind to the cost of
        # everything it let through. A tenant provider slot runs on the
        # TENANT's key: it is not platform spend, so it is priced at 0 and is
        # exempt from the platform daily spend cap.
        estimated_cost = (
            0.0
            if slot is not None
            else service.estimate_request_cost(body.provider, body.duration_s)
        )
        result = await quota_service.check_quota(
            tenant_id, estimated_cost=estimated_cost, tenant_funded=slot is not None
        )
        if not result.allowed:
            logger.warning("Quota exceeded for tenant=%s: %s", tenant_id, result.reason)
            raise HTTPException(status_code=429, detail=result.reason)

    try:
        job_id = await service.create_job(
            db,
            tenant_id=tenant_id,
            prompt=body.prompt,
            # A tenant slot REPLACES provider selection: the row names the slot's
            # provider and carries no fallbacks (tenant slots never fall back).
            provider=video_provider_name(slot.kind) if slot is not None else body.provider,
            negative_prompt=body.negative_prompt,
            image_url=body.image_url,
            duration_s=body.duration_s,
            width=body.width,
            height=body.height,
            style=body.style,
            seed=body.seed,
            extra_params=body.extra_params,
            fallback_providers=None if slot is not None else body.fallback_providers,
            callback_url=body.callback_url,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=fingerprint if idempotency_key is not None else None,
        )
    except IdempotentReplayError as replay:
        # Lost a race to a concurrent duplicate: our INSERT wrote nothing.
        return _replayed_video_submission(
            cast(VideoGenerationJob, replay.job), fingerprint, response
        )
    if slot is not None:
        provider_slot, sealed = seal_slot(slot, job_id=job_id)
        await JobManager.attach_provider_slot(
            db, job_id, provider_slot=provider_slot, provider_secret_sealed=sealed
        )
    settings = get_settings()
    # Durable dispatch: mark the job queue-owned + acquire a lease so the reaper
    # on ANY replica can reclaim it if this pod dies before it finishes.
    await JobManager.register_dispatch(
        db,
        job_id,
        options=build_video_single_options(
            budget_limit=body.budget_limit, callback_url=body.callback_url
        ),
        worker_id=pod_id(),
        lease_seconds=settings.dispatch.lease_seconds,
    )
    await db.commit()

    if quota_service is not None:
        await quota_service.record_generation_start(tenant_id, job_id)

    task = asyncio.create_task(
        run_with_lease(
            job_id=job_id,
            lease_seconds=settings.dispatch.lease_seconds,
            renew=JobManager.renew_lease,
            work=_process_job_bg(
                service,
                job_id,
                tenant_id,
                budget_limit=body.budget_limit,
                callback_url=body.callback_url,
                quota_service=quota_service,
            ),
            heartbeat_enabled=settings.dispatch.heartbeat_enabled,
        )
    )
    service.register_task(job_id, task)

    logger.info("Generate job created: job_id=%s tenant=%s", job_id, tenant_id)
    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"/api/content/videos/jobs/{job_id}",
    )


@router.post(
    "/generate/batch",
    operation_id="start_video_generation_batch",
    responses={
        422: {"description": "Validation error"},
        429: {"description": "Quota exceeded"},
    },
)
async def generate_batch(
    body: GenerateBatchRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> BatchJobResponse:
    """Generate multiple AI video clips asynchronously.

    Args:
        body: Batch generation parameters with list of items.
        request: FastAPI request with tenant context.
        db: Async database session.

    Returns:
        BatchJobResponse with batch_id and list of job_ids.

    Raises:
        HTTPException: 422 for validation errors.
    """
    tenant_id = _get_tenant_id(request)
    logger.info(
        "Batch generate request received: tenant=%s items=%d budget=%.2f",
        tenant_id,
        len(body.items),
        body.budget_limit,
    )
    if provider_slots_enforced_or_503():
        # enforce: item ``provider_slot``s are ignored; the RUN's slot decides.
        reject_tenant_slot_on_batch(await resolve_run_video_slot())
        for item in body.items:
            _validate_generate_request(request, item, slot=None)
    else:
        for item in body.items:
            reject_slot_on_batch(item.provider_slot)
            _validate_generate_request(request, item)

    service = _get_video_generation_service(request)
    tracker = _get_batch_tracker(request)
    quota_service = _get_quota_service(request)

    # Admission check, exactly as ``/generate`` does. Without it a tenant already
    # over its daily/concurrent limit was refused on the single endpoint and
    # admitted on this one — the batch endpoint was entirely quota-free.
    if quota_service is not None:
        # A batch is admitted as one unit, so it is priced as one unit: the sum
        # of every clip it would buy. Checking with 0.0 admitted an arbitrarily
        # expensive batch against a nearly-full daily cap.
        batch_cost = sum(
            service.estimate_request_cost(item.provider, item.duration_s) for item in body.items
        )
        admission = await quota_service.check_quota(tenant_id, estimated_cost=batch_cost)
        if not admission.allowed:
            logger.warning("Quota exceeded for tenant=%s (batch): %s", tenant_id, admission.reason)
            raise HTTPException(status_code=429, detail=admission.reason)

    batch_id = str(uuid.uuid7())
    job_ids: list[str] = []

    for item in body.items:
        job_id = await service.create_job(
            db,
            tenant_id=tenant_id,
            prompt=item.prompt,
            provider=item.provider or body.provider,
            negative_prompt=item.negative_prompt,
            image_url=item.image_url,
            duration_s=item.duration_s,
            width=item.width,
            height=item.height,
            style=item.style,
            seed=item.seed,
            extra_params=item.extra_params,
            fallback_providers=item.fallback_providers,
            callback_url=item.callback_url or body.callback_url,
        )
        job_ids.append(job_id)

    settings = get_settings()
    # Durable dispatch: each batch child job carries its batch context so the
    # reaper can resume an individual clip on any replica after a pod crash.
    for job_id in job_ids:
        await JobManager.register_dispatch(
            db,
            job_id,
            options=build_video_batch_options(batch_id=batch_id, budget_limit=body.budget_limit),
            worker_id=pod_id(),
            lease_seconds=settings.dispatch.lease_seconds,
        )

    await db.commit()

    await tracker.create_batch(
        batch_id, job_ids, budget_limit=body.budget_limit, tenant_id=tenant_id
    )

    for job_id in job_ids:
        task = asyncio.create_task(
            run_with_lease(
                job_id=job_id,
                lease_seconds=settings.dispatch.lease_seconds,
                renew=JobManager.renew_lease,
                work=_process_batch_job_bg(
                    service,
                    job_id,
                    tenant_id,
                    batch_id,
                    tracker,
                    body.budget_limit,
                    quota_service=quota_service,
                ),
                heartbeat_enabled=settings.dispatch.heartbeat_enabled,
            )
        )
        service.register_task(job_id, task)

    logger.info("Batch created: batch_id=%s jobs=%d tenant=%s", batch_id, len(job_ids), tenant_id)
    return BatchJobResponse(batch_id=batch_id, job_ids=job_ids, status="pending")


async def _process_job_bg(
    service: VideoGenerationService,
    job_id: str,
    tenant_id: str,
    budget_limit: float = 5.0,
    spent_so_far: float = 0.0,
    callback_url: str | None = None,
    quota_service: QuotaService | None = None,
    tenant_source: TenantSource | _Keep | None = KEEP_TENANT_SOURCE,
) -> None:
    """Background task that processes a single video generation job.

    Args:
        service: VideoGenerationService orchestrating the generation.
        job_id: Unique identifier for the job.
        tenant_id: Tenant owning the job.
        budget_limit: Maximum spend in USD.
        spent_so_far: Amount already spent in the current context.
        callback_url: Optional webhook URL for completion notification.
        quota_service: Optional quota enforcement service.
    """
    # Phase D RLS: bind the tenant for this bg task's sessions (after_begin re-binds each txn).
    # ⚠️ tenant_source IS LOAD-BEARING — omit it and every metered outbound hop
    # this task makes fails closed. `set_identity_context` sets
    # `_tenant_source_var` UNCONDITIONALLY, so calling it without `tenant_source`
    # does not merely "fail to set" provenance: it DESTROYS the provenance this
    # task already inherited. `asyncio.create_task` copies the caller's context,
    # so AuthMiddleware's provenance IS present here until this line overwrites
    # it with None — which `build_outbound_identity` reports as 'unrecorded' and
    # refuses:
    #   refusing to delegate tenant <t> to metered destination <d>:
    #   provenance is 'unrecorded', which is not vouched
    # Measured 2026-08-28: this is why modality-gateway refused panel image
    # generation in the w2v_frontend E2E post-promote matrix.
    # Reading it back and passing it through is therefore a PRESERVE, not a
    # re-derive — it keeps the provenance the request already established.
    # ⛔ Do NOT copy this line into a reaper/worker loop that iterates rows for
    # DIFFERENT tenants (see services/cleanup_service.py): there the inherited
    # value belongs to another tenant, and preserving it would vouch for a tenant
    # whose provenance was never established. Those must stay None.
    #
    # ...which is precisely why ``tenant_source`` is a PARAMETER and not read from
    # the context here. ReaperService enters this function too, on another pod,
    # from a loop walking rows for many tenants — so the ambient value is either
    # unset (this task then fails closed at its first metered hop) or the PREVIOUS
    # row's. The reaper passes the value persisted on the job row at enqueue,
    # restored via ``durable_dispatch.restore_tenant_source``; the request path
    # leaves it at ``KEEP_TENANT_SOURCE``, which preserves the provenance this
    # task inherited from the request context — byte-for-byte the old behaviour.
    set_identity_context(tenant_id, None, None, tenant_source=tenant_source)
    # The route already called record_generation_start(tenant, job_id), so this task
    # OWES exactly one release. ``cost`` must exist on every failure path or the
    # release in ``finally`` would raise NameError and leak the slot it exists to
    # return.
    #
    # ⚠️ This function is ALSO entered from ``ReaperService._dispatch_video`` on a
    # reclaim, where no route ran and no start was recorded. That used to make the
    # release below a bare DECR against zero INCRs — one free concurrency slot per
    # reclaim, and one per in-flight job on every rollout, silently disabling the
    # cap on paid generation. It is safe now only because the release names
    # ``job_id`` and is therefore idempotent (ZREM of an absent member is a no-op),
    # and because the slot the crashed run took is the SAME member, so the reclaimed
    # run inherits it instead of needing a second one. Do NOT change the release
    # back to an unnamed decrement.
    cost = 0.0
    try:
        async with get_db_session() as db:
            cost = await service.process_job(
                db, job_id, tenant_id, budget_limit=budget_limit, spent_so_far=spent_so_far
            )

            # Fire callback if configured
            if callback_url:
                try:
                    from src.services.callback_service import deliver_callback

                    job = await JobManager.get_job(db, job_id)
                    payload = {
                        "job_id": job_id,
                        "status": job.status if job else "unknown",
                        "cost": cost,
                    }
                    await deliver_callback(db, job_id, callback_url, payload)
                except Exception:
                    # Callback delivery is best-effort and must NEVER fail the
                    # job: the video is already generated + stored. An
                    # unreachable customer webhook (httpx.ConnectError, DNS
                    # failure, etc.) is logged, not fatal. Mirrors the sibling
                    # concept-image path (_deliver_concept_image_callback).
                    logger.exception("Callback delivery failed for job %s", job_id)

    except asyncio.CancelledError:
        logger.info("Job %s was cancelled", job_id)
    except Exception:
        # Catch-all so an unexpected error (httpx, SQLAlchemy, etc.) raised
        # outside the provider retry loop can never leave the job wedged in
        # ``processing`` — always drive it to a terminal ``failed`` state.
        logger.exception("Background video generation job %s failed", job_id)
        try:
            async with get_db_session() as db:
                await JobManager.mark_failed(db, job_id, "Unexpected error during video generation")
        except JobStateConflictError:
            logger.info("Job %s already terminal; skipping top-level mark_failed", job_id)
        except OSError, SQLAlchemyError:
            logger.exception("Failed to mark job %s as failed", job_id)
    finally:
        service.unregister_task(job_id)
        # Release the tenant concurrency slot on EVERY exit path. Living inside
        # the ``try`` meant any exception, early return or CancelledError leaked
        # a slot until its 1h TTL, progressively throttling the tenant to zero.
        # Best-effort like callback delivery above: a quota-store failure is
        # logged, never re-raised out of ``finally`` (which would replace the
        # original exception). CancelledError is a BaseException and still
        # propagates.
        if quota_service is not None:
            try:
                await quota_service.record_generation_complete(tenant_id, job_id, cost=cost)
            except Exception:
                logger.exception("Failed to release quota slot for job %s", job_id)


async def _admit_batch_clip(
    db: AsyncSession,
    service: VideoGenerationService,
    job_id: str,
    batch_id: str,
    tracker: BatchTracker,
    tenant_id: str,
    quota_service: QuotaService | None,
) -> bool:
    """Take a quota slot for ONE batch clip. False = rejected (already accounted).

    The video analogue of ``content_routes._check_quota_for_batch_job``. Returns
    True (having recorded the start) when the clip may run, or False after
    marking the clip failed and accounting it against the batch.

    The clip row already exists here, so its cost is priced from the persisted
    provider + duration — the same shape ``_check_quota_for_batch_job`` uses.
    """
    if quota_service is None:
        return True
    job = await JobManager.get_job(db, job_id)
    estimated_cost = (
        service.estimate_request_cost(job.provider, job.duration_s) if job is not None else 0.0
    )
    check = await quota_service.check_quota(tenant_id, estimated_cost=estimated_cost)
    if not check.allowed:
        logger.warning(
            "Quota exceeded for batch clip %s (batch %s): %s", job_id, batch_id, check.reason
        )
        try:
            await JobManager.mark_failed(db, job_id, check.reason or "Quota exceeded")
        except JobStateConflictError:
            logger.info("Batch job %s already terminal; quota path", job_id)
        await tracker.record_clip_failed(batch_id)
        return False
    await quota_service.record_generation_start(tenant_id, job_id)
    return True


async def _process_batch_job_bg(
    service: VideoGenerationService,
    job_id: str,
    tenant_id: str,
    batch_id: str,
    tracker: BatchTracker,
    budget_limit: float = 5.0,
    quota_service: QuotaService | None = None,
    tenant_source: TenantSource | _Keep | None = KEEP_TENANT_SOURCE,
) -> None:
    """Background task that processes a single job within a batch.

    Args:
        service: VideoGenerationService orchestrating the generation.
        job_id: Unique identifier for the job.
        tenant_id: Tenant owning the batch.
        batch_id: Identifier of the parent batch.
        tracker: BatchTracker for updating batch progress.
        budget_limit: Maximum total spend for the batch.
        quota_service: Optional quota enforcement service.
    """

    # Phase D RLS: bind the tenant for this bg task's sessions (after_begin re-binds each txn).
    # ⚠️ tenant_source IS LOAD-BEARING — omit it and every metered outbound hop
    # this task makes fails closed. `set_identity_context` sets
    # `_tenant_source_var` UNCONDITIONALLY, so calling it without `tenant_source`
    # does not merely "fail to set" provenance: it DESTROYS the provenance this
    # task already inherited. `asyncio.create_task` copies the caller's context,
    # so AuthMiddleware's provenance IS present here until this line overwrites
    # it with None — which `build_outbound_identity` reports as 'unrecorded' and
    # refuses:
    #   refusing to delegate tenant <t> to metered destination <d>:
    #   provenance is 'unrecorded', which is not vouched
    # Measured 2026-08-28: this is why modality-gateway refused panel image
    # generation in the w2v_frontend E2E post-promote matrix.
    # Reading it back and passing it through is therefore a PRESERVE, not a
    # re-derive — it keeps the provenance the request already established.
    # ⛔ Do NOT copy this line into a reaper/worker loop that iterates rows for
    # DIFFERENT tenants (see services/cleanup_service.py): there the inherited
    # value belongs to another tenant, and preserving it would vouch for a tenant
    # whose provenance was never established. Those must stay None.
    #
    # ...which is precisely why ``tenant_source`` is a PARAMETER and not read from
    # the context here. ReaperService enters this function too, on another pod,
    # from a loop walking rows for many tenants — so the ambient value is either
    # unset (this task then fails closed at its first metered hop) or the PREVIOUS
    # row's. The reaper passes the value persisted on the job row at enqueue,
    # restored via ``durable_dispatch.restore_tenant_source``; the request path
    # leaves it at ``KEEP_TENANT_SOURCE``, which preserves the provenance this
    # task inherited from the request context — byte-for-byte the old behaviour.
    set_identity_context(tenant_id, None, None, tenant_source=tenant_source)
    # Unlike the single-job path, the route does NOT record a start for batch
    # children (they are also spawned by the reaper on reclaim, which has no
    # route to do it). Each child takes its OWN slot here and the release below
    # is gated on ``quota_started`` — mirroring the concept-image batch child.
    # Previously the ``finally`` released unconditionally against a start that
    # was never recorded: one DECR per clip against zero INCRs, which zeroed the
    # tenant's concurrency gauge (cancelling slots held by its other in-flight
    # generations) and left every batch clip uncounted against the daily
    # generation quota.
    quota_started = False
    # Always defined so the release in ``finally`` cannot NameError. See
    # _process_job_bg for the full rationale.
    cost = 0.0
    try:
        async with get_db_session() as db:
            if not await _admit_batch_clip(
                db, service, job_id, batch_id, tracker, tenant_id, quota_service
            ):
                return
            quota_started = quota_service is not None
            batch = await tracker.get_batch(batch_id)
            spent = batch.total_cost if batch else 0.0
            cost = await service.process_job(
                db, job_id, tenant_id, budget_limit=budget_limit, spent_so_far=spent
            )
            if cost > 0:
                await tracker.record_clip_completed(batch_id, cost=cost)
            else:
                job = await JobManager.get_job(db, job_id)
                if job and job.status == "failed":
                    err = job.error_message or ""
                    if "Budget exceeded" in str(err):
                        await tracker.record_clip_skipped(batch_id)
                    else:
                        await tracker.record_clip_failed(batch_id)
                else:
                    await tracker.record_clip_failed(batch_id)

    except asyncio.CancelledError:
        # Account the clip. A cancelled child is TERMINAL --
        # it will never produce a clip -- but it used to bump no counter at all,
        # so ``done < total`` held forever: the batch's ``computed_status``
        # stayed ``processing`` and ``GET /batches/{id}/result`` 409'd until the
        # 2h TTL, then 404'd. Every other exit path from this worker accounts
        # itself; this one silently did not.
        logger.info("Batch job %s was cancelled", job_id)
        try:
            await tracker.record_clip_cancelled(batch_id)
        except Exception:
            # Best-effort: we are already unwinding a cancellation, and failing
            # to record must not replace it with a different error.
            logger.exception("Failed to account cancelled batch clip %s", job_id)
    except Exception:
        # Catch-all so an unexpected error can never leave the job wedged in
        # ``processing`` or the batch clip unaccounted-for.
        logger.exception("Background batch job %s (batch %s) failed", job_id, batch_id)
        await tracker.record_clip_failed(batch_id)
    finally:
        service.unregister_task(job_id)
        # The ONE release, on every exit path, only if a start was recorded.
        # Releasing a slot this child never took handed the tenant free
        # concurrency — see _handle_batch_job_exception in content_routes.py.
        if quota_service is not None and quota_started:
            try:
                await quota_service.record_generation_complete(tenant_id, job_id, cost=cost)
            except Exception:
                logger.exception("Failed to release quota slot for batch job %s", job_id)
