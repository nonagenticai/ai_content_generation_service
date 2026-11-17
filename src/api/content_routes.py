"""Routes for AI-generated concept images."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.auth import LICENSE_GATE
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

from src.api.content_schemas import (
    ConceptImageAssetResponse,
    ConceptImageBatchRequest,
    ConceptImageBatchResponse,
    ConceptImageBatchSubmissionResponse,
    ConceptImageJobResponse,
    ConceptImageJobSubmissionResponse,
    ConceptImageRequest,
)
from src.api.routes.dependencies import (
    _get_imagegen_service,
    _get_quota_service,
    _get_redis,
    _get_storage,
    _get_tenant_id,
    _validate_generate_image_request,
    effective_image_slot,
    provider_slots_enforced_or_503,
    reject_slot_on_batch,
    reject_tenant_slot_on_batch,
    require_slot_keyring_or_503,
    resolve_run_image_slot,
)
from src.config import get_settings
from src.database.models import ImageGenJob
from src.database.session import get_db, get_db_session
from src.services.durable_dispatch import (
    build_image_batch_options,
    build_image_single_options,
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
from src.services.image_batch_manager import ImageBatchManager
from src.services.image_job_manager import ImageJobManager
from src.services.job_manager import JobStateConflictError
from src.services.provider_slots import (
    image_provider_name,
    seal_slot,
)

logger = logging.getLogger(__name__)

# Entitlement gate, mounted router-wide -- see src/api/routes/__init__.py for
# the full rationale. Concept-image generation is metered per image, so an
# unentitled tenant reaching these routes is unbilled spend on a paid
# provider. Dormant until ``W2V_LICENSE_GATE_ENABLED=true``.
router = APIRouter(
    prefix="/api/concept-images",
    tags=["concept-images"],
    dependencies=list(LICENSE_GATE),
)


def _job_status_path(job_id: str) -> str:
    return f"/api/concept-images/jobs/{job_id}"


def _job_result_path(job_id: str) -> str:
    return f"/api/concept-images/jobs/{job_id}/result"


def _batch_status_path(batch_id: str) -> str:
    return f"/api/concept-images/batches/{batch_id}"


def _batch_result_path(batch_id: str) -> str:
    return f"/api/concept-images/batches/{batch_id}/result"


def _asset_download_path(asset_id: str) -> str:
    return f"/api/concept-images/assets/{asset_id}/download"


def _preferred_asset_url(
    *,
    download_url: str | None,
    image_url: str | None,
) -> str | None:
    """Pick the best URL for consumers.

    The stream-through `download_url` (served by this service) is the
    canonical path — browsers cannot reach `minio.test:8389` directly.
    Falls back to a third-party-hosted `image_url` (e.g. provider CDN).
    """
    return download_url or image_url


def _serialize_concept_image_job(storage: Any, job: Any) -> ConceptImageJobResponse:
    # presigned_url intentionally None: browsers can't reach minio.test:8389.
    # Consumers use download_url (stream-through) instead.
    del storage  # no longer consulted for presigned URLs
    download_url = (
        _asset_download_path(job.asset_id) if job.status == "completed" and job.asset_id else None
    )
    return ConceptImageJobResponse(
        job_id=job.job_id,
        asset_id=job.asset_id,
        status=job.status,
        provider=job.provider,
        url=_preferred_asset_url(
            download_url=download_url,
            image_url=job.image_url,
        ),
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error_message,
        download_url=download_url,
        presigned_url=None,
        image_url=job.image_url,
    )


def _serialize_concept_image_asset(storage: Any, job: Any) -> ConceptImageAssetResponse:
    # presigned_url intentionally None: browsers can't reach minio.test:8389.
    del storage  # no longer consulted for presigned URLs
    download_url = (
        _asset_download_path(job.asset_id) if job.status == "completed" and job.asset_id else None
    )
    metadata = {"provider_job_id": job.provider_job_id} if job.provider_job_id else None
    return ConceptImageAssetResponse(
        job_id=job.job_id,
        asset_id=job.asset_id,
        status=job.status,
        provider=job.provider,
        prompt=job.prompt,
        url=_preferred_asset_url(
            download_url=download_url,
            image_url=job.image_url,
        ),
        image_path=job.image_path,
        image_url=job.image_url,
        download_url=download_url,
        presigned_url=None,
        mime_type=job.mime_type,
        width=job.width,
        height=job.height,
        file_size_bytes=job.file_size_bytes,
        cost=job.cost,
        usage_context=job.usage_context,
        target_usage=job.target_usage,
        error=job.error_message,
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        metadata=metadata,
    )


def _replayed_image_submission(
    job: ImageGenJob, fingerprint: str, response: Response
) -> ConceptImageJobSubmissionResponse:
    """The ORIGINAL job for a repeated ``Idempotency-Key`` -- no new work started."""
    ensure_same_request(job, fingerprint)
    response.headers[REPLAYED_HEADER] = "true"
    logger.info("Concept-image Idempotency-Key replay -> existing job %s", job.job_id)
    return ConceptImageJobSubmissionResponse(
        job_id=job.job_id,
        status=job.status,
        poll_url=_job_status_path(job.job_id),
        result_url=_job_result_path(job.job_id),
    )


@router.post(
    "",
    operation_id="create_concept_image",
    responses={
        422: {"description": "Validation error, or Idempotency-Key reused with a different body"},
    },
)
async def create_concept_image(
    body: ConceptImageRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> ConceptImageJobSubmissionResponse:
    """Create one concept-image job.

    Honours ``Idempotency-Key``: a repeat with the same key (same tenant, same
    body) returns the original job -- same id, its CURRENT status, header
    ``Idempotent-Replayed: true`` -- and starts no second generation. A failed
    or cancelled job releases its key, so a retry after one gets a fresh job.
    """
    tenant_id = _get_tenant_id(request)
    enforce_slots = provider_slots_enforced_or_503()
    if not enforce_slots:
        _validate_generate_image_request(request, body)
    idempotency_key = read_idempotency_key(request)
    # Fingerprint of the body AS SENT (secret masked), in both modes: a retry with
    # the same Idempotency-Key replays exactly as before, whatever the switch says.
    fingerprint = request_fingerprint(body)
    if idempotency_key is not None:
        existing = await find_live_job(db, ImageGenJob, tenant_id=tenant_id, key=idempotency_key)
        if existing is not None:
            return _replayed_image_submission(cast(ImageGenJob, existing), fingerprint, response)
    # enforce: the RUN's image slot from the orchestrator, body ``provider_slot``
    # IGNORED; off: the body's. After the replay check, so a replay never resolves.
    slot = await effective_image_slot(request, body, enforce=enforce_slots)
    if slot is not None:
        require_slot_keyring_or_503()
        logger.info("Concept-image request uses tenant provider slot %s", slot.label())

    service = _get_imagegen_service(request)
    # A tenant slot REPLACES provider selection: ``provider`` is ignored and the
    # row names the slot's provider class.
    provider = image_provider_name(slot.kind) if slot is not None else body.provider
    quota_service = _get_quota_service(request)
    if quota_service is not None:
        # Price the image we are about to admit, as the batch-child path at
        # ``_check_quota_for_batch_job`` already did. Without it the spend cap
        # could only trip on budget already spent. A tenant provider slot runs
        # on the TENANT's key: not platform spend, so priced at 0 and exempt
        # from the platform daily spend cap.
        estimated_cost = (
            0.0
            if slot is not None
            else service.estimate_request_cost(provider, size=body.size, quality=body.quality)
        )
        result = await quota_service.check_quota(
            tenant_id, estimated_cost=estimated_cost, tenant_funded=slot is not None
        )
        if not result.allowed:
            raise HTTPException(status_code=429, detail=result.reason)

    try:
        job_id = await service.create_job(
            db,
            tenant_id=tenant_id,
            prompt=body.prompt,
            provider=provider,
            negative_prompt=body.negative_prompt,
            style=body.style,
            size=body.size,
            quality=body.quality,
            reference_image_urls=body.reference_image_urls,
            usage_context=body.usage_context,
            target_usage=body.target_usage,
            extra_params=body.extra_params,
            callback_url=body.callback_url,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=fingerprint if idempotency_key is not None else None,
        )
    except IdempotentReplayError as replay:
        # Lost a race to a concurrent duplicate: our INSERT wrote nothing.
        return _replayed_image_submission(cast(ImageGenJob, replay.job), fingerprint, response)
    if slot is not None:
        provider_slot, sealed = seal_slot(slot, job_id=job_id)
        await ImageJobManager.attach_provider_slot(
            db, job_id, provider_slot=provider_slot, provider_secret_sealed=sealed
        )
    settings = get_settings()
    # Durable dispatch: queue-own + lease so the reaper can reclaim on pod crash.
    await ImageJobManager.register_dispatch(
        db,
        job_id,
        options=build_image_single_options(budget_limit=5.0),
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
            renew=ImageJobManager.renew_lease,
            work=_process_concept_image_job_bg(
                service,
                job_id,
                tenant_id,
                budget_limit=5.0,
                quota_service=quota_service,
            ),
            heartbeat_enabled=settings.dispatch.heartbeat_enabled,
        )
    )
    service.register_task(job_id, task)

    return ConceptImageJobSubmissionResponse(
        job_id=job_id,
        status="pending",
        poll_url=_job_status_path(job_id),
        result_url=_job_result_path(job_id),
    )


@router.post("/batches", operation_id="create_concept_image_batch")
async def create_concept_image_batch(
    body: ConceptImageBatchRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ConceptImageBatchSubmissionResponse:
    tenant_id = _get_tenant_id(request)
    service = _get_imagegen_service(request)

    if provider_slots_enforced_or_503():
        # enforce: item ``provider_slot``s are ignored; the RUN's slot decides.
        reject_tenant_slot_on_batch(await resolve_run_image_slot())
        for item in body.items:
            _validate_generate_image_request(request, item, slot=None)
    else:
        for item in body.items:
            reject_slot_on_batch(item.provider_slot)
            _validate_generate_image_request(request, item)

    batch = await ImageBatchManager.create_batch(
        db,
        tenant_id=tenant_id,
        budget_limit=body.budget_limit,
        job_count=len(body.items),
    )

    job_ids: list[str] = []
    for item in body.items:
        job_id = await service.create_job(
            db,
            tenant_id=tenant_id,
            prompt=item.prompt,
            provider=item.provider or body.provider,
            batch_id=batch.batch_id,
            negative_prompt=item.negative_prompt,
            style=item.style,
            size=item.size,
            quality=item.quality,
            reference_image_urls=item.reference_image_urls,
            usage_context=item.usage_context,
            target_usage=item.target_usage,
            extra_params=item.extra_params,
            callback_url=item.callback_url or body.callback_url,
        )
        job_ids.append(job_id)

    settings = get_settings()
    quota_service = _get_quota_service(request)
    # Durable dispatch: route each batch CHILD through the claim-queue (queue-own +
    # lease), exactly like the video batch. Each child is an independently leased,
    # cross-pod-reclaimable job — if this pod dies mid-batch, the reaper resumes the
    # unfinished children on another replica instead of leaving them wedged in
    # ``pending`` under a single fire-and-forget orchestrator task.
    for job_id in job_ids:
        await ImageJobManager.register_dispatch(
            db,
            job_id,
            options=build_image_batch_options(
                batch_id=batch.batch_id, budget_limit=body.budget_limit
            ),
            worker_id=pod_id(),
            lease_seconds=settings.dispatch.lease_seconds,
        )

    await ImageBatchManager.mark_processing(db, batch.batch_id)
    await db.commit()

    for job_id in job_ids:
        task = asyncio.create_task(
            run_with_lease(
                job_id=job_id,
                lease_seconds=settings.dispatch.lease_seconds,
                renew=ImageJobManager.renew_lease,
                work=_process_concept_image_batch_job_bg(
                    service,
                    job_id,
                    tenant_id,
                    batch.batch_id,
                    budget_limit=body.budget_limit,
                    quota_service=quota_service,
                ),
                heartbeat_enabled=settings.dispatch.heartbeat_enabled,
            )
        )
        service.register_task(job_id, task)

    return ConceptImageBatchSubmissionResponse(
        batch_id=batch.batch_id,
        job_ids=job_ids,
        status="pending",
        status_url=_batch_status_path(batch.batch_id),
        result_url=_batch_result_path(batch.batch_id),
    )


@router.post(
    "/jobs/{job_id}/cancel",
    operation_id="cancel_concept_image_job",
    responses={
        409: {"description": "Job cannot be cancelled"},
    },
)
async def cancel_concept_image_job(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Cancel a pending or in-flight concept-image job.

    Mirrors ``POST /api/content/videos/jobs/{id}/cancel``: a cross-pod Redis
    flag plus the local task / DB transition. A foreign or unknown job and an
    already-terminal job all get the same 409, so cancel never confirms another
    tenant's job exists. A cancelled job releases its ``Idempotency-Key``.
    """
    tenant_id = _get_tenant_id(request)
    service = _get_imagegen_service(request)
    if not await service.cancel_job(db, job_id, tenant_id):
        logger.warning("Concept-image cancel rejected: job_id=%s (not cancellable)", job_id)
        raise HTTPException(status_code=409, detail="Job cannot be cancelled")
    logger.info("Concept-image job cancelled: job_id=%s", job_id)
    return {"job_id": job_id, "status": "cancelled"}


@router.get(
    "/jobs/{job_id}",
    operation_id="get_concept_image_job",
    responses={
        404: {"description": "Job not found"},
    },
)
async def get_concept_image_job(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ConceptImageJobResponse:
    redis = _get_redis(request)
    tenant_id = _get_tenant_id(request)
    # Must match ImageJobManager._invalidate_cache's key (``content:image:...``)
    # or state transitions never evict this entry.
    cache_key = f"content:image:{job_id}:status"
    storage = _get_storage(request)
    if redis:
        try:
            cached = await redis.get(cache_key)
            # The cache KEY is not tenant-scoped, only the cached VALUE is
            # tenant-tagged — so a foreign tenant never hits this branch and
            # always falls through to the DB read below, warm or cold. That
            # read is now tenant-scoped in the application layer (RLS is the
            # backstop underneath it, no longer the only layer).
            if isinstance(cached, dict) and cached.get("tenant_id") == tenant_id:
                return ConceptImageJobResponse(**cached["response"])
        except OSError:
            logger.debug("Redis cache operation failed", exc_info=True)

    job = await ImageJobManager.get_job(db, job_id, tenant_id=tenant_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    response = _serialize_concept_image_job(storage, job)

    if redis:
        try:
            await redis.set(
                cache_key,
                {"tenant_id": tenant_id, "response": response.model_dump()},
                ttl=30,
            )
        except OSError:
            logger.debug("Redis cache operation failed", exc_info=True)

    return response


@router.get(
    "/jobs/{job_id}/result",
    operation_id="get_concept_image_job_result",
    responses={
        404: {"description": "Job not found"},
    },
)
async def get_concept_image_job_result(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ConceptImageAssetResponse:
    job = await ImageJobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job not completed (status: {job.status})")

    storage = _get_storage(request)
    return _serialize_concept_image_asset(storage, job)


@router.get("/batches/{batch_id}", operation_id="get_concept_image_batch")
async def get_concept_image_batch(
    batch_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ConceptImageBatchResponse:
    tenant_id = _get_tenant_id(request)
    batch = await ImageBatchManager.get_batch(db, batch_id, tenant_id=tenant_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    jobs = await ImageBatchManager.get_batch_jobs(db, batch_id, tenant_id=tenant_id)
    storage = _get_storage(request)

    return ConceptImageBatchResponse(
        batch_id=batch.batch_id,
        status=batch.status,
        job_ids=[job.job_id for job in jobs],
        job_count=batch.job_count,
        completed_jobs=batch.completed_jobs,
        failed_jobs=batch.failed_jobs,
        pending_jobs=max(0, batch.job_count - batch.completed_jobs - batch.failed_jobs),
        budget_limit=batch.budget_limit,
        total_cost=batch.total_cost,
        created_at=batch.created_at.isoformat() if batch.created_at else None,
        completed_at=batch.completed_at.isoformat() if batch.completed_at else None,
        results=[
            _serialize_concept_image_asset(storage, job)
            for job in jobs
            if job.status in {"completed", "failed", "cancelled"}
        ],
    )


@router.get("/batches/{batch_id}/result", operation_id="get_concept_image_batch_result")
async def get_concept_image_batch_result(
    batch_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ConceptImageBatchResponse:
    response = await get_concept_image_batch(batch_id, request, db)
    if response.status not in {"completed", "partial", "failed"}:
        raise HTTPException(
            status_code=409,
            detail=f"Batch is not finished (status: {response.status})",
        )
    return response


@router.get(
    "/assets/{asset_id}",
    operation_id="get_concept_image_asset",
    responses={404: {"description": "Asset not found"}},
)
async def get_concept_image_asset(
    asset_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ConceptImageAssetResponse:
    job = await ImageJobManager.get_job_by_asset_id(db, asset_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")

    storage = _get_storage(request)
    return _serialize_concept_image_asset(storage, job)


@router.get(
    "/assets/{asset_id}/download",
    operation_id="download_concept_image_asset",
    responses={404: {"description": "Asset not found"}, 409: {"description": "Asset not ready"}},
)
async def download_concept_image_asset(
    asset_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    job = await ImageJobManager.get_job_by_asset_id(db, asset_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Asset not ready (status: {job.status})")

    # Always stream the bytes through this service — browsers cannot reach
    # minio.test:8389. Matches the org-wide stream-through pattern
    # (cf. artifact_service:download_artifact).
    storage = _get_storage(request)
    storage_key = job.minio_key or job.image_path
    if not storage_key:
        raise HTTPException(status_code=404, detail="Asset data unavailable")
    data = await storage.retrieve(storage_key)
    if data is None:
        raise HTTPException(status_code=404, detail="Asset file not found on disk")

    return Response(
        content=data,
        media_type=job.mime_type or "image/png",
        headers={"Content-Disposition": f'inline; filename="{asset_id}"'},
    )


async def _deliver_concept_image_callback(
    *,
    db: AsyncSession,
    job_id: str,
    callback_url: str | None,
    storage: Any,
) -> None:
    if not callback_url:
        return
    try:
        from src.services.callback_service import deliver_callback

        job = await ImageJobManager.get_job(db, job_id)
        if job is None:
            return
        payload = _serialize_concept_image_asset(storage, job).model_dump()
        await deliver_callback(db, job_id, callback_url, payload)
    except Exception:
        logger.exception("Callback delivery failed for concept-image job %s", job_id)


async def _process_concept_image_job_bg(
    service: Any,
    job_id: str,
    tenant_id: str,
    *,
    budget_limit: float,
    quota_service: Any,
    tenant_source: TenantSource | _Keep | None = KEEP_TENANT_SOURCE,
) -> None:
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
    # create_concept_image() already called record_generation_start(tenant, job_id),
    # so this task OWES exactly one release. ``cost`` must exist on every failure
    # path or the release in ``finally`` would NameError and leak the slot.
    #
    # ⚠️ Also entered from ``ReaperService._dispatch_image`` on a reclaim, where no
    # route ran — see the twin note in routes/generation.py::_process_job_bg. The
    # release is safe there only because it names ``job_id`` and is idempotent.
    cost = 0.0
    try:
        async with get_db_session() as db:
            job = await ImageJobManager.get_job(db, job_id)
            callback_url = job.callback_url if job else None
            cost = await service.process_job(
                db,
                job_id,
                tenant_id,
                budget_limit=budget_limit,
            )
            storage = service._storage
            await _deliver_concept_image_callback(
                db=db,
                job_id=job_id,
                callback_url=callback_url,
                storage=storage,
            )

    except asyncio.CancelledError:
        logger.info("Concept-image job %s was cancelled", job_id)
    except Exception as exc:
        # Catch-all so an unexpected error (e.g. an httpx error raised by the
        # OpenAI image provider, which is not wrapped in process_job) can never
        # leave the job wedged in ``processing`` — always mark it failed.
        logger.exception("Background concept-image job %s failed", job_id)
        try:
            async with get_db_session() as db:
                await ImageJobManager.mark_failed(
                    db,
                    job_id,
                    f"Unexpected error during concept image generation: {exc}",
                )
        except JobStateConflictError:
            logger.info("Concept-image job %s already terminal; skipping mark_failed", job_id)
        except OSError, SQLAlchemyError:
            logger.exception("Failed to mark concept-image job %s as failed", job_id)
    finally:
        service.unregister_task(job_id)
        # Release the tenant concurrency slot on EVERY exit path. Inside the
        # ``try`` it was skipped by any exception or CancelledError, leaking a
        # slot until its 1h TTL. Best-effort like callback delivery: never
        # re-raise out of ``finally`` (that would mask the original exception).
        if quota_service is not None:
            try:
                await quota_service.record_generation_complete(tenant_id, job_id, cost=cost)
            except Exception:
                logger.exception("Failed to release quota slot for concept-image job %s", job_id)


async def _check_quota_for_batch_job(
    db: Any,
    service: Any,
    job: Any,
    tenant_id: str,
    batch_id: str,
    quota_service: Any,
) -> bool:
    """Returns True if quota allows the job; False if rejected (and marks failure)."""
    if quota_service is None:
        return True
    estimated_cost = service.estimate_cost(job.provider, size=job.size, quality=job.quality) or 0.0
    quota_check = await quota_service.check_quota(tenant_id, estimated_cost=estimated_cost)
    if not quota_check.allowed:
        try:
            await ImageJobManager.mark_failed(
                db, job.job_id, quota_check.reason or "Quota exceeded"
            )
        except JobStateConflictError:
            logger.info("Concept-image job %s already terminal; quota path", job.job_id)
        await ImageBatchManager.record_job_failed(db, batch_id)
        await db.commit()
        return False
    await quota_service.record_generation_start(tenant_id, job.job_id)
    return True


async def _execute_batch_job(
    db: Any,
    service: Any,
    job: Any,
    tenant_id: str,
    batch_id: str,
    *,
    budget_limit: float,
    spent_so_far: float,
) -> tuple[float, float]:
    """Execute a single batch job. Returns (new_spent_so_far, cost)."""
    cost = await service.process_job(
        db,
        job.job_id,
        tenant_id,
        budget_limit=budget_limit,
        spent_so_far=spent_so_far,
    )
    refreshed_job = await ImageJobManager.get_job(db, job.job_id)
    if refreshed_job and refreshed_job.status == "completed":
        spent_so_far += cost
        await ImageBatchManager.record_job_completed(db, batch_id, cost=cost)
    else:
        await ImageBatchManager.record_job_failed(db, batch_id)
    await _deliver_concept_image_callback(
        db=db,
        job_id=job.job_id,
        callback_url=refreshed_job.callback_url if refreshed_job else None,
        storage=service._storage,
    )
    await db.commit()
    return spent_so_far, cost


async def _handle_batch_job_exception(db: Any, job: Any, batch_id: str) -> None:
    """Mark a failed batch child failed and account it against the batch.

    Deliberately does NOT release the quota slot: the single release now lives in
    ``_process_concept_image_batch_job_bg``'s ``finally``, gated on whether
    ``record_generation_start`` actually ran. Releasing here too would
    double-decrement the concurrency gauge (and it fired even on paths where no
    start had been recorded, handing the tenant free concurrency).
    """
    try:
        await ImageJobManager.mark_failed(
            db, job.job_id, "Unexpected error during concept image generation"
        )
    except JobStateConflictError:
        logger.info("Concept-image job %s already terminal; batch-exc path", job.job_id)
    await ImageBatchManager.record_job_failed(db, batch_id)
    await db.commit()


async def _process_concept_image_batch_job_bg(
    service: Any,
    job_id: str,
    tenant_id: str,
    batch_id: str,
    *,
    budget_limit: float,
    quota_service: Any,
    tenant_source: TenantSource | _Keep | None = KEEP_TENANT_SOURCE,
) -> None:
    """Process ONE child of a concept-image batch under its own lease.

    The batch analog of the single-job ``_process_concept_image_job_bg`` and a
    mirror of the video batch's ``_process_batch_job_bg``: one leased job per
    child, reclaimable by the reaper on any replica after a pod crash. Batch
    progress is recorded via ``ImageBatchManager`` (atomic cross-pod counters).

    Note: like the single-image and video paths, this inherits the reaper's
    at-least-once side-effect semantics — a reclaim after a crash mid-generation
    can re-run generate/upload/cost-emit for that child. Exactly-once (dedup on
    generate/upload/cost-event) is a separate, out-of-scope hardening.
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
    # Unlike the single-job paths, the start is recorded HERE (inside
    # _check_quota_for_batch_job), not by the route — and only on the
    # quota-allowed path. So the release must be gated on ``quota_started``:
    # an unconditional release on the vanished-job / quota-rejected paths would
    # decrement a slot that was never taken. ``cost`` is pre-bound for the same
    # NameError reason as elsewhere.
    quota_started = False
    cost = 0.0
    try:
        async with get_db_session() as db:
            job = await ImageJobManager.get_job(db, job_id)
            if job is None:
                logger.warning("Concept-image batch job %s vanished before processing", job_id)
                return
            if not await _check_quota_for_batch_job(
                db, service, job, tenant_id, batch_id, quota_service
            ):
                return
            quota_started = True
            # Approximate the batch's remaining budget from the durable running
            # total (cross-pod authoritative), mirroring the video batch's
            # ``spent = batch.total_cost``.
            batch = await ImageBatchManager.get_batch(db, batch_id)
            spent_so_far = batch.total_cost if batch else 0.0
            _, cost = await _execute_batch_job(
                db,
                service,
                job,
                tenant_id,
                batch_id,
                budget_limit=budget_limit,
                spent_so_far=spent_so_far,
            )
    except asyncio.CancelledError:
        logger.info("Concept-image batch job %s was cancelled", job_id)
    except Exception:
        logger.exception("Background concept-image batch job %s failed", job_id)
        try:
            async with get_db_session() as db:
                job = await ImageJobManager.get_job(db, job_id)
                if job is not None:
                    await _handle_batch_job_exception(db, job, batch_id)
        except OSError, SQLAlchemyError:
            logger.exception("Failed to record failure for concept-image batch job %s", job_id)
    finally:
        service.unregister_task(job_id)
        # The ONE release, on every exit path, only if a start was recorded.
        # Best-effort: never re-raise out of ``finally``.
        if quota_service is not None and quota_started:
            try:
                await quota_service.record_generation_complete(tenant_id, job_id, cost=cost)
            except Exception:
                logger.exception(
                    "Failed to release quota slot for concept-image batch job %s", job_id
                )
