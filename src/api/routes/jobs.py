"""Job status endpoints: list, detail, result, costs, video, cancel."""

from __future__ import annotations

import logging
import math
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.dependencies import (
    _get_redis,
    _get_storage,
    _get_tenant_id,
    _get_video_generation_service,
)
from src.api.schemas import (
    CostReport,
    GenerationResultResponse,
    JobListItem,
    JobListResponse,
    JobStatusResponse,
)
from src.constants import CACHE_TTL_JOB_STATUS_S
from src.database.session import get_db
from src.services.job_manager import JobManager

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Job Cancellation
# ============================================================================


@router.post(
    "/jobs/{job_id}/cancel",
    operation_id="cancel_video_job",
    responses={
        409: {"description": "Job cannot be cancelled"},
    },
)
async def cancel_job(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Cancel an in-flight or pending job.

    Args:
        job_id: Job identifier to cancel.
        request: FastAPI request with tenant context.
        db: Async database session.

    Returns:
        Dict with job_id and cancelled status.

    Raises:
        HTTPException: 409 if job cannot be cancelled. A job belonging to
            another tenant is indistinguishable from an unknown job — both
            reach this same 409, so cancel never confirms existence.
    """
    tenant_id = _get_tenant_id(request)
    logger.info("Cancel request received: job_id=%s tenant=%s", job_id, tenant_id)
    service = _get_video_generation_service(request)
    cancelled = await service.cancel_job(db, job_id, tenant_id)
    if not cancelled:
        logger.warning("Cancel rejected: job_id=%s (not cancellable)", job_id)
        raise HTTPException(status_code=409, detail="Job cannot be cancelled")
    logger.info("Job cancelled: job_id=%s", job_id)
    return {"job_id": job_id, "status": "cancelled"}


# ============================================================================
# Job Status Endpoints
# ============================================================================


@router.get("/jobs", operation_id="list_video_jobs")
async def list_jobs(
    request: Request,
    db: AsyncSession = Depends(get_db),
    status: str | None = Query(None, description="Filter by status"),
    provider: str | None = Query(None, description="Filter by provider"),
    created_after: str | None = Query(None, description="Filter: created after (ISO)"),
    created_before: str | None = Query(None, description="Filter: created before (ISO)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
) -> JobListResponse:
    """List jobs with filtering and pagination.

    Args:
        request: FastAPI request with tenant context.
        db: Async database session.
        status: Optional status filter.
        provider: Optional provider filter.
        created_after: Optional ISO datetime lower bound.
        created_before: Optional ISO datetime upper bound.
        page: Page number (1-indexed).
        page_size: Items per page.
        sort_by: Column to sort by.
        sort_order: Sort direction (asc/desc).

    Returns:
        JobListResponse with paginated results.
    """
    tenant_id = _get_tenant_id(request)
    logger.debug("List jobs request: tenant=%s status=%s page=%d", tenant_id, status, page)
    jobs, total = await JobManager.list_jobs(
        db,
        tenant_id,
        status=status,
        provider=provider,
        created_after=created_after,
        created_before=created_before,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    items = [
        JobListItem(
            job_id=j.job_id,
            status=j.status,
            provider=j.provider,
            created_at=j.created_at.isoformat() if j.created_at else None,
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
            cost=j.cost,
        )
        for j in jobs
    ]

    total_pages = max(1, math.ceil(total / page_size))

    return JobListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/jobs/{job_id}",
    operation_id="get_video_job_status",
    responses={
        404: {"description": "Job not found"},
    },
)
async def get_job_status(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    """Get the current status of a video generation job.

    Args:
        job_id: Job identifier.
        request: FastAPI request.
        db: Async database session.

    Returns:
        JobStatusResponse with current status and metadata.

    Raises:
        HTTPException: 404 if job not found.
    """
    redis = _get_redis(request)
    tenant_id = _get_tenant_id(request)
    cache_key = f"content_generation:job:{job_id}:status"

    if redis:
        try:
            cached = await redis.get(cache_key)
            # The cache KEY is not tenant-scoped, only the cached VALUE is
            # tenant-tagged — so a foreign tenant never hits this branch and
            # always falls through to the DB read below, warm or cold. That
            # read is now tenant-scoped in the application layer (RLS is the
            # backstop underneath it, no longer the only layer).
            if isinstance(cached, dict) and cached.get("tenant_id") == tenant_id:
                return JobStatusResponse(**cached["response"])
        except OSError:
            logger.debug("Redis cache operation failed", exc_info=True)

    job = await JobManager.get_job(db, job_id, tenant_id=tenant_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    result_url = f"/api/content/videos/jobs/{job_id}/result" if job.status == "completed" else None

    response = JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        provider=job.provider,
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error_message,
        degradation=job.degradation,
        result_url=result_url,
    )

    # Cache for short TTL since status changes during processing
    if redis:
        try:
            await redis.set(
                cache_key,
                {"tenant_id": tenant_id, "response": response.model_dump()},
                ttl=CACHE_TTL_JOB_STATUS_S,
            )
        except OSError:
            logger.debug("Redis cache operation failed", exc_info=True)

    return response


@router.get(
    "/jobs/{job_id}/result",
    operation_id="get_video_job_result",
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job not completed yet"},
    },
)
async def get_job_result(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> GenerationResultResponse:
    """Get the generation result for a completed job.

    Args:
        job_id: Job identifier.
        request: FastAPI request.
        db: Async database session.

    Returns:
        GenerationResultResponse with video path, cost, and metadata.

    Raises:
        HTTPException: 404 if not found, 409 if not completed.
    """
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job not completed (status: {job.status})",
            headers={"Retry-After": "30"},
        )

    return GenerationResultResponse(
        job_id=job.job_id,
        status=job.status,
        provider=job.provider,
        prompt=job.prompt,
        video_path=job.video_path,
        video_url=_preferred_video_url(job),
        file_size_bytes=job.file_size_bytes,
        actual_duration_s=job.actual_duration_s,
        generation_time_seconds=job.generation_time_seconds,
        cost=job.cost,
        error=job.error_message,
        degradation=job.degradation,
    )


def _video_download_path(job_id: str) -> str:
    return f"/api/content/videos/jobs/{job_id}/video"


class _StoredVideoJob(Protocol):
    """The read-only surface ``_preferred_video_url`` needs from a job row."""

    job_id: str
    video_url: str | None
    minio_key: str | None
    video_path: str | None


def _preferred_video_url(job: _StoredVideoJob) -> str | None:
    """Pick a URL a CONSUMER can actually fetch.

    Video sibling of ``content_routes._preferred_asset_url``. When this service
    holds the bytes (MinIO object or local artifact) the canonical answer is our
    own stream-through path: it is license-gated, and the orchestrator's
    ``ai_video_gen._republish_relative`` re-publishes any ``/``-prefixed URL into
    the run's artifact store, which is what makes the clip reachable by the
    anonymous video-compile service.

    ``job.video_url`` is only ever a fallback, and by then it has already been
    filtered by ``providers.base.publishable_video_url`` -- so it is either a real
    ``http(s)`` provider CDN URL or ``None``. ``None`` is the deliberate loud
    state: the orchestrator's post-gather guard turns a missing video_url into a
    hard stage failure, instead of the previous silent shot-drop.
    """
    if job.minio_key or job.video_path:
        return _video_download_path(job.job_id)
    return job.video_url


@router.get(
    "/jobs/{job_id}/costs",
    operation_id="get_video_job_costs",
    responses={
        404: {"description": "Job not found"},
    },
)
async def get_job_costs(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CostReport:
    """Get cost data for a specific job.

    Args:
        job_id: Job identifier.
        request: FastAPI request.
        db: Async database session.

    Returns:
        CostReport with per-clip cost details.

    Raises:
        HTTPException: 404 if job not found.
    """
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    from src.api.schemas import ClipCostItem

    per_clip = []
    if job.cost is not None:
        per_clip.append(
            ClipCostItem(
                job_id=job.job_id,
                provider=job.provider,
                duration_s=job.duration_s or 0.0,
                cost=job.cost,
                status=job.status,
            )
        )

    return CostReport(
        total_cost=job.cost or 0.0,
        clips_generated=1 if job.status == "completed" else 0,
        clips_failed=1 if job.status == "failed" else 0,
        per_clip_costs=per_clip,
    )


@router.get(
    "/jobs/{job_id}/video",
    operation_id="download_video_job_video",
    responses={
        404: {"description": "Job or video not found"},
    },
)
async def get_job_video(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download the generated video clip.

    Args:
        job_id: Job identifier.
        request: FastAPI request.
        db: Async database session.

    Returns:
        Video file as binary response or redirect to MinIO presigned URL.

    Raises:
        HTTPException: 404 if job or video not found.
    """
    job = await JobManager.get_job(db, job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    storage = _get_storage(request)

    # Try MinIO presigned URL first
    if job.minio_key:
        presigned = await storage.presign_url(job.minio_key)
        if presigned:
            return RedirectResponse(url=presigned, status_code=307)

    if not job.video_path:
        raise HTTPException(status_code=404, detail="No video available")

    data = await storage.retrieve(job.video_path)
    if data is None:
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    return Response(
        content=data,
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{job.provider}_clip.mp4"'},
    )
