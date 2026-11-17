"""Batch status endpoints: batch status, batch result."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.dependencies import _get_batch_tracker, _get_tenant_id
from src.api.schemas import (
    BatchResultResponse,
    BatchStatusResponse,
    ClipStatusItem,
    CostReport,
)
from src.database.session import get_db
from src.services.job_manager import JobManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/batches/{batch_id}/status",
    operation_id="get_video_batch_status",
    responses={
        404: {"description": "Batch not found"},
    },
)
async def get_batch_status(
    batch_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> BatchStatusResponse:
    """Get per-clip status, overall progress, and cost report for a batch.

    Args:
        batch_id: Batch identifier.
        request: FastAPI request.
        db: Async database session.

    Returns:
        BatchStatusResponse with clip statuses and cost report.

    Raises:
        HTTPException: 404 if batch not found.
    """
    logger.debug("Batch status request: batch_id=%s", batch_id)
    tenant_id = _get_tenant_id(request)
    tracker = _get_batch_tracker(request)
    batch = await tracker.get_batch(batch_id, tenant_id=tenant_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    clips: list[ClipStatusItem] = []
    for jid in batch.job_ids:
        # A clip that does not resolve for this tenant is SKIPPED, not fatal:
        # the batch itself is already proven to belong to the caller, so a
        # stray/foreign job id must not 404 the whole (otherwise legitimate)
        # status response — it just does not appear.
        job = await JobManager.get_job(db, jid, tenant_id=tenant_id)
        if job:
            clips.append(
                ClipStatusItem(
                    job_id=jid,
                    status=job.status,
                    provider=job.provider,
                    cost=job.cost,
                    error=job.error_message,
                )
            )

    cost_report = CostReport(
        total_cost=batch.total_cost,
        budget_limit=batch.budget_limit,
        clips_generated=batch.clips_completed,
        clips_skipped=batch.clips_skipped,
        clips_failed=batch.clips_failed,
    )

    return BatchStatusResponse(
        batch_id=batch.batch_id,
        status=batch.computed_status,
        clips_submitted=batch.clips_submitted,
        clips_completed=batch.clips_completed,
        clips_failed=batch.clips_failed,
        clips_skipped=batch.clips_skipped,
        clips_cancelled=batch.clips_cancelled,
        total_cost=batch.total_cost,
        budget_limit=batch.budget_limit,
        clips=clips,
        cost_report=cost_report,
    )


@router.get(
    "/batches/{batch_id}/result",
    operation_id="get_video_batch_result",
    responses={
        404: {"description": "Batch not found"},
        409: {"description": "Batch still processing"},
    },
)
async def get_batch_result(
    batch_id: str,
    request: Request,
) -> BatchResultResponse:
    """Get final results of a batch job (409 if still processing).

    Args:
        batch_id: Batch identifier.
        request: FastAPI request.

    Returns:
        BatchResultResponse with completion data and cost report.

    Raises:
        HTTPException: 404 if batch not found, 409 if still processing.
    """
    tracker = _get_batch_tracker(request)
    batch = await tracker.get_batch(batch_id, tenant_id=_get_tenant_id(request))
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    status = batch.computed_status
    if status in ("pending", "processing"):
        raise HTTPException(
            status_code=409,
            detail=f"Batch still processing (status: {status})",
            headers={"Retry-After": "30"},
        )

    cost_report = CostReport(
        total_cost=batch.total_cost,
        budget_limit=batch.budget_limit,
        clips_generated=batch.clips_completed,
        clips_skipped=batch.clips_skipped,
        clips_failed=batch.clips_failed,
    )

    return BatchResultResponse(
        batch_id=batch.batch_id,
        status=status,
        job_ids=batch.job_ids,
        clips_completed=batch.clips_completed,
        clips_failed=batch.clips_failed,
        total_cost=batch.total_cost,
        cost_report=cost_report,
    )
