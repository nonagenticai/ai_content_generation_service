"""Usage and quota endpoints: GET /usage, GET /quotas/{tenant_id}."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.dependencies import _get_quota_service, _get_tenant_id
from src.api.schemas import (
    QuotaResponse,
    UsageByProviderItem,
    UsagePeriodItem,
    UsageResponse,
)
from src.database.session import get_db
from src.services.job_manager import JobManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/usage", operation_id="get_video_usage")
async def get_usage(
    request: Request,
    db: AsyncSession = Depends(get_db),
    date_from: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    group_by: str = Query("day", description="Group by: day, week, month"),
) -> UsageResponse:
    """Get usage dashboard data.

    Args:
        request: FastAPI request with tenant context.
        db: Async database session.
        date_from: Optional start date (YYYY-MM-DD).
        date_to: Optional end date (YYYY-MM-DD).
        group_by: Grouping period (day, week, month).

    Returns:
        UsageResponse with period breakdown and cost totals.
    """
    tenant_id = _get_tenant_id(request)
    logger.debug("Usage request: tenant=%s from=%s to=%s", tenant_id, date_from, date_to)
    usage_rows, provider_breakdown = await JobManager.get_usage(
        db,
        tenant_id,
        date_from=date_from,
        date_to=date_to,
    )

    periods = [
        UsagePeriodItem(
            date=row.date,
            generations_started=row.generations_started or 0,
            generations_completed=row.generations_completed or 0,
            generations_failed=row.generations_failed or 0,
            total_duration_generated_s=row.total_duration_generated_s or 0.0,
            total_file_size_bytes=row.total_file_size_bytes or 0,
        )
        for row in usage_rows
    ]

    by_provider = [
        UsageByProviderItem(
            provider=prov,
            total_cost=float(cost or 0.0),
            job_count=int(count),
        )
        for prov, cost, count in provider_breakdown
    ]

    total_cost = sum(p.total_cost for p in by_provider)
    total_gens = sum(p.generations_completed for p in periods)

    return UsageResponse(
        tenant_id=UUID(tenant_id),
        date_from=date_from,
        date_to=date_to,
        periods=periods,
        by_provider=by_provider,
        total_cost=total_cost,
        total_generations=total_gens,
    )


# ============================================================================
# Quota Endpoint
# ============================================================================


@router.get(
    "/quotas/{tenant_id}",
    operation_id="get_video_quota_status",
    responses={
        403: {"description": "Requested tenant is not the caller's tenant"},
        501: {"description": "Quota service not enabled"},
    },
)
async def get_quota_status(tenant_id: str, request: Request) -> QuotaResponse:
    """Get quota status for the calling tenant.

    The tenant is still a path parameter for contract stability, but it is
    now only ever an ASSERTION about the caller's own identity: the id in the
    URL must equal the request identity, and the quota service is not even
    consulted otherwise. Quota state is Redis-backed and every tenant id
    synthesises a status, so there is no RLS backstop and no natural
    "not found" -- without this check the route returned another tenant's
    ``daily_spend_used`` and limits at HTTP 200 to anyone who could name them.

    The refusal is a **403**, not the 404 used by the by-id job/asset/batch
    reads. Those return 404 to avoid an existence oracle over unguessable
    ids; here the refusal is identical for every foreign id -- existing or
    not -- so there is nothing to disclose, and 403 states the truth
    (authenticated, not authorised) rather than inventing a missing resource.
    A super-admin exemption, if it is ever wanted, belongs exactly here.

    Args:
        tenant_id: Tenant identifier. Must match the caller's own tenant.
        request: FastAPI request with tenant context.

    Returns:
        QuotaResponse with usage and limit data.

    Raises:
        HTTPException: 403 if ``tenant_id`` is not the caller's tenant,
            501 if quota service not enabled.
    """
    caller_tenant = _get_tenant_id(request)
    if tenant_id != caller_tenant:
        logger.warning(
            "Cross-tenant quota read refused: caller=%s requested=%s",
            caller_tenant,
            tenant_id,
        )
        raise HTTPException(
            status_code=403,
            detail="Quota is readable only for your own tenant",
        )

    quota_service = _get_quota_service(request)
    if quota_service is None:
        raise HTTPException(status_code=501, detail="Quota service not enabled")

    status = await quota_service.get_quota_status(caller_tenant)
    return QuotaResponse(**status)
