"""Provider endpoints: list providers, provider health."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from src.api.routes.dependencies import _get_redis, _get_registry
from src.api.schemas import (
    ProviderHealthResponse,
    ProviderInfoResponse,
    ProvidersListResponse,
)
from src.constants import CACHE_TTL_PROVIDERS_S

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/providers", operation_id="list_video_providers")
async def list_providers(request: Request) -> ProvidersListResponse:
    """List available AI video generation providers (cached 60s).

    Args:
        request: FastAPI request.

    Returns:
        ProvidersListResponse with available providers and default provider.
    """
    logger.debug("List providers request received")
    from src.config import get_settings

    redis = _get_redis(request)
    cache_key = "content_generation:providers"

    if redis:
        try:
            cached = await redis.get(cache_key)
            if cached:
                return ProvidersListResponse(**cached)
        except OSError:
            logger.debug("Redis cache operation failed", exc_info=True)

    registry = _get_registry(request)
    providers = [
        ProviderInfoResponse(
            name=info.name,
            available=info.available,
            models=info.models,
            styles=info.styles,
            max_duration_s=info.max_duration_s,
            supports_image_to_video=info.supports_image_to_video,
        )
        for info in registry.list_providers()
    ]
    result = ProvidersListResponse(
        providers=providers,
        default_provider=get_settings().generation.default_provider,
    )

    if redis:
        try:
            await redis.set(cache_key, result, ttl=CACHE_TTL_PROVIDERS_S)
        except OSError:
            logger.debug("Redis cache operation failed", exc_info=True)

    return result


@router.get(
    "/providers/{name}/health",
    operation_id="get_video_provider_health",
    responses={
        404: {"description": "Unknown provider"},
        503: {"description": "Health monitoring not enabled"},
    },
)
async def get_provider_health(name: str, request: Request) -> ProviderHealthResponse:
    """Get health status for a specific provider.

    Args:
        name: Provider name.
        request: FastAPI request.

    Returns:
        ProviderHealthResponse with health state.

    Raises:
        HTTPException: 404 if provider unknown, 503 if health monitoring disabled.
    """
    logger.debug("Provider health request: provider=%s", name)
    # Validate provider is registered
    registry = _get_registry(request)
    if registry.get(name) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown provider: {name}. Available: {registry.provider_names}",
        )

    health_monitor = getattr(request.app.state, "health_monitor", None)
    if health_monitor is None:
        raise HTTPException(status_code=503, detail="Health monitoring not enabled")

    state = health_monitor.get_health(name)
    return ProviderHealthResponse(
        provider=name,
        is_healthy=state.is_healthy,
        consecutive_failures=state.consecutive_failures,
        average_latency_ms=state.average_latency_ms,
        disabled_at=str(state.disabled_at) if state.disabled_at else None,
        cooldown_until=str(state.cooldown_until) if state.cooldown_until else None,
    )
