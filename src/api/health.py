"""Health check endpoints for ai_content_generation_service."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.cache.redis_client import get_redis_client
from src.constants import CACHE_TTL_HEALTH_CHECK_S
from src.database.connection import get_engine
from src.services.exceptions import format_error

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_health_cache: dict[str, Any] | JSONResponse | None = None
_health_cache_time: float = 0.0
_ready_cache: dict[str, Any] | JSONResponse | None = None
_ready_cache_time: float = 0.0


async def _probe_storage(request: Request) -> str:
    """Probe the storage the APP actually uses — not a fresh, uninitialised one.

    This used to do ``StorageService()`` per request. ``MinioBackend.__init__`` sets
    ``_minio_client = None``, and only ``initialize()`` — called once at startup by
    ``asgi._init_storage`` — builds it; ``check_health`` returns False when the client
    is None. So a brand-new instance **always** reported unhealthy, whatever MinIO was
    actually doing: the probe measured "has this object been initialised", never "is
    the object store reachable".

    It went unnoticed for as long as ``storage_status`` was excluded from ``overall``.
    Folding it into the verdict turned a permanently-false probe into a permanent 503,
    which is how it was finally caught — measured on the running pod as
    ``_minio_client is None`` → True, with valid endpoint, bucket and credentials.
    """
    try:
        storage = getattr(request.app.state, "storage_service", None)
        if storage is None:
            # Startup never reached `_init_storage`. A real fault, not an unknown: the
            # service cannot persist anything. Reporting it healthy would be the
            # absent-means-fine default this whole sweep exists to remove.
            return "unhealthy: storage service not initialised"
        return "healthy" if await storage.health_check() else "unhealthy"
    except Exception as e:  # noqa: BLE001 — a probe must report, never raise
        return f"unhealthy: {format_error(e)}"


def _probe_video_providers(request: Request) -> str:
    """Report whether ANY video provider registered at start-up.

    Zero registered providers is the failure this probe exists for: on
    2026-08-20 both prod replicas ran with an empty registry (the image shipped
    without ffmpeg, so the stub provider -- the only one activated, every cloud
    key being blank -- failed its health check) and still reported Ready. The
    only trace was a WARNING that scrolled past on start-up and was never
    emitted again. Nothing polls a start-up log; something has to poll a probe.
    """
    registry = getattr(request.app.state, "provider_registry", None)
    if registry is None:
        return "unhealthy: video provider registry not initialised"
    names = registry.provider_names
    if not names:
        return "unhealthy: no video providers registered"
    return "healthy"


@router.get("/health", response_model=None, operation_id="health_check")
async def health_check(request: Request) -> dict[str, Any] | JSONResponse:
    """Combined health check for all components (cached for 5s)."""
    global _health_cache, _health_cache_time
    now = time.monotonic()
    if _health_cache is not None and (now - _health_cache_time) < CACHE_TTL_HEALTH_CHECK_S:
        return _health_cache

    try:
        db_status = "healthy"
        try:
            engine = await get_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except (OSError, SQLAlchemyError) as e:
            db_status = f"unhealthy: {format_error(e)}"

        redis_status = "healthy"
        try:
            redis_client = get_redis_client()
            if redis_client:
                await redis_client.ping()
            else:
                redis_status = "not configured"
        except OSError as e:
            redis_status = f"unhealthy: {format_error(e)}"

        storage_status = await _probe_storage(request)
        video_provider_status = _probe_video_providers(request)

        # Severity model. "degraded" = reduced capability, the service can still
        # deliver its product. "unhealthy" = it cannot, and requests are actively
        # harmful -- the same meaning the except-branch below already assigns, so
        # it carries the same 503.
        #
        # Storage is fatal, not degraded: every generated asset is persisted
        # through it, and the write happens AFTER the provider call has been paid
        # for. With storage down each request burns provider spend and produces
        # no artifact, and no already-stored asset can be retrieved or presigned.
        # That is total loss of function, not partial. The database is fatal for
        # the same reason (no job can be accepted or tracked). Redis is a cache /
        # cross-pod coordinator and "not configured" is a supported deployment,
        # so it only degrades.
        #
        # Zero video providers is fatal, not degraded: video generation is a
        # product of this service, and with an empty registry EVERY video
        # request fails -- that is total loss of that function, not a reduced
        # one. It is deliberately NOT folded into /health/ready: the condition
        # is a property of the image, so it is identical on every replica, and
        # failing readiness would evict all of them at once and take the live
        # local-image path down with it -- turning a video outage into a total
        # one. Nothing routes on /health, so a 503 here is loud and harmless.
        if (
            db_status != "healthy"
            or storage_status != "healthy"
            or video_provider_status != "healthy"
        ):
            overall = "unhealthy"
        elif redis_status != "healthy":
            overall = "degraded"
        else:
            overall = "healthy"

        # One annotation, reused: a second `dict[str, Any]` here would trip the
        # any-ratchet, and this payload is the same object either way -- an
        # unhealthy verdict just wraps it in the 503 the severity model demands.
        result: dict[str, Any] | JSONResponse = {
            "status": overall,
            "components": {
                "database": db_status,
                "redis": redis_status,
                "storage": storage_status,
                "video_providers": video_provider_status,
                "api": "healthy",
            },
        }
        if overall == "unhealthy":
            result = JSONResponse(status_code=503, content=result)
        _health_cache = result
        _health_cache_time = now
        return result
    except (OSError, SQLAlchemyError, RuntimeError) as e:
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": format_error(e)}
        )


@router.get("/health/live", response_model=None, operation_id="health_live")
async def health_live() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "ok"}


@router.get("/healthz", operation_id="health.live")
async def healthz() -> dict[str, str]:
    """Modern K8s liveness probe path (alias of /health/live)."""
    return {"status": "ok"}


@router.get("/health/ready", response_model=None, operation_id="health_ready")
async def health_ready() -> dict[str, Any] | JSONResponse:
    """Kubernetes readiness probe -- checks database connectivity (cached for 5s).

    Deliberately DB-only, and deliberately not widened to storage: readiness
    gates endpoint membership, so a shared-MinIO blip would evict every replica
    at once and turn a partial outage into a total one. Storage truth is
    reported (and 503s) on ``/health``, which nothing routes on.
    """
    global _ready_cache, _ready_cache_time
    now = time.monotonic()
    if _ready_cache is not None and (now - _ready_cache_time) < CACHE_TTL_HEALTH_CHECK_S:
        return _ready_cache

    try:
        engine = await get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except (OSError, SQLAlchemyError) as e:
        # SUCCESS-ONLY cache. A failed probe is deliberately NOT stored: caching it
        # pinned the pod to "not ready" for the whole TTL after the database had
        # already come back, because the next probe short-circuited on the cached
        # 503 instead of re-probing. Drop any stale entry too, so a cache written
        # before this failure can never be served as if it were current.
        _ready_cache = None
        _ready_cache_time = 0.0
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "ready": False,
                "checks": {"database": {"status": "down", "error": format_error(e)}},
            },
        )

    # Healthy verdicts keep the existing TTL -- the cache exists to stop a 5s-interval
    # kubelet probe hammering the database on a pod that is fine.
    result: dict[str, Any] | JSONResponse = {
        "status": "healthy",
        "ready": True,
        "checks": {"database": {"status": "up"}},
    }
    _ready_cache = result
    _ready_cache_time = now
    return result


@router.get("/readyz", response_model=None, operation_id="health.ready")
async def readyz() -> dict[str, Any] | JSONResponse:
    """Modern K8s readiness probe path (alias of /health/ready)."""
    return await health_ready()
