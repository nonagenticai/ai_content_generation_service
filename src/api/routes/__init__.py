"""API routes package for ai_content_generation_service.

Composes all sub-routers into a single ``router`` that is importable as::

    from src.api.routes import router
"""

from __future__ import annotations

from fastapi import APIRouter
from w2v_common.auth import LICENSE_GATE

from src.api.routes import batches, generation, jobs, preview, providers, usage
from src.api.routes.dependencies import (
    _get_batch_tracker,
    _get_quota_service,
    _get_redis,
    _get_registry,
    _get_storage,
    _get_tenant_id,
    _get_video_generation_service,
    _validate_generate_request,
)

# Entitlement gate, mounted router-wide: EVERY route below this prefix is
# tenant-facing and BILLED (this service is in w2v_common's
# METERED_DESTINATION_SERVICES), so authenticating "who are you"
# (AuthMiddleware) without ever asking "may you" served revoked and expired
# tenants for free. Mounting it on the PARENT covers the six sub-routers
# included below, including ones added later.
#
# Dormant on merge: ``require_license`` is a no-op unless
# ``W2V_LICENSE_GATE_ENABLED=true``. Probe/doc paths are exempted inside
# w2v_common (``PUBLIC_ENDPOINTS`` u {/health,/healthz,/ready}) -- never
# hand-roll that list here, a gated /health/live crashloops the pod.
router = APIRouter(
    prefix="/api/content/videos",
    tags=["content-videos"],
    dependencies=list(LICENSE_GATE),
)

router.include_router(generation.router)
router.include_router(jobs.router)
router.include_router(providers.router)
router.include_router(batches.router)
router.include_router(preview.router)
router.include_router(usage.router)

# Re-export dependency helpers so existing imports like
# ``from src.api.routes import _validate_generate_request`` keep working.
__all__ = [
    "_get_batch_tracker",
    "_get_quota_service",
    "_get_redis",
    "_get_registry",
    "_get_storage",
    "_get_tenant_id",
    "_get_video_generation_service",
    "_validate_generate_request",
    "router",
]
