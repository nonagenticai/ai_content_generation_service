"""Dependency functions for API route injection."""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from fastapi import HTTPException, Request
from w2v_common.net import SSRFError, validate_public_url
from w2v_common.providers import (
    ProviderHopRefusedError,
    ProviderSlotConfigError,
    ProviderSlotRequiredError,
    ProviderSlotUnavailableError,
    provider_slots_mode,
    resolve_provider_slot,
)
from w2v_common.providers import ProviderSlotError as ResolveError
from w2v_common.tenant import UnvouchedTenantError

if TYPE_CHECKING:
    from src.services.quota_service import QuotaService

from src.api.content_schemas import ConceptImageRequest
from src.api.schemas import GenerateRequest
from src.services.batch_tracker import BatchTracker
from src.services.image_providers.base import ImageProviderRegistry
from src.services.imagegen_service import ImageGenService
from src.services.job_manager import JobManager, get_default_job_manager
from src.services.metering import MissingTenantError, require_tenant
from src.services.provider_secrets import KeyringNotConfiguredError
from src.services.provider_slots import (
    ImageProviderSlot,
    ProviderSlot,
    ProviderSlotError,
    VideoProviderSlot,
    require_keyring,
    slot_from_resolved,
)
from src.services.providers.base import VideoProviderRegistry
from src.services.storage_service import StorageService
from src.services.video_generation_service import VideoGenerationService


def _get_tenant_id(request: Request) -> str:
    """Return the caller's real forwarded tenant, or HTTP 400 -- never a sentinel.

    ``request.state.tenant_id`` is set by ``w2v_common.auth.middleware.AuthMiddleware``
    from the license-token introspection or the JWT ``tenant_id`` claim.

    This used to fall back to ``DEFAULT_TENANT_ID`` (the nil UUID). Product-owner
    ruling: an absent tenant is an ERROR, never a substituted default. The nil
    UUID is a valid uuid, so it passed every cast, and billing then bucketed it
    as unattributed and NEVER CHARGED IT -- real usage, silently unbilled.

    **400, not 401**, matching the sibling ``video_compiler_service`` route layer:
    a request that reaches here has already cleared ``AuthMiddleware`` (a public
    endpoint or a trusted-network caller passes without a license token). The
    credential is not missing; the request is simply unattributable. Answering
    401 would tell a trusted-network caller to re-authenticate, which fixes
    nothing -- the caller must forward a real registered tenant.
    """
    try:
        return require_tenant(getattr(request.state, "tenant_id", None))
    except MissingTenantError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _get_video_generation_service(request: Request) -> VideoGenerationService:
    """Retrieve the VideoGenerationService instance from application state."""
    return request.app.state.video_generation_service


def get_job_manager(request: Request) -> JobManager:
    """Retrieve the JobManager instance from application state.

    Falls back to ``get_default_job_manager()`` (an unconfigured, lazily
    constructed singleton) if lifespan has not run -- which keeps tests
    that bypass the FastAPI app boot sequence working.
    """
    jm = getattr(request.app.state, "job_manager", None)
    if jm is None:
        return get_default_job_manager()
    return jm


def _get_registry(request: Request) -> VideoProviderRegistry:
    """Retrieve the provider registry from application state."""
    return request.app.state.provider_registry


def _get_image_registry(request: Request) -> ImageProviderRegistry:
    """Retrieve the image provider registry from application state."""
    return request.app.state.image_provider_registry


def _get_storage(request: Request) -> StorageService:
    """Retrieve the storage service from application state."""
    return request.app.state.storage_service


def _get_imagegen_service(request: Request) -> ImageGenService:
    """Retrieve the ImageGenService instance from application state."""
    return request.app.state.imagegen_service


def _get_batch_tracker(request: Request) -> BatchTracker:
    """Retrieve the batch tracker from application state."""
    return request.app.state.batch_tracker


def _get_redis(request: Request) -> Any:
    """Get Redis client (may be None if not configured)."""
    return getattr(request.app.state, "redis_client", None)


def _get_quota_service(request: Request) -> QuotaService | None:
    """Get QuotaService (may be None if not configured)."""
    return getattr(request.app.state, "quota_service", None)


_INTERNAL_URL_DETAIL = "must not resolve to a private/internal IP address"


def _reject_internal_url(url: str, field_name: str) -> None:
    """Resolve a URL's hostname and reject it if the IP is in a private/reserved range.

    The block-list is the fleet-wide one in ``w2v_common.net`` (>=0.51.0). The
    local list this replaced was RFC1918-only: it missed CGNAT
    ``100.64.0.0/10`` — the split-horizon private range, where internal
    services live and where
    ``ip_address(...).is_private`` is **False** — plus ``0.0.0.0/8``,
    ``224.0.0.0/4``, ``240.0.0.0/4``, the broadcast address, and it had **no
    IPv6 entries at all**, so ``::1`` / ``fd00::`` sailed through
    (``getaddrinfo`` is not family-filtered).

    Every rejection is an HTTP **400**, unchanged from the previous
    implementation. Notably a scoped IPv6 literal (``fe80::1%eth0``) used to
    reach a bare ``ipaddress.ip_address()`` and raise an uncaught ``ValueError``
    -> HTTP 500; it now fails closed as a 400.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        raise HTTPException(status_code=400, detail=f"{field_name} has no valid hostname")

    try:
        pinned = validate_public_url(url)
    except SSRFError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} {_INTERNAL_URL_DETAIL}") from exc
    except ValueError as exc:  # bad scheme / unextractable hostname
        raise HTTPException(status_code=400, detail=f"{field_name} has no valid hostname") from exc

    if pinned is None:
        return
    # NOTE: the pin is ADVISORY here. ``validate_public_url`` returns the IP the
    # connection must be pinned to, but this boundary check only parse-checks it
    # and throws it away -- it cannot pin a fetch it does not perform. Passing
    # this check therefore does NOT mean the URL is safe to fetch: between here
    # and the fetch the hostname is re-resolved, which is a DNS-rebinding TOCTOU
    # window. The real enforcement is at every fetch site, which must use
    # ``w2v_common.net.safe_get`` (it re-validates AND pins). Do not treat a 400
    # here as the security boundary.
    try:
        ipaddress.ip_address(pinned)
    except ValueError as exc:
        # A zone-scoped / otherwise unparseable resolved address (``fe80::1%eth0``)
        # cannot be range-checked, so it is rejected rather than trusted.
        raise HTTPException(status_code=400, detail=f"{field_name} {_INTERNAL_URL_DETAIL}") from exc


# ============================================================================
# Request Validation
# ============================================================================


def _validate_provider_for_generate(registry: Any, body: GenerateRequest) -> None:
    if body.provider is None:
        return
    provider = registry.get(body.provider)
    if provider is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown provider: {body.provider}. Available: {registry.provider_names}",
        )
    if body.duration_s is not None:
        max_dur = provider.get_max_duration_s()
        if max_dur is not None and body.duration_s > max_dur:
            raise HTTPException(
                status_code=422,
                detail=f"Duration {body.duration_s}s exceeds provider max {max_dur}s",
            )
    if body.image_url is not None and not provider.supports_image_to_video():
        raise HTTPException(
            status_code=422,
            detail=f"Provider {body.provider} does not support image-to-video",
        )


def _validate_fallback_providers(registry: Any, body: GenerateRequest) -> None:
    if not body.fallback_providers:
        return
    for fp in body.fallback_providers:
        if registry.get(fp) is None:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown fallback provider: {fp}",
            )


def _validate_external_url(value: str | None, field_name: str) -> None:
    if value is None:
        return
    if not value.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be an HTTP(S) URL",
        )
    _reject_internal_url(value, field_name)


def require_slot_keyring_or_503() -> None:
    """Fail CLOSED when a request carries ``provider_slot`` but ACG has no keyring.

    The slot's secret must be sealed on the job row before the job may exist
    (the reaper resumes from the row). Without ``PROVIDER_SECRET_KEYS`` it cannot
    be, and running the job anyway -- or on a platform provider -- is not an
    option. 503: a deployment fault the caller cannot fix by changing the request.
    """
    try:
        require_keyring()
    except KeyringNotConfiguredError as exc:
        raise HTTPException(
            status_code=503,
            detail="tenant provider slots are unavailable: provider secret keyring not configured",
        ) from exc


def reject_slot_on_batch(provider_slot: object | None) -> None:
    """``provider_slot`` is a SINGLE-job contract; a batch item carrying one is refused.

    Silently ignoring it would run that item on a platform provider -- exactly
    the fallback a tenant slot must never get.
    """
    if provider_slot is not None:
        raise HTTPException(
            status_code=422,
            detail="provider_slot is only supported on the single-job endpoint, not in batches",
        )


def provider_slots_enforced_or_503() -> bool:
    """``W2V_PROVIDER_SLOTS``: ``True`` = enforce (ACG resolves the run's slot itself).

    Read ONCE per request. A typo is a deployment fault, never a silent ``off``
    (w2v_common raises ``ValueError``): 503, fail closed.
    """
    try:
        return provider_slots_mode() == "enforce"
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="tenant provider slots are misconfigured on this service (W2V_PROVIDER_SLOTS)",
        ) from exc


async def _resolve_run_slot[SlotT: ProviderSlot](
    slot_name: str, model: type[SlotT]
) -> SlotT | None:
    """The RUN's ``slot_name`` provider from the orchestrator (``enforce`` only).

    ``None`` = the platform default. A tenant answer comes back in this service's
    ``provider_slot`` contract, so everything downstream (seal, persist, resume,
    retry, cancel) is the exact path a body slot takes. Every failure is an HTTP
    error -- NEVER a silent platform fallback for a tenant that pinned its own:

    * 403 ``byo_provider_required`` -- the licence forbids the platform default;
    * 403 unvouched tenant / provider-hop loop guard;
    * 503 orchestrator unreachable/rolling (transient) or a deployment fault;
    * 409 the run's provider cannot be used (permanent).

    Messages are w2v_common's, which are secret-free by contract.
    """
    try:
        resolved = await resolve_provider_slot(slot_name)
        return slot_from_resolved(resolved, model)
    except ProviderSlotRequiredError as exc:
        raise HTTPException(status_code=403, detail=exc.failure_detail) from None
    except UnvouchedTenantError:
        raise HTTPException(
            status_code=403,
            detail=f"provider slot {slot_name!r}: the tenant's provenance is not vouched",
        ) from None
    except ProviderHopRefusedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except ProviderSlotConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except ProviderSlotUnavailableError as exc:
        status = 503 if exc.transient else 409
        raise HTTPException(
            status_code=status,
            detail={"code": "provider_unavailable", "modality": slot_name, "message": str(exc)},
        ) from None
    except (ResolveError, ProviderSlotError) as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "provider_unavailable", "modality": slot_name, "message": str(exc)},
        ) from None


async def resolve_run_image_slot() -> ImageProviderSlot | None:
    """The run's ``image`` slot (``enforce``): tenant slot or ``None`` (platform)."""
    return await _resolve_run_slot("image", ImageProviderSlot)


async def resolve_run_video_slot() -> VideoProviderSlot | None:
    """The run's ``video_gen`` slot (``enforce``): tenant slot or ``None`` (platform)."""
    return await _resolve_run_slot("video_gen", VideoProviderSlot)


async def effective_image_slot(
    request: Request, body: ConceptImageRequest, *, enforce: bool
) -> ImageProviderSlot | None:
    """The tenant image slot a single job runs on (``None`` = platform).

    ``off``: the body's ``provider_slot`` (already validated by the caller).
    ``enforce``: the run's resolved slot, validated here; the body's is ignored.
    """
    if not enforce:
        return body.provider_slot
    slot = await resolve_run_image_slot()
    _validate_generate_image_request(request, body, slot=slot)
    return slot


async def effective_video_slot(
    request: Request, body: GenerateRequest, *, enforce: bool
) -> VideoProviderSlot | None:
    """Video twin of :func:`effective_image_slot`."""
    if not enforce:
        return body.provider_slot
    slot = await resolve_run_video_slot()
    _validate_generate_request(request, body, slot=slot)
    return slot


def reject_tenant_slot_on_batch(slot: ProviderSlot | None) -> None:
    """``enforce`` twin of :func:`reject_slot_on_batch` for the RESOLVED run slot.

    A batch is platform-only; a run that pinned a tenant provider must not have
    its batch silently run on platform providers.
    """
    if slot is not None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"this run uses tenant provider slot {slot.label()}; tenant provider "
                "slots are only supported on the single-job endpoint, not in batches"
            ),
        )


def _validate_provider_slot_base_url(base_url: str | None) -> None:
    """A tenant slot's ``base_url`` is a URL this service will send a credential to.

    Same boundary rule as every other caller-supplied URL here: HTTP(S) only and
    never a private/internal address (``_reject_internal_url``) -- otherwise a
    tenant could point "their provider" at an in-cluster service.
    """
    _validate_external_url(base_url, "provider_slot.base_url")


class _BodySlot:
    """Sentinel: validate against the request body's own ``provider_slot`` (``off``)."""


BODY_SLOT = _BodySlot()


def _validate_generate_request(
    request: Request,
    body: GenerateRequest,
    *,
    slot: ProviderSlot | _BodySlot | None = BODY_SLOT,
) -> None:
    """Validate a generate request at the API boundary.

    ``slot`` is the EFFECTIVE tenant slot. Default: the body's ``provider_slot``
    (``W2V_PROVIDER_SLOTS=off``). Under ``enforce`` the caller passes the slot it
    resolved itself (or ``None`` for the platform) and the body's is ignored.
    """
    effective = body.provider_slot if isinstance(slot, _BodySlot) else slot
    if effective is not None:
        # The slot REPLACES provider selection: ``provider`` and
        # ``fallback_providers`` are ignored, so they are not validated against
        # the platform registry either (a tenant slot never uses it).
        _validate_provider_slot_base_url(effective.base_url)
    else:
        registry = _get_registry(request)
        _validate_provider_for_generate(registry, body)
        _validate_fallback_providers(registry, body)
    _validate_external_url(body.image_url, "image_url")
    _validate_external_url(body.callback_url, "callback_url")


def _validate_generate_image_request(
    request: Request,
    body: ConceptImageRequest,
    *,
    slot: ProviderSlot | _BodySlot | None = BODY_SLOT,
) -> None:
    """Validate an image generation request at the API boundary (``slot``: see video twin)."""
    registry = _get_image_registry(request)
    effective = body.provider_slot if isinstance(slot, _BodySlot) else slot

    if effective is not None:
        # ``provider`` is ignored when a tenant slot is present.
        _validate_provider_slot_base_url(effective.base_url)
    elif body.provider is not None and registry.get(body.provider) is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown image provider: {body.provider}. Available: {registry.provider_names}",
        )

    if body.reference_image_urls:
        for value in body.reference_image_urls:
            if not value.startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=422,
                    detail="reference_image_urls must be HTTP(S) URLs",
                )
            _reject_internal_url(value, "reference_image_urls")

    if body.callback_url is not None:
        if not body.callback_url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=422,
                detail="callback_url must be an HTTP(S) URL",
            )
        _reject_internal_url(body.callback_url, "callback_url")
