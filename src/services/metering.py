"""Cost metering — delegates to the centralized w2v_common cost-bus emitter.

One cost event is emitted per successful video generation
(Sora/Kling/Veo/Wan/Runway/Seedance/...) or per successful image generation
(OpenAI DALL-E + future image providers), following the minimum-viable shape
in ``metering_service/docs/EMITTER_GUIDE.md``.

The emit logic, field marshalling, Redis-connection lifecycle, GC-safe task
scheduling, and canonical-stream resolution (``METERING_STREAM`` env →
``improvement-loop:costs``) all live in ``w2v_common.metering`` so the next
stream rename is a one-line change in one place. This module only carries the
service-specific routing identity (``service_key`` / ``cost_source``) and the
fail-loud tenant guard.
"""

from __future__ import annotations

import uuid

from w2v_common.auth import W2V_APP_KEY
from w2v_common.metering import (
    Funding,
    close_metering_redis,
    cost_event_key,
    emit_cost_event,
    init_metering_redis,
    resolve_stream,
    schedule_cost_event,
)

#: The licences app this service's spend is attributed to (billing resolves the
#: app KEY through ``split_asserted_app``). Every event carries it; never "".
APP_ID = W2V_APP_KEY

# Routing identity for the cost bus — billing keys events to this service.
COST_SOURCE = "w2v-ai-content"
SERVICE_KEY = "w2v-ai-content"  # pragma: allowlist secret -- routing slug, not a credential

# ---------------------------------------------------------------------------
# NO TENANT SENTINEL LIVES HERE. Do not re-add one.
#
# ``DEFAULT_TENANT_ID`` (env-overridable, defaulting to w2v_common's nil-UUID
# ``UNATTRIBUTED_TENANT_ID``) used to sit on this line. Product-owner ruling:
# a ``tenant_id`` may NEVER hold a sentinel. Every message carries a real
# registered tenant forwarded from the requestor; an absent tenant is an ERROR,
# never a substituted default.
#
# The nil UUID was the most dangerous spelling of that lie: it is a VALID uuid,
# so it passes every uuid cast and every dirty-row check, and billing-service
# then buckets it as unattributed and NEVER CHARGES IT
# (``_parsers.py`` ``_normalize_tenant_id`` + ``cost_flusher.py``). Real usage,
# silently unbilled. ``""``/``None`` are no better -- billing maps those to
# ``"unknown"``, a phantom tenant that IS charged.
#
# Fail loud instead: :func:`require_tenant` below. Both the sentinel and
# ``UNATTRIBUTED_TENANT_ID`` are also removed from ``w2v_common`` in 0.67.0, so
# importing either would break this service at import time.
# ---------------------------------------------------------------------------


# The nil UUID, spelled out locally ON PURPOSE. This is the value the retired
# ``DEFAULT_TENANT_ID`` resolved to, and w2v_common's ``AuthMiddleware`` can still
# write it onto ``request.state.tenant_id`` on the legacy trusted-network opt-in
# path under 0.66.1. It is NOT imported from w2v_common: 0.67.0 deletes the name,
# and this guard must keep working on both. It is listed here only to be REJECTED.
_NIL_TENANT_SENTINEL = "00000000-0000-0000-0000-000000000000"


class MissingTenantError(ValueError):
    """No real tenant was forwarded for an operation that MUST be tenant-scoped.

    FAIL LOUD, never substitute. Raised by :func:`require_tenant`.
    """

    def __init__(self, field: str = "tenant_id") -> None:
        super().__init__(
            f"No tenant identity on the request: {field} is required. "
            "ai_content_generation_service is tenant-scoped -- authenticate with a "
            "license token or JWT carrying a real registered tenant_id. "
            "No default tenant is substituted."
        )
        self.field = field


def require_tenant(tenant_id: object, field: str = "tenant_id") -> str:
    """Return a real forwarded tenant, or raise :class:`MissingTenantError`.

    Accepts a ``uuid.UUID`` (``jobs.tenant_id`` / ``image_gen_jobs.tenant_id`` are
    native ``uuid`` columns) or a non-blank string, and returns the canonical
    string form the cost bus and the tenant-scoped queries both take.

    Deliberately does NOT parse a plain string as a uuid: the tenant string is
    threaded straight through to queries whose columns do that cast, and widening
    this into a format gate is a separate change. What it DOES guarantee is the
    thing the ruling is about -- that no absent/blank tenant, and no nil-UUID
    sentinel, is quietly threaded on to the DB or to the cost bus.
    """
    value = str(tenant_id) if isinstance(tenant_id, uuid.UUID) else tenant_id
    if not isinstance(value, str) or not value.strip():
        raise MissingTenantError(field)
    value = value.strip()
    # A sentinel is not a tenant. It is a valid uuid, so nothing downstream
    # would have caught it -- billing would bucket it as unattributed and never
    # charge the usage. Compare BY VALUE, not against the one canonical
    # spelling: ``00000000000000000000000000000000``, ``{...}``, ``urn:uuid:...``
    # and any casing all parse to the same nil uuid and a string equality check
    # lets every one of them through.
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        # Not uuid-shaped at all. Format-gating the tenant string is a separate
        # change (see the docstring); the sentinel guard has nothing to do here.
        # NOTE: the raise below must sit OUTSIDE this try -- MissingTenantError
        # subclasses ValueError, so raising inside would be swallowed here.
        parsed = None
    if parsed is not None and parsed.int == 0:
        raise MissingTenantError(field)
    return value


#: WHO FUNDS a PLATFORM-key job, by provider ``name`` (the registry key that is
#: emitted as ``provider``). A tenant provider-slot job is always ``tenant_byo``
#: and never consults this table. Self-hosted compute -> ``local_gpu`` (no vendor
#: invoice); an external vendor billing the platform's key -> ``platform_external``.
#: The test doubles (``stub``/``mock``/``mock_images``) call no vendor at all.
_PLATFORM_FUNDING: dict[str, Funding] = {
    "local_image": "local_gpu",  # mflux/FLUX backend on mini1
    "wan": "local_gpu",  # Wan2.2 on local CUDA
    "stub": "local_gpu",
    "mock": "local_gpu",
    "mock_images": "local_gpu",
    "openai_images": "platform_external",
    "sora": "platform_external",
    "veo": "platform_external",
    "kling": "platform_external",
    "runway": "platform_external",
    "seedance": "platform_external",
}


class UnknownFundingProviderError(ValueError):
    """A platform job ran on a provider with no declared funding. FAIL LOUD.

    A guessed ``funding`` is stored verbatim by billing, so a new provider must
    be added to :data:`_PLATFORM_FUNDING` deliberately, never defaulted.
    """


def platform_funding(provider: str) -> Funding:
    """``funding`` for a PLATFORM-key job on ``provider``; raises when undeclared."""
    try:
        return _PLATFORM_FUNDING[provider]
    except KeyError:
        raise UnknownFundingProviderError(
            f"provider {provider!r} has no declared funding (local_gpu/platform_external)"
        ) from None


def job_event_key(job_id: str, kind: str) -> str:
    """Deterministic billing ``event_key`` for ONE generation job's cost event.

    A job completes (and is metered) at most once, so ``(service, job, kind)``
    identifies the physical call; a re-emit after a restart dedupes at billing.
    """
    return cost_event_key(SERVICE_KEY, job_id, kind)


__all__ = [
    "APP_ID",
    "COST_SOURCE",
    "SERVICE_KEY",
    "MissingTenantError",
    "UnknownFundingProviderError",
    "close_metering_redis",
    "emit_cost_event",
    "init_metering_redis",
    "job_event_key",
    "platform_funding",
    "require_tenant",
    "resolve_stream",
    "schedule_cost_event",
]
