"""Tenant provider slots (P5-2): per-job provider instances built from the request.

The orchestrator resolves a run's pinned ``image`` / ``video_gen`` slot and, ONLY
when that slot is tenant-sourced, sends it on the generation request as
``provider_slot`` (cross-repo contract, fixed -- w2v-p5-slots-blueprint.md)::

    {"slot": "image"|"video_gen"|"device_farm", "kind": str, "base_url": str|null,
     "model": str|null, "options": {..}, "secret": str|null,
     "ref": {"connection_id": uuid, "version": int}}

When it is present this service:

* ignores the request's ``provider`` name and never falls back to a platform
  provider -- the job runs on the tenant's credentials or fails;
* builds a PER-JOB provider instance of ``kind`` (never registered, never
  health-checked, never sharing the platform HTTP pool);
* persists the non-secret slot fields on the job row (``provider_slot``) and the
  secret SEALED (``provider_secret_sealed``, see ``provider_secrets``) so the
  reaper can resume the job after a pod crash; the sealed secret is purged when
  the job reaches a terminal state.

Absent -> today's behaviour, byte-for-byte.

Secrets: :class:`ProviderSlot` holds the secret as a ``SecretStr`` (redacted in
every repr), the persisted metadata never contains it, and every message this
module builds about a slot names the slot by ``slot/kind`` + connection ref only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID

import httpx
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from w2v_common.net import SSRFError

from src.services.provider_secrets import (
    KeyringNotConfiguredError,
    ProviderSecretError,
    get_keyring,
)

if TYPE_CHECKING:
    from w2v_common.providers import ProviderSlot as ResolvedProviderSlot

    from src.services.image_providers.base import ImageGenProvider
    from src.services.providers.base import VideoGenerationProvider

logger = logging.getLogger(__name__)

#: Image kinds a tenant slot may name (contract). ``openai`` = the OpenAI Images
#: API; ``nonagentic_endpoint`` = a tenant self-hosted GPU speaking the platform
#: ``POST /api/image/generate`` protocol; ``openai_compatible`` = a tenant
#: self-hosted server speaking the OpenAI ``POST /images/generations`` shape.
IMAGE_SLOT_KINDS: frozenset[str] = frozenset({"openai", "nonagentic_endpoint", "openai_compatible"})
#: Image kinds with no contract default endpoint: ``base_url`` is REQUIRED.
SELF_HOSTED_IMAGE_SLOT_KINDS: frozenset[str] = frozenset(
    {"nonagentic_endpoint", "openai_compatible"}
)
#: Video kinds a tenant slot may name (contract) -- one per cloud video provider.
VIDEO_SLOT_KINDS: frozenset[str] = frozenset({"runway", "kling", "veo", "sora", "seedance"})

#: Cost-event ``funding`` value (owner-decided taxonomy: local_gpu /
#: platform_external / tenant_byo) for spend on a tenant's own key. Emitted as
#: the top-level ``funding`` kwarg of ``schedule_cost_event`` (w2v_common >= 0.94).
FUNDING_TENANT_BYO: Literal["tenant_byo"] = "tenant_byo"

#: Contract default for an ``openai`` image slot with no ``base_url`` / ``model``.
#: Deliberately NOT ``DALLE_BASE_URL``: a platform override of that setting (a
#: proxy, a gateway) must never receive a tenant's key.
OPENAI_IMAGE_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_IMAGE_DEFAULT_MODEL = "gpt-image-1"


class ProviderSlotRef(BaseModel):
    """Which tenant connection version the orchestrator resolved the slot from."""

    connection_id: UUID
    version: int


class ProviderSlot(BaseModel):
    """A tenant-sourced provider slot as sent by the orchestrator (contract-shaped)."""

    slot: Literal["image", "video_gen", "device_farm"]
    kind: str = Field(..., min_length=1, max_length=64)
    base_url: str | None = Field(default=None, max_length=2048)
    model: str | None = Field(default=None, max_length=256)
    options: dict[str, object] = Field(default_factory=dict)
    secret: SecretStr | None = None
    ref: ProviderSlotRef

    def label(self) -> str:
        """Secret-free name for logs and job errors: ``video_gen/kling (connection <id> v<n>)``."""
        return f"{self.slot}/{self.kind} (connection {self.ref.connection_id} v{self.ref.version})"

    def to_persisted(self) -> dict[str, object]:
        """The NON-secret fields persisted on the job row (``provider_slot``)."""
        return {
            "slot": self.slot,
            "kind": self.kind,
            "base_url": self.base_url,
            "model": self.model,
            "options": dict(self.options),
            "ref": {"connection_id": str(self.ref.connection_id), "version": self.ref.version},
            "has_secret": self.secret is not None,
        }


class ImageProviderSlot(ProviderSlot):
    """``provider_slot`` on ``POST /api/concept-images``: an ``image`` slot."""

    slot: Literal["image"]  # ty: ignore[incompatible-mutable-override]  # pydantic narrows the field per request shape

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in IMAGE_SLOT_KINDS:
            raise ValueError(f"unknown image slot kind; expected one of {sorted(IMAGE_SLOT_KINDS)}")
        return value

    @model_validator(mode="after")
    def _self_hosted_needs_base_url(self) -> ImageProviderSlot:
        if self.kind in SELF_HOSTED_IMAGE_SLOT_KINDS and not self.base_url:
            raise ValueError(f"image slot kind {self.kind!r} requires base_url")
        return self


class VideoProviderSlot(ProviderSlot):
    """``provider_slot`` on ``POST /api/content/videos/generate``: a ``video_gen`` slot."""

    slot: Literal["video_gen"]  # ty: ignore[incompatible-mutable-override]  # pydantic narrows the field per request shape

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in VIDEO_SLOT_KINDS:
            raise ValueError(f"unknown video slot kind; expected one of {sorted(VIDEO_SLOT_KINDS)}")
        return value


def slot_from_resolved[SlotT: ProviderSlot](
    resolved: ResolvedProviderSlot, model: type[SlotT]
) -> SlotT | None:
    """Turn ``w2v_common.providers.resolve_provider_slot``'s answer into this contract.

    ``None`` = the platform default (``source == "deployment"``) -- today's
    platform path. A tenant answer is re-validated through ``model``
    (:class:`ImageProviderSlot` / :class:`VideoProviderSlot`) exactly as a body
    ``provider_slot`` is, so an unknown kind or a self-hosted kind without
    ``base_url`` fails the same way. Raises :class:`ProviderSlotError` (secret-free)
    for a tenant answer this service cannot run.
    """
    if not resolved.is_tenant:
        return None
    ref = resolved.ref
    if ref is None:  # parser guarantees a ref on a tenant slot; defence only
        raise ProviderSlotError(f"tenant provider slot {resolved.slot}: missing connection ref")
    try:
        return model.model_validate(
            {
                "slot": resolved.slot,
                "kind": resolved.kind,
                "base_url": resolved.base_url,
                "model": resolved.model,
                "options": dict(resolved.options),
                "secret": resolved.secret_value(),
                "ref": {"connection_id": ref.connection_id, "version": ref.version},
            }
        )
    except ValidationError as exc:
        # ``include_input=False``: the input dict carries the secret.
        reasons = "; ".join(
            str(err.get("msg", "invalid")) for err in exc.errors(include_input=False)
        )
        raise ProviderSlotError(
            f"tenant provider slot {resolved.slot}/{resolved.kind} "
            f"(connection {ref.connection_id} v{ref.version}) is not usable here: {reasons}"
        ) from None


class ProviderSlotError(RuntimeError):
    """A persisted slot cannot be turned into a provider. Message is secret-free."""


@dataclass(frozen=True, slots=True)
class SlotJob:
    """What the generation services need to run a slot job.

    ``secret`` is the UNSEALED tenant secret (``""`` for a slot without one) and
    lives only in memory for the duration of one ``process_job`` call; it is
    held here solely so :meth:`scrub` can strip it from provider error text.
    """

    slot: str
    kind: str
    label: str
    base_url: str | None
    model: str | None
    secret: str
    connection_id: str | None = None
    version: int | None = None

    def __repr__(self) -> str:  # never render the secret
        return f"SlotJob(label={self.label!r})"

    def byo_usage_extra(self) -> dict[str, object]:
        """Cost-event provenance marking this job's spend as TENANT-funded.

        The job ran on the tenant's own key, so the platform paid nothing
        upstream and must never bill platform provider prices for it.
        billing-service derives BYO from ``usage_extra.byo == "true"`` (the
        modality_gateway ``byo_cost`` convention: under ``byo_platform_fee`` the
        markup is forced to 0, and the ``byo`` rollup dimension), so ``byo``
        stays here. ``funding`` is NOT here: it is emitted top-level via the
        ``funding=`` kwarg (:data:`FUNDING_TENANT_BYO`).
        """
        return {
            "byo": "true",
            "slot": self.slot,
            "slot_kind": self.kind,
            "connection_id": self.connection_id or "",
            "version": self.version,
        }

    def scrub(self, text: str) -> str:
        """Remove the secret from provider-originated text (belt and braces)."""
        if self.secret:
            return text.replace(self.secret, "[redacted]")
        return text

    def failure_message(self, exc: BaseException) -> str:
        """Permanent, slot-naming job error for a provider failure. Never the secret."""
        from src.services.exceptions import format_error

        status = _http_status(exc)
        if _ssrf_error(exc) is not None:
            reason = (
                "provider host resolves to a non-public address at connect time -- "
                "refused (SSRF guard)"
            )
        elif status in (401, 403):
            reason = (
                f"provider rejected the tenant credentials (HTTP {status}) -- "
                "fix or replace the connection"
            )
        elif status is not None and 400 <= status < 500:
            reason = f"provider refused the request (HTTP {status})"
        else:
            reason = self.scrub(format_error(exc) if isinstance(exc, Exception) else str(exc))
        return (
            f"tenant provider slot {self.label} failed: {reason}. "
            "Not retried on a platform provider (tenant slots never fall back)."
        )


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status behind ``exc`` (directly or via ``__cause__``), if any."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            return current.response.status_code
        current = current.__cause__
    return None


def _ssrf_error(exc: BaseException) -> SSRFError | None:
    """The connect-time SSRF refusal behind ``exc`` (directly or via ``__cause__``)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, SSRFError):
            return current
        current = current.__cause__ or current.__context__
    return None


def is_permanent_slot_error(exc: BaseException) -> bool:
    """Final for a slot job: an SSRF refusal, or a provider 4xx other than 429."""
    if _ssrf_error(exc) is not None:
        return True
    status = _http_status(exc)
    return status is not None and 400 <= status < 500 and status != 429


# ---------------------------------------------------------------------------
# Sealing at enqueue / unsealing at process time
# ---------------------------------------------------------------------------


def require_keyring() -> None:
    """Raise :class:`KeyringNotConfiguredError` when ACG has no keyring (route -> 503)."""
    if not get_keyring().configured:
        raise KeyringNotConfiguredError("PROVIDER_SECRET_KEYS is not configured")


def seal_slot(slot: ProviderSlot, *, job_id: str) -> tuple[dict[str, object], bytes | None]:
    """``(persisted metadata, sealed secret envelope | None)`` for ``job_id``'s row."""
    sealed = None
    if slot.secret is not None:
        sealed = get_keyring().encrypt(slot.secret.get_secret_value(), job_id=job_id)
    return slot.to_persisted(), sealed


def open_slot_job(job_id: str, persisted: dict[str, object], sealed: bytes | None) -> SlotJob:
    """Rebuild the in-memory :class:`SlotJob` from a job row.

    Raises :class:`ProviderSlotError` (secret-free message) when the row cannot be
    used: malformed metadata, a secret that was expected but is gone, or an
    envelope the keyring cannot open (unknown kid, wrong job binding, tampered,
    empty keyring). The caller fails the job permanently -- it must never fall
    back to a platform provider.
    """
    slot, kind, ref = persisted.get("slot"), persisted.get("kind"), persisted.get("ref")
    if not isinstance(slot, str) or not isinstance(kind, str) or not isinstance(ref, dict):
        raise ProviderSlotError("tenant provider slot metadata on the job is malformed")
    label = f"{slot}/{kind} (connection {ref.get('connection_id')} v{ref.get('version')})"
    has_secret = bool(persisted.get("has_secret"))
    base_url, model = persisted.get("base_url"), persisted.get("model")

    secret = ""
    if has_secret:
        if not sealed:
            raise ProviderSlotError(
                f"tenant provider slot {label}: sealed secret is missing from the job"
            )
        try:
            secret = get_keyring().decrypt(bytes(sealed), job_id=job_id)
        except ProviderSecretError as exc:
            raise ProviderSlotError(
                f"tenant provider slot {label}: secret could not be unsealed ({exc})"
            ) from None
    version = ref.get("version")
    return SlotJob(
        slot=slot,
        kind=kind,
        label=label,
        base_url=base_url if isinstance(base_url, str) else None,
        model=model if isinstance(model, str) else None,
        secret=secret,
        connection_id=str(ref.get("connection_id") or "") or None,
        version=version if isinstance(version, int) else None,
    )


# ---------------------------------------------------------------------------
# Per-job provider factories
# ---------------------------------------------------------------------------


def _video_provider_class(kind: str) -> type[VideoGenerationProvider]:
    from src.services.providers.kling import KlingProvider
    from src.services.providers.runway import RunwayProvider
    from src.services.providers.seedance import SeedanceProvider
    from src.services.providers.sora import SoraProvider
    from src.services.providers.veo import VeoProvider

    classes: dict[str, type[VideoGenerationProvider]] = {
        "runway": RunwayProvider,
        "kling": KlingProvider,
        "veo": VeoProvider,
        "sora": SoraProvider,
        "seedance": SeedanceProvider,
    }
    cls = classes.get(kind)
    if cls is None:
        raise ProviderSlotError(f"unknown video slot kind {kind!r}")
    return cls


def build_video_provider(job: SlotJob) -> VideoGenerationProvider:
    """A per-job video provider on the slot's key/base_url/model.

    ``api_key`` is ALWAYS passed explicitly (``""`` for a keyless slot): an
    omitted key would make the provider fall back to the PLATFORM key, i.e. bill
    a tenant-slot job to the platform account.
    """
    # The ABC's ``__init__`` takes no arguments; every concrete cloud class here
    # accepts the slot keywords, which ``_video_provider_class`` guarantees.
    cls = cast(Any, _video_provider_class(job.kind))
    return cls(api_key=job.secret, base_url=job.base_url or None, model=job.model or None)


def _image_provider_class(kind: str) -> type[ImageGenProvider]:
    from src.services.image_providers.openai import OpenAIImageProvider
    from src.services.image_providers.tenant_endpoints import (
        NonagenticEndpointImageProvider,
        OpenAICompatibleImageProvider,
    )

    classes: dict[str, type[ImageGenProvider]] = {
        "openai": OpenAIImageProvider,
        "nonagentic_endpoint": NonagenticEndpointImageProvider,
        "openai_compatible": OpenAICompatibleImageProvider,
    }
    cls = classes.get(kind)
    if cls is None:
        raise ProviderSlotError(f"unknown image slot kind {kind!r}")
    return cls


def build_image_provider(job: SlotJob) -> ImageGenProvider:
    """A per-job image provider on the slot's key/base_url/model (see video twin).

    An unknown kind, or a self-hosted kind without ``base_url``, raises
    :class:`ProviderSlotError` -- never a silent platform provider.
    """
    cls = _image_provider_class(job.kind)
    if job.kind == "openai":
        return cast(Any, cls)(
            api_key=job.secret,
            base_url=job.base_url or OPENAI_IMAGE_DEFAULT_BASE_URL,
            model=job.model or OPENAI_IMAGE_DEFAULT_MODEL,
        )
    if not job.base_url:
        raise ProviderSlotError(f"tenant provider slot {job.label}: base_url is required")
    return cast(Any, cls)(base_url=job.base_url, secret=job.secret, model=job.model)


def video_provider_name(kind: str) -> str:
    """The ``provider`` column value for a video slot job (= the provider's name)."""
    return _video_provider_class(kind).name


def image_provider_name(kind: str) -> str:
    """The ``provider`` column value for an image slot job (= the provider's name)."""
    return _image_provider_class(kind).name
