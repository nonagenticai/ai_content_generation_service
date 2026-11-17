"""Request/response Pydantic models for ai_content_generation_service."""

from __future__ import annotations

import json
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from src.services.provider_slots import VideoProviderSlot

# --- Request Schemas ---


class GenerateRequest(BaseModel):
    """Request to generate an AI video clip."""

    prompt: str = Field(
        ..., min_length=1, max_length=10000, description="Text prompt for video generation"
    )
    provider: str | None = Field(
        None, description="Provider name (runway, kling, veo, sora, seedance)"
    )
    negative_prompt: str | None = Field(None, description="Negative prompt")
    image_url: str | None = Field(None, description="Source image URL for image-to-video")
    duration_s: float | None = Field(
        None, gt=0, le=300, description="Desired clip duration in seconds"
    )
    width: int | None = Field(None, ge=128, le=4096, description="Video width")
    height: int | None = Field(None, ge=128, le=4096, description="Video height")
    style: str | None = Field(None, description="Style preset name")
    seed: int | None = Field(
        None, ge=0, le=4294967295, description="Random seed for reproducibility"
    )
    extra_params: dict | None = Field(None, description="Provider-specific extra parameters")

    @field_validator("extra_params")
    @classmethod
    def validate_extra_params(cls, v: dict | None) -> dict | None:
        if v is None:
            return v
        if len(v) > 20:
            raise ValueError("extra_params must have at most 20 keys")
        serialized = json.dumps(v)
        if len(serialized) > 10240:
            raise ValueError("extra_params must be at most 10KB when serialized")
        return v

    # Bounded like the concept-image sibling (``ConceptImageBatchRequest.budget_limit``
    # is ``gt=0``), plus an upper bound it lacks.
    #
    # What was actually broken: a BARE ``float`` (no bounds at all) let
    # ``{"budget_limit": 1e400}`` parse to ``inf``, and
    # ``VideoGenerationService.check_budget`` then computes ``inf - spent == inf``
    # so every clip is affordable — the only per-request spend cap silently off.
    # ``nan`` failed closed there but still reached a ``Float`` column and the
    # durable-dispatch JSONB options as bare ``NaN`` (not valid RFC-8259).
    #
    # The constraint that rejects all of that is the RANGE, not the flag:
    # ``inf > le`` and every comparison against ``nan`` is False, so ``gt=0,
    # le=100.0`` alone already rejects ``1e400``, ``-1e400``, ``nan``, ``inf``,
    # ``"nan"``, ``"inf"`` and ``"Infinity"``. ``allow_inf_nan=False`` is kept as
    # defence-in-depth (it survives a future widening of the range and gives a
    # clearer error), but no input distinguishes it — do not read it as the guard.
    #
    # * ``le=100.0`` — grounded in the reachable worst case, not invented: the
    #   priciest provider is sora at ``0.10 * duration_s / 5`` = $0.02/s and
    #   ``duration_s`` is capped at 300s, so one clip costs at most $6.00 and a
    #   full batch (10 items) at most $60.00. $100 clears both and also clears
    #   ``QuotaSettings.default_max_daily_spend`` ($50/tenant/day), so the
    #   request bound never becomes the binding constraint on real traffic —
    #   the quota does.
    budget_limit: float = Field(
        default=5.0,
        gt=0,
        le=100.0,
        allow_inf_nan=False,
        description="Maximum budget in USD for this generation",
    )
    fallback_providers: list[str] | None = Field(None, description="Ordered fallback providers")
    callback_url: str | None = Field(None, description="Webhook URL for completion notification")
    provider_slot: VideoProviderSlot | None = Field(
        None,
        description=(
            "Tenant video_gen provider slot (sent by the orchestrator only for a "
            "tenant-sourced slot). Present => `provider`/`fallback_providers` are "
            "ignored and the job runs on the slot's credentials, never a platform "
            "provider. Single-job endpoint only."
        ),
    )


class GenerateBatchRequest(BaseModel):
    """Request to generate multiple AI video clips."""

    # Bounded like the concept-image sibling (``ConceptImageBatchRequest.items``,
    # ``max_length=10``). One request fans out to one DB row + one leased asyncio
    # task + one PAID provider call PER item, and video is the more expensive of
    # the two media paths — so it gets the sibling's cap, not a looser one.
    items: list[GenerateRequest] = Field(..., min_length=1, max_length=10)
    provider: str | None = Field(None, description="Default provider for all items")
    # Same bounds as GenerateRequest.budget_limit — see the note there.
    budget_limit: float = Field(
        default=5.0,
        gt=0,
        le=100.0,
        allow_inf_nan=False,
        description="Maximum total budget in USD for batch",
    )
    callback_url: str | None = Field(None, description="Webhook URL for batch completion")


class PreviewRequest(BaseModel):
    """Request for a quick preview generation (no DB record)."""

    prompt: str = Field(..., min_length=1, max_length=10000, description="Text prompt")
    provider: str | None = Field(None, description="Provider name")
    duration_s: float = Field(default=5.0, gt=0, le=10, description="Duration capped at 10s")


# --- Response Schemas ---


class AsyncJobResponse(BaseModel):
    """Response for an asynchronous video generation job submission."""

    job_id: str
    status: str = "pending"
    poll_url: str


class JobStatusResponse(BaseModel):
    """Current status of a video generation job."""

    job_id: str
    status: str
    provider: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    # Set on a COMPLETED job that is not what the pipeline promised (e.g. a clip
    # delivered without normalization). Distinct from ``error``: the artifact
    # exists and is usable, it is just degraded. None = clean.
    degradation: str | None = None
    result_url: str | None = None


class GenerationResultResponse(BaseModel):
    """Detailed result of a completed video generation job."""

    job_id: str
    status: str
    provider: str | None = None
    prompt: str | None = None
    video_path: str | None = None
    video_url: str | None = None
    file_size_bytes: int | None = None
    actual_duration_s: float | None = None
    generation_time_seconds: float | None = None
    cost: float | None = None
    error: str | None = None
    degradation: str | None = None


class ClipCostItem(BaseModel):
    """Cost info for a single clip."""

    job_id: str
    provider: str
    duration_s: float
    cost: float
    status: str


class CostReport(BaseModel):
    """Budget/cost report for a generation request or batch."""

    total_cost: float = 0.0
    budget_limit: float = 5.0
    clips_generated: int = 0
    clips_skipped: int = 0
    clips_failed: int = 0
    per_clip_costs: list[ClipCostItem] = []


class ClipStatusItem(BaseModel):
    """Status of a single clip within a batch."""

    job_id: str
    status: str
    provider: str | None = None
    cost: float | None = None
    error: str | None = None


class BatchJobResponse(BaseModel):
    """Response for a batch video generation submission."""

    batch_id: str
    job_ids: list[str]
    status: str = "pending"
    cost_report: CostReport | None = None


class BatchStatusResponse(BaseModel):
    """Status of a batch generation job."""

    batch_id: str
    status: str
    clips_submitted: int
    clips_completed: int
    clips_failed: int
    clips_skipped: int
    # Terminal-but-cancelled children. Defaulted so an older client that never
    # sends/reads it is unaffected; exposed so a cancelled clip is VISIBLE rather
    # than merely counted internally.
    clips_cancelled: int = 0
    total_cost: float
    budget_limit: float
    clips: list[ClipStatusItem] = []
    cost_report: CostReport | None = None


class BatchResultResponse(BaseModel):
    """Final results of a batch generation job."""

    batch_id: str
    status: str
    job_ids: list[str]
    clips_completed: int
    clips_failed: int
    total_cost: float
    cost_report: CostReport | None = None


class ProviderInfoResponse(BaseModel):
    """Information about a single video generation provider."""

    name: str
    available: bool
    models: list[str] | None = None
    styles: list[str] | None = None
    max_duration_s: float | None = None
    supports_image_to_video: bool = False


class ProvidersListResponse(BaseModel):
    """List of available providers with the default selection."""

    providers: list[ProviderInfoResponse]
    default_provider: str


class PreviewResponse(BaseModel):
    """Response from preview generation."""

    status: str
    provider: str
    video_base64: str | None = None
    estimated_cost: float | None = None
    error: str | None = None


# --- Job Listing Schemas (Phase 3) ---


class JobListItem(BaseModel):
    """Summary of a single job for listing."""

    job_id: str
    status: str
    provider: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    cost: float | None = None


class JobListResponse(BaseModel):
    """Paginated job listing."""

    items: list[JobListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


# --- Usage Dashboard Schemas (Phase 3) ---


class UsageByProviderItem(BaseModel):
    """Usage breakdown by provider."""

    provider: str
    total_cost: float
    job_count: int


class UsagePeriodItem(BaseModel):
    """Usage for a time period."""

    date: str
    generations_started: int = 0
    generations_completed: int = 0
    generations_failed: int = 0
    total_duration_generated_s: float = 0.0
    total_file_size_bytes: int = 0


class UsageResponse(BaseModel):
    """Usage dashboard response."""

    tenant_id: UUID
    date_from: str | None = None
    date_to: str | None = None
    periods: list[UsagePeriodItem] = []
    by_provider: list[UsageByProviderItem] = []
    total_cost: float = 0.0
    total_generations: int = 0


# --- Provider Health Schemas (Phase 4) ---


class ProviderHealthResponse(BaseModel):
    """Health status for a provider."""

    provider: str
    is_healthy: bool
    consecutive_failures: int = 0
    average_latency_ms: float | None = None
    disabled_at: str | None = None
    cooldown_until: str | None = None


# --- Quota Schemas (Phase 5) ---


class QuotaResponse(BaseModel):
    """Quota status for a tenant."""

    tenant_id: UUID
    daily_generations_used: int = 0
    daily_generations_limit: int = 100
    daily_spend_used: float = 0.0
    daily_spend_limit: float = 50.0
    concurrent_jobs: int = 0
    concurrent_jobs_limit: int = 10
