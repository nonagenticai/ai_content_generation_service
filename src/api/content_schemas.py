"""Pydantic models for concept-image generation endpoints."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator

from src.services.provider_slots import ImageProviderSlot


class ConceptImageRequest(BaseModel):
    """Request to generate a single concept image."""

    prompt: str = Field(..., min_length=1, max_length=10000)
    provider: str | None = Field(default=None)
    negative_prompt: str | None = Field(default=None)
    style: str | None = Field(default=None)
    size: str | None = Field(default=None, pattern=r"^\d{2,5}x\d{2,5}$")
    quality: str | None = Field(default=None)
    reference_image_urls: list[str] | None = Field(default=None, max_length=5)
    usage_context: dict | None = Field(default=None)
    target_usage: str | None = Field(default=None, max_length=64)
    extra_params: dict | None = Field(default=None)
    callback_url: str | None = Field(default=None)
    provider_slot: ImageProviderSlot | None = Field(
        default=None,
        description=(
            "Tenant image provider slot (sent by the orchestrator only for a "
            "tenant-sourced slot). Present => `provider` is ignored and the job "
            "runs on the slot's credentials, never a platform provider. "
            "Single-job endpoint only."
        ),
    )

    @field_validator("extra_params")
    @classmethod
    def validate_extra_params(cls, value: dict | None) -> dict | None:
        if value is None:
            return value
        if len(value) > 20:
            raise ValueError("extra_params must have at most 20 keys")
        serialized = json.dumps(value)
        if len(serialized) > 10240:
            raise ValueError("extra_params must be at most 10KB when serialized")
        return value


class ConceptImageBatchRequest(BaseModel):
    """Request to generate multiple concept images."""

    items: list[ConceptImageRequest] = Field(..., min_length=1, max_length=10)
    provider: str | None = Field(default=None)
    budget_limit: float = Field(default=5.0, gt=0)
    callback_url: str | None = Field(default=None)


class ConceptImageJobSubmissionResponse(BaseModel):
    """Submission response for an async concept-image job."""

    job_id: str
    status: str = "pending"
    poll_url: str
    result_url: str


class ConceptImageBatchSubmissionResponse(BaseModel):
    """Submission response for an async concept-image batch."""

    batch_id: str
    job_ids: list[str]
    status: str = "pending"
    status_url: str
    result_url: str


class ConceptImageJobResponse(BaseModel):
    """Current status of a concept-image generation job."""

    job_id: str
    asset_id: str | None = None
    status: str
    provider: str | None = None
    url: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    download_url: str | None = None
    presigned_url: str | None = None
    image_url: str | None = None


class ConceptImageAssetResponse(BaseModel):
    """Resolved concept-image asset metadata."""

    job_id: str
    asset_id: str
    status: str
    provider: str | None = None
    prompt: str | None = None
    url: str | None = None
    image_path: str | None = None
    image_url: str | None = None
    download_url: str | None = None
    presigned_url: str | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    file_size_bytes: int | None = None
    cost: float | None = None
    usage_context: dict | None = None
    target_usage: str | None = None
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    metadata: dict | None = None


class ConceptImageBatchResponse(BaseModel):
    """Status and result set for a submitted concept-image batch."""

    batch_id: str
    status: str
    job_ids: list[str]
    job_count: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    budget_limit: float
    total_cost: float = 0.0
    created_at: str | None = None
    completed_at: str | None = None
    results: list[ConceptImageAssetResponse] = Field(default_factory=list)
