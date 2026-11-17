"""SQLAlchemy ORM models for ai_content_generation_service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from w2v_common.database.rls import register_rls_ddl


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""

    pass


class VideoGenerationJob(Base):
    """An AI video generation job."""

    __tablename__ = "video_generation_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, default=lambda: str(uuid.uuid7())
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")

    # Generation parameters
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # For image-to-video
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    style: Mapped[str | None] = mapped_column(String(128), nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra_params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Provider-side tracking
    provider_job_id: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # Results
    video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # External URL from provider
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    generation_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    minio_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metadata
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # A job that COMPLETED but is not what the pipeline promised. Distinct from
    # ``error_message``, which belongs to a failed job: a degraded job still has
    # a usable artifact and still bills. Currently written when a clip was
    # delivered without normalization (see ``format_normalization_degradation``);
    # NULL means a clean completion.
    degradation: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Fallback & retry (Phase 2)
    fallback_providers: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    # Callback (Phase 3)
    callback_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    callback_status: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Durable multi-replica dispatch (claim-lease queue). A pod-local
    # ``asyncio.create_task`` dies with its pod; these columns let the reaper on
    # ANY replica reclaim a crashed pod's job. ``job_options`` is the dispatch
    # bundle (kind + budget + batch) so a different pod can reconstruct the work,
    # and doubles as the reclaim-ownership flag (NULL = not managed by the queue).
    claimed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    job_options: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # When the job FIRST started running (first dispatch / first reclaim), never
    # overwritten. The wall-clock lifetime bound is measured from here, NOT from
    # ``created_at``: a job legitimately sitting in a backlog has consumed nothing
    # and must not age out before it ever ran. NULL = never claimed → exempt.
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Tenant provider slot (P5-2). ``provider_slot`` = the NON-secret slot fields
    # (kind/base_url/model/options/ref) sent by the orchestrator; NULL = a
    # platform job, i.e. today's behaviour. ``provider_secret_sealed`` = the
    # slot's secret as an AES-256-GCM envelope bound to ``job_id`` (see
    # ``services/provider_secrets.py``) so the reaper can resume on any pod. The
    # trigger ``purge_provider_secret_on_terminal`` NULLs it the moment the row
    # turns terminal -- whoever writes that status (ORM or SQL reclaim fn).
    provider_slot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    provider_secret_sealed: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    # Inbound ``Idempotency-Key`` (migration 016). Unique per tenant via a partial
    # index; NULL = a submission that sent no key. ``idempotency_fingerprint`` is
    # the sha256 of the request body, so a reused key with a different body is
    # refused instead of silently returning the wrong job. See
    # ``services/idempotency.py``.
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    idempotency_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_video_generation_jobs_tenant_status", "tenant_id", "status"),
        Index("ix_video_generation_jobs_completed_at", "completed_at"),
        Index("ix_video_generation_jobs_status_completed_at", "status", "completed_at"),
        Index(
            "ix_video_generation_jobs_claimable",
            "status",
            "lease_expires_at",
            postgresql_where=text("job_options IS NOT NULL"),
        ),
        Index(
            "uq_video_generation_jobs_tenant_idempotency_key",
            "tenant_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
    )


class VideoGenerationUsage(Base):
    """Daily usage aggregation per tenant."""

    __tablename__ = "video_generation_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    date: Mapped[str] = mapped_column(String(10), nullable=False)
    generations_started: Mapped[int] = mapped_column(Integer, default=0)
    generations_completed: Mapped[int] = mapped_column(Integer, default=0)
    generations_failed: Mapped[int] = mapped_column(Integer, default=0)
    total_duration_generated_s: Mapped[float] = mapped_column(Float, default=0.0)
    total_file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint("tenant_id", "date", name="uq_video_generation_usage_tenant_date"),
    )


class ImageGenJob(Base):
    """An AI image generation job."""

    __tablename__ = "imagegen_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, default=lambda: str(uuid.uuid7())
    )
    asset_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, default=lambda: str(uuid.uuid7())
    )
    batch_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("imagegen_batches.batch_id"),
        nullable=True,
        index=True,
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    style: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size: Mapped[str | None] = mapped_column(String(32), nullable=True)
    quality: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reference_image_urls: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    usage_context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    target_usage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    extra_params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    provider_job_id: Mapped[str | None] = mapped_column(String(256), nullable=True)

    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    minio_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    callback_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    callback_status: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Durable multi-replica dispatch (see VideoGenerationJob for rationale).
    claimed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    job_options: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Wall-clock lifetime anchor — see VideoGenerationJob.started_at.
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Tenant provider slot (P5-2) — see VideoGenerationJob.provider_slot.
    provider_slot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    provider_secret_sealed: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    # Inbound ``Idempotency-Key`` (migration 016). Unique per tenant via a partial
    # index; NULL = a submission that sent no key. ``idempotency_fingerprint`` is
    # the sha256 of the request body, so a reused key with a different body is
    # refused instead of silently returning the wrong job. See
    # ``services/idempotency.py``.
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    idempotency_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_imagegen_jobs_tenant_status", "tenant_id", "status"),
        Index("ix_imagegen_jobs_completed_at", "completed_at"),
        Index("ix_imagegen_jobs_status_completed_at", "status", "completed_at"),
        Index(
            "ix_imagegen_jobs_claimable",
            "status",
            "lease_expires_at",
            postgresql_where=text("job_options IS NOT NULL"),
        ),
        Index(
            "uq_imagegen_jobs_tenant_idempotency_key",
            "tenant_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
    )


class ImageGenBatch(Base):
    """A persisted batch of concept-image generation jobs."""

    __tablename__ = "imagegen_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, default=lambda: str(uuid.uuid7())
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    budget_limit: Mapped[float] = mapped_column(Float, nullable=False, default=5.0)
    total_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    job_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_jobs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_jobs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_imagegen_batches_status", "status"),
        Index("ix_imagegen_batches_tenant_status", "tenant_id", "status"),
    )


# Phase D RLS: strict per-table policies on all 4 tenant_id tables (w2v_common) + the
# SECURITY DEFINER cross-tenant cleanup READ resolver. Registered as after_create
# events so tests' create_all matches the 008_rls migration. Inert until the runtime
# connects as restricted w2v_ai_content_app with TENANT_RLS_MODE=enforce.
register_rls_ddl(Base.metadata)

from src.database.rls_resolvers import register_reclaim_fns, register_resolvers  # noqa: E402

register_resolvers(Base.metadata)
# Durable claim-queue: SECURITY DEFINER cross-tenant reclaim functions for the
# reaper (owner bypasses RLS ENABLE, exactly like the cleanup read resolver).
register_reclaim_fns(Base.metadata)

from src.database.rls_resolvers import register_slot_secret_purge  # noqa: E402

# Tenant provider-slot secrets are purged by trigger on any terminal transition.
register_slot_secret_purge(Base.metadata)
