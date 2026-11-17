"""Core AI video generation orchestration service."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from src.config import Settings
    from src.database.models import VideoGenerationJob
    from src.services.cancel_registry import CancelRegistry
    from src.services.provider_slots import SlotJob
    from src.services.providers.base import OnSubmit, VideoGenerationProvider

from src.observability.metrics import (
    record_job_completed,
    record_job_failed,
    record_job_finished,
    record_job_started,
    record_normalization_degraded,
)
from src.services.job_manager import JobManager, JobStateConflictError
from src.services.normalizer import NormalizationOutcome, VideoNormalizer
from src.services.provider_health import ProviderHealthMonitor
from src.services.provider_selector import ProviderSelector
from src.services.providers.base import VideoProviderRegistry
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)

_DEGRADATION_PREFIX = "video_not_normalized"


def format_normalization_degradation(reason: str | None) -> str | None:
    """Render a normalization ``reason`` as the job row's degradation marker.

    Returns None for ``reason is None`` -- i.e. normalized fine, nothing to
    normalize, or normalization switched off. The reason is kept IN the string
    because "degraded" alone cannot distinguish a one-off ffmpeg timeout from
    ``ffmpeg_missing``, which means the pod can never normalize anything.
    """
    if reason is None:
        return None
    return f"{_DEGRADATION_PREFIX}: {reason}"


def observe_normalization(outcome: NormalizationOutcome) -> str | None:
    """Count a degraded normalization and render its persisted marker.

    Kept out of ``_attempt_provider`` so the success path stays one straight
    line -- the metric is a side effect of the SAME decision, not a branch the
    retry loop has to carry.
    """
    degradation = format_normalization_degradation(outcome.reason)
    if degradation is not None:
        record_normalization_degraded(outcome.reason or "unknown")
    return degradation


@dataclass
class _AdoptableTask:
    """A provider task a PREVIOUS run of this job already paid for.

    Read from ``video_generation_jobs.provider_job_id``, which is now written
    the instant the provider accepts the task rather than only at completion.
    Its presence on a job that is being processed again means the earlier run
    died (pod roll / eviction / crash) after the money was spent, and the
    lease-reclaim in ``010_claim_refund_lifetime`` handed the row back as
    ``pending``. Re-attaching to ``task_id`` is what stops that reclaim from
    submitting — and paying for — a second generation.

    ``provider`` is load-bearing: a task id is only meaningful to the provider
    that issued it, so a fallback provider in the chain must never adopt it.
    """

    provider: str
    task_id: str


class VideoGenerationService:
    """Orchestrates AI video generation across providers.

    Manages the full lifecycle of video generation jobs including
    provider selection, budget enforcement, retry logic with
    fallback, video normalization, and storage.

    Attributes:
        _storage: Storage backend for video files.
        _registry: Registry of available video providers.
        _default_provider: Default provider name when none specified.
        _semaphore: Concurrency limiter for parallel generations.
        _max_retries: Maximum retry attempts per provider.
        _fallback_enabled: Whether to try fallback providers on failure.
        _health_monitor: Optional provider health tracker.
        _selector: Optional strategy-based provider selector.
        _normalizer: Optional FFmpeg-based video normalizer.
        _active_tasks: Map of job_id to running asyncio tasks.
    """

    def __init__(
        self,
        storage: StorageService,
        registry: VideoProviderRegistry,
        default_provider: str,
        parallel_requests: int = 3,
        max_retries: int = 2,
        fallback_enabled: bool = True,
        health_monitor: ProviderHealthMonitor | None = None,
        selector: ProviderSelector | None = None,
        normalizer: VideoNormalizer | None = None,
        metering_redis: object | None = None,
        cancel_registry: CancelRegistry | None = None,
    ) -> None:
        self._storage = storage
        self._registry = registry
        self._default_provider = default_provider
        self._semaphore = asyncio.Semaphore(parallel_requests)
        self._max_retries = max_retries
        self._fallback_enabled = fallback_enabled
        self._health_monitor = health_monitor
        self._selector = selector
        self._normalizer = normalizer
        self._active_tasks: dict[str, asyncio.Task] = {}
        # Cross-pod cancellation: a cancel request on ANY replica sets a Redis
        # flag that this job's processor polls at its checkpoints, so a task on
        # another pod is reachable (the pod-local ``_active_tasks`` never was).
        self._cancel_registry = cancel_registry
        # Metering: None = disabled (METERING_REDIS_URL unset/unreachable).
        # Emission is GC-safe fire-and-forget via w2v_common.metering, which
        # owns the in-flight task set and resolves the cost-bus stream.
        self._metering_redis = metering_redis

    def register_task(self, job_id: str, task: asyncio.Task) -> None:
        """Register an active asyncio.Task for a job."""
        self._active_tasks[job_id] = task

    def unregister_task(self, job_id: str) -> None:
        """Unregister a completed task."""
        self._active_tasks.pop(job_id, None)

    async def cancel_job(self, db: AsyncSession, job_id: str, tenant_id: str) -> bool:
        """Cancel an in-flight job owned by ``tenant_id``. True if cancelled.

        Ownership is resolved BEFORE any side effect, and that ordering is
        the security boundary: the Redis cancel flag and the in-process
        ``task.cancel()`` are not database operations, so RLS cannot guard
        them. Firing either one first would let a caller kill another
        tenant's in-flight generation with nothing but a job id. A foreign
        job is therefore treated exactly like an unknown one — ``False``,
        no flag, no task touched, no existence disclosed.

        Guarantees that if a task was registered for ``job_id`` it is
        unregistered exactly once on every exit path — including
        unexpected exceptions raised by ``mark_cancelled`` or
        ``task.cancel()`` / ``await task``. Without this guarantee the
        in-memory registry would leak entries and a later scheduled
        callback could fire on a job we believed cancelled.
        """
        from src.services.job_manager import JobStateConflictError

        # --- Ownership gate. Must stay ABOVE the registry/task calls. ---
        job = await JobManager.get_job(db, job_id, tenant_id=tenant_id)
        if job is None:
            logger.warning("Cancel denied: job_id=%s not owned by tenant=%s", job_id, tenant_id)
            return False

        # Cross-pod: set the cancel flag as the first side effect so a task
        # running on any other replica observes it at its next checkpoint,
        # even when this pod holds no local task for the job.
        if self._cancel_registry is not None:
            await self._cancel_registry.request_cancel(job_id)

        task = self._active_tasks.get(job_id)
        if task is None:
            # No local task: a pending job is cancellable at the DB level; a
            # processing job is running on another pod (or a just-cleared local
            # task) — the flag will drive it to cancelled at its next checkpoint.
            if job.status == "pending":
                try:
                    await JobManager.mark_cancelled(db, job_id)
                except JobStateConflictError:
                    return False
                return True
            # A processing job is running on another pod — the flag drives it to
            # cancelled at its next checkpoint; report accepted.
            return job.status == "processing"

        try:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except (RuntimeError, OSError) as exc:
                logger.warning(
                    "Awaiting cancelled task for job %s raised %s: %s",
                    job_id,
                    type(exc).__name__,
                    exc,
                )
            try:
                await JobManager.mark_cancelled(db, job_id)
            except JobStateConflictError:
                # Job already transitioned to a terminal state
                # (completed/failed) by the time we got here — nothing
                # to cancel at the DB level. The finally block still
                # unregisters the task.
                return False
            return True
        finally:
            self.unregister_task(job_id)

    def estimate_cost(self, provider_name: str, duration_s: float) -> float | None:
        """Estimate cost for a generation request. Returns None if provider not found."""
        provider = self._registry.get(provider_name)
        if provider is None:
            return None
        return provider.estimate_cost(duration_s)

    def estimate_request_cost(self, provider: str | None, duration_s: float | None) -> float:
        """USD estimate for a request that has NOT been created yet.

        Admission (``check_quota``) needs the cost of the job it is about to
        admit; without it the spend cap degenerates to ``daily_spend + 0.0 >
        max``, which can only ever trip on budget already spent and never on
        the generation being let through.

        Resolves provider and duration exactly the way ``create_job`` will, so
        the number priced here is the number that will be charged. An unknown
        provider yields 0.0 — the pre-existing behaviour, and the only safe
        direction for something we cannot price.
        """
        from src.config import get_settings

        effective_duration = duration_s or get_settings().generation.default_duration_s
        provider_name = provider or self._resolve_provider(effective_duration)
        return self.estimate_cost(provider_name, effective_duration) or 0.0

    async def create_job(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        prompt: str,
        provider: str | None = None,
        negative_prompt: str | None = None,
        image_url: str | None = None,
        duration_s: float | None = None,
        width: int | None = None,
        height: int | None = None,
        style: str | None = None,
        seed: int | None = None,
        extra_params: dict | None = None,
        fallback_providers: list[str] | None = None,
        callback_url: str | None = None,
        idempotency_key: str | None = None,
        idempotency_fingerprint: str | None = None,
    ) -> str:
        """Create a new video generation job in the database.

        Args:
            db: Async database session.
            tenant_id: Tenant owning the job.
            prompt: Text prompt for video generation.
            provider: Explicit provider name, or None to auto-select.
            negative_prompt: Optional negative prompt.
            image_url: Optional source image for image-to-video.
            duration_s: Desired clip duration in seconds.
            width: Video width in pixels.
            height: Video height in pixels.
            style: Style preset name.
            seed: Random seed for reproducibility.
            extra_params: Provider-specific extra parameters.
            fallback_providers: Ordered list of fallback provider names.
            callback_url: Webhook URL for completion notification.
            idempotency_key: Inbound ``Idempotency-Key``; a live duplicate raises
                ``IdempotentReplayError`` carrying the existing job.
            idempotency_fingerprint: sha256 of the request body bound to the key.

        Returns:
            The job_id of the newly created job.
        """
        effective_provider = provider or self._resolve_provider(duration_s)
        job = await JobManager.create_job(
            db,
            tenant_id=tenant_id,
            provider=effective_provider,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_url=image_url,
            duration_s=duration_s,
            width=width,
            height=height,
            style=style,
            seed=seed,
            extra_params=extra_params,
            fallback_providers=fallback_providers,
            callback_url=callback_url,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
        )
        return job.job_id

    def _resolve_provider(self, duration_s: float | None = None) -> str:
        """Resolve provider using selector strategy or default."""
        if self._selector is not None:
            try:
                return self._selector.select(self._registry, self._health_monitor, duration_s)
            except RuntimeError, OSError:
                logger.warning("Provider selector failed, falling back to default")
        return self._default_provider

    async def check_budget(
        self,
        provider_name: str,
        duration_s: float,
        budget_limit: float,
        spent: float,
    ) -> tuple[bool, float]:
        """Check if a clip fits within remaining budget.

        Returns (allowed, estimated_cost).
        """
        estimated = self.estimate_cost(provider_name, duration_s)
        if estimated is None:
            return True, 0.0
        remaining = budget_limit - spent
        return estimated <= remaining, estimated

    async def _is_cancelled(self, job_id: str) -> bool:
        """True if a cross-pod cancel has been requested for this job."""
        if self._cancel_registry is None:
            return False
        return await self._cancel_registry.is_cancelled(job_id)

    async def process_job(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: str,
        budget_limit: float = 5.0,
        spent_so_far: float = 0.0,
    ) -> float:
        """Process a pending video generation job.

        Returns the cost of this job (0.0 if skipped/failed).
        """
        async with self._semaphore:
            return await self._process_job_inner(db, job_id, tenant_id, budget_limit, spent_so_far)

    @staticmethod
    def _build_provider_chain(job: Any, fallback_enabled: bool) -> list[str]:
        """Build the ordered provider chain: primary first, then fallbacks.

        Deduplicates so the primary provider is never retried as a
        "fallback" of itself. ``fallback_enabled=False`` collapses the
        chain to just the primary provider.
        """
        chain = [job.provider]
        if fallback_enabled and job.fallback_providers:
            chain.extend(p for p in job.fallback_providers if p not in chain)
        return chain

    def _provider_is_available(self, provider_name: str) -> bool | None:
        """Return health-monitor availability for a provider.

        Returns:
            True/False if the health monitor reports a status,
            None if no monitor is configured (caller treats as available),
            True (best-effort) if the monitor itself raises -- we never
            want a flaky monitor to take down all generation.
        """
        if self._health_monitor is None:
            return None
        try:
            return self._health_monitor.is_available(provider_name)
        except OSError, RuntimeError:
            logger.debug("Failed to record health metric", exc_info=True)
            return True

    async def _store_video_artifacts(
        self, job_id: str, provider_name: str, video_data: bytes | None
    ) -> tuple[str | None, str | None, int]:
        """Persist video bytes to local + MinIO storage.

        This runs INSIDE ``_attempt_provider``'s ``for attempt`` try, i.e. after
        the provider has already been paid for the clip we are holding. MinIO and
        httpx surface a bucket 5xx / 429 as ``httpx.HTTPStatusError``, which
        ``is_retriable_error`` classifies as retriable — which would send the
        retry gate back through ``_try_generate_once`` and buy the SAME clip a
        second time, with no idempotency key anywhere on that path. Every failure
        here is therefore re-raised as ``StorageError``, which is never retriable,
        so a storage fault fails the job instead of double-charging it.

        Returns:
            (stored_path, minio_key, file_size). All three are None/0
            when ``video_data`` is empty (provider returned a URL only).

        Raises:
            StorageError: any local- or MinIO-side persistence failure.
        """
        from src.services.exceptions import StorageError

        if not video_data:
            return None, None, 0
        filename = f"{provider_name}_clip.mp4"
        try:
            stored_path = await self._storage.store_video(job_id, video_data, filename)
            file_size = len(video_data)
            minio_key = await self._storage.upload_to_minio(job_id, video_data, filename)
        # Deliberately broad: the point is to reclassify EVERY storage fault,
        # whatever SDK it came from (httpx, minio S3Error, OSError), as
        # non-retriable. CancelledError is a BaseException and still passes through.
        except Exception as exc:
            raise StorageError(
                f"Failed to store video artifacts for job {job_id}: {exc}", cause=exc
            ) from exc
        return stored_path, minio_key, file_size

    async def _maybe_normalize(self, video_data: bytes | None, job_id: str) -> NormalizationOutcome:
        """Normalize video via FFmpeg if configured, reporting what happened.

        Returns an outcome rather than bytes so the caller can tell a
        normalized clip from one that came back untouched. Falling back to the
        raw bytes is still correct -- we already paid for the clip -- but a
        fallback is a DEGRADATION and must be recorded as one, otherwise
        video_compiler concatenates a clip of the wrong resolution/fps (or with
        the provider's audio track still on it) into a delivered video while
        every job in the run reports success.

        ``reason is None`` covers the two non-faults: no bytes to normalize,
        and normalization deliberately switched off (``self._normalizer is
        None`` when ``settings.normalization.enabled`` is false).
        """
        if not video_data or self._normalizer is None:
            return NormalizationOutcome(data=video_data or b"", normalized=False, reason=None)
        try:
            return await self._normalizer.normalize(video_data, job_id)
        except (OSError, RuntimeError) as norm_err:
            # The normalizer's own arms already return an outcome; this catches
            # a crash of the normalizer itself, which is equally a degradation.
            logger.warning("Normalization failed for %s, using raw: %s", job_id, norm_err)
            return NormalizationOutcome(
                data=video_data,
                normalized=False,
                reason=f"normalizer_error_{type(norm_err).__name__}",
            )

    def _record_health_success(
        self, provider_name: str, elapsed_s: float, slot_job: SlotJob | None = None
    ) -> None:
        """Record provider success in the health monitor (best effort).

        Skipped for a tenant slot run: its outcome says nothing about the
        PLATFORM provider of the same name.
        """
        if self._health_monitor is None or slot_job is not None:
            return
        try:
            self._health_monitor.record_success(provider_name, elapsed_s * 1000)
        except OSError, RuntimeError:
            logger.debug("Failed to record health metric", exc_info=True)

    def _record_health_failure(self, provider_name: str) -> None:
        """Record provider failure in the health monitor (best effort)."""
        if self._health_monitor is None:
            return
        try:
            self._health_monitor.record_failure(provider_name)
        except OSError, RuntimeError:
            logger.debug("Failed to record health metric", exc_info=True)

    def _emit_cost_event(
        self,
        *,
        job_id: str,
        tenant_id: str,
        provider: str,
        duration_s: float,
        estimated_cost: float,
        elapsed_s: float,
        provider_request_id: str,
        file_size_bytes: int,
        slot_job: SlotJob | None = None,
    ) -> None:
        """Fire-and-forget cost event to improvement-loop:costs.

        No-ops when METERING_REDIS_URL is unset. license_id + subject_id come
        from per-request ContextVars set by w2v_common.auth.middleware.

        A tenant provider-slot job ran on the TENANT's key: it is emitted as
        tenant-funded BYO (``SlotJob.byo_usage_extra``) with ``cost_usd=0`` so it
        is never billed at platform provider prices.
        """
        if self._metering_redis is None:
            return
        from typing import Any, cast

        from w2v_common.auth.identity_context import (
            get_attribution_subject_id_from_context,
            get_license_id_from_context,
        )

        from src.services.metering import (
            APP_ID,
            COST_SOURCE,
            SERVICE_KEY,
            job_event_key,
            platform_funding,
            schedule_cost_event,
        )
        from src.services.provider_slots import FUNDING_TENANT_BYO

        schedule_cost_event(
            cast(Any, self._metering_redis),
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id=tenant_id,
            license_id=get_license_id_from_context(),
            subject_id=get_attribution_subject_id_from_context(),
            app_id=APP_ID,
            provider=provider,
            model=provider,  # video providers expose one model per provider
            # video_seconds per EMITTER_GUIDE. Billing recomputes from the emitted
            # UNITS, so ``int()`` here was a money bug twice over: a sub-second
            # clip emitted 0 billable units (free generation) and every fractional
            # duration was floored, systematically under-billing by up to 1s.
            # ``max(1, ...)`` mirrors the image path's megapixel guard.
            input_tokens=max(1, round(duration_s)),
            output_tokens=0,
            latency_ms=int(elapsed_s * 1000),
            # local estimate; metering recomputes. 0 for a tenant-key (BYO) job.
            cost_usd=estimated_cost if slot_job is None else 0.0,
            provider_request_id=provider_request_id,
            usage_extra={
                "unit": "video_seconds",
                "duration_s": duration_s,
                "file_size_bytes": file_size_bytes,
                **(slot_job.byo_usage_extra() if slot_job is not None else {}),
            },
            # Tenant key -> tenant_byo; a platform job states its provider's
            # declared funding (an undeclared provider raises -- never guessed).
            funding=FUNDING_TENANT_BYO if slot_job is not None else platform_funding(provider),
            event_key=job_event_key(job_id, "video"),
        )

    async def _persist_success(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: str,
        result: Any,
        stored_path: str | None,
        minio_key: str | None,
        file_size: int,
        elapsed: float,
        estimated_cost: float,
        provider_name: str,
        degradation: str | None = None,
    ) -> bool:
        """Persist a successful generation: provider_job_id, mark_completed, usage.

        Args:
            degradation: Non-None when the clip completed but is not what the
                pipeline promised -- currently, a clip that was NOT normalized
                while normalization was enabled. It is written to the terminal
                row so a downstream consumer (and any audit of a delivered
                run) can tell a degraded completion from a clean one.

        Returns True on success, False if a state conflict makes the
        success no longer relevant (e.g. job was cancelled while we
        were storing the bytes).
        """
        if result.provider_job_id:
            try:
                await JobManager.set_provider_job_id(db, job_id, result.provider_job_id)
            except JobStateConflictError:
                logger.info(
                    "Job %s no longer processing; cannot attach provider_job_id",
                    job_id,
                )

        try:
            await JobManager.mark_completed(
                db,
                job_id,
                video_path=stored_path,
                video_url=result.video_url,
                file_size_bytes=file_size,
                actual_duration_s=result.duration_s,
                generation_time_seconds=elapsed,
                cost=estimated_cost,
                minio_key=minio_key,
                degradation=degradation,
            )
        except JobStateConflictError:
            logger.info(
                "Job %s no longer processing when result arrived; cancelled or superseded",
                job_id,
            )
            return False

        await JobManager.record_usage(
            db,
            tenant_id=tenant_id,
            generations_completed=1,
            duration_generated=result.duration_s or 0.0,
            file_size=file_size,
        )
        await db.commit()
        record_job_completed(
            provider=provider_name,
            duration_s=elapsed,
            cost=estimated_cost,
        )
        return True

    async def _enforce_budget(
        self,
        db: AsyncSession,
        job_id: str,
        provider_name: str,
        duration_s: float,
        budget_limit: float,
        spent_so_far: float,
    ) -> tuple[bool, float]:
        """Pre-flight budget check; mark_failed + return (False, _) if blocked.

        Returns (allowed, estimated_cost). When False, the job has
        already been transitioned to failed and the caller must return
        0.0 to the dispatcher.
        """
        allowed, estimated_cost = await self.check_budget(
            provider_name, duration_s, budget_limit, spent_so_far
        )
        if allowed:
            return True, estimated_cost
        try:
            await JobManager.mark_failed(
                db,
                job_id,
                f"Budget exceeded: estimated ${estimated_cost:.2f}, "
                f"remaining ${budget_limit - spent_so_far:.2f}",
            )
        except JobStateConflictError:
            logger.info(
                "Job %s already terminal; skipping budget-exceeded mark_failed",
                job_id,
            )
        record_job_failed(provider=provider_name)
        return False, estimated_cost

    def _make_submit_recorder(self, db: AsyncSession, job_id: str) -> OnSubmit:
        """Build the callback that durably records a provider task id at submit.

        This is the whole cross-pod double-billing guard. ``provider_job_id``
        used to be written only in ``_persist_success``, i.e. after download —
        so a pod torn down mid-generation (``replicas: 1`` + ``maxSurge: 0``
        means a roll kills the pod before its replacement exists) left the row
        with NO record of the paid task. The reclaim SQL then reset it to
        ``pending`` and the reaper re-ran the full pipeline, submitting again.

        ``set_provider_job_id`` commits, so the id survives the process.
        """

        async def _record(task_id: str) -> None:
            try:
                await JobManager.set_provider_job_id(db, job_id, task_id)
            except JobStateConflictError:
                logger.info(
                    "Job %s no longer processing; cannot attach provider_job_id %s",
                    job_id,
                    task_id,
                )
            except Exception:
                # NEVER propagate. The provider has ALREADY been paid by the
                # time this runs; letting a DB fault escape would unwind into
                # the retry loop and buy the same clip a second time — exactly
                # the defect this callback exists to prevent. Worst case on a
                # swallowed failure is the pre-fix behaviour, logged loudly.
                logger.exception(
                    "Failed to persist provider_job_id %s for job %s; a reclaim of this "
                    "job can no longer re-attach and would re-submit at provider cost",
                    task_id,
                    job_id,
                )

        return _record

    async def _try_generate_once(
        self,
        db: AsyncSession,
        provider: Any,
        job: Any,
        job_id: str,
        duration_s: float,
        settings: Any,
        resume_task_id: str | None = None,
    ) -> tuple[Any, float]:
        """Call provider.generate and return (result, elapsed_seconds).

        The result may indicate failure -- the caller is responsible
        for raising RuntimeError when result.success is False.

        ``resume_task_id`` re-attaches to an already-paid provider task; in that
        case no submit happens, so there is nothing new to record.
        """
        start = time.monotonic()
        result = await provider.generate(
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            image_url=job.image_url,
            duration_s=duration_s,
            width=job.width or settings.generation.default_width,
            height=job.height or settings.generation.default_height,
            style=job.style,
            seed=job.seed,
            extra_params=job.extra_params,
            resume_task_id=resume_task_id,
            on_submit=None if resume_task_id else self._make_submit_recorder(db, job_id),
        )
        elapsed = time.monotonic() - start
        return result, elapsed

    @staticmethod
    def _resume_task_id(
        job_id: str,
        provider_name: str,
        attempt: int,
        adoptable: _AdoptableTask | None,
    ) -> str | None:
        """The already-paid task this attempt should re-attach to, if any.

        Adoption is allowed exactly once, on ``attempt == 0`` of the provider
        that issued the id:

        * a LATER attempt only happens because the adopted task turned out to
          be unusable, so retrying it would just re-fail forever; and
        * a DIFFERENT provider cannot interpret another provider's id.

        Both fall through to a normal submit, which is correct — the adopted
        task is dead, so there is nothing left to double-bill against.
        """
        if attempt != 0 or adoptable is None or adoptable.provider != provider_name:
            return None
        logger.warning(
            "Job %s was reclaimed with provider task %s already submitted to %s; "
            "re-attaching instead of re-submitting (avoids a second charge)",
            job_id,
            adoptable.task_id,
            provider_name,
        )
        return adoptable.task_id

    async def _attempt_provider(
        self,
        db: AsyncSession,
        job: Any,
        job_id: str,
        tenant_id: str,
        provider: Any,
        provider_name: str,
        duration_s: float,
        estimated_cost: float,
        provider_errors: dict[str, str],
        settings: Any,
        adoptable: _AdoptableTask | None = None,
        slot_job: SlotJob | None = None,
    ) -> tuple[str, float]:
        """Run the retry loop for a single provider.

        ``slot_job`` marks a tenant provider-slot run: health is not recorded
        (a tenant's bad key must not mark the PLATFORM provider unhealthy) and
        failures are classified by ``_decide_after_attempt_error``'s slot rules.

        ``adoptable`` carries a task a previous run of this job already paid
        for. It is consumed at most ONCE, on the first attempt of the provider
        that issued it: a later attempt only happens because the adopted task
        turned out to be unusable (failed / expired provider-side), and a
        fallback provider cannot make sense of another provider's id.

        Returns:
            ("completed", cost): job finished -- caller should return cost.
            ("cancelled_or_conflict", 0.0): terminal state already reached.
            ("terminal", 0.0): storage failed; job already marked failed and
                the chain must STOP (a different generation provider cannot
                fix MinIO, it can only re-buy the clip).
            ("exhausted", 0.0): retries exhausted, try next provider in chain.
        """
        for attempt in range(self._max_retries + 1):
            resume_task_id = self._resume_task_id(job_id, provider_name, attempt, adoptable)
            try:
                # Cross-pod cancel: a cancel requested on another replica lands
                # here as a raised CancelledError, handled below like a local
                # task.cancel() — the job is driven to ``cancelled``.
                if await self._is_cancelled(job_id):
                    raise asyncio.CancelledError
                result, elapsed = await self._try_generate_once(
                    db, provider, job, job_id, duration_s, settings, resume_task_id
                )
                if not result.success:
                    raise RuntimeError(result.error or "Generation failed")

                normalization = await self._maybe_normalize(result.video_data, job_id)
                degradation = observe_normalization(normalization)
                stored_path, minio_key, file_size = await self._store_video_artifacts(
                    job_id, provider_name, normalization.data
                )
                self._record_health_success(provider_name, elapsed, slot_job)

                persisted = await self._persist_success(
                    db,
                    job_id,
                    tenant_id,
                    result,
                    stored_path,
                    minio_key,
                    file_size,
                    elapsed,
                    estimated_cost,
                    provider_name,
                    degradation=degradation,
                )
                record_job_finished()
                if not persisted:
                    return "cancelled_or_conflict", 0.0
                self._emit_cost_event(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    provider=provider_name,
                    duration_s=duration_s,
                    estimated_cost=estimated_cost,
                    elapsed_s=elapsed,
                    provider_request_id=str(result.provider_job_id or ""),
                    file_size_bytes=file_size,
                    slot_job=slot_job,
                )
                return "completed", estimated_cost

            except asyncio.CancelledError:
                try:
                    await JobManager.mark_cancelled(db, job_id)
                except JobStateConflictError:
                    logger.info("Job %s already terminal on CancelledError", job_id)
                except Exception:
                    # mark_cancelled's own write failed for a reason OTHER
                    # than a state conflict (DB down, pool exhausted, ...).
                    # Letting this propagate would replace the
                    # CancelledError with a generic exception -- the outer
                    # _process_job_bg catch-all reclassifies anything it
                    # doesn't recognize as CancelledError as "Unexpected
                    # error during video generation" and marks the job
                    # FAILED instead of CANCELLED, exactly backwards for a
                    # caller who asked to cancel. Log loudly and still
                    # report the cancel as accepted; the row is left
                    # non-terminal and is reclaimed by the reaper/lease path
                    # like any other stranded job.
                    logger.exception(
                        "Failed to persist cancellation for job %s after CancelledError",
                        job_id,
                    )
                record_job_finished()
                return "cancelled_or_conflict", 0.0

            except Exception as exc:  # noqa: BLE001 -- logged by _log_attempt_failure
                self._log_attempt_failure(job_id, provider_name, attempt, exc, slot_job)
                decision = await self._decide_after_attempt_error(
                    db, exc, job_id, tenant_id, provider_name, attempt, provider_errors, slot_job
                )
                if decision == "retry":
                    continue
                return decision, 0.0

        return "exhausted", 0.0

    @staticmethod
    def _log_attempt_failure(
        job_id: str,
        provider_name: str,
        attempt: int,
        exc: Exception,
        slot_job: SlotJob | None,
    ) -> None:
        """Log one failed attempt. A slot job logs no traceback, only scrubbed text."""
        if slot_job is None:
            logger.error(
                "Video generation job %s failed (provider=%s, attempt=%d): %s",
                job_id,
                provider_name,
                attempt + 1,
                exc,
                exc_info=True,
            )
            return
        logger.error(
            "Video generation job %s failed (slot=%s, attempt=%d): %s",
            job_id,
            slot_job.label,
            attempt + 1,
            slot_job.failure_message(exc),
        )

    async def _decide_after_attempt_error(
        self,
        db: AsyncSession,
        exc: Exception,
        job_id: str,
        tenant_id: str,
        provider_name: str,
        attempt: int,
        provider_errors: dict[str, str],
        slot_job: SlotJob | None = None,
    ) -> str:
        """Classify one failed attempt into the loop's next move.

        A tenant slot (``slot_job``) records a slot-naming, secret-free error,
        never touches platform provider health, and treats a provider 4xx
        (auth / bad request, not 429) as final -- no retry, and since a slot
        chain has exactly one provider, no fallback either.

        Returns one of "retry" (caller continues the attempt loop),
        "terminal", "cancelled_or_conflict", or "exhausted" -- the last
        three are returned straight out of ``_attempt_provider``.
        """
        from src.services.errors import is_retriable_error
        from src.services.exceptions import StorageError, format_error
        from src.services.provider_slots import is_permanent_slot_error

        provider_errors[provider_name] = (
            format_error(exc) if slot_job is None else slot_job.failure_message(exc)
        )

        # Storage is a chain-WIDE dependency, not a provider fault. The provider
        # has already been paid for the clip we are holding, and a MinIO /
        # local-disk outage fails identically for every other provider in the
        # chain. Answering "exhausted" here makes ``_process_job_inner``
        # ``continue`` to the next provider, so one storage outage buys
        # ``1 + len(fallback_providers)`` clips and discards every one of them.
        # ``_store_video_artifacts`` re-raises every storage fault as
        # StorageError, so this branch is the whole storage surface. "terminal"
        # is the sentinel ``_process_job_inner`` short-circuits on; its contract
        # is that the job is ALREADY in a terminal DB state, which is what
        # ``_finalize_storage_failure`` establishes.
        if isinstance(exc, StorageError):
            await self._finalize_storage_failure(
                db, job_id, tenant_id, provider_name, provider_errors
            )
            return "terminal"

        if slot_job is not None:
            if is_permanent_slot_error(exc):
                return "exhausted"
        else:
            self._record_health_failure(provider_name)

        if is_retriable_error(exc) and attempt < self._max_retries:
            if not await self._await_retry(db, job_id, attempt, provider_errors):
                return "cancelled_or_conflict"
            return "retry"
        return "exhausted"

    async def _await_retry(
        self,
        db: AsyncSession,
        job_id: str,
        attempt: int,
        provider_errors: dict[str, str],
    ) -> bool:
        """Sleep with exponential backoff and re-mark the job processing.

        Returns False if a state conflict means the job is no longer
        ours to retry (caller should bail out). Centralizes the
        mark_retrying / sleep / mark_processing dance so the retry path
        does not contribute three branches to ``_attempt_provider``.
        """
        backoff = min(2**attempt * 2, 30)
        logger.info(
            "Retrying job %s in %.1fs (attempt %d/%d)",
            job_id,
            backoff,
            attempt + 2,
            self._max_retries + 1,
        )
        try:
            await JobManager.mark_retrying(
                db, job_id, attempt + 1, provider_errors[next(reversed(provider_errors))]
            )
        except JobStateConflictError:
            logger.info("Job %s not processing; retry path bailed out", job_id)
            record_job_finished()
            return False
        await asyncio.sleep(backoff)
        try:
            await JobManager.mark_processing(db, job_id)
        except JobStateConflictError:
            logger.info("Job %s not pending after retry; bailing out", job_id)
            record_job_finished()
            return False
        return True

    async def _finalize_all_failed(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: str,
        provider_errors: dict[str, str],
    ) -> None:
        """Mark the job failed after every provider in the chain failed."""
        if provider_errors:
            aggregated_error = "; ".join(f"{name}: {err}" for name, err in provider_errors.items())
        else:
            aggregated_error = "All providers failed"
        try:
            await JobManager.mark_failed(db, job_id, aggregated_error)
        except JobStateConflictError:
            logger.info("Job %s already terminal at end of all-providers-failed", job_id)
        await JobManager.record_usage(db, tenant_id=tenant_id, is_error=True)
        await db.commit()

    async def _finalize_storage_failure(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: str,
        provider_name: str,
        provider_errors: dict[str, str],
    ) -> None:
        """Mark the job failed on a storage fault and stop the chain.

        Same terminal bookkeeping as ``_finalize_all_failed`` plus the
        per-provider failure metrics that ``_try_provider_in_chain``
        records on its "exhausted" arm -- which this path deliberately
        never reaches. Provider HEALTH is untouched on purpose: the
        provider delivered the clip, MinIO is what broke, and marking
        every provider unhealthy during a bucket outage would circuit-
        break generation long after storage recovered.
        """
        await self._finalize_all_failed(db, job_id, tenant_id, provider_errors)
        record_job_failed(provider=provider_name)
        record_job_finished()

    async def _try_provider_in_chain(
        self,
        db: AsyncSession,
        job: Any,
        job_id: str,
        tenant_id: str,
        provider_name: str,
        provider_idx: int,
        duration_s: float,
        budget_limit: float,
        spent_so_far: float,
        provider_errors: dict[str, str],
        settings: Any,
        adoptable: _AdoptableTask | None = None,
        slot_provider: VideoGenerationProvider | None = None,
        slot_job: SlotJob | None = None,
    ) -> tuple[str, float]:
        """Run all per-provider preamble + the retry loop for one chain slot.

        ``slot_provider`` / ``slot_job``: a tenant provider-slot run. The per-job
        instance is used as-is -- no registry lookup, no health gate (a tenant
        slot is never health-checked) -- and priced from that instance.

        Returns:
            ("completed", cost): job done, caller should return cost.
            ("terminal", 0.0): job reached a terminal state (cancelled,
                budget-exceeded, conflict). Caller should return 0.0.
            ("skip", 0.0): provider is unknown or unhealthy. Caller
                should continue to the next provider in the chain.
            ("exhausted", 0.0): retries exhausted; failure metrics
                already recorded. Caller should continue.
        """
        if provider_idx > 0:
            logger.warning(
                "Falling back to provider %s for job %s (previous errors: %s)",
                provider_name,
                job_id,
                provider_errors,
            )
        provider: VideoGenerationProvider | None
        if slot_provider is not None:
            provider = slot_provider
        else:
            provider = self._registry.get(provider_name)
            if provider is None:
                provider_errors[provider_name] = f"Unknown provider: {provider_name}"
                logger.warning("Provider %s not found for job %s", provider_name, job_id)
                return "skip", 0.0

            if self._provider_is_available(provider_name) is False:
                provider_errors[provider_name] = f"Provider {provider_name} is unhealthy"
                logger.warning("Skipping unhealthy provider %s for job %s", provider_name, job_id)
                return "skip", 0.0

        if slot_job is not None:
            # A tenant provider slot runs on the TENANT's key: the platform pays
            # nothing, so it is outside the platform budget and its row cost,
            # quota spend and platform cost metric are all 0.
            estimated_cost = 0.0
        else:
            allowed, estimated_cost = await self._enforce_budget(
                db, job_id, provider_name, duration_s, budget_limit, spent_so_far
            )
            if not allowed:
                return "terminal", 0.0

        try:
            await JobManager.mark_processing(db, job_id)
        except JobStateConflictError:
            logger.info("Job %s not pending; another worker or cancel won the race", job_id)
            return "terminal", 0.0

        record_job_started(provider=provider_name)
        outcome, cost = await self._attempt_provider(
            db,
            job,
            job_id,
            tenant_id,
            provider,
            provider_name,
            duration_s,
            estimated_cost,
            provider_errors,
            settings,
            adoptable,
            slot_job,
        )
        if outcome == "completed":
            return "completed", cost
        if outcome in ("cancelled_or_conflict", "terminal"):
            # "terminal" already did its own mark_failed + metrics.
            return "terminal", 0.0
        # exhausted -- record per-provider failure metrics
        record_job_failed(provider=provider_name)
        record_job_finished()
        return "exhausted", 0.0

    async def _process_job_inner(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: str,
        budget_limit: float,
        spent_so_far: float,
    ) -> float:
        """Execute the generation pipeline for a single job.

        Thin orchestrator: load the job, build the provider chain, then
        delegate per-provider work to ``_attempt_provider``. All the
        retry / health / budget / persistence logic lives in helpers
        above so this function stays readable.

        Returns:
            Cost of the completed job, or 0.0 on failure/skip/cancel.
        """
        from src.config import get_settings
        from src.services.exceptions import JobNotFoundError

        job = await JobManager.get_job(db, job_id)
        if job is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        if job.status == "cancelled":
            return 0.0

        # Cross-pod cancel requested before we started work: honour it up-front.
        if await self._is_cancelled(job_id):
            with contextlib.suppress(JobStateConflictError):
                await JobManager.mark_cancelled(db, job_id)
            return 0.0

        settings = get_settings()
        duration_s = job.duration_s or settings.generation.default_duration_s

        # ``provider_slot`` is only ever written as a JSON object by
        # ``attach_provider_slot``; NULL (None) is a platform job.
        if isinstance(job.provider_slot, dict):
            return await self._process_slot_job(
                db, job, job_id, tenant_id, duration_s, budget_limit, spent_so_far, settings
            )

        provider_chain = self._build_provider_chain(job, self._fallback_enabled)

        # A provider_job_id already on the row means an EARLIER run of this job
        # got as far as the provider accepting (and charging for) the task. This
        # run is therefore a reclaim of a crashed one: adopt the paid task rather
        # than submit a new one. Read once, here, so the chain's fallback
        # providers and the in-provider retries cannot re-use a stale id.
        adoptable = (
            _AdoptableTask(provider=job.provider, task_id=job.provider_job_id)
            if job.provider_job_id
            else None
        )

        provider_errors: dict[str, str] = {}
        for provider_idx, provider_name in enumerate(provider_chain):
            outcome, cost = await self._try_provider_in_chain(
                db,
                job,
                job_id,
                tenant_id,
                provider_name,
                provider_idx,
                duration_s,
                budget_limit,
                spent_so_far,
                provider_errors,
                settings,
                adoptable,
            )
            if outcome == "completed":
                return cost
            if outcome == "terminal":
                return 0.0
            # outcome == "skip" or "exhausted": continue to next provider

        await self._finalize_all_failed(db, job_id, tenant_id, provider_errors)
        return 0.0

    async def _process_slot_job(
        self,
        db: AsyncSession,
        job: VideoGenerationJob,
        job_id: str,
        tenant_id: str,
        duration_s: float,
        budget_limit: float,
        spent_so_far: float,
        settings: Settings,
    ) -> float:
        """Run a tenant provider-slot job on a PER-JOB provider (P5-2).

        The slot is rebuilt from the row on every run -- the request path and a
        reaper resume alike -- so a resumed job unseals the secret the request
        sealed. The chain is exactly ONE provider: ``fallback_providers`` and the
        registry are never consulted, so there is no path to a platform provider.
        Every failure (unseal, auth, exhausted retries) is a permanent job
        failure whose message names the slot and never contains the secret.
        """
        from src.services.provider_slots import (
            ProviderSlotError,
            build_video_provider,
            open_slot_job,
        )

        try:
            meta = job.provider_slot if isinstance(job.provider_slot, dict) else {}
            slot_job = open_slot_job(job_id, meta, job.provider_secret_sealed)
            provider = build_video_provider(slot_job)
        except ProviderSlotError as exc:
            logger.warning("Video job %s: %s", job_id, exc)
            await self._finalize_slot_failed(db, job_id, tenant_id, str(exc))
            return 0.0

        adoptable = (
            _AdoptableTask(provider=job.provider, task_id=job.provider_job_id)
            if job.provider_job_id
            else None
        )
        provider_errors: dict[str, str] = {}
        try:
            outcome, cost = await self._try_provider_in_chain(
                db,
                job,
                job_id,
                tenant_id,
                job.provider,
                0,
                duration_s,
                budget_limit,
                spent_so_far,
                provider_errors,
                settings,
                adoptable,
                slot_provider=provider,
                slot_job=slot_job,
            )
        finally:
            await provider.aclose()
        if outcome == "completed":
            return cost
        if outcome == "exhausted":
            await self._finalize_slot_failed(
                db,
                job_id,
                tenant_id,
                provider_errors.get(job.provider)
                or f"tenant provider slot {slot_job.label} failed",
            )
        return 0.0

    async def _finalize_slot_failed(
        self, db: AsyncSession, job_id: str, tenant_id: str, message: str
    ) -> None:
        """Permanently fail a slot job with its (secret-free, slot-naming) message."""
        try:
            await JobManager.mark_failed(db, job_id, message)
        except JobStateConflictError:
            logger.info("Job %s already terminal at slot failure", job_id)
        await JobManager.record_usage(db, tenant_id=tenant_id, is_error=True)
        await db.commit()
