"""Core AI image generation orchestration service."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from src.services.image_job_manager import ImageJobManager
from src.services.image_providers.base import ImageProviderRegistry
from src.services.storage_service import StorageService

if TYPE_CHECKING:
    from src.database.models import ImageGenJob
    from src.services.cancel_registry import CancelRegistry
    from src.services.image_providers.base import ImageGenProvider, ImageGenResult
    from src.services.provider_slots import SlotJob

logger = logging.getLogger(__name__)


def _filename_for_mime(mime_type: str | None) -> str:
    if mime_type == "image/jpeg":
        return "image.jpg"
    if mime_type == "image/webp":
        return "image.webp"
    return "image.png"


class ImageGenService:
    """Orchestrates AI image generation across providers."""

    def __init__(
        self,
        storage: StorageService,
        registry: ImageProviderRegistry,
        default_provider: str,
        parallel_requests: int = 4,
        metering_redis: object | None = None,
        cancel_registry: CancelRegistry | None = None,
    ) -> None:
        self._storage = storage
        self._registry = registry
        self._default_provider = default_provider
        self._semaphore = asyncio.Semaphore(parallel_requests)
        # Batch children now register here too (one leased task each), so the
        # reaper's shutdown drain cancels them before releasing their leases —
        # closing the double-spend window the old orchestrator-level
        # ``_active_batch_tasks`` set left open.
        self._active_tasks: dict[str, asyncio.Task] = {}
        # Cross-pod cancellation (see VideoGenerationService for rationale).
        self._cancel_registry = cancel_registry
        # Metering: None = disabled (METERING_REDIS_URL unset/unreachable).
        # Emission is GC-safe fire-and-forget via w2v_common.metering, which
        # owns the in-flight task set and resolves the cost-bus stream.
        self._metering_redis = metering_redis

    def register_task(self, job_id: str, task: asyncio.Task) -> None:
        self._active_tasks[job_id] = task

    def unregister_task(self, job_id: str) -> None:
        self._active_tasks.pop(job_id, None)

    async def _is_cancelled(self, job_id: str) -> bool:
        """True if a cross-pod cancel has been requested for this image job."""
        if self._cancel_registry is None:
            return False
        return await self._cancel_registry.is_cancelled(job_id)

    async def cancel_job(self, db: AsyncSession, job_id: str, tenant_id: str) -> bool:
        """Cancel an image job owned by ``tenant_id``. True if cancelled.

        Same ordering rule as :meth:`VideoGenerationService.cancel_job`: the
        ownership read comes BEFORE the Redis cancel flag and the in-process
        ``task.cancel()``, because neither of those is a database operation
        and RLS therefore cannot backstop them. A foreign job is treated
        exactly like an unknown one.
        """
        from src.services.job_manager import JobStateConflictError

        # --- Ownership gate. Must stay ABOVE the registry/task calls. ---
        job = await ImageJobManager.get_job(db, job_id, tenant_id=tenant_id)
        if job is None:
            logger.warning("Cancel denied: job_id=%s not owned by tenant=%s", job_id, tenant_id)
            return False

        # Cross-pod: set the flag as the first side effect so a task on any
        # replica observes it.
        if self._cancel_registry is not None:
            await self._cancel_registry.request_cancel(job_id)

        task = self._active_tasks.get(job_id)
        if task is None:
            if job.status == "pending":
                try:
                    await ImageJobManager.mark_cancelled(db, job_id)
                except JobStateConflictError:
                    return False
                return True
            # Running on another pod — the flag drives it to cancelled.
            return job.status == "processing"
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        try:
            await ImageJobManager.mark_cancelled(db, job_id)
        except JobStateConflictError:
            self.unregister_task(job_id)
            return False
        self.unregister_task(job_id)
        return True

    async def create_job(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        prompt: str,
        provider: str | None = None,
        batch_id: str | None = None,
        negative_prompt: str | None = None,
        style: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        reference_image_urls: list[str] | None = None,
        usage_context: dict | None = None,
        target_usage: str | None = None,
        extra_params: dict | None = None,
        callback_url: str | None = None,
        idempotency_key: str | None = None,
        idempotency_fingerprint: str | None = None,
    ) -> str:
        job = await ImageJobManager.create_job(
            db,
            tenant_id=tenant_id,
            provider=provider or self._default_provider,
            prompt=prompt,
            batch_id=batch_id,
            negative_prompt=negative_prompt,
            style=style,
            size=size,
            quality=quality,
            reference_image_urls=reference_image_urls,
            usage_context=usage_context,
            target_usage=target_usage,
            extra_params=extra_params,
            callback_url=callback_url,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
        )
        return job.job_id

    def estimate_cost(
        self, provider_name: str, *, size: str | None = None, quality: str | None = None
    ) -> float | None:
        provider = self._registry.get(provider_name)
        if provider is None:
            return None
        return provider.estimate_cost(size=size, quality=quality)

    def estimate_request_cost(
        self, provider: str | None, *, size: str | None = None, quality: str | None = None
    ) -> float:
        """USD estimate for a request that has NOT been created yet.

        Resolves the default provider the way ``create_job`` will, so the
        admission check prices the job it is admitting instead of 0.0. Mirrors
        ``VideoGenerationService.estimate_request_cost``.
        """
        return (
            self.estimate_cost(provider or self._default_provider, size=size, quality=quality)
            or 0.0
        )

    async def process_job(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: str,
        budget_limit: float = 5.0,
        spent_so_far: float = 0.0,
    ) -> float:
        del tenant_id
        async with self._semaphore:
            return await self._process_job_inner(db, job_id, budget_limit, spent_so_far)

    @staticmethod
    async def _safe_mark_failed(
        db: AsyncSession, job_id: str, error: str, log_context: str
    ) -> None:
        """Mark the image job as failed, swallowing JobStateConflictError."""
        from src.services.job_manager import JobStateConflictError

        try:
            await ImageJobManager.mark_failed(db, job_id, error)
        except JobStateConflictError:
            logger.info("Image job %s already terminal; %s", job_id, log_context)

    async def _generate_image(self, provider: ImageGenProvider, job: ImageGenJob) -> ImageGenResult:
        """Invoke the provider's generate() with the job's parameters."""
        return await provider.generate(
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            style=job.style,
            size=job.size,
            quality=job.quality,
            reference_image_urls=job.reference_image_urls,
            extra_params=job.extra_params,
            usage_context=job.usage_context,
            target_usage=job.target_usage,
        )

    async def _store_and_complete(
        self,
        db: AsyncSession,
        job_id: str,
        result: ImageGenResult,
        estimated_cost: float,
    ) -> bool:
        """Persist generated image bytes and mark the job completed.

        Returns True when this call completed the job, False when it did not
        (no bytes, or the job was no longer processing).
        """
        from src.services.job_manager import JobStateConflictError

        image_data = result.image_data
        if image_data is None:  # defensive — caller already checked
            return False
        filename = _filename_for_mime(result.mime_type)
        image_path: str | None = await self._storage.store_image(
            job_id,
            image_data,
            filename,
            content_type=result.mime_type,
        )
        minio_key: str | None = None
        if image_path is not None and image_path.startswith("content_generation/"):
            minio_key = image_path
            image_path = None

        try:
            await ImageJobManager.mark_completed(
                db,
                job_id,
                image_path=image_path,
                image_url=result.image_url,
                mime_type=result.mime_type,
                width=result.width,
                height=result.height,
                file_size_bytes=len(image_data),
                cost=estimated_cost,
                minio_key=minio_key,
            )
        except JobStateConflictError:
            logger.info("Image job %s no longer processing when result arrived", job_id)
            return False
        return True

    async def _process_job_inner(
        self,
        db: AsyncSession,
        job_id: str,
        budget_limit: float,
        spent_so_far: float,
    ) -> float:
        from src.services.job_manager import JobStateConflictError

        job = await ImageJobManager.get_job(db, job_id)
        if job is None:
            raise ValueError(f"Image generation job not found: {job_id}")
        if job.status == "cancelled":
            return 0.0

        # Cross-pod cancel requested before we started: honour it up-front.
        if await self._is_cancelled(job_id):
            with contextlib.suppress(JobStateConflictError):
                await ImageJobManager.mark_cancelled(db, job_id)
            return 0.0

        # ``provider_slot`` is only ever written as a JSON object by
        # ``attach_provider_slot``; NULL (None) is a platform job.
        if isinstance(job.provider_slot, dict):
            return await self._process_slot_job(db, job, job_id, budget_limit, spent_so_far)

        provider = self._registry.get(job.provider)
        if provider is None:
            await self._safe_mark_failed(
                db, job_id, f"Unknown provider: {job.provider}", "unknown-provider path"
            )
            return 0.0
        return await self._run_provider(db, job, job_id, provider, budget_limit, spent_so_far)

    async def _process_slot_job(
        self,
        db: AsyncSession,
        job: ImageGenJob,
        job_id: str,
        budget_limit: float,
        spent_so_far: float,
    ) -> float:
        """Run a tenant provider-slot job on a PER-JOB provider (P5-2).

        The slot is rebuilt from the row on every run -- request path and reaper
        resume alike -- so a resumed job unseals the same secret the request
        sealed. Any unseal failure, or any provider failure, fails the job
        PERMANENTLY with a slot-naming, secret-free error. There is no fallback
        to a platform provider on any path.
        """
        from src.services.provider_slots import (
            ProviderSlotError,
            build_image_provider,
            open_slot_job,
        )

        try:
            meta = job.provider_slot if isinstance(job.provider_slot, dict) else {}
            slot_job = open_slot_job(job_id, meta, job.provider_secret_sealed)
            provider = build_image_provider(slot_job)
        except ProviderSlotError as exc:
            logger.warning("Image job %s: %s", job_id, exc)
            await self._safe_mark_failed(db, job_id, str(exc), "slot-unseal path")
            return 0.0
        try:
            return await self._run_provider(
                db, job, job_id, provider, budget_limit, spent_so_far, slot_job=slot_job
            )
        finally:
            await provider.aclose()

    async def _run_provider(
        self,
        db: AsyncSession,
        job: ImageGenJob,
        job_id: str,
        provider: ImageGenProvider,
        budget_limit: float,
        spent_so_far: float,
        slot_job: SlotJob | None = None,
    ) -> float:
        from src.services.job_manager import JobStateConflictError

        # A tenant provider slot runs on the TENANT's key: the platform pays
        # nothing, so it is outside the platform budget and its row cost and
        # quota spend are 0.
        estimated_cost = (
            0.0
            if slot_job is not None
            else provider.estimate_cost(size=job.size, quality=job.quality)
        )
        if slot_job is None and spent_so_far + estimated_cost > budget_limit:
            await self._safe_mark_failed(
                db,
                job_id,
                f"Budget exceeded: estimated ${estimated_cost:.2f}, "
                f"remaining ${budget_limit - spent_so_far:.2f}",
                "budget path",
            )
            return 0.0

        try:
            await ImageJobManager.mark_processing(db, job_id)
        except JobStateConflictError:
            logger.info("Image job %s not pending; another worker or cancel won", job_id)
            return 0.0

        import time as _time

        t0 = _time.monotonic()
        if slot_job is None:
            result = await self._generate_image(provider, job)
        else:
            try:
                result = await self._generate_image(provider, job)
            except Exception as exc:  # noqa: BLE001 -- recorded on the job, see below
                # A tenant slot's failure is final and names the slot. Letting it
                # escape would reach the bg catch-all, whose generic message
                # interpolates raw exception text and does not name the slot.
                message = slot_job.failure_message(exc)
                logger.warning("Image job %s: %s", job_id, message)
                await self._safe_mark_failed(db, job_id, message, "slot failure path")
                return 0.0
        elapsed_ms = int((_time.monotonic() - t0) * 1000)

        if not result.success or result.image_data is None:
            error = result.error or "Image generation failed"
            if slot_job is not None:
                error = (
                    f"tenant provider slot {slot_job.label} failed: {slot_job.scrub(error)}. "
                    "Not retried on a platform provider (tenant slots never fall back)."
                )
            await self._safe_mark_failed(db, job_id, error, "failure path")
            return 0.0

        # Cross-pod cancel requested while generating: drop the result, cancel.
        if await self._is_cancelled(job_id):
            with contextlib.suppress(JobStateConflictError):
                await ImageJobManager.mark_cancelled(db, job_id)
            return 0.0

        if not await self._store_and_complete(db, job_id, result, estimated_cost):
            return 0.0
        # A slot job completes at cost 0 and must still emit its tenant-funded
        # BYO usage event; a zero-cost platform job stays silent as before.
        if estimated_cost > 0 or slot_job is not None:
            self._emit_cost_event(
                job=job,
                result=result,
                estimated_cost=estimated_cost,
                latency_ms=elapsed_ms,
                slot_job=slot_job,
            )
        return estimated_cost

    def _emit_cost_event(
        self,
        *,
        job: ImageGenJob,
        result: ImageGenResult,
        estimated_cost: float,
        latency_ms: int,
        slot_job: SlotJob | None = None,
    ) -> None:
        """Fire-and-forget cost event to improvement-loop:costs. No-ops when disabled.

        Encoding per EMITTER_GUIDE.md (non-token workload):
            input_tokens  = (width * height) // 1_000_000 (megapixels)
            output_tokens = 0
            usage_extra   = {"unit": "megapixels", "size": ..., "quality": ...}

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
            require_tenant,
            schedule_cost_event,
        )
        from src.services.provider_slots import FUNDING_TENANT_BYO

        width = result.width or 0
        height = result.height or 0
        megapixels = max(1, (width * height) // 1_000_000)
        schedule_cost_event(
            cast(Any, self._metering_redis),
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id=require_tenant(job.tenant_id),
            license_id=get_license_id_from_context(),
            subject_id=get_attribution_subject_id_from_context(),
            app_id=APP_ID,
            provider=job.provider,
            model=job.provider,
            input_tokens=megapixels,
            output_tokens=0,
            latency_ms=latency_ms,
            cost_usd=estimated_cost if slot_job is None else 0.0,
            usage_extra={
                "unit": "megapixels",
                "size": job.size or "",
                "quality": job.quality or "",
                "width": width,
                "height": height,
                **(slot_job.byo_usage_extra() if slot_job is not None else {}),
            },
            # Tenant key -> tenant_byo; a platform job states its provider's
            # declared funding (an undeclared provider raises -- never guessed).
            funding=FUNDING_TENANT_BYO if slot_job is not None else platform_funding(job.provider),
            event_key=job_event_key(job.job_id, "image"),
        )
