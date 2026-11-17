"""Integration tests for fallback and retry logic (Phase 2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.services.providers.base import (
    VideoGenerationProvider,
    VideoGenerationResult,
    VideoProviderRegistry,
)
from src.services.video_generation_service import VideoGenerationService


class SlowFailProvider(VideoGenerationProvider):
    """Provider that fails with a retriable error."""

    name = "slow_fail"

    async def submit(self, prompt: str, **kwargs) -> str:
        return "task-fail"

    async def check_status(self, task_id: str) -> dict:
        return {"status": "failed", "error": "timeout"}

    async def download(self, task_id: str, video_url: str) -> bytes:
        return b""

    def estimate_cost(self, duration_s: float) -> float:
        return 0.01

    async def generate(self, **kwargs) -> VideoGenerationResult:
        raise RuntimeError("timeout error")


class SuccessProvider(VideoGenerationProvider):
    """Provider that always succeeds."""

    name = "success"

    async def submit(self, prompt: str, **kwargs) -> str:
        return "task-ok"

    async def check_status(self, task_id: str) -> dict:
        return {"status": "completed", "video_url": "https://example.com/v.mp4"}

    async def download(self, task_id: str, video_url: str) -> bytes:
        return b"video-bytes"

    def estimate_cost(self, duration_s: float) -> float:
        return 0.02

    async def generate(self, **kwargs) -> VideoGenerationResult:
        return VideoGenerationResult(
            success=True,
            video_data=b"video-bytes",
            video_url="https://example.com/v.mp4",
            provider_job_id="task-ok",
            duration_s=5.0,
        )


def _make_mock_job(provider="slow_fail", fallback_providers=None, status="pending"):
    job = MagicMock()
    job.job_id = "test-job"
    job.provider = provider
    job.status = status
    job.prompt = "test prompt"
    job.negative_prompt = None
    job.image_url = None
    job.duration_s = 5.0
    job.width = 1920
    job.height = 1080
    job.style = None
    job.seed = None
    job.extra_params = None
    job.fallback_providers = fallback_providers
    job.callback_url = None
    job.error_message = None
    job.cost = None
    return job


def _mock_settings():
    s = MagicMock()
    s.generation.default_duration_s = 5.0
    s.generation.default_width = 1920
    s.generation.default_height = 1080
    return s


async def test_primary_fails_secondary_succeeds():
    """When primary fails, fallback provider should be tried."""
    registry = VideoProviderRegistry()
    registry.register(SlowFailProvider())
    registry.register(SuccessProvider())

    storage = AsyncMock()
    storage.store_video = AsyncMock(return_value="/tmp/test.mp4")
    storage.upload_to_minio = AsyncMock(return_value=None)

    svc = VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider="slow_fail",
        max_retries=0,
        fallback_enabled=True,
    )

    job = _make_mock_job(provider="slow_fail", fallback_providers=["success"])

    with (
        patch("src.services.video_generation_service.JobManager") as mock_jm,
        patch("src.config.get_settings", return_value=_mock_settings()),
    ):
        mock_jm.get_job = AsyncMock(return_value=job)
        mock_jm.mark_processing = AsyncMock()
        mock_jm.set_provider_job_id = AsyncMock()
        mock_jm.mark_completed = AsyncMock()
        mock_jm.mark_failed = AsyncMock()
        mock_jm.mark_retrying = AsyncMock()
        mock_jm.record_usage = AsyncMock()

        db = AsyncMock()
        cost = await svc.process_job(db, "test-job", "tenant-1")

    assert cost > 0
    mock_jm.mark_completed.assert_called_once()


async def test_all_providers_fail():
    """When all providers fail, job should be marked failed."""
    registry = VideoProviderRegistry()
    registry.register(SlowFailProvider())

    storage = AsyncMock()
    svc = VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider="slow_fail",
        max_retries=0,
        fallback_enabled=True,
    )

    job = _make_mock_job(provider="slow_fail", fallback_providers=[])

    with (
        patch("src.services.video_generation_service.JobManager") as mock_jm,
        patch("src.config.get_settings", return_value=_mock_settings()),
    ):
        mock_jm.get_job = AsyncMock(return_value=job)
        mock_jm.mark_processing = AsyncMock()
        mock_jm.set_provider_job_id = AsyncMock()
        mock_jm.mark_completed = AsyncMock()
        mock_jm.mark_failed = AsyncMock()
        mock_jm.mark_retrying = AsyncMock()
        mock_jm.record_usage = AsyncMock()

        db = AsyncMock()
        cost = await svc.process_job(db, "test-job", "tenant-1")

    assert cost == 0.0
    mock_jm.mark_failed.assert_called_once()


async def test_storage_5xx_does_not_buy_the_clip_twice():
    """A storage-layer 5xx must not re-enter the PAID generation call.

    _store_video_artifacts runs inside the ``for attempt`` try. MinIO/httpx
    surface a bucket 503 as httpx.HTTPStatusError, which is_retriable_error()
    classifies retriable — so the retry gate re-runs _try_generate_once and the
    provider is charged a second time for a clip we already generated. There is
    no idempotency key anywhere in this path.
    """
    calls = 0

    class CountingProvider(VideoGenerationProvider):
        name = "counting"

        async def submit(self, prompt, **kwargs):
            return "c1"

        async def check_status(self, task_id):
            return {"status": "completed", "video_url": "https://x.com/v.mp4"}

        async def download(self, task_id, video_url):
            return b"data"

        def estimate_cost(self, duration_s):
            return 0.01

        async def generate(self, **kwargs):
            nonlocal calls
            calls += 1
            return VideoGenerationResult(
                success=True,
                video_data=b"video-bytes",
                video_url="https://x.com/v.mp4",
                provider_job_id="c1",
                duration_s=5.0,
            )

    registry = VideoProviderRegistry()
    registry.register(CountingProvider())

    storage = AsyncMock()
    storage.store_video = AsyncMock(return_value="/tmp/test.mp4")
    storage.upload_to_minio = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=httpx.Request("PUT", "https://minio.test:8389/w2v-videos/x.mp4"),
            response=httpx.Response(503),
        )
    )

    svc = VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider="counting",
        max_retries=2,
        fallback_enabled=False,
    )

    job = _make_mock_job(provider="counting", fallback_providers=[])

    with (
        patch("src.services.video_generation_service.JobManager") as mock_jm,
        patch("src.config.get_settings", return_value=_mock_settings()),
        patch("src.services.video_generation_service.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_jm.get_job = AsyncMock(return_value=job)
        mock_jm.mark_processing = AsyncMock()
        mock_jm.set_provider_job_id = AsyncMock()
        mock_jm.mark_completed = AsyncMock()
        mock_jm.mark_failed = AsyncMock()
        mock_jm.mark_retrying = AsyncMock()
        mock_jm.record_usage = AsyncMock()

        db = AsyncMock()
        cost = await svc.process_job(db, "test-job", "tenant-1")

    assert calls == 1, f"storage 503 re-ran the PAID provider call {calls}x — double-charged"
    assert cost == 0.0
    mock_jm.mark_failed.assert_called_once()


async def test_storage_5xx_does_not_walk_the_fallback_chain():
    """A storage 5xx must not buy a second clip from the NEXT provider either.

    Narrowing the retry gate (StorageError is non-retriable) only stops the
    SAME provider being paid twice. The chain loop in ``_process_job_inner``
    answers ``"exhausted"`` with ``continue``, so a MinIO outage still bought
    ``1 + len(fallback_providers)`` clips -- one per provider, every one of
    them discarded, because storage is down for all of them equally. Storage
    must terminate the job, not rotate the generation provider.
    """
    paid: dict[str, int] = {"a": 0, "b": 0}

    def _make_counting_provider(provider_name: str):
        class _Counting(VideoGenerationProvider):
            name = provider_name

            async def submit(self, prompt, **kwargs):
                return f"{provider_name}-1"

            async def check_status(self, task_id):
                return {"status": "completed", "video_url": "https://x.com/v.mp4"}

            async def download(self, task_id, video_url):
                return b"data"

            def estimate_cost(self, duration_s):
                return 0.01

            async def generate(self, **kwargs):
                paid[provider_name] += 1
                return VideoGenerationResult(
                    success=True,
                    video_data=b"video-bytes",
                    video_url="https://x.com/v.mp4",
                    provider_job_id=f"{provider_name}-1",
                    duration_s=5.0,
                )

        return _Counting()

    registry = VideoProviderRegistry()
    registry.register(_make_counting_provider("a"))
    registry.register(_make_counting_provider("b"))

    storage = AsyncMock()
    storage.store_video = AsyncMock(return_value="/tmp/test.mp4")
    storage.upload_to_minio = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=httpx.Request("PUT", "https://minio.test:8389/w2v-videos/x.mp4"),
            response=httpx.Response(503),
        )
    )

    svc = VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider="a",
        max_retries=2,
        fallback_enabled=True,
    )

    job = _make_mock_job(provider="a", fallback_providers=["b"])

    with (
        patch("src.services.video_generation_service.JobManager") as mock_jm,
        patch("src.config.get_settings", return_value=_mock_settings()),
        patch("src.services.video_generation_service.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_jm.get_job = AsyncMock(return_value=job)
        mock_jm.mark_processing = AsyncMock()
        mock_jm.set_provider_job_id = AsyncMock()
        mock_jm.mark_completed = AsyncMock()
        mock_jm.mark_failed = AsyncMock()
        mock_jm.mark_retrying = AsyncMock()
        mock_jm.record_usage = AsyncMock()

        db = AsyncMock()
        cost = await svc.process_job(db, "test-job", "tenant-1")

    assert paid == {"a": 1, "b": 0}, f"storage 503 walked the fallback chain: {paid}"
    assert cost == 0.0
    # Terminal state: failed exactly once (not once here plus once again from
    # _finalize_all_failed at the end of the chain), attributed to storage.
    mock_jm.mark_failed.assert_called_once()
    failure_reason = mock_jm.mark_failed.call_args.args[2]
    assert "503" in failure_reason
    # Billing: the tenant is charged one error-usage record, and no completion.
    mock_jm.record_usage.assert_awaited_once_with(db, tenant_id="tenant-1", is_error=True)
    mock_jm.mark_completed.assert_not_called()


async def test_retry_on_transient_error():
    """Transient errors should trigger retry with backoff."""
    call_count = 0

    class TransientProvider(VideoGenerationProvider):
        name = "transient"

        async def submit(self, prompt, **kwargs):
            return "t1"

        async def check_status(self, task_id):
            return {"status": "completed", "video_url": "https://x.com/v.mp4"}

        async def download(self, task_id, video_url):
            return b"data"

        def estimate_cost(self, duration_s):
            return 0.01

        async def generate(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("timeout in generation")
            return VideoGenerationResult(
                success=True,
                video_data=b"data",
                video_url="https://x.com/v.mp4",
                provider_job_id="t1",
                duration_s=5.0,
            )

    registry = VideoProviderRegistry()
    registry.register(TransientProvider())

    storage = AsyncMock()
    storage.store_video = AsyncMock(return_value="/tmp/test.mp4")
    storage.upload_to_minio = AsyncMock(return_value=None)

    svc = VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider="transient",
        max_retries=2,
        fallback_enabled=False,
    )

    job = _make_mock_job(provider="transient", fallback_providers=[])

    with (
        patch("src.services.video_generation_service.JobManager") as mock_jm,
        patch("src.config.get_settings", return_value=_mock_settings()),
        patch("src.services.video_generation_service.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_jm.get_job = AsyncMock(return_value=job)
        mock_jm.mark_processing = AsyncMock()
        mock_jm.set_provider_job_id = AsyncMock()
        mock_jm.mark_completed = AsyncMock()
        mock_jm.mark_failed = AsyncMock()
        mock_jm.mark_retrying = AsyncMock()
        mock_jm.record_usage = AsyncMock()

        db = AsyncMock()
        cost = await svc.process_job(db, "test-job", "tenant-1")

    assert cost > 0
    assert call_count == 2
    mock_jm.mark_completed.assert_called_once()
