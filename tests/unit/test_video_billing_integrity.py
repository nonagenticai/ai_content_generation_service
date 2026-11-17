"""Regression: two money defects on the video generation path.

1. ``VideoGenerationService._emit_cost_event`` metered ``input_tokens=int(duration_s)``.
   The cost bus bills from the emitted *units* (``cost_usd`` is documented in the
   same call as a "local estimate; metering recomputes"), so a sub-second clip
   emitted 0 video_seconds and was billed nothing, and every fractional duration
   under-metered by up to a full second. The sibling image path already guards
   this exact case — ``ImageGenService._emit_cost_event`` uses
   ``max(1, (width * height) // 1_000_000)``.

2. ``src.services.errors.is_retriable_error`` did not know ``httpx.RequestError``.
   ``httpx.ConnectError`` / ``ReadError`` / ``RemoteProtocolError`` are NOT
   ``OSError`` subclasses (verified: ``ConnectError -> NetworkError ->
   TransportError -> RequestError -> HTTPError -> Exception``), so they escape
   ``VideoGenerationProvider.generate``'s ``except (OSError, RuntimeError,
   ValueError)`` and reach ``_attempt_provider``, which classified them
   non-retriable. A TCP reset therefore burned the provider chain in ONE attempt
   and jumped straight to a different PAID provider — while a plain HTTP 503
   from the same endpoint got the full ``max_retries + 1``.
"""

from __future__ import annotations

from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.services.errors import is_retriable_error
from src.services.providers.base import (
    VideoGenerationProvider,
    VideoGenerationResult,
    VideoProviderRegistry,
)
from src.services.video_generation_service import VideoGenerationService


class _Emitted(NamedTuple):
    """The two fields of the cost event this test cares about."""

    billable_units: int  # ``input_tokens``, i.e. video_seconds
    reported_duration_s: float  # ``usage_extra["duration_s"]``


class TestMeteredVideoSeconds:
    def _emit(self, duration_s: float) -> _Emitted:
        svc = VideoGenerationService(
            storage=AsyncMock(),
            registry=VideoProviderRegistry(),
            default_provider="stub",
            metering_redis=MagicMock(),
        )
        with patch(
            "src.services.metering.schedule_cost_event",
        ) as spy:
            svc._emit_cost_event(
                job_id="job-1",
                tenant_id="tenant-1",
                provider="sora",
                duration_s=duration_s,
                estimated_cost=0.05,
                elapsed_s=12.0,
                provider_request_id="req-1",
                file_size_bytes=1024,
            )
        assert spy.call_count == 1, "precondition: metering enabled, one event emitted"
        kwargs = spy.call_args.kwargs
        return _Emitted(
            billable_units=int(kwargs["input_tokens"]),
            reported_duration_s=float(kwargs["usage_extra"]["duration_s"]),
        )

    def test_a_full_second_clip_meters_its_seconds(self) -> None:
        """Precondition: the harness reads the billable unit correctly."""
        assert self._emit(5.0).billable_units == 5

    def test_sub_second_clip_is_not_metered_as_zero(self) -> None:
        emitted = self._emit(0.5)
        assert emitted.reported_duration_s == 0.5  # a real clip was produced
        assert emitted.billable_units > 0, (
            "a 0.5s clip was emitted with 0 billable video_seconds — "
            "the cost bus bills from units, so this generation is free"
        )

    def test_fractional_duration_is_not_floored_away(self) -> None:
        emitted = self._emit(7.9)
        assert emitted.billable_units == 8, (
            f"7.9s metered as {emitted.billable_units} video_seconds — "
            "truncation systematically under-bills every fractional clip"
        )


class TestTransportErrorsAreRetriable:
    def test_http_503_is_retriable(self) -> None:
        """Precondition: the classifier does treat transient server errors as retriable."""
        exc = httpx.HTTPStatusError(
            "503",
            request=httpx.Request("GET", "https://api.example.com/x"),
            response=httpx.Response(503),
        )
        assert is_retriable_error(exc) is True

    def test_connect_error_is_retriable(self) -> None:
        assert not issubclass(httpx.ConnectError, OSError)  # precondition
        assert is_retriable_error(httpx.ConnectError("connection refused")) is True

    def test_read_error_is_retriable(self) -> None:
        assert is_retriable_error(httpx.ReadError("peer reset")) is True


class _FlakyTransportProvider(VideoGenerationProvider):
    """Raises a transport error until ``fail_times`` attempts have been made."""

    name = "flaky"

    def __init__(self, exc: Exception, fail_times: int) -> None:
        self._exc = exc
        self._fail_times = fail_times
        self.calls = 0

    async def submit(self, prompt: str, **kwargs) -> str:
        return "t1"

    async def check_status(self, task_id: str) -> dict:
        return {"status": "completed", "video_url": "https://x.example.com/v.mp4"}

    async def download(self, task_id: str, video_url: str) -> bytes:
        return b"data"

    def estimate_cost(self, duration_s: float) -> float:
        return 0.01

    async def generate(self, **kwargs) -> VideoGenerationResult:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return VideoGenerationResult(
            success=True,
            video_data=b"data",
            video_url="https://x.example.com/v.mp4",
            provider_job_id="t1",
            duration_s=5.0,
        )


def _mock_job() -> MagicMock:
    job = MagicMock()
    job.job_id = "test-job"
    job.provider = "flaky"
    job.status = "pending"
    job.prompt = "test prompt"
    job.negative_prompt = None
    job.image_url = None
    job.duration_s = 5.0
    job.width = 1920
    job.height = 1080
    job.style = None
    job.seed = None
    job.extra_params = None
    job.fallback_providers = []
    job.callback_url = None
    job.error_message = None
    job.cost = None
    return job


def _mock_settings() -> MagicMock:
    s = MagicMock()
    s.generation.default_duration_s = 5.0
    s.generation.default_width = 1920
    s.generation.default_height = 1080
    return s


async def _attempts_for(exc: Exception) -> int:
    """Run one job whose provider raises ``exc`` once; return provider.generate call count."""
    provider = _FlakyTransportProvider(exc, fail_times=1)
    registry = VideoProviderRegistry()
    registry.register(provider)

    storage = AsyncMock()
    storage.store_video = AsyncMock(return_value="/tmp/test.mp4")
    storage.upload_to_minio = AsyncMock(return_value=None)

    svc = VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider="flaky",
        max_retries=2,
        fallback_enabled=False,
    )

    with (
        patch("src.services.video_generation_service.JobManager") as mock_jm,
        patch("src.config.get_settings", return_value=_mock_settings()),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_jm.get_job = AsyncMock(return_value=_mock_job())
        mock_jm.mark_processing = AsyncMock()
        mock_jm.set_provider_job_id = AsyncMock()
        mock_jm.mark_completed = AsyncMock()
        mock_jm.mark_failed = AsyncMock()
        mock_jm.mark_retrying = AsyncMock()
        mock_jm.record_usage = AsyncMock()
        await svc.process_job(AsyncMock(), "test-job", "tenant-1")

    return provider.calls


class TestTransportErrorRetryBehaviour:
    async def test_runtime_timeout_is_retried(self) -> None:
        """Precondition: a retriable-classified error DOES get a second attempt."""
        assert await _attempts_for(RuntimeError("timeout during generation")) == 2

    async def test_connect_error_is_retried_not_burned(self) -> None:
        calls = await _attempts_for(httpx.ConnectError("connection refused"))
        assert calls == 2, (
            f"provider was called {calls}x for a transient TCP failure — the chain "
            "burned the provider in one attempt and would fall through to a "
            "different PAID provider"
        )
