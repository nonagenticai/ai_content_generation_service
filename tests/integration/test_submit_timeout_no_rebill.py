"""A client-side timeout on provider SUBMIT must not trigger a re-submit.

Defect: ``httpx.TimeoutException`` was classified retriable regardless of
which lifecycle phase raised it. A submit-phase timeout is AMBIGUOUS -- the
provider may have accepted (and billed) the task before the response
reached us -- so retrying calls ``submit()`` again and can create a second
billable provider task for one logical job. No provider here exposes an
outbound idempotency key or a resume-by-task_id path, so the only safe move
is to stop retrying, exactly like ``_store_video_artifacts`` does one step
later in the pipeline for storage faults (see
``test_fallback_retry.test_storage_5xx_does_not_buy_the_clip_twice``, the
sibling precedent this file mirrors).

Poll and download are read/no-side-effect operations, so a timeout in
either phase must keep its existing retriable behaviour -- this file also
pins that down so a future change cannot regress reliability while fixing
billing safety.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.services.providers.base import VideoGenerationProvider, VideoProviderRegistry
from src.services.video_generation_service import VideoGenerationService


def _mock_settings():
    s = MagicMock()
    s.generation.default_duration_s = 5.0
    s.generation.default_width = 1920
    s.generation.default_height = 1080
    return s


def _make_mock_job(provider: str, fallback_providers=None):
    job = MagicMock()
    job.job_id = "test-job"
    job.provider = provider
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
    # MUST be None, and this is not cosmetic. #119 added a re-attach path that
    # skips submit() entirely when the row already carries a provider task id.
    # A bare MagicMock auto-creates a TRUTHY attribute here, so without this the
    # service adopts a phantom task and submit() is never called at all -- these
    # tests would then assert nothing about the submit boundary they exist for.
    # Every scenario in this module is a FIRST submit that has never been accepted.
    job.provider_job_id = None
    job.fallback_providers = fallback_providers or []
    job.callback_url = None
    job.error_message = None
    job.cost = None
    return job


def _submit_request() -> httpx.Request:
    return httpx.Request("POST", "https://provider.example.com/submit")


async def _run(svc: VideoGenerationService, provider: str, fallback=None):
    job = _make_mock_job(provider, fallback_providers=fallback)
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
        return cost, mock_jm


class SubmitTimeoutProvider(VideoGenerationProvider):
    """Every submit() call times out. Does NOT override generate(), so the
    real ``VideoGenerationProvider.generate()`` lifecycle in base.py runs."""

    name = "submit_timeout"

    def __init__(self) -> None:
        self.submit_calls = 0

    async def submit(self, prompt: str, **kwargs) -> str:
        self.submit_calls += 1
        raise httpx.TimeoutException("submit timed out", request=_submit_request())

    async def check_status(self, task_id: str) -> dict:
        return {"status": "completed", "video_url": "https://provider.example.com/v.mp4"}

    async def download(self, task_id: str, video_url: str) -> bytes:
        return b"video-bytes"

    def estimate_cost(self, duration_s: float) -> float:
        return 0.01


class PollTimeoutThenSuccessProvider(VideoGenerationProvider):
    """submit() succeeds every time; check_status() times out on the FIRST
    call only, then succeeds. Poll timeouts must stay retriable."""

    name = "poll_timeout"

    def __init__(self) -> None:
        self.submit_calls = 0
        self.check_status_calls = 0

    async def submit(self, prompt: str, **kwargs) -> str:
        self.submit_calls += 1
        return f"task-{self.submit_calls}"

    async def check_status(self, task_id: str) -> dict:
        self.check_status_calls += 1
        if self.check_status_calls == 1:
            raise httpx.TimeoutException(
                "poll timed out",
                request=httpx.Request("GET", "https://provider.example.com/status"),
            )
        return {"status": "completed", "video_url": "https://provider.example.com/v.mp4"}

    async def download(self, task_id: str, video_url: str) -> bytes:
        return b"video-bytes"

    def estimate_cost(self, duration_s: float) -> float:
        return 0.01


class ConnectErrorThenSuccessProvider(VideoGenerationProvider):
    """submit() raises a ConnectError (nothing sent) on the FIRST call, then
    succeeds. A pre-send connect failure is NOT ambiguous -- it is safe (and
    desirable) to keep retrying it, unlike a submit-phase timeout."""

    name = "connect_error"

    def __init__(self) -> None:
        self.submit_calls = 0

    async def submit(self, prompt: str, **kwargs) -> str:
        self.submit_calls += 1
        if self.submit_calls == 1:
            raise httpx.ConnectError("connection refused", request=_submit_request())
        return f"task-{self.submit_calls}"

    async def check_status(self, task_id: str) -> dict:
        return {"status": "completed", "video_url": "https://provider.example.com/v.mp4"}

    async def download(self, task_id: str, video_url: str) -> bytes:
        return b"video-bytes"

    def estimate_cost(self, duration_s: float) -> float:
        return 0.01


async def test_submit_phase_timeout_does_not_resubmit():
    """A submit-phase timeout must call submit() exactly ONCE and fail the
    job -- retrying would risk a second billable provider task."""
    provider = SubmitTimeoutProvider()
    registry = VideoProviderRegistry()
    registry.register(provider)

    storage = AsyncMock()
    svc = VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider="submit_timeout",
        max_retries=2,
        fallback_enabled=False,
    )

    cost, mock_jm = await _run(svc, "submit_timeout")

    assert provider.submit_calls == 1, (
        f"submit() called {provider.submit_calls}x on a submit-phase timeout "
        "-- this re-submits an ambiguous (possibly already-billed) request"
    )
    assert cost == 0.0
    mock_jm.mark_failed.assert_called_once()
    mock_jm.mark_completed.assert_not_called()


async def test_submit_phase_timeout_does_not_walk_fallback_chain_either():
    """Same guarantee, but confirm a fallback provider isn't tried instead --
    the current single-provider assertion already proves no re-submit; this
    pins that a submit timeout doesn't quietly succeed via another provider,
    which would mask the failure mode in a differently-configured job."""
    provider = SubmitTimeoutProvider()
    registry = VideoProviderRegistry()
    registry.register(provider)

    storage = AsyncMock()
    svc = VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider="submit_timeout",
        max_retries=1,
        fallback_enabled=True,
    )

    cost, mock_jm = await _run(svc, "submit_timeout", fallback=[])

    assert provider.submit_calls == 1
    assert cost == 0.0
    mock_jm.mark_failed.assert_called_once()


async def test_poll_phase_timeout_still_retries():
    """Unchanged behaviour: a timeout while POLLING (a GET, idempotent) must
    still retry -- it carries no double-billing risk and this must not
    regress reliability while fixing submit-phase safety."""
    provider = PollTimeoutThenSuccessProvider()
    registry = VideoProviderRegistry()
    registry.register(provider)

    storage = AsyncMock()
    storage.store_video = AsyncMock(return_value="/tmp/test.mp4")
    storage.upload_to_minio = AsyncMock(return_value="key")

    svc = VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider="poll_timeout",
        max_retries=2,
        fallback_enabled=False,
    )

    cost, mock_jm = await _run(svc, "poll_timeout")

    assert provider.check_status_calls >= 2, "poll timeout was not retried"
    assert cost > 0
    mock_jm.mark_completed.assert_called_once()


async def test_pre_send_connect_error_still_retries():
    """A connect failure before anything was sent is NOT ambiguous -- the
    request never reached the provider, so nothing could have been billed.
    This must stay retriable, unlike a submit-phase timeout."""
    provider = ConnectErrorThenSuccessProvider()
    registry = VideoProviderRegistry()
    registry.register(provider)

    storage = AsyncMock()
    storage.store_video = AsyncMock(return_value="/tmp/test.mp4")
    storage.upload_to_minio = AsyncMock(return_value="key")

    svc = VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider="connect_error",
        max_retries=2,
        fallback_enabled=False,
    )

    cost, mock_jm = await _run(svc, "connect_error")

    assert provider.submit_calls == 2, "pre-send connect error was not retried"
    assert cost > 0
    mock_jm.mark_completed.assert_called_once()
