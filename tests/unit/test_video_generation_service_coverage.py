"""Tests for video_generation_service.py to cover missing lines.

Covers: create_job, _resolve_provider, process_job (budget, cancelled,
unhealthy provider, health recording, normalization, storage, retry,
CancelledError, all-providers-exhausted), cancel_job.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.exceptions import JobNotFoundError
from src.services.normalizer import NormalizationOutcome
from src.services.providers.base import VideoGenerationResult, VideoProviderRegistry
from src.services.video_generation_service import VideoGenerationService

# ============================================================================
# Helpers
# ============================================================================


class FakeProvider:
    """Minimal concrete provider for testing."""

    name = "fake"

    async def generate(self, **kwargs) -> VideoGenerationResult:
        return VideoGenerationResult(
            success=True,
            video_data=b"fake-video-data",
            video_url="https://cdn.example.com/v.mp4",
            provider_job_id="pj-1",
            duration_s=5.0,
        )

    def estimate_cost(self, duration_s: float) -> float:
        return 0.01 * duration_s


class FailingGenerateProvider:
    """Provider whose generate raises RuntimeError."""

    name = "failing"

    async def generate(self, **kwargs) -> VideoGenerationResult:
        return VideoGenerationResult(success=False, error="Generation failed")

    def estimate_cost(self, duration_s: float) -> float:
        return 0.01


def _make_registry(*providers) -> VideoProviderRegistry:
    reg = VideoProviderRegistry()
    for p in providers:
        reg.register(p)
    return reg


def _make_job_mock(
    *,
    job_id="job-1",
    status="pending",
    provider="fake",
    prompt="a cat",
    duration_s=5.0,
    width=None,
    height=None,
    style=None,
    seed=None,
    extra_params=None,
    negative_prompt=None,
    image_url=None,
    fallback_providers=None,
    callback_url=None,
    error_message=None,
):
    job = MagicMock()
    job.job_id = job_id
    job.status = status
    job.provider = provider
    job.prompt = prompt
    job.duration_s = duration_s
    job.width = width
    job.height = height
    job.style = style
    job.seed = seed
    job.extra_params = extra_params
    job.negative_prompt = negative_prompt
    job.image_url = image_url
    job.fallback_providers = fallback_providers
    job.callback_url = callback_url
    job.error_message = error_message
    return job


def _make_settings():
    settings = MagicMock()
    settings.generation.default_duration_s = 5.0
    settings.generation.default_width = 1920
    settings.generation.default_height = 1080
    return settings


# ============================================================================
# create_job
# ============================================================================


class TestCreateJob:
    """Tests for VideoGenerationService.create_job (lines 106-123)."""

    async def test_create_job_with_provider(self) -> None:
        """create_job passes params to JobManager.create_job."""
        registry = _make_registry(FakeProvider())
        storage = MagicMock()

        svc = VideoGenerationService(
            storage=storage,
            registry=registry,
            default_provider="fake",
        )

        fake_job = MagicMock()
        fake_job.job_id = "job-abc"

        db = AsyncMock()

        with patch("src.services.video_generation_service.JobManager") as mock_jm:
            mock_jm.create_job = AsyncMock(return_value=fake_job)
            result = await svc.create_job(
                db,
                tenant_id="t1",
                prompt="hello",
                provider="fake",
                negative_prompt="bad things",
                image_url="https://example.com/img.png",
                duration_s=10.0,
                width=1280,
                height=720,
                style="cinematic",
                seed=42,
                extra_params={"fps": 30},
                fallback_providers=["other"],
                callback_url="https://example.com/cb",
            )

        assert result == "job-abc"
        mock_jm.create_job.assert_awaited_once()

    async def test_create_job_uses_default_provider(self) -> None:
        """When provider is None, _resolve_provider is used."""
        registry = _make_registry(FakeProvider())
        storage = MagicMock()

        svc = VideoGenerationService(
            storage=storage,
            registry=registry,
            default_provider="fake",
        )

        fake_job = MagicMock()
        fake_job.job_id = "job-def"

        db = AsyncMock()

        with patch("src.services.video_generation_service.JobManager") as mock_jm:
            mock_jm.create_job = AsyncMock(return_value=fake_job)
            result = await svc.create_job(
                db,
                tenant_id="t1",
                prompt="test",
                provider=None,
            )

        assert result == "job-def"
        call_kwargs = mock_jm.create_job.call_args[1]
        assert call_kwargs["provider"] == "fake"


# ============================================================================
# _resolve_provider
# ============================================================================


class TestResolveProvider:
    """Tests for _resolve_provider (lines 125-132)."""

    def test_resolve_uses_selector(self) -> None:
        selector = MagicMock()
        selector.select.return_value = "kling"

        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=_make_registry(),
            default_provider="fake",
            selector=selector,
        )

        result = svc._resolve_provider(duration_s=5.0)
        assert result == "kling"
        selector.select.assert_called_once()

    def test_resolve_selector_failure_falls_back(self) -> None:
        """When selector raises, return default_provider."""
        selector = MagicMock()
        selector.select.side_effect = RuntimeError("boom")

        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=_make_registry(),
            default_provider="fake",
            selector=selector,
        )

        result = svc._resolve_provider(duration_s=5.0)
        assert result == "fake"

    def test_resolve_no_selector(self) -> None:
        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=_make_registry(),
            default_provider="fake",
            selector=None,
        )

        result = svc._resolve_provider()
        assert result == "fake"


# ============================================================================
# process_job - happy path & variations
# ============================================================================


class TestProcessJob:
    """Tests for process_job and _process_job_inner covering all branches."""

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_completed")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_process_job_success(self, mock_started, mock_completed, mock_finished) -> None:
        """Full happy-path: submit, generate, store, mark completed."""
        provider = FakeProvider()
        registry = _make_registry(provider)
        storage = AsyncMock()
        storage.store_video = AsyncMock(return_value="/tmp/job-1/clip.mp4")
        storage.upload_to_minio = AsyncMock(return_value="content_generation/job-1/clip.mp4")

        svc = VideoGenerationService(
            storage=storage,
            registry=registry,
            default_provider="fake",
            max_retries=0,
        )

        job = _make_job_mock(status="pending", provider="fake")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1", budget_limit=5.0)

        assert cost >= 0
        mock_jm.mark_completed.assert_awaited_once()
        mock_jm.record_usage.assert_awaited_once()
        mock_started.assert_called_once()
        mock_completed.assert_called_once()
        mock_finished.assert_called_once()

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_failed")
    async def test_process_job_budget_exceeded(self, mock_failed, mock_finished) -> None:
        """Budget check fails -> mark_failed and return 0.0."""
        provider = FakeProvider()
        registry = _make_registry(provider)

        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=registry,
            default_provider="fake",
        )

        job = _make_job_mock(status="pending", provider="fake", duration_s=1000.0)
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_failed = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1", budget_limit=0.001)

        assert cost == 0.0
        mock_jm.mark_failed.assert_awaited_once()
        mock_failed.assert_called_once()

    async def test_process_job_cancelled(self) -> None:
        """Cancelled job returns 0.0 immediately."""
        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=_make_registry(),
            default_provider="fake",
        )

        job = _make_job_mock(status="cancelled")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            cost = await svc.process_job(db, "job-1", "t1")

        assert cost == 0.0

    async def test_process_job_not_found(self) -> None:
        """Job not found raises ValueError."""
        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=_make_registry(),
            default_provider="fake",
        )

        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=None)
            with pytest.raises(JobNotFoundError, match="Job not found"):
                await svc.process_job(db, "nonexistent", "t1")

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_failed")
    async def test_process_job_unhealthy_provider_skipped(self, mock_failed, mock_finished) -> None:
        """Unhealthy provider is skipped; falls through to failure."""
        provider = FakeProvider()
        registry = _make_registry(provider)

        health_monitor = MagicMock()
        health_monitor.is_available.return_value = False

        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=registry,
            default_provider="fake",
            health_monitor=health_monitor,
            fallback_enabled=False,
        )

        job = _make_job_mock(status="pending", provider="fake")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_failed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1")

        assert cost == 0.0
        mock_jm.mark_failed.assert_awaited_once()

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_completed")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_process_job_health_monitor_check_exception(
        self, mock_started, mock_completed, mock_finished
    ) -> None:
        """When health_monitor.is_available() raises, processing still continues."""
        provider = FakeProvider()
        registry = _make_registry(provider)
        storage = AsyncMock()
        storage.store_video = AsyncMock(return_value="/tmp/clip.mp4")
        storage.upload_to_minio = AsyncMock(return_value=None)

        health_monitor = MagicMock()
        health_monitor.is_available.side_effect = OSError("redis down")
        health_monitor.record_success = MagicMock()

        svc = VideoGenerationService(
            storage=storage,
            registry=registry,
            default_provider="fake",
            health_monitor=health_monitor,
            fallback_enabled=False,
            max_retries=0,
        )

        job = _make_job_mock(status="pending", provider="fake")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1")

        # The exception is caught, processing continues
        assert cost >= 0

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_completed")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_process_job_with_normalizer(
        self, mock_started, mock_completed, mock_finished
    ) -> None:
        """Normalizer is called on video data."""
        provider = FakeProvider()
        registry = _make_registry(provider)
        storage = AsyncMock()
        storage.store_video = AsyncMock(return_value="/tmp/job-1/clip.mp4")
        storage.upload_to_minio = AsyncMock(return_value=None)

        normalizer = AsyncMock()
        normalizer.normalize = AsyncMock(
            return_value=NormalizationOutcome(data=b"normalized-data", normalized=True, reason=None)
        )

        svc = VideoGenerationService(
            storage=storage,
            registry=registry,
            default_provider="fake",
            max_retries=0,
            normalizer=normalizer,
        )

        job = _make_job_mock(status="pending", provider="fake")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1")

        normalizer.normalize.assert_awaited_once()
        assert cost >= 0

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_completed")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_process_job_normalizer_failure_uses_raw(
        self, mock_started, mock_completed, mock_finished
    ) -> None:
        """When normalizer fails, raw video data is used."""
        provider = FakeProvider()
        registry = _make_registry(provider)
        storage = AsyncMock()
        storage.store_video = AsyncMock(return_value="/tmp/job-1/clip.mp4")
        storage.upload_to_minio = AsyncMock(return_value=None)

        normalizer = AsyncMock()
        normalizer.normalize = AsyncMock(side_effect=RuntimeError("ffmpeg crashed"))

        svc = VideoGenerationService(
            storage=storage,
            registry=registry,
            default_provider="fake",
            max_retries=0,
            normalizer=normalizer,
        )

        job = _make_job_mock(status="pending", provider="fake")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1")

        assert cost >= 0
        mock_jm.mark_completed.assert_awaited_once()

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_completed")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_process_job_with_health_monitor_records_success(
        self, mock_started, mock_completed, mock_finished
    ) -> None:
        """Health monitor records success on successful generation."""
        provider = FakeProvider()
        registry = _make_registry(provider)
        storage = AsyncMock()
        storage.store_video = AsyncMock(return_value="/tmp/clip.mp4")
        storage.upload_to_minio = AsyncMock(return_value=None)

        health_monitor = MagicMock()
        health_monitor.is_available.return_value = True
        health_monitor.record_success = MagicMock()

        svc = VideoGenerationService(
            storage=storage,
            registry=registry,
            default_provider="fake",
            max_retries=0,
            health_monitor=health_monitor,
        )

        job = _make_job_mock(status="pending", provider="fake")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1")

        assert cost >= 0
        health_monitor.record_success.assert_called_once()

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_completed")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_process_job_health_monitor_record_success_exception(
        self, mock_started, mock_completed, mock_finished
    ) -> None:
        """When health_monitor.record_success raises, generation still succeeds."""
        provider = FakeProvider()
        registry = _make_registry(provider)
        storage = AsyncMock()
        storage.store_video = AsyncMock(return_value="/tmp/clip.mp4")
        storage.upload_to_minio = AsyncMock(return_value=None)

        health_monitor = MagicMock()
        health_monitor.is_available.return_value = True
        health_monitor.record_success.side_effect = OSError("redis down")

        svc = VideoGenerationService(
            storage=storage,
            registry=registry,
            default_provider="fake",
            max_retries=0,
            health_monitor=health_monitor,
        )

        job = _make_job_mock(status="pending", provider="fake")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1")

        assert cost >= 0

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_failed")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_process_job_generation_fails_records_health_failure(
        self, mock_started, mock_failed, mock_finished
    ) -> None:
        """When generation fails, health_monitor.record_failure is called."""

        class FailProvider:
            name = "fail"

            async def generate(self, **kw) -> VideoGenerationResult:
                raise RuntimeError("Generation explosion")

            def estimate_cost(self, duration_s: float) -> float:
                return 0.01

        registry = _make_registry(FailProvider())

        health_monitor = MagicMock()
        health_monitor.is_available.return_value = True
        health_monitor.record_failure = MagicMock()

        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=registry,
            default_provider="fail",
            max_retries=0,
            fallback_enabled=False,
            health_monitor=health_monitor,
        )

        job = _make_job_mock(status="pending", provider="fail")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_failed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1")

        assert cost == 0.0
        health_monitor.record_failure.assert_called_once_with("fail")

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_failed")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_process_job_health_monitor_record_failure_exception(
        self, mock_started, mock_failed, mock_finished
    ) -> None:
        """When health_monitor.record_failure raises, processing continues."""

        class FailProvider:
            name = "fail"

            async def generate(self, **kw) -> VideoGenerationResult:
                raise RuntimeError("Generation explosion")

            def estimate_cost(self, duration_s: float) -> float:
                return 0.01

        registry = _make_registry(FailProvider())

        health_monitor = MagicMock()
        health_monitor.is_available.return_value = True
        health_monitor.record_failure.side_effect = OSError("redis fail")

        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=registry,
            default_provider="fail",
            max_retries=0,
            fallback_enabled=False,
            health_monitor=health_monitor,
        )

        job = _make_job_mock(status="pending", provider="fail")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_failed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1")

        assert cost == 0.0

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_process_job_cancelled_error(self, mock_started, mock_finished) -> None:
        """CancelledError during generation -> mark_cancelled."""

        class CancelProvider:
            name = "cancel"

            async def generate(self, **kw) -> VideoGenerationResult:
                raise asyncio.CancelledError()

            def estimate_cost(self, duration_s: float) -> float:
                return 0.01

        registry = _make_registry(CancelProvider())

        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=registry,
            default_provider="cancel",
            max_retries=0,
        )

        job = _make_job_mock(status="pending", provider="cancel")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_cancelled = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1")

        assert cost == 0.0
        mock_jm.mark_cancelled.assert_awaited_once()
        mock_finished.assert_called_once()

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_failed")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_process_job_unknown_provider(
        self, mock_started, mock_failed, mock_finished
    ) -> None:
        """Unknown provider in chain is skipped."""
        registry = _make_registry()  # empty

        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=registry,
            default_provider="fake",
            fallback_enabled=False,
        )

        job = _make_job_mock(status="pending", provider="nonexistent")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_failed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1")

        assert cost == 0.0
        mock_jm.mark_failed.assert_awaited_once()

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_failed")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_process_job_fallback_chain(
        self, mock_started, mock_failed, mock_finished
    ) -> None:
        """Fallback chain: first provider fails, second succeeds."""

        class FlakeyProvider:
            name = "flakey"

            async def generate(self, **kw) -> VideoGenerationResult:
                raise RuntimeError("flakey failure")

            def estimate_cost(self, d: float) -> float:
                return 0.01

        class BackupProvider:
            name = "backup"

            async def generate(self, **kw) -> VideoGenerationResult:
                return VideoGenerationResult(
                    success=True,
                    video_data=b"backup-data",
                    provider_job_id="bk-1",
                    duration_s=5.0,
                )

            def estimate_cost(self, d: float) -> float:
                return 0.02

        registry = _make_registry(FlakeyProvider(), BackupProvider())
        storage = AsyncMock()
        storage.store_video = AsyncMock(return_value="/tmp/clip.mp4")
        storage.upload_to_minio = AsyncMock(return_value=None)

        svc = VideoGenerationService(
            storage=storage,
            registry=registry,
            default_provider="flakey",
            max_retries=0,
            fallback_enabled=True,
        )

        job = _make_job_mock(
            status="pending",
            provider="flakey",
            fallback_providers=["backup"],
        )
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1")

        assert cost > 0
        mock_jm.mark_completed.assert_awaited_once()

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_completed")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_process_job_generation_returns_failure_result(
        self, mock_started, mock_completed, mock_finished
    ) -> None:
        """When generate() returns success=False, RuntimeError is raised internally."""

        class FailResultProvider:
            name = "failresult"

            async def generate(self, **kw) -> VideoGenerationResult:
                return VideoGenerationResult(success=False, error="Server overloaded")

            def estimate_cost(self, d: float) -> float:
                return 0.01

        registry = _make_registry(FailResultProvider())

        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=registry,
            default_provider="failresult",
            max_retries=0,
            fallback_enabled=False,
        )

        job = _make_job_mock(status="pending", provider="failresult")
        db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_failed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(db, "job-1", "t1")

        assert cost == 0.0


# ============================================================================
# cancel_job
# ============================================================================


class TestCancelJob:
    """Tests for cancel_job covering task/no-task branches."""

    async def test_cancel_active_task(self) -> None:
        """Cancel an active asyncio task."""
        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=_make_registry(),
            default_provider="fake",
        )

        started = asyncio.Event()

        async def fake_coro():
            started.set()
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                raise

        task = asyncio.create_task(fake_coro())
        svc.register_task("job-1", task)

        # Wait for the task to actually start
        await started.wait()

        db = AsyncMock()

        with patch("src.services.video_generation_service.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=_make_job_mock(status="processing"))
            mock_jm.mark_cancelled = AsyncMock()

            result = await svc.cancel_job(db, "job-1", "tenant-1")

        assert result is True
        assert task.cancelled() or task.done()
        assert "job-1" not in svc._active_tasks

    async def test_cancel_pending_job_no_task(self) -> None:
        """Cancel pending job without active task."""
        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=_make_registry(),
            default_provider="fake",
        )

        job = _make_job_mock(status="pending")
        db = AsyncMock()

        with patch("src.services.video_generation_service.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_cancelled = AsyncMock()

            result = await svc.cancel_job(db, "job-1", "tenant-1")

        assert result is True
        mock_jm.mark_cancelled.assert_awaited_once()

    async def test_cancel_completed_job_returns_false(self) -> None:
        """Cannot cancel a completed job."""
        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=_make_registry(),
            default_provider="fake",
        )

        job = _make_job_mock(status="completed")
        db = AsyncMock()

        with patch("src.services.video_generation_service.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=job)

            result = await svc.cancel_job(db, "job-1", "tenant-1")

        assert result is False

    async def test_cancel_nonexistent_job_returns_false(self) -> None:
        """Cannot cancel a job that does not exist."""
        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=_make_registry(),
            default_provider="fake",
        )

        db = AsyncMock()

        with patch("src.services.video_generation_service.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=None)

            result = await svc.cancel_job(db, "nonexistent", "tenant-1")

        assert result is False


# ============================================================================
# Extracted phase-method tests (decomposition of _process_job_inner)
# ============================================================================


def _make_svc(**kwargs) -> VideoGenerationService:
    """Build a VideoGenerationService with override-able defaults."""
    return VideoGenerationService(
        storage=kwargs.pop("storage", AsyncMock()),
        registry=kwargs.pop("registry", _make_registry(FakeProvider())),
        default_provider=kwargs.pop("default_provider", "fake"),
        **kwargs,
    )


class TestBuildProviderChain:
    """_build_provider_chain: deduplication + fallback gating."""

    def test_chain_with_fallbacks_enabled(self) -> None:
        job = MagicMock()
        job.provider = "primary"
        job.fallback_providers = ["fb1", "fb2"]
        chain = VideoGenerationService._build_provider_chain(job, fallback_enabled=True)
        assert chain == ["primary", "fb1", "fb2"]

    def test_chain_fallbacks_disabled(self) -> None:
        job = MagicMock()
        job.provider = "primary"
        job.fallback_providers = ["fb1", "fb2"]
        chain = VideoGenerationService._build_provider_chain(job, fallback_enabled=False)
        assert chain == ["primary"]

    def test_chain_dedupes_primary_in_fallbacks(self) -> None:
        """If primary is also listed in fallback_providers, it appears once."""
        job = MagicMock()
        job.provider = "primary"
        job.fallback_providers = ["primary", "fb1"]
        chain = VideoGenerationService._build_provider_chain(job, fallback_enabled=True)
        assert chain == ["primary", "fb1"]

    def test_chain_no_fallbacks_attribute_value(self) -> None:
        job = MagicMock()
        job.provider = "only"
        job.fallback_providers = None
        chain = VideoGenerationService._build_provider_chain(job, fallback_enabled=True)
        assert chain == ["only"]


class TestProviderIsAvailable:
    """_provider_is_available: monitor None / False / raises."""

    def test_returns_none_when_no_monitor(self) -> None:
        svc = _make_svc(health_monitor=None)
        assert svc._provider_is_available("fake") is None

    def test_returns_false_when_monitor_says_unhealthy(self) -> None:
        monitor = MagicMock()
        monitor.is_available.return_value = False
        svc = _make_svc(health_monitor=monitor)
        assert svc._provider_is_available("fake") is False

    def test_returns_true_when_monitor_raises(self) -> None:
        """A flaky monitor must NOT take down the provider -- best effort."""
        monitor = MagicMock()
        monitor.is_available.side_effect = RuntimeError("monitor crashed")
        svc = _make_svc(health_monitor=monitor)
        # Best-effort: treat as available so we don't lose all providers
        assert svc._provider_is_available("fake") is True


class TestStoreVideoArtifacts:
    """_store_video_artifacts: empty bytes vs real bytes."""

    async def test_empty_video_data_returns_zeros(self) -> None:
        svc = _make_svc()
        path, key, size = await svc._store_video_artifacts("job1", "fake", None)
        assert path is None
        assert key is None
        assert size == 0

    async def test_stores_and_uploads(self) -> None:
        storage = AsyncMock()
        storage.store_video = AsyncMock(return_value="/tmp/foo.mp4")
        storage.upload_to_minio = AsyncMock(return_value="minio/foo.mp4")
        svc = _make_svc(storage=storage)
        path, key, size = await svc._store_video_artifacts("job1", "fake", b"x" * 42)
        assert path == "/tmp/foo.mp4"
        assert key == "minio/foo.mp4"
        assert size == 42
        storage.store_video.assert_awaited_once()
        storage.upload_to_minio.assert_awaited_once()


class TestMaybeNormalize:
    """_maybe_normalize: passthrough / normalize / fallback on error."""

    async def test_no_normalizer_returns_input(self) -> None:
        svc = _make_svc(normalizer=None)
        out = await svc._maybe_normalize(b"raw", "job1")
        assert out.data == b"raw"
        assert out.degraded is False  # switched off is a posture, not a fault

    async def test_empty_bytes_passthrough(self) -> None:
        normalizer = MagicMock()
        normalizer.normalize = AsyncMock()
        svc = _make_svc(normalizer=normalizer)
        out = await svc._maybe_normalize(None, "job1")
        assert out.data == b""
        assert out.degraded is False
        normalizer.normalize.assert_not_called()

    async def test_normalizes_when_configured(self) -> None:
        normalizer = MagicMock()
        normalizer.normalize = AsyncMock(
            return_value=NormalizationOutcome(data=b"normalized", normalized=True, reason=None)
        )
        svc = _make_svc(normalizer=normalizer)
        out = await svc._maybe_normalize(b"raw", "job1")
        assert out.data == b"normalized"
        assert out.degraded is False

    async def test_falls_back_to_raw_on_normalization_error(self) -> None:
        """A normalizer crash must not lose the generated video."""
        normalizer = MagicMock()
        normalizer.normalize = AsyncMock(side_effect=OSError("ffmpeg crashed"))
        svc = _make_svc(normalizer=normalizer)
        out = await svc._maybe_normalize(b"raw", "job1")
        assert out.data == b"raw"
        assert out.degraded is True, "a raw fallback is a degradation, not a clean pass"


class TestHealthRecording:
    """_record_health_success / _record_health_failure -- best-effort + None-safe."""

    def test_success_with_no_monitor_is_noop(self) -> None:
        svc = _make_svc(health_monitor=None)
        svc._record_health_success("fake", 1.5)  # must not raise

    def test_failure_with_no_monitor_is_noop(self) -> None:
        svc = _make_svc(health_monitor=None)
        svc._record_health_failure("fake")  # must not raise

    def test_success_records_milliseconds(self) -> None:
        monitor = MagicMock()
        svc = _make_svc(health_monitor=monitor)
        svc._record_health_success("fake", 1.5)
        monitor.record_success.assert_called_once_with("fake", 1500.0)

    def test_success_swallows_monitor_error(self) -> None:
        monitor = MagicMock()
        monitor.record_success.side_effect = OSError("redis down")
        svc = _make_svc(health_monitor=monitor)
        svc._record_health_success("fake", 1.0)  # must not raise

    def test_failure_swallows_monitor_error(self) -> None:
        monitor = MagicMock()
        monitor.record_failure.side_effect = RuntimeError("monitor flaky")
        svc = _make_svc(health_monitor=monitor)
        svc._record_health_failure("fake")  # must not raise


class TestCancelledErrorMarkCancelledDbFailure:
    """A ``mark_cancelled`` failure during a real CancelledError must not
    replace/erase the cancellation: the original CancelledError semantics
    (job treated as cancelled, not a generic failure) must survive."""

    @patch("src.services.video_generation_service.record_job_finished")
    @patch("src.services.video_generation_service.record_job_started")
    async def test_mark_cancelled_db_error_does_not_mask_cancellation(
        self, mock_started, mock_finished, caplog
    ) -> None:
        class CancelProvider:
            name = "cancel"

            async def generate(self, **kw) -> VideoGenerationResult:
                raise asyncio.CancelledError()

            def estimate_cost(self, duration_s: float) -> float:
                return 0.01

        registry = _make_registry(CancelProvider())

        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=registry,
            default_provider="cancel",
            max_retries=0,
        )

        job = _make_job_mock(status="pending", provider="cancel")
        db = AsyncMock()

        caplog.set_level("ERROR")
        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings", return_value=_make_settings()),
        ):
            mock_jm.get_job = AsyncMock(return_value=job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_cancelled = AsyncMock(side_effect=ConnectionError("db pool exhausted"))
            mock_jm.mark_failed = AsyncMock()

            # Must NOT raise -- a ConnectionError escaping here would be
            # caught by the outer _process_job_bg generic `except Exception`
            # handler and reclassified as "Unexpected error during video
            # generation" -> job marked FAILED instead of CANCELLED.
            cost = await svc.process_job(db, "job-1", "t1")

        assert cost == 0.0
        mock_jm.mark_cancelled.assert_awaited_once()
        # The generic top-level failure path (mark_failed) must never be
        # reached from inside process_job for a cancellation -- that mapping
        # only exists in the OUTER bg-task wrapper, and only as a fallback
        # for exceptions process_job itself doesn't already handle.
        mock_jm.mark_failed.assert_not_awaited()
        mock_finished.assert_called_once()

        messages = [r.getMessage() for r in caplog.records]
        assert any("Failed to persist cancellation for job job-1" in m for m in messages)
