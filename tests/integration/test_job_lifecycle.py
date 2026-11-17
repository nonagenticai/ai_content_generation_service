"""Integration test: job lifecycle using mock provider (no DB required)."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.providers.base import VideoProviderRegistry
from src.services.providers.mock import MockProvider
from src.services.video_generation_service import VideoGenerationService


def _make_service(tmpdir: str) -> VideoGenerationService:
    registry = VideoProviderRegistry()
    registry.register(MockProvider(delay_s=0.0))

    with patch("src.services.storage_service.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            storage=MagicMock(
                local_path=tmpdir,
                backend="local",
                minio_endpoint="",
                minio_access_key="",
                minio_secret_key="",
                minio_bucket="test",
                minio_secure=False,
            )
        )
        from src.services.storage_service import StorageService

        storage = StorageService()

    return VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider="mock",
    )


async def test_create_and_process_job():
    """End-to-end: create job -> process -> result via mock provider."""
    with tempfile.TemporaryDirectory() as tmpdir:
        svc = _make_service(tmpdir)

        # Mock DB session
        mock_job = MagicMock()
        mock_job.job_id = "test-job-1"
        mock_job.provider = "mock"
        mock_job.prompt = "a beautiful sunset"
        mock_job.negative_prompt = None
        mock_job.image_url = None
        mock_job.duration_s = 5.0
        mock_job.width = 1920
        mock_job.height = 1080
        mock_job.style = None
        mock_job.seed = None
        mock_job.extra_params = None
        mock_job.status = "pending"
        mock_job.cost = None
        mock_job.minio_key = None
        mock_job.fallback_providers = None
        mock_job.callback_url = None

        mock_db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings") as mock_gs,
        ):
            mock_gs.return_value = MagicMock(
                generation=MagicMock(
                    default_duration_s=5.0,
                    default_width=1920,
                    default_height=1080,
                )
            )
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            mock_jm.mark_processing = AsyncMock()
            mock_jm.set_provider_job_id = AsyncMock()
            mock_jm.mark_completed = AsyncMock()
            mock_jm.mark_failed = AsyncMock()
            mock_jm.mark_retrying = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(mock_db, "test-job-1", "tenant-1")

            mock_jm.mark_processing.assert_called()
            mock_jm.mark_completed.assert_called_once()
            assert cost > 0


async def test_process_job_unknown_provider():
    """Job with unknown provider should be marked failed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        svc = _make_service(tmpdir)

        mock_job = MagicMock()
        mock_job.job_id = "test-job-2"
        mock_job.provider = "nonexistent"
        mock_job.status = "pending"
        mock_job.fallback_providers = None

        mock_db = AsyncMock()

        with patch("src.services.video_generation_service.JobManager") as mock_jm:
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            mock_jm.mark_failed = AsyncMock()
            mock_jm.record_usage = AsyncMock()

            cost = await svc.process_job(mock_db, "test-job-2", "tenant-1")

            mock_jm.mark_failed.assert_called_once()
            assert cost == 0.0


async def test_process_job_budget_exceeded():
    """Job exceeding budget should be marked failed with budget message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        svc = _make_service(tmpdir)

        mock_job = MagicMock()
        mock_job.job_id = "test-job-3"
        mock_job.provider = "mock"
        mock_job.prompt = "test"
        mock_job.duration_s = 5.0
        mock_job.status = "pending"
        mock_job.fallback_providers = None

        mock_db = AsyncMock()

        with (
            patch("src.services.video_generation_service.JobManager") as mock_jm,
            patch("src.config.get_settings") as mock_gs,
        ):
            mock_gs.return_value = MagicMock(generation=MagicMock(default_duration_s=5.0))
            mock_jm.get_job = AsyncMock(return_value=mock_job)
            mock_jm.mark_failed = AsyncMock()

            cost = await svc.process_job(
                mock_db,
                "test-job-3",
                "tenant-1",
                budget_limit=0.001,
                spent_so_far=0.0,
            )

            mock_jm.mark_failed.assert_called_once()
            call_args = mock_jm.mark_failed.call_args
            assert "Budget exceeded" in call_args[0][2]
            assert cost == 0.0
