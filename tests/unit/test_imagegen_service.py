"""Unit tests for ImageGenService orchestration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.image_providers.base import ImageGenResult
from src.services.imagegen_service import ImageGenService, _filename_for_mime


class TestFilenameMime:
    def test_jpeg(self) -> None:
        assert _filename_for_mime("image/jpeg") == "image.jpg"

    def test_webp(self) -> None:
        assert _filename_for_mime("image/webp") == "image.webp"

    def test_png(self) -> None:
        assert _filename_for_mime("image/png") == "image.png"

    def test_none(self) -> None:
        assert _filename_for_mime(None) == "image.png"

    def test_unknown(self) -> None:
        assert _filename_for_mime("image/bmp") == "image.png"


class TestImageGenServiceInit:
    def test_register_and_unregister_task(self) -> None:
        service = ImageGenService(
            storage=MagicMock(),
            registry=MagicMock(),
            default_provider="mock_images",
        )
        task = MagicMock()
        service.register_task("j1", task)
        assert "j1" in service._active_tasks
        service.unregister_task("j1")
        assert "j1" not in service._active_tasks

    def test_unregister_nonexistent_task(self) -> None:
        service = ImageGenService(
            storage=MagicMock(),
            registry=MagicMock(),
            default_provider="mock_images",
        )
        service.unregister_task("missing")


class TestEstimateCost:
    def test_known_provider(self) -> None:
        provider = MagicMock()
        provider.estimate_cost.return_value = 0.04
        registry = MagicMock()
        registry.get.return_value = provider

        service = ImageGenService(
            storage=MagicMock(),
            registry=registry,
            default_provider="openai_images",
        )
        cost = service.estimate_cost("openai_images", size="1024x1024", quality="hd")
        assert cost == 0.04
        provider.estimate_cost.assert_called_once_with(size="1024x1024", quality="hd")

    def test_unknown_provider(self) -> None:
        registry = MagicMock()
        registry.get.return_value = None

        service = ImageGenService(
            storage=MagicMock(),
            registry=registry,
            default_provider="openai_images",
        )
        assert service.estimate_cost("unknown") is None


class TestCreateJob:
    async def test_create_job_delegates_to_manager(self) -> None:
        fake_job = MagicMock()
        fake_job.job_id = "img-job-1"

        registry = MagicMock()
        service = ImageGenService(
            storage=MagicMock(),
            registry=registry,
            default_provider="mock_images",
        )

        db = AsyncMock()
        with patch(
            "src.services.imagegen_service.ImageJobManager.create_job",
            new_callable=AsyncMock,
            return_value=fake_job,
        ) as mock_create:
            job_id = await service.create_job(
                db,
                tenant_id="t1",
                prompt="a sunset",
                provider="openai_images",
                negative_prompt="ugly",
                style="editorial",
            )

        assert job_id == "img-job-1"
        mock_create.assert_called_once()

    async def test_create_job_uses_default_provider_when_none(self) -> None:
        fake_job = MagicMock()
        fake_job.job_id = "img-job-2"

        service = ImageGenService(
            storage=MagicMock(),
            registry=MagicMock(),
            default_provider="mock_images",
        )

        db = AsyncMock()
        with patch(
            "src.services.imagegen_service.ImageJobManager.create_job",
            new_callable=AsyncMock,
            return_value=fake_job,
        ) as mock_create:
            await service.create_job(db, tenant_id="t1", prompt="test", provider=None)

        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs["provider"] == "mock_images"


class TestCancelJob:
    async def test_cancel_active_task(self) -> None:
        service = ImageGenService(
            storage=MagicMock(),
            registry=MagicMock(),
            default_provider="mock_images",
        )

        # Create a real coroutine that will be cancelled
        async def _long_running():
            await asyncio.sleep(100)

        task = asyncio.create_task(_long_running())
        service.register_task("j1", task)
        db = AsyncMock()
        db.commit = AsyncMock()

        with (
            patch(
                "src.services.imagegen_service.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=MagicMock(status="processing"),
            ),
            patch(
                "src.services.imagegen_service.ImageJobManager.mark_cancelled",
                new_callable=AsyncMock,
            ),
        ):
            result = await service.cancel_job(db, "j1", "tenant-1")

        assert result is True
        assert "j1" not in service._active_tasks

    async def test_cancel_pending_job_no_task(self) -> None:
        service = ImageGenService(
            storage=MagicMock(),
            registry=MagicMock(),
            default_provider="mock_images",
        )

        fake_job = MagicMock()
        fake_job.status = "pending"
        db = AsyncMock()
        db.commit = AsyncMock()

        with (
            patch(
                "src.services.imagegen_service.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=fake_job,
            ),
            patch(
                "src.services.imagegen_service.ImageJobManager.mark_cancelled",
                new_callable=AsyncMock,
            ),
        ):
            result = await service.cancel_job(db, "j2", "tenant-1")

        assert result is True

    async def test_cancel_completed_job_returns_false(self) -> None:
        service = ImageGenService(
            storage=MagicMock(),
            registry=MagicMock(),
            default_provider="mock_images",
        )

        fake_job = MagicMock()
        fake_job.status = "completed"
        db = AsyncMock()

        with patch(
            "src.services.imagegen_service.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=fake_job,
        ):
            result = await service.cancel_job(db, "j3", "tenant-1")

        assert result is False

    async def test_cancel_nonexistent_job_returns_false(self) -> None:
        service = ImageGenService(
            storage=MagicMock(),
            registry=MagicMock(),
            default_provider="mock_images",
        )

        db = AsyncMock()
        with patch(
            "src.services.imagegen_service.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await service.cancel_job(db, "j4", "tenant-1")

        assert result is False


class TestProcessJob:
    async def test_process_job_success(self) -> None:
        provider = MagicMock()
        provider.estimate_cost.return_value = 0.04
        provider.generate = AsyncMock(
            return_value=ImageGenResult(
                success=True,
                image_data=b"png-bytes",
                mime_type="image/png",
                width=1024,
                height=768,
            )
        )

        registry = MagicMock()
        registry.get.return_value = provider

        storage = MagicMock()
        storage.store_image = AsyncMock(return_value="/local/img.png")

        service = ImageGenService(
            storage=storage,
            registry=registry,
            default_provider="mock_images",
        )

        fake_job = MagicMock()
        fake_job.status = "pending"
        fake_job.provider = "openai_images"
        fake_job.prompt = "test"
        fake_job.negative_prompt = None
        fake_job.style = None
        fake_job.size = "1024x768"
        fake_job.quality = None
        fake_job.reference_image_urls = None
        fake_job.extra_params = None
        fake_job.usage_context = None
        fake_job.target_usage = None

        db = AsyncMock()
        db.commit = AsyncMock()

        with (
            patch(
                "src.services.imagegen_service.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=fake_job,
            ),
            patch(
                "src.services.imagegen_service.ImageJobManager.mark_processing",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.imagegen_service.ImageJobManager.mark_completed",
                new_callable=AsyncMock,
            ) as mock_completed,
        ):
            cost = await service.process_job(db, "j1", "t1")

        assert cost == 0.04
        mock_completed.assert_called_once()

    async def test_process_job_not_found_raises(self) -> None:
        service = ImageGenService(
            storage=MagicMock(),
            registry=MagicMock(),
            default_provider="mock_images",
        )

        db = AsyncMock()
        with patch(
            "src.services.imagegen_service.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(ValueError, match="not found"):
                await service.process_job(db, "j-missing", "t1")

    async def test_process_job_cancelled(self) -> None:
        service = ImageGenService(
            storage=MagicMock(),
            registry=MagicMock(),
            default_provider="mock_images",
        )

        fake_job = MagicMock()
        fake_job.status = "cancelled"

        db = AsyncMock()
        with patch(
            "src.services.imagegen_service.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=fake_job,
        ):
            cost = await service.process_job(db, "j-cancel", "t1")

        assert cost == 0.0

    async def test_process_job_unknown_provider(self) -> None:
        registry = MagicMock()
        registry.get.return_value = None

        service = ImageGenService(
            storage=MagicMock(),
            registry=registry,
            default_provider="mock_images",
        )

        fake_job = MagicMock()
        fake_job.status = "pending"
        fake_job.provider = "nonexistent"

        db = AsyncMock()
        db.commit = AsyncMock()

        with (
            patch(
                "src.services.imagegen_service.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=fake_job,
            ),
            patch(
                "src.services.imagegen_service.ImageJobManager.mark_failed",
                new_callable=AsyncMock,
            ) as mock_failed,
        ):
            cost = await service.process_job(db, "j-bad-provider", "t1")

        assert cost == 0.0
        mock_failed.assert_called_once()

    async def test_process_job_budget_exceeded(self) -> None:
        provider = MagicMock()
        provider.estimate_cost.return_value = 10.0

        registry = MagicMock()
        registry.get.return_value = provider

        service = ImageGenService(
            storage=MagicMock(),
            registry=registry,
            default_provider="mock_images",
        )

        fake_job = MagicMock()
        fake_job.status = "pending"
        fake_job.provider = "openai_images"
        fake_job.size = None
        fake_job.quality = None

        db = AsyncMock()
        db.commit = AsyncMock()

        with (
            patch(
                "src.services.imagegen_service.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=fake_job,
            ),
            patch(
                "src.services.imagegen_service.ImageJobManager.mark_failed",
                new_callable=AsyncMock,
            ) as mock_failed,
        ):
            cost = await service.process_job(db, "j-budget", "t1", budget_limit=5.0)

        assert cost == 0.0
        mock_failed.assert_called_once()
        assert "Budget exceeded" in mock_failed.call_args.args[2]

    async def test_process_job_generation_fails(self) -> None:
        provider = MagicMock()
        provider.estimate_cost.return_value = 0.04
        provider.generate = AsyncMock(return_value=ImageGenResult(success=False, error="GPU OOM"))

        registry = MagicMock()
        registry.get.return_value = provider

        service = ImageGenService(
            storage=MagicMock(),
            registry=registry,
            default_provider="mock_images",
        )

        fake_job = MagicMock()
        fake_job.status = "pending"
        fake_job.provider = "openai_images"
        fake_job.prompt = "test"
        fake_job.negative_prompt = None
        fake_job.style = None
        fake_job.size = None
        fake_job.quality = None
        fake_job.reference_image_urls = None
        fake_job.extra_params = None
        fake_job.usage_context = None
        fake_job.target_usage = None

        db = AsyncMock()
        db.commit = AsyncMock()

        with (
            patch(
                "src.services.imagegen_service.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=fake_job,
            ),
            patch(
                "src.services.imagegen_service.ImageJobManager.mark_processing",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.imagegen_service.ImageJobManager.mark_failed",
                new_callable=AsyncMock,
            ) as mock_failed,
        ):
            cost = await service.process_job(db, "j-fail", "t1")

        assert cost == 0.0
        mock_failed.assert_called_once()
        assert "GPU OOM" in mock_failed.call_args.args[2]

    async def test_process_job_minio_storage(self) -> None:
        """Test that minio_key is set when storage returns content_generation/ path."""
        provider = MagicMock()
        provider.estimate_cost.return_value = 0.01
        provider.generate = AsyncMock(
            return_value=ImageGenResult(
                success=True,
                image_data=b"png",
                mime_type="image/png",
                width=512,
                height=512,
            )
        )

        registry = MagicMock()
        registry.get.return_value = provider

        storage = MagicMock()
        storage.store_image = AsyncMock(return_value="content_generation/img.png")

        service = ImageGenService(
            storage=storage,
            registry=registry,
            default_provider="mock_images",
        )

        fake_job = MagicMock()
        fake_job.status = "pending"
        fake_job.provider = "mock_images"
        fake_job.prompt = "test"
        fake_job.negative_prompt = None
        fake_job.style = None
        fake_job.size = None
        fake_job.quality = None
        fake_job.reference_image_urls = None
        fake_job.extra_params = None
        fake_job.usage_context = None
        fake_job.target_usage = None

        db = AsyncMock()
        db.commit = AsyncMock()

        with (
            patch(
                "src.services.imagegen_service.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=fake_job,
            ),
            patch(
                "src.services.imagegen_service.ImageJobManager.mark_processing",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.imagegen_service.ImageJobManager.mark_completed",
                new_callable=AsyncMock,
            ) as mock_completed,
        ):
            cost = await service.process_job(db, "j-minio", "t1")

        assert cost == 0.01
        call_kwargs = mock_completed.call_args.kwargs
        assert call_kwargs["minio_key"] == "content_generation/img.png"
        assert call_kwargs["image_path"] is None
