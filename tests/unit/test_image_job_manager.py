"""Unit tests for ImageJobManager — mock-only paths.

Real-db state-transition coverage lives in ``test_job_managers_real_db.py``.
This file targets ``create_job``, ``get_job``, ``get_job_by_asset_id``,
``get_jobs_for_batch``, and ``_invalidate_cache`` — the non-state-changing
paths that don't depend on atomic UPDATE semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.image_job_manager import ImageJobManager


@dataclass
class _FakeImageJob:
    job_id: str = "img-j1"
    asset_id: str = "asset-1"
    batch_id: str | None = None
    tenant_id: str = "t1"
    provider: str = "openai_images"
    prompt: str = "sunset"
    negative_prompt: str | None = None
    style: str | None = None
    size: str | None = None
    quality: str | None = None
    reference_image_urls: list[str] | None = None
    usage_context: dict | None = None
    target_usage: str | None = None
    extra_params: dict | None = None
    callback_url: str | None = None
    status: str = "pending"
    provider_job_id: str | None = None
    image_path: str | None = None
    image_url: str | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    file_size_bytes: int | None = None
    cost: float | None = None
    minio_key: str | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class TestInvalidateCache:
    async def test_invalidate_cache_with_redis(self) -> None:
        mock_redis = AsyncMock()
        with patch(
            "src.services.image_job_manager.get_redis_client",
            return_value=mock_redis,
            create=True,
        ):
            with patch(
                "src.cache.redis_client.get_redis_client",
                return_value=mock_redis,
            ):
                await ImageJobManager._invalidate_cache("j1")
                mock_redis.delete.assert_called_once_with("content:image:j1:status")

    async def test_invalidate_cache_no_redis(self) -> None:
        with patch(
            "src.cache.redis_client.get_redis_client",
            return_value=None,
        ):
            await ImageJobManager._invalidate_cache("j1")

    async def test_invalidate_cache_exception_swallowed(self) -> None:
        with patch(
            "src.cache.redis_client.get_redis_client",
            side_effect=OSError("redis down"),
        ):
            await ImageJobManager._invalidate_cache("j1")


class TestCreateJob:
    async def test_create_job_adds_to_db(self) -> None:
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        job = await ImageJobManager.create_job(
            db,
            tenant_id="t1",
            provider="openai_images",
            prompt="a sunset over the ocean",
        )

        db.add.assert_called_once()
        db.flush.assert_called_once()
        assert job.tenant_id == "t1"
        assert job.provider == "openai_images"
        assert job.status == "pending"

    async def test_create_job_with_all_params(self) -> None:
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        job = await ImageJobManager.create_job(
            db,
            tenant_id="t2",
            provider="mock_images",
            prompt="a cat",
            batch_id="batch-1",
            negative_prompt="ugly",
            style="editorial",
            size="512x512",
            quality="hd",
            reference_image_urls=["https://example.com/img.png"],
            usage_context={"brand": "Acme"},
            target_usage="redesign_concept",
            extra_params={"background": "transparent"},
            callback_url="https://example.com/cb",
        )

        assert job.batch_id == "batch-1"
        assert job.negative_prompt == "ugly"
        assert job.callback_url == "https://example.com/cb"


class TestGetJob:
    async def test_get_job_found(self) -> None:
        fake_job = _FakeImageJob()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = fake_job

        db = AsyncMock()
        db.execute.return_value = result_mock

        job = await ImageJobManager.get_job(db, "img-j1")
        assert job is not None
        assert job.job_id == "img-j1"

    async def test_get_job_not_found(self) -> None:
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None

        db = AsyncMock()
        db.execute.return_value = result_mock

        job = await ImageJobManager.get_job(db, "missing")
        assert job is None


class TestGetJobByAssetId:
    async def test_get_job_by_asset_id(self) -> None:
        fake_job = _FakeImageJob(asset_id="asset-99")
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = fake_job

        db = AsyncMock()
        db.execute.return_value = result_mock

        job = await ImageJobManager.get_job_by_asset_id(db, "asset-99")
        assert job is not None
        assert job.asset_id == "asset-99"


class TestGetJobsForBatch:
    async def test_get_jobs_for_batch(self) -> None:
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [_FakeImageJob(), _FakeImageJob(job_id="j2")]
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock

        db = AsyncMock()
        db.execute.return_value = result_mock

        jobs = await ImageJobManager.get_jobs_for_batch(db, "batch-1")
        assert len(jobs) == 2
