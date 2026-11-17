"""Real-database tests for JobManager, ImageJobManager, and ImageBatchManager.

Uses postgres+psycopg via the ``db_session`` fixture from conftest.
Redis cache invalidation is monkeypatched to a no-op so tests run without Redis.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.services.image_batch_manager import ImageBatchManager
from src.services.image_job_manager import ImageJobManager
from src.services.job_manager import JobManager
from tests.conftest import TENANT_A, TENANT_B

# ---------------------------------------------------------------------------
# Disable Redis cache invalidation for all tests in this module
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_cache_invalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace _invalidate_cache with a no-op on both managers."""

    async def _noop(_job_id: str) -> None:
        pass

    monkeypatch.setattr(JobManager, "_invalidate_cache", staticmethod(_noop))
    monkeypatch.setattr(ImageJobManager, "_invalidate_cache", staticmethod(_noop))


# ============================================================================
# TestJobManager
# ============================================================================


class TestJobManager:
    """CRUD and lifecycle tests for VideoGenerationJob via JobManager."""

    async def test_create_and_get(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="runway",
            prompt="A sunset over the ocean",
        )
        assert job.job_id
        assert str(job.tenant_id) == TENANT_A
        assert job.provider == "runway"
        assert job.prompt == "A sunset over the ocean"
        assert job.status == "pending"

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id

    async def test_get_nonexistent_returns_none(self, db_session) -> None:
        result = await JobManager.get_job(db_session, "does-not-exist")
        assert result is None

    async def test_create_with_all_optional_fields(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session,
            tenant_id=TENANT_B,
            provider="kling",
            prompt="Dancing robots",
            negative_prompt="blurry",
            image_url="https://example.com/img.png",
            duration_s=10.0,
            width=1920,
            height=1080,
            style="cinematic",
            seed=42,
            extra_params={"fps": 24, "model": "v2"},
            fallback_providers=["veo", "sora"],
            callback_url="https://hook.example.com/done",
        )
        assert job.negative_prompt == "blurry"
        assert job.image_url == "https://example.com/img.png"
        assert job.duration_s == 10.0
        assert job.width == 1920
        assert job.height == 1080
        assert job.style == "cinematic"
        assert job.seed == 42
        assert job.callback_url == "https://hook.example.com/done"

    async def test_extra_params_jsonb_roundtrip(self, db_session) -> None:
        params = {"fps": 24, "nested": {"key": [1, 2, 3]}}
        job = await JobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="runway",
            prompt="test",
            extra_params=params,
        )
        await db_session.flush()

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.extra_params == params

    async def test_fallback_providers_jsonb_roundtrip(self, db_session) -> None:
        providers = ["veo", "sora", "seedance"]
        job = await JobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="kling",
            prompt="test",
            fallback_providers=providers,
        )
        await db_session.flush()

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.fallback_providers == providers

    async def test_mark_processing(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test"
        )
        await JobManager.mark_processing(db_session, job.job_id)
        await JobManager.set_provider_job_id(db_session, job.job_id, "ext-123")

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "processing"
        assert fetched.provider_job_id == "ext-123"

    async def test_mark_completed_persists_a_degradation(self, db_session) -> None:
        """A completed-but-degraded clip must be readable back off the ROW.

        Asserting only that the caller passed ``degradation=`` proves nothing:
        ``mark_completed`` builds its UPDATE from a dict, and a marker that never
        reaches ``values`` is dropped silently -- the job still commits as a clean
        completion, which is the exact defect this column exists to close.
        """
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test"
        )
        await JobManager.mark_processing(db_session, job.job_id)
        await JobManager.mark_completed(
            db_session,
            job.job_id,
            video_path="/videos/out.mp4",
            degradation="video_not_normalized: ffmpeg_missing",
        )

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.degradation == "video_not_normalized: ffmpeg_missing"

    async def test_mark_completed_without_degradation_is_clean(self, db_session) -> None:
        """Control: the default must stay NULL, or every completion reads degraded."""
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test"
        )
        await JobManager.mark_processing(db_session, job.job_id)
        await JobManager.mark_completed(db_session, job.job_id, video_path="/videos/out.mp4")

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.degradation is None

    async def test_mark_completed(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test"
        )
        await JobManager.mark_processing(db_session, job.job_id)
        await JobManager.mark_completed(
            db_session,
            job.job_id,
            video_path="/videos/out.mp4",
            video_url="https://cdn.example.com/out.mp4",
            file_size_bytes=5_000_000,
            actual_duration_s=4.5,
            generation_time_seconds=30.0,
            cost=0.15,
            minio_key="bucket/key.mp4",
        )

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.video_path == "/videos/out.mp4"
        assert fetched.video_url == "https://cdn.example.com/out.mp4"
        assert fetched.file_size_bytes == 5_000_000
        assert fetched.actual_duration_s == 4.5
        assert fetched.generation_time_seconds == 30.0
        assert fetched.cost == 0.15
        assert fetched.minio_key == "bucket/key.mp4"
        assert fetched.completed_at is not None

    async def test_mark_failed(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test"
        )
        await JobManager.mark_failed(db_session, job.job_id, "Provider timeout")

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "failed"
        assert fetched.error_message == "Provider timeout"
        assert fetched.completed_at is not None

    async def test_mark_failed_truncates_error(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test"
        )
        long_error = "x" * 5000
        await JobManager.mark_failed(db_session, job.job_id, long_error)

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.error_message is not None
        assert len(fetched.error_message) == 4000

    async def test_mark_cancelled(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test"
        )
        await JobManager.mark_cancelled(db_session, job.job_id)

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "cancelled"
        assert fetched.completed_at is not None

    async def test_mark_retrying(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test"
        )
        # mark_retrying now requires the job to be in "processing" state.
        await JobManager.mark_processing(db_session, job.job_id)
        await JobManager.mark_retrying(db_session, job.job_id, retry_count=2, error="Transient")

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "pending"
        assert fetched.retry_count == 2
        assert fetched.error_message == "Transient"

    async def test_list_jobs_basic(self, db_session) -> None:
        for i in range(5):
            await JobManager.create_job(
                db_session, tenant_id=TENANT_A, provider="runway", prompt=f"prompt {i}"
            )
        await db_session.flush()

        jobs, total = await JobManager.list_jobs(db_session, TENANT_A)
        assert total == 5
        assert len(jobs) == 5

    async def test_list_jobs_tenant_isolation(self, db_session) -> None:
        await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="tenant 1"
        )
        await JobManager.create_job(
            db_session, tenant_id=TENANT_B, provider="runway", prompt="tenant 2"
        )
        await db_session.flush()

        jobs_t1, total_t1 = await JobManager.list_jobs(db_session, TENANT_A)
        jobs_t2, total_t2 = await JobManager.list_jobs(db_session, TENANT_B)
        assert total_t1 == 1
        assert total_t2 == 1
        assert str(jobs_t1[0].tenant_id) == TENANT_A
        assert str(jobs_t2[0].tenant_id) == TENANT_B

    async def test_list_jobs_filter_by_status(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test"
        )
        await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test2"
        )
        await JobManager.mark_processing(db_session, job.job_id)
        await JobManager.mark_completed(db_session, job.job_id)

        completed, total = await JobManager.list_jobs(db_session, TENANT_A, status="completed")
        assert total == 1
        assert completed[0].status == "completed"

    async def test_list_jobs_filter_by_provider(self, db_session) -> None:
        await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test"
        )
        await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="kling", prompt="test2"
        )
        await db_session.flush()

        jobs, total = await JobManager.list_jobs(db_session, TENANT_A, provider="kling")
        assert total == 1
        assert jobs[0].provider == "kling"

    async def test_list_jobs_pagination(self, db_session) -> None:
        for i in range(7):
            await JobManager.create_job(
                db_session, tenant_id=TENANT_A, provider="runway", prompt=f"p{i}"
            )
        await db_session.flush()

        page1, total = await JobManager.list_jobs(db_session, TENANT_A, page=1, page_size=3)
        assert total == 7
        assert len(page1) == 3

        page2, _ = await JobManager.list_jobs(db_session, TENANT_A, page=2, page_size=3)
        assert len(page2) == 3

        page3, _ = await JobManager.list_jobs(db_session, TENANT_A, page=3, page_size=3)
        assert len(page3) == 1

    async def test_find_expired_jobs(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test"
        )
        await JobManager.mark_processing(db_session, job.job_id)
        await JobManager.mark_completed(db_session, job.job_id)

        # The completed_at was set to now; search for jobs completed before 1 hour from now
        cutoff = datetime.now(UTC) + timedelta(hours=1)
        expired = await JobManager.find_expired_jobs(db_session, "completed", cutoff)
        assert len(expired) == 1
        assert expired[0].job_id == job.job_id

    async def test_find_expired_jobs_excludes_recent(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="test"
        )
        await JobManager.mark_processing(db_session, job.job_id)
        await JobManager.mark_completed(db_session, job.job_id)

        # Cutoff in the past should exclude jobs completed just now
        cutoff = datetime.now(UTC) - timedelta(hours=1)
        expired = await JobManager.find_expired_jobs(db_session, "completed", cutoff)
        assert len(expired) == 0

    async def test_delete_jobs(self, db_session) -> None:
        j1 = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="delete me"
        )
        j2 = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="keep me"
        )
        await db_session.flush()

        deleted = await JobManager.delete_jobs(db_session, [j1.job_id])
        assert deleted == 1

        assert await JobManager.get_job(db_session, j1.job_id) is None
        assert await JobManager.get_job(db_session, j2.job_id) is not None

    async def test_delete_jobs_empty_list(self, db_session) -> None:
        deleted = await JobManager.delete_jobs(db_session, [])
        assert deleted == 0

    async def test_status_transition_pending_to_processing_to_completed(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="lifecycle"
        )
        assert job.status == "pending"

        await JobManager.mark_processing(db_session, job.job_id)
        await JobManager.set_provider_job_id(db_session, job.job_id, "ext-1")
        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "processing"
        assert fetched.provider_job_id == "ext-1"

        await JobManager.mark_completed(db_session, job.job_id, cost=0.50)
        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.cost == 0.50

    async def test_status_transition_pending_to_failed(self, db_session) -> None:
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="runway", prompt="fail"
        )
        await JobManager.mark_failed(db_session, job.job_id, "out of quota")
        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "failed"


# ============================================================================
# TestImageJobManager
# ============================================================================


class TestImageJobManager:
    """CRUD and lifecycle tests for ImageGenJob via ImageJobManager."""

    async def _create_batch(self, db_session) -> str:
        """Helper: create a batch and return its batch_id."""
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=2
        )
        await db_session.flush()
        return batch.batch_id

    async def test_create_and_get(self, db_session) -> None:
        job = await ImageJobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="dall-e-3",
            prompt="A watercolor cat",
        )
        assert job.job_id
        assert job.asset_id
        assert job.status == "pending"
        assert str(job.tenant_id) == TENANT_A

        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id

    async def test_get_nonexistent_returns_none(self, db_session) -> None:
        assert await ImageJobManager.get_job(db_session, "no-such-id") is None

    async def test_get_by_asset_id(self, db_session) -> None:
        job = await ImageJobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="dall-e-3",
            prompt="A pastel sunset",
        )
        await db_session.flush()

        fetched = await ImageJobManager.get_job_by_asset_id(db_session, job.asset_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id

    async def test_get_by_asset_id_nonexistent(self, db_session) -> None:
        assert await ImageJobManager.get_job_by_asset_id(db_session, "nope") is None

    async def test_create_with_batch_fk(self, db_session) -> None:
        batch_id = await self._create_batch(db_session)
        job = await ImageJobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="dall-e-3",
            prompt="batch image",
            batch_id=batch_id,
        )
        await db_session.flush()

        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.batch_id == batch_id

    async def test_create_with_all_optional_fields(self, db_session) -> None:
        batch_id = await self._create_batch(db_session)
        job = await ImageJobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="dall-e-3",
            prompt="detailed image",
            batch_id=batch_id,
            negative_prompt="blurry, low quality",
            style="photographic",
            size="1024x1024",
            quality="hd",
            reference_image_urls=["https://ex.com/a.png", "https://ex.com/b.png"],
            usage_context={"scene": "intro", "slide_index": 0},
            target_usage="thumbnail",
            extra_params={"guidance_scale": 7.5},
            callback_url="https://hook.example.com/img",
        )
        assert job.negative_prompt == "blurry, low quality"
        assert job.style == "photographic"
        assert job.size == "1024x1024"
        assert job.quality == "hd"
        assert job.target_usage == "thumbnail"
        assert job.callback_url == "https://hook.example.com/img"

    async def test_reference_image_urls_jsonb_roundtrip(self, db_session) -> None:
        urls = ["https://ex.com/1.png", "https://ex.com/2.png", "https://ex.com/3.png"]
        job = await ImageJobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="dall-e-3",
            prompt="ref images",
            reference_image_urls=urls,
        )
        await db_session.flush()

        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.reference_image_urls == urls

    async def test_usage_context_jsonb_roundtrip(self, db_session) -> None:
        ctx = {"scene": "hero", "position": {"x": 100, "y": 200}, "tags": ["important"]}
        job = await ImageJobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="dall-e-3",
            prompt="ctx test",
            usage_context=ctx,
        )
        await db_session.flush()

        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.usage_context == ctx

    async def test_extra_params_jsonb_roundtrip(self, db_session) -> None:
        params = {"guidance_scale": 7.5, "steps": 50, "scheduler": "euler"}
        job = await ImageJobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="dall-e-3",
            prompt="params test",
            extra_params=params,
        )
        await db_session.flush()

        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.extra_params == params

    async def test_mark_processing(self, db_session) -> None:
        job = await ImageJobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="dall-e-3", prompt="test"
        )
        await ImageJobManager.mark_processing(db_session, job.job_id, provider_job_id="prov-99")
        await db_session.flush()

        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "processing"
        assert fetched.provider_job_id == "prov-99"

    async def test_mark_completed(self, db_session) -> None:
        job = await ImageJobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="dall-e-3", prompt="test"
        )
        await ImageJobManager.mark_processing(db_session, job.job_id)
        await ImageJobManager.mark_completed(
            db_session,
            job.job_id,
            image_path="/images/out.png",
            image_url="https://cdn.example.com/out.png",
            mime_type="image/png",
            width=1024,
            height=1024,
            file_size_bytes=2_000_000,
            cost=0.04,
            minio_key="bucket/img.png",
        )

        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.image_path == "/images/out.png"
        assert fetched.image_url == "https://cdn.example.com/out.png"
        assert fetched.mime_type == "image/png"
        assert fetched.width == 1024
        assert fetched.height == 1024
        assert fetched.file_size_bytes == 2_000_000
        assert fetched.cost == 0.04
        assert fetched.minio_key == "bucket/img.png"
        assert fetched.completed_at is not None

    async def test_mark_failed(self, db_session) -> None:
        job = await ImageJobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="dall-e-3", prompt="test"
        )
        await ImageJobManager.mark_failed(db_session, job.job_id, "Content policy violation")
        await db_session.flush()

        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "failed"
        assert fetched.error_message == "Content policy violation"
        assert fetched.completed_at is not None

    async def test_mark_failed_truncates_error(self, db_session) -> None:
        job = await ImageJobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="dall-e-3", prompt="test"
        )
        await ImageJobManager.mark_failed(db_session, job.job_id, "e" * 5000)
        await db_session.flush()

        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.error_message is not None
        assert len(fetched.error_message) == 4000

    async def test_mark_cancelled(self, db_session) -> None:
        job = await ImageJobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="dall-e-3", prompt="test"
        )
        await ImageJobManager.mark_cancelled(db_session, job.job_id)
        await db_session.flush()

        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "cancelled"
        assert fetched.completed_at is not None

    async def test_get_jobs_for_batch(self, db_session) -> None:
        batch_id = await self._create_batch(db_session)
        j1 = await ImageJobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="dall-e-3",
            prompt="batch img 1",
            batch_id=batch_id,
        )
        j2 = await ImageJobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="dall-e-3",
            prompt="batch img 2",
            batch_id=batch_id,
        )
        await db_session.flush()

        jobs = await ImageJobManager.get_jobs_for_batch(db_session, batch_id)
        assert len(jobs) == 2
        job_ids = {j.job_id for j in jobs}
        assert j1.job_id in job_ids
        assert j2.job_id in job_ids

    async def test_get_jobs_for_batch_empty(self, db_session) -> None:
        batch_id = await self._create_batch(db_session)
        jobs = await ImageJobManager.get_jobs_for_batch(db_session, batch_id)
        assert jobs == []

    async def test_status_lifecycle(self, db_session) -> None:
        job = await ImageJobManager.create_job(
            db_session, tenant_id=TENANT_A, provider="dall-e-3", prompt="lifecycle"
        )
        assert job.status == "pending"

        await ImageJobManager.mark_processing(db_session, job.job_id)
        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "processing"

        await ImageJobManager.mark_completed(db_session, job.job_id, cost=0.02)
        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "completed"


# ============================================================================
# TestImageBatchManager
# ============================================================================


class TestImageBatchManager:
    """CRUD and lifecycle tests for ImageGenBatch via ImageBatchManager."""

    async def test_create_and_get(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=25.0, job_count=5
        )
        assert batch.batch_id
        assert batch.tenant_id == TENANT_A
        assert batch.budget_limit == 25.0
        assert batch.job_count == 5
        assert batch.status == "pending"
        assert batch.completed_jobs == 0
        assert batch.failed_jobs == 0
        assert batch.total_cost == 0.0

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.batch_id == batch.batch_id

    async def test_get_nonexistent_returns_none(self, db_session) -> None:
        assert await ImageBatchManager.get_batch(db_session, "no-batch") is None

    async def test_mark_processing(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=3
        )
        await ImageBatchManager.mark_processing(db_session, batch.batch_id)
        await db_session.flush()

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.status == "processing"

    async def test_mark_processing_nonexistent_is_noop(self, db_session) -> None:
        # Should not raise
        await ImageBatchManager.mark_processing(db_session, "ghost-batch")

    async def test_record_job_completed_accumulates_cost(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=3
        )
        await ImageBatchManager.record_job_completed(db_session, batch.batch_id, cost=0.05)
        await ImageBatchManager.record_job_completed(db_session, batch.batch_id, cost=0.10)
        await db_session.flush()

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.completed_jobs == 2
        assert fetched.total_cost == pytest.approx(0.15)

    async def test_record_job_failed_increments(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=2
        )
        await ImageBatchManager.record_job_failed(db_session, batch.batch_id)
        await db_session.flush()

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.failed_jobs == 1

    async def test_batch_status_completed_when_all_succeed(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=2
        )
        await ImageBatchManager.record_job_completed(db_session, batch.batch_id, cost=0.05)
        await ImageBatchManager.record_job_completed(db_session, batch.batch_id, cost=0.05)
        await db_session.flush()

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.completed_at is not None

    async def test_batch_status_failed_when_all_fail(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=2
        )
        await ImageBatchManager.record_job_failed(db_session, batch.batch_id)
        await ImageBatchManager.record_job_failed(db_session, batch.batch_id)
        await db_session.flush()

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.status == "failed"
        assert fetched.completed_at is not None

    async def test_batch_status_partial_when_mixed(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=2
        )
        await ImageBatchManager.record_job_completed(db_session, batch.batch_id, cost=0.05)
        await ImageBatchManager.record_job_failed(db_session, batch.batch_id)
        await db_session.flush()

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.status == "partial"
        assert fetched.completed_at is not None

    async def test_batch_status_processing_when_incomplete(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=3
        )
        await ImageBatchManager.record_job_completed(db_session, batch.batch_id, cost=0.05)
        await db_session.flush()

        fetched = await ImageBatchManager.get_batch(db_session, batch.batch_id)
        assert fetched is not None
        assert fetched.status == "processing"
        # Not yet finished, so completed_at should be None
        assert fetched.completed_at is None

    async def test_get_batch_jobs(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=2
        )
        await db_session.flush()

        j1 = await ImageJobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="dall-e-3",
            prompt="batch 1",
            batch_id=batch.batch_id,
        )
        j2 = await ImageJobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="dall-e-3",
            prompt="batch 2",
            batch_id=batch.batch_id,
        )
        await db_session.flush()

        jobs = await ImageBatchManager.get_batch_jobs(db_session, batch.batch_id)
        assert len(jobs) == 2
        job_ids = {j.job_id for j in jobs}
        assert j1.job_id in job_ids
        assert j2.job_id in job_ids

    async def test_get_batch_jobs_empty(self, db_session) -> None:
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=10.0, job_count=0
        )
        await db_session.flush()

        jobs = await ImageBatchManager.get_batch_jobs(db_session, batch.batch_id)
        assert jobs == []

    async def test_record_completed_on_nonexistent_batch_is_noop(self, db_session) -> None:
        # Should not raise
        await ImageBatchManager.record_job_completed(db_session, "ghost-batch", cost=1.0)

    async def test_record_failed_on_nonexistent_batch_is_noop(self, db_session) -> None:
        # Should not raise
        await ImageBatchManager.record_job_failed(db_session, "ghost-batch")

    async def test_batch_fk_relationship_with_image_jobs(self, db_session) -> None:
        """Verify that image jobs correctly reference their parent batch via FK."""
        batch = await ImageBatchManager.create_batch(
            db_session, tenant_id=TENANT_A, budget_limit=5.0, job_count=1
        )
        await db_session.flush()

        job = await ImageJobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="dall-e-3",
            prompt="fk test",
            batch_id=batch.batch_id,
        )
        await db_session.flush()

        # Fetch via batch manager
        batch_jobs = await ImageBatchManager.get_batch_jobs(db_session, batch.batch_id)
        assert len(batch_jobs) == 1
        assert batch_jobs[0].job_id == job.job_id
        assert batch_jobs[0].batch_id == batch.batch_id

        # Fetch via image job manager
        fetched = await ImageJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.batch_id == batch.batch_id
