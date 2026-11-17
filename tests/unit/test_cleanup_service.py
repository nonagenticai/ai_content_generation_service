"""Integration tests for cleanup service (Phase 5 + Phase D RLS).

Under Phase D RLS the cross-tenant expired-job scan runs through the
``expired_video_generation_jobs`` SECURITY DEFINER resolver, then deletes per-tenant
via the existing JobManager. These tests exercise the real resolver + delete path
against a live Postgres (the db_session_factory fixture's create_all schema includes
the resolver), seeding real rows — stronger than the previous JobManager mocks.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

from sqlalchemy import func, select

from src.database.models import VideoGenerationJob
from src.services.cleanup_service import CleanupService
from tests.conftest import TENANT_A


async def _seed(
    factory, *, job_id, status, days_ago, tenant_id=TENANT_A, video_path=None, minio_key=None
):
    now = datetime.now(UTC)
    async with factory() as db:
        db.add(
            VideoGenerationJob(
                job_id=job_id,
                tenant_id=tenant_id,
                provider="runway",
                prompt="p",
                status=status,
                video_path=video_path,
                minio_key=minio_key,
                created_at=now,
                updated_at=now,
                completed_at=now - timedelta(days=days_ago),
            )
        )
        await db.commit()


async def _count(factory) -> int:
    async with factory() as db:
        return (
            await db.execute(select(func.count()).select_from(VideoGenerationJob))
        ).scalar() or 0


async def test_delete_expired_completed(db_session_factory):
    await _seed(
        db_session_factory,
        job_id="j1",
        status="completed",
        days_ago=31,
        video_path="/tmp/j1/clip.mp4",
        minio_key="content_generation/j1/clip.mp4",
    )
    storage = AsyncMock()
    svc = CleanupService(
        completed_ttl_days=30,
        failed_ttl_days=7,
        storage=storage,
        session_factory=db_session_factory,
    )
    deleted = await svc.run_cleanup()
    assert deleted >= 1
    assert await _count(db_session_factory) == 0
    storage.delete_local.assert_called()
    storage.delete_from_minio.assert_called()


async def test_delete_expired_failed(db_session_factory):
    await _seed(db_session_factory, job_id="j2", status="failed", days_ago=8)
    storage = AsyncMock()
    svc = CleanupService(
        completed_ttl_days=30,
        failed_ttl_days=7,
        storage=storage,
        session_factory=db_session_factory,
    )
    deleted = await svc.run_cleanup()
    assert deleted >= 1
    assert await _count(db_session_factory) == 0


async def test_no_session_factory():
    svc = CleanupService(session_factory=None)
    deleted = await svc.run_cleanup()
    assert deleted == 0


async def test_handle_missing_files(db_session_factory):
    """Cleanup should not fail if storage deletes raise — the row is still removed."""
    await _seed(
        db_session_factory,
        job_id="j3",
        status="completed",
        days_ago=31,
        video_path="/tmp/j3/clip.mp4",
        minio_key="k",
    )
    storage = AsyncMock()
    storage.delete_local = AsyncMock(side_effect=OSError("file not found"))
    storage.delete_from_minio = AsyncMock(side_effect=OSError("not found"))
    svc = CleanupService(
        completed_ttl_days=30,
        failed_ttl_days=7,
        storage=storage,
        session_factory=db_session_factory,
    )
    deleted = await svc.run_cleanup()
    assert deleted >= 1
    assert await _count(db_session_factory) == 0


async def test_start_stop():
    svc = CleanupService(interval_seconds=3600, session_factory=None)
    await svc.start()
    assert svc._task is not None
    await svc.stop()
    assert svc._task is None


async def test_batch_limit(db_session_factory):
    """Cleanup runs both statuses: one completed + one failed expired job are removed."""
    await _seed(db_session_factory, job_id="c1", status="completed", days_ago=31)
    await _seed(db_session_factory, job_id="f1", status="failed", days_ago=8)
    # A recent completed job must survive.
    await _seed(db_session_factory, job_id="recent", status="completed", days_ago=0)
    storage = AsyncMock()
    svc = CleanupService(
        completed_ttl_days=30,
        failed_ttl_days=7,
        storage=storage,
        session_factory=db_session_factory,
    )
    deleted = await svc.run_cleanup()
    assert deleted == 2
    assert await _count(db_session_factory) == 1
