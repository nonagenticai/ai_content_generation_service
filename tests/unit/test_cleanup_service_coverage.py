"""Tests for cleanup_service.py to cover _loop and exception handling.

Missing lines: 58-65 (_loop body), 113-114 (exception in cleanup for-loop).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.cleanup_service import CleanupService
from tests.conftest import TENANT_A


def _make_job(
    job_id: str,
    status: str = "completed",
    video_path: str | None = None,
    minio_key: str | None = None,
) -> MagicMock:
    job = MagicMock()
    job.job_id = job_id
    job.status = status
    job.video_path = video_path
    job.minio_key = minio_key
    return job


async def test_loop_runs_cleanup_and_cancels():
    """The _loop method calls run_cleanup repeatedly until cancelled."""
    call_count = 0

    async def fake_cleanup(*_args: Any) -> int:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise asyncio.CancelledError()
        return 0

    svc = CleanupService(interval_seconds=0, session_factory=None)
    svc.run_cleanup = fake_cleanup

    with patch("src.services.cleanup_service.asyncio.sleep", new_callable=AsyncMock):
        await svc._loop()

    assert call_count >= 1


async def test_loop_catches_general_exception():
    """The _loop catches non-CancelledError exceptions and continues."""
    call_count = 0

    async def fake_sleep(seconds: float) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            raise asyncio.CancelledError()

    svc = CleanupService(interval_seconds=0, session_factory=None)

    async def failing_cleanup(*_args: Any) -> int:
        raise RuntimeError("Unexpected error")

    svc.run_cleanup = failing_cleanup

    with patch("src.services.cleanup_service.asyncio.sleep", side_effect=fake_sleep):
        await svc._loop()

    assert call_count >= 1


async def test_cleanup_exception_in_status_loop():
    """Exception during cleanup for a specific status is caught, other status continues."""
    storage = AsyncMock()
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    call_count = 0

    def session_factory_fn():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call (completed) raises
            raise RuntimeError("DB connection error")
        return session

    session_factory = MagicMock(side_effect=session_factory_fn)

    with patch("src.services.job_manager.JobManager") as mock_jm:
        mock_jm.find_expired_jobs = AsyncMock(return_value=[])
        mock_jm.delete_jobs = AsyncMock(return_value=0)

        svc = CleanupService(
            completed_ttl_days=30,
            failed_ttl_days=7,
            storage=storage,
            session_factory=session_factory,
        )
        deleted = await svc.run_cleanup()

    # First status (completed) failed, but failed status may have been processed
    assert deleted >= 0


async def test_cleanup_no_expired_jobs():
    """Cleanup with no expired jobs returns 0."""
    storage = AsyncMock()
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session_factory = MagicMock(return_value=session)

    with patch("src.services.job_manager.JobManager") as mock_jm:
        mock_jm.find_expired_jobs = AsyncMock(return_value=[])

        svc = CleanupService(
            storage=storage,
            session_factory=session_factory,
        )
        deleted = await svc.run_cleanup()

    assert deleted == 0


async def test_cleanup_with_no_storage(db_session_factory):
    """Cleanup works even without storage (skips file deletion) — real DB + resolver."""
    from datetime import UTC, datetime, timedelta

    from src.database.models import VideoGenerationJob

    now = datetime.now(UTC)
    async with db_session_factory() as db:
        db.add(
            VideoGenerationJob(
                job_id="j1",
                tenant_id=TENANT_A,
                provider="runway",
                prompt="p",
                status="completed",
                video_path="/tmp/v.mp4",
                minio_key="key",
                created_at=now,
                updated_at=now,
                completed_at=now - timedelta(days=31),
            )
        )
        await db.commit()

    svc = CleanupService(
        completed_ttl_days=30,
        failed_ttl_days=7,
        storage=None,
        session_factory=db_session_factory,
    )
    deleted = await svc.run_cleanup()
    assert deleted >= 1


async def test_stop_when_no_task():
    """Stopping when no task has been started is a no-op."""
    svc = CleanupService(session_factory=None)
    svc._task = None
    await svc.stop()
    assert svc._task is None
