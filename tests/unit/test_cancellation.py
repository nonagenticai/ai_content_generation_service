"""Unit tests for job cancellation (Phase 2)."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.video_generation_service import VideoGenerationService

TENANT_A = "11111111-1111-7111-8111-111111111111"
TENANT_B = "22222222-2222-7222-8222-222222222222"


def _make_service():
    return VideoGenerationService(
        storage=MagicMock(),
        registry=MagicMock(),
        default_provider="fake",
    )


async def test_cancel_pending_job():
    """Cancel a pending job (no active task)."""
    svc = _make_service()
    db = AsyncMock()

    mock_job = MagicMock()
    mock_job.status = "pending"

    with patch("src.services.video_generation_service.JobManager") as mock_jm:
        mock_jm.get_job = AsyncMock(return_value=mock_job)
        mock_jm.mark_cancelled = AsyncMock()

        result = await svc.cancel_job(db, "job-1", TENANT_A)

    assert result is True
    mock_jm.mark_cancelled.assert_called_once_with(db, "job-1")
    mock_jm.get_job.assert_awaited_once_with(db, "job-1", tenant_id=TENANT_A)


async def test_cancel_completed_job_returns_false():
    """Completed jobs cannot be cancelled."""
    svc = _make_service()
    db = AsyncMock()

    mock_job = MagicMock()
    mock_job.status = "completed"

    with patch("src.services.video_generation_service.JobManager") as mock_jm:
        mock_jm.get_job = AsyncMock(return_value=mock_job)

        result = await svc.cancel_job(db, "job-1", TENANT_A)

    assert result is False


async def test_cancel_unknown_job_returns_false():
    """Unknown jobs return False."""
    svc = _make_service()
    db = AsyncMock()

    with patch("src.services.video_generation_service.JobManager") as mock_jm:
        mock_jm.get_job = AsyncMock(return_value=None)

        result = await svc.cancel_job(db, "nonexistent", TENANT_A)

    assert result is False


async def test_cancel_active_task():
    """Cancel a job with an active asyncio.Task."""
    svc = _make_service()
    db = AsyncMock()

    async def long_running():
        await asyncio.sleep(100)

    task = asyncio.create_task(long_running())
    svc.register_task("job-1", task)

    with patch("src.services.video_generation_service.JobManager") as mock_jm:
        mock_jm.get_job = AsyncMock(return_value=MagicMock(status="processing"))
        mock_jm.mark_cancelled = AsyncMock()

        result = await svc.cancel_job(db, "job-1", TENANT_A)

    assert result is True
    assert task.cancelled() or task.done()
    assert "job-1" not in svc._active_tasks


async def test_register_unregister_task():
    svc = _make_service()
    task = MagicMock()

    svc.register_task("job-1", task)
    assert "job-1" in svc._active_tasks

    svc.unregister_task("job-1")
    assert "job-1" not in svc._active_tasks

    # Unregistering non-existent is safe
    svc.unregister_task("nonexistent")


# ============================================================================
# cancel_job registry hygiene — uses the REAL in-memory registry, not a mock,
# to prove unregister_task fires on every exit path of cancel_job.
# ============================================================================


async def test_cancel_job_only_removes_target_from_registry():
    """Register 3 tasks, cancel 1; the real registry has exactly 2 entries left."""
    svc = _make_service()
    db = AsyncMock()

    async def long_running():
        await asyncio.sleep(100)

    tasks = {}
    for jid in ("job-a", "job-b", "job-c"):
        t = asyncio.create_task(long_running())
        svc.register_task(jid, t)
        tasks[jid] = t

    assert set(svc._active_tasks) == {"job-a", "job-b", "job-c"}

    with patch("src.services.video_generation_service.JobManager") as mock_jm:
        mock_jm.get_job = AsyncMock(return_value=MagicMock(status="processing"))
        mock_jm.mark_cancelled = AsyncMock()
        result = await svc.cancel_job(db, "job-b", TENANT_A)

    assert result is True
    assert set(svc._active_tasks) == {"job-a", "job-c"}
    assert len(svc._active_tasks) == 2

    # Cleanup the surviving tasks so pytest doesn't warn about pending tasks
    for jid in ("job-a", "job-c"):
        tasks[jid].cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tasks[jid]


async def test_cancel_job_unregisters_even_when_mark_cancelled_raises():
    """If mark_cancelled raises an unexpected DB error, unregister_task STILL fires.

    Before the fix the broad `except` masked this and the registry leaked
    the entry, so a later scheduled callback could fire on a job we
    believed cancelled.
    """
    svc = _make_service()
    db = AsyncMock()

    async def long_running():
        await asyncio.sleep(100)

    for jid in ("job-a", "job-b"):
        svc.register_task(jid, asyncio.create_task(long_running()))

    assert len(svc._active_tasks) == 2

    with patch("src.services.video_generation_service.JobManager") as mock_jm:
        mock_jm.get_job = AsyncMock(return_value=MagicMock(status="processing"))
        mock_jm.mark_cancelled = AsyncMock(side_effect=OSError("postgres down"))

        # Unexpected exceptions propagate (no longer silently swallowed),
        # but the finally block still unregisters the task.
        with contextlib.suppress(OSError):
            await svc.cancel_job(db, "job-a", TENANT_A)

    # Registry dropped from 2 -> 1 even though mark_cancelled raised.
    assert "job-a" not in svc._active_tasks
    assert len(svc._active_tasks) == 1

    # Cleanup
    remaining = next(iter(svc._active_tasks.values()))
    remaining.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await remaining


async def test_cancel_job_unregisters_when_state_conflict():
    """JobStateConflictError still triggers unregister via finally."""
    from src.services.job_manager import JobStateConflictError

    svc = _make_service()
    db = AsyncMock()

    async def long_running():
        await asyncio.sleep(100)

    svc.register_task("job-x", asyncio.create_task(long_running()))
    assert "job-x" in svc._active_tasks

    with patch("src.services.video_generation_service.JobManager") as mock_jm:
        mock_jm.get_job = AsyncMock(return_value=MagicMock(status="processing"))
        mock_jm.mark_cancelled = AsyncMock(side_effect=JobStateConflictError("already terminal"))
        result = await svc.cancel_job(db, "job-x", TENANT_A)

    assert result is False
    assert "job-x" not in svc._active_tasks
    assert len(svc._active_tasks) == 0


# ============================================================================
# Cross-tenant cancel — the ORDERING is the fix.
#
# The Redis cancel flag and the in-process ``task.cancel()`` are NOT database
# operations, so RLS cannot backstop them: whichever runs before the ownership
# read is reachable by any caller holding a job id. These tests therefore assert
# the SIDE EFFECTS, not the return value — a version that fired the flag and
# then returned False would still pass a status-only assertion.
# ============================================================================


def _tenant_scoped_get_job(owner_tenant: str, job):
    """Stand-in for JobManager.get_job's tenant predicate: foreign reads as absent."""

    async def _get_job(_db, _job_id, tenant_id=None):
        if tenant_id is not None and tenant_id != owner_tenant:
            return None
        return job

    return _get_job


async def test_foreign_tenant_cancel_leaves_flag_unset_and_task_running():
    """Tenant B cancelling tenant A's job: no Redis flag, task still running."""
    from w2v_common.cache.redis_client import RedisClient
    from w2v_common.testing import inject_fakeredis

    from src.services.cancel_registry import CancelRegistry

    redis = RedisClient(redis_url="redis://localhost:6379/0")
    inject_fakeredis(redis)
    cancel_registry = CancelRegistry(redis)

    svc = VideoGenerationService(
        storage=MagicMock(),
        registry=MagicMock(),
        default_provider="fake",
        cancel_registry=cancel_registry,
    )

    started = asyncio.Event()

    async def long_running():
        started.set()
        await asyncio.sleep(100)

    task = asyncio.create_task(long_running())
    svc.register_task("job-a", task)
    await started.wait()

    owned_job = MagicMock(status="processing")
    db = AsyncMock()

    with patch("src.services.video_generation_service.JobManager") as mock_jm:
        mock_jm.get_job = AsyncMock(side_effect=_tenant_scoped_get_job(TENANT_A, owned_job))
        mock_jm.mark_cancelled = AsyncMock()

        result = await svc.cancel_job(db, "job-a", TENANT_B)

    assert result is False
    # The Redis flag was never set — a task on ANY other replica keeps running.
    assert await cancel_registry.is_cancelled("job-a") is False
    # The local task was never cancelled and is still registered.
    assert not task.cancelled()
    assert not task.done()
    assert svc._active_tasks.get("job-a") is task
    # And no DB write was attempted either.
    mock_jm.mark_cancelled.assert_not_awaited()

    # The owner CAN still cancel it — the gate scopes, it does not disable.
    with patch("src.services.video_generation_service.JobManager") as mock_jm:
        mock_jm.get_job = AsyncMock(side_effect=_tenant_scoped_get_job(TENANT_A, owned_job))
        mock_jm.mark_cancelled = AsyncMock()

        assert await svc.cancel_job(AsyncMock(), "job-a", TENANT_A) is True

    assert await cancel_registry.is_cancelled("job-a") is True
    assert task.cancelled() or task.done()
    assert "job-a" not in svc._active_tasks


async def test_imagegen_foreign_tenant_cancel_leaves_flag_unset_and_task_running():
    """Same ordering guarantee on the image path."""
    from w2v_common.cache.redis_client import RedisClient
    from w2v_common.testing import inject_fakeredis

    from src.services.cancel_registry import CancelRegistry
    from src.services.imagegen_service import ImageGenService

    redis = RedisClient(redis_url="redis://localhost:6379/0")
    inject_fakeredis(redis)
    cancel_registry = CancelRegistry(redis)

    svc = ImageGenService(
        storage=MagicMock(),
        registry=MagicMock(),
        default_provider="mock_images",
        cancel_registry=cancel_registry,
    )

    started = asyncio.Event()

    async def long_running():
        started.set()
        await asyncio.sleep(100)

    task = asyncio.create_task(long_running())
    svc.register_task("img-a", task)
    await started.wait()

    owned_job = MagicMock(status="processing")

    with (
        patch(
            "src.services.imagegen_service.ImageJobManager.get_job",
            new=AsyncMock(side_effect=_tenant_scoped_get_job(TENANT_A, owned_job)),
        ),
        patch(
            "src.services.imagegen_service.ImageJobManager.mark_cancelled",
            new_callable=AsyncMock,
        ) as mock_mark,
    ):
        result = await svc.cancel_job(AsyncMock(), "img-a", TENANT_B)

    assert result is False
    assert await cancel_registry.is_cancelled("img-a") is False
    assert not task.cancelled()
    assert not task.done()
    assert svc._active_tasks.get("img-a") is task
    mock_mark.assert_not_awaited()

    # Cleanup
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
