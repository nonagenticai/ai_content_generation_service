"""Unit tests for concurrency semaphore (Phase 1)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.video_generation_service import VideoGenerationService


def _make_service(parallel_requests=3):
    """Create a VideoGenerationService with mocked dependencies."""
    storage = MagicMock()
    registry = MagicMock()
    return VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider="fake",
        parallel_requests=parallel_requests,
    )


def test_semaphore_created_with_correct_value():
    svc = _make_service(parallel_requests=5)
    assert svc._semaphore._value == 5


def test_semaphore_default_value():
    svc = _make_service()
    assert svc._semaphore._value == 3


async def test_only_n_jobs_run_simultaneously():
    """Verify that only parallel_requests jobs run at the same time."""
    max_concurrent = 2
    svc = _make_service(parallel_requests=max_concurrent)

    concurrent_count = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def fake_process(db, job_id, tenant_id, budget_limit=5.0, spent_so_far=0.0):
        nonlocal concurrent_count, max_seen
        async with lock:
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
        await asyncio.sleep(0.05)
        async with lock:
            concurrent_count -= 1
        return 0.0

    svc._process_job_inner = AsyncMock(side_effect=fake_process)

    db = MagicMock()
    tasks = [asyncio.create_task(svc.process_job(db, f"job-{i}", "tenant-1")) for i in range(6)]
    await asyncio.gather(*tasks)

    assert max_seen <= max_concurrent


async def test_semaphore_releases_on_error():
    """Verify semaphore releases even when job fails."""
    svc = _make_service(parallel_requests=1)

    async def failing_process(db, job_id, tenant_id, budget_limit=5.0, spent_so_far=0.0):
        raise RuntimeError("test failure")

    svc._process_job_inner = AsyncMock(side_effect=failing_process)
    db = MagicMock()

    with pytest.raises(RuntimeError):
        await svc.process_job(db, "job-1", "tenant-1")

    # Semaphore should be released
    assert svc._semaphore._value == 1
