"""Cross-pod cancellation via the Redis flag registry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from w2v_common.cache.redis_client import RedisClient
from w2v_common.testing import inject_fakeredis

from src.services.cancel_registry import CancelRegistry


def _shared_redis() -> RedisClient:
    client = RedisClient(redis_url="redis://localhost:6379/0")
    inject_fakeredis(client)
    return client


async def test_cancel_flag_reaches_another_pod():
    """A cancel requested via 'pod B' is observed by a poller on 'pod A'."""
    redis = _shared_redis()
    # Two registries over the SAME Redis, standing in for two replicas.
    pod_a = CancelRegistry(redis)
    pod_b = CancelRegistry(redis)

    job_id = "job-cross-pod"
    assert await pod_a.is_cancelled(job_id) is False

    # Cancel lands on pod B; pod A (which runs the task) must see it.
    assert await pod_b.request_cancel(job_id) is True
    assert await pod_a.is_cancelled(job_id) is True


async def test_cancel_flag_clear_and_isolation():
    redis = _shared_redis()
    reg = CancelRegistry(redis)
    await reg.request_cancel("j1")
    assert await reg.is_cancelled("j1") is True
    # A different job is unaffected.
    assert await reg.is_cancelled("j2") is False
    await reg.clear("j1")
    assert await reg.is_cancelled("j1") is False


async def test_cancel_registry_degrades_without_redis():
    """No Redis → flag is a no-op, never raises (falls back to local/DB cancel)."""
    reg = CancelRegistry(None)
    assert await reg.request_cancel("j1") is False
    assert await reg.is_cancelled("j1") is False
    await reg.clear("j1")  # no raise


async def test_service_cancel_job_sets_cross_pod_flag_for_remote_processing():
    """cancel_job on a pod with no local task still flags a processing job as cancelled."""
    from src.services.video_generation_service import VideoGenerationService

    redis = _shared_redis()
    registry = CancelRegistry(redis)
    svc = VideoGenerationService(
        storage=MagicMock(),
        registry=MagicMock(),
        default_provider="stub",
        cancel_registry=registry,
    )

    job_id = "remote-job"
    processing_job = MagicMock(status="processing")

    # No local task registered (job runs on another replica).
    with patch(
        "src.services.video_generation_service.JobManager.get_job",
        new=AsyncMock(return_value=processing_job),
    ):
        accepted = await svc.cancel_job(AsyncMock(), job_id, "tenant-1")

    assert accepted is True
    # The cross-pod flag is now set, so the remote task will self-cancel.
    assert await registry.is_cancelled(job_id) is True
