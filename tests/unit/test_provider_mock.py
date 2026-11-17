"""Unit tests for MockProvider."""

from __future__ import annotations

import uuid

from src.services.providers.mock import MockProvider

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


async def test_submit_returns_uuid():
    p = MockProvider(delay_s=0.0)
    task_id = await p.submit("test prompt")
    # Should be a valid UUID
    parsed = uuid.UUID(task_id)
    assert str(parsed) == task_id


async def test_submit_accepts_kwargs():
    p = MockProvider(delay_s=0.0)
    task_id = await p.submit(
        "test prompt",
        duration_s=10,
        width=1280,
        height=720,
        style="default",
    )
    assert isinstance(task_id, str)
    assert len(task_id) > 0


# ---------------------------------------------------------------------------
# Check status
# ---------------------------------------------------------------------------


async def test_check_status_completed():
    p = MockProvider(delay_s=0.0)
    task_id = await p.submit("test")
    result = await p.check_status(task_id)
    assert result["status"] == "completed"
    assert result["video_url"] == f"mock://video/{task_id}.mp4"
    assert result["error"] is None


async def test_check_status_contains_task_id_in_url():
    p = MockProvider(delay_s=0.0)
    task_id = await p.submit("test")
    result = await p.check_status(task_id)
    assert task_id in result["video_url"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


async def test_download_returns_valid_bytes():
    p = MockProvider(delay_s=0.0)
    data = await p.download("any-task", "mock://video/any-task.mp4")
    assert isinstance(data, bytes)
    assert len(data) > 0
    # Should start with a valid ftyp box (MP4 signature)
    assert b"ftyp" in data


async def test_download_is_consistent():
    """Multiple downloads return the same bytes (deterministic mock)."""
    p = MockProvider(delay_s=0.0)
    data1 = await p.download("task-a", "mock://video/task-a.mp4")
    data2 = await p.download("task-b", "mock://video/task-b.mp4")
    assert data1 == data2


# ---------------------------------------------------------------------------
# Estimate cost
# ---------------------------------------------------------------------------


async def test_estimate_cost():
    p = MockProvider(delay_s=0.0)
    # $0.01 per second, min 1s
    assert p.estimate_cost(1.0) == 0.01
    assert p.estimate_cost(5.0) == 0.05
    assert abs(p.estimate_cost(10.0) - 0.10) < 1e-9
    # Below 1s still charges at least 1s
    assert p.estimate_cost(0.5) == 0.01


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


async def test_health_check():
    p = MockProvider(delay_s=0.0)
    assert await p.health_check() is True


# ---------------------------------------------------------------------------
# Model info / metadata
# ---------------------------------------------------------------------------


async def test_get_models():
    p = MockProvider(delay_s=0.0)
    assert p.get_models() == ["mock-v1"]


async def test_get_styles():
    p = MockProvider(delay_s=0.0)
    assert p.get_styles() == ["default"]


async def test_get_max_duration_s():
    p = MockProvider(delay_s=0.0)
    assert p.get_max_duration_s() == 60.0


async def test_supports_image_to_video():
    p = MockProvider(delay_s=0.0)
    assert p.supports_image_to_video() is True


async def test_name():
    p = MockProvider(delay_s=0.0)
    assert p.name == "mock"
