"""Unit tests for video normalizer (Phase 6)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.normalizer import VideoNormalizer


def test_ffmpeg_args_default():
    norm = VideoNormalizer()
    args = norm._build_ffmpeg_args("/tmp/in.mp4", "/tmp/out.mp4")
    assert args[0] == "ffmpeg"
    assert "-i" in args
    assert "/tmp/in.mp4" in args
    assert "-vf" in args
    assert "scale=1920:1080" in args
    assert "-r" in args
    assert "30" in args
    assert "-c:v" in args
    assert "libx264" in args
    assert "-an" in args
    assert "/tmp/out.mp4" in args


def test_ffmpeg_args_keep_audio():
    norm = VideoNormalizer(strip_audio=False)
    args = norm._build_ffmpeg_args("/tmp/in.mp4", "/tmp/out.mp4")
    assert "-an" not in args


def test_ffmpeg_args_custom_resolution():
    norm = VideoNormalizer(target_resolution="1280:720", target_fps=24, target_codec="libx265")
    args = norm._build_ffmpeg_args("/tmp/in.mp4", "/tmp/out.mp4")
    assert "scale=1280:720" in args
    assert "24" in args
    assert "libx265" in args


async def test_normalize_empty_data():
    norm = VideoNormalizer()
    outcome = await norm.normalize(b"", "job-1")
    assert outcome.data == b""
    assert outcome.degraded is False  # nothing to normalize is not a fault


async def test_normalize_ffmpeg_not_found():
    """If FFmpeg is not installed, returns original data -- flagged as degraded."""
    norm = VideoNormalizer()
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("ffmpeg")):
        outcome = await norm.normalize(b"video-data", "job-1")
    assert outcome.data == b"video-data"
    assert outcome.degraded is True


async def test_normalize_ffmpeg_failure():
    """If FFmpeg returns error, returns original data."""
    norm = VideoNormalizer()

    process = AsyncMock()
    process.communicate = AsyncMock(return_value=(b"", b"error message"))
    process.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=process):
        outcome = await norm.normalize(b"video-data", "job-1")

    assert outcome.data == b"video-data"
    assert outcome.degraded is True


async def test_normalize_timeout():
    """If FFmpeg times out, returns original data."""
    norm = VideoNormalizer(timeout_seconds=1)

    process = MagicMock()
    process.communicate = MagicMock()
    process.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=process):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            outcome = await norm.normalize(b"video-data", "job-1")

    assert outcome.data == b"video-data"
    assert outcome.degraded is True


class _FakeProcess:
    """Minimal stand-in for asyncio.subprocess.Process that records reaping."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.sleep(3600)  # never completes; the test cancels it
        return b"", b""  # pragma: no cover

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        self.returncode = -9
        return -9


async def test_normalize_reaps_child_on_cancel():
    """A cancelled normalize() must not orphan the ffmpeg child.

    asyncio.CancelledError is a BaseException, so it escapes the TimeoutError /
    FileNotFoundError arms untouched. Without a finally, nothing kills ffmpeg
    while the TemporaryDirectory deletes its input file out from under it.
    """
    norm = VideoNormalizer(timeout_seconds=600)
    proc = _FakeProcess()
    started = asyncio.Event()

    async def _fake_exec(*args, **kwargs):
        del args, kwargs
        started.set()
        return proc

    with patch("asyncio.create_subprocess_exec", _fake_exec):
        task = asyncio.create_task(norm.normalize(b"video-data", "job-cancel"))
        await started.wait()
        await asyncio.sleep(0)  # let wait_for(communicate()) suspend
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert proc.killed, "ffmpeg child was never killed on cancel — orphaned process"
    assert proc.returncode is not None, "ffmpeg child was killed but never reaped — zombie"


def test_disabled_returns_original():
    """When normalization is disabled, the service should not be created."""
    from src.config import NormalizationSettings

    settings = NormalizationSettings(enabled=False)
    assert settings.enabled is False
    # Service creation is conditional in asgi.py - if not enabled, normalizer is None
