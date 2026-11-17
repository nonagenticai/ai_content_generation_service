"""Mock video generation provider for testing and local development."""

from __future__ import annotations

import asyncio
import logging
import struct
import uuid
from typing import Any

from src.services.providers.base import VideoGenerationProvider

logger = logging.getLogger(__name__)

# Minimal valid MP4: ftyp box + minimal moov box
_FTYP_BOX = struct.pack(">I4s4sI4s", 20, b"ftyp", b"isom", 512, b"isom")
_MOOV_BOX = struct.pack(">I4s", 8, b"moov")
_MINIMAL_MP4 = _FTYP_BOX + _MOOV_BOX


class MockProvider(VideoGenerationProvider):
    """Mock provider that returns synthetic video data without external API calls."""

    name = "mock"

    def __init__(self, delay_s: float = 0.0) -> None:
        self._delay_s = delay_s

    async def submit(self, prompt: str, **kwargs: Any) -> str:
        """Submit a mock generation request.

        Args:
            prompt: Text prompt (logged for debugging).
            **kwargs: Ignored.

        Returns:
            Random UUID as task ID.
        """
        task_id = str(uuid.uuid7())
        logger.info("Mock submit: task_id=%s prompt=%s", task_id, prompt[:80])
        return task_id

    async def check_status(self, task_id: str) -> dict:
        """Return completed status immediately (with optional delay).

        Args:
            task_id: Task identifier.

        Returns:
            Dict with completed status and mock video URL.
        """
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        return {
            "status": "completed",
            "video_url": f"mock://video/{task_id}.mp4",
            "error": None,
        }

    async def download(self, task_id: str, video_url: str) -> bytes:
        """Return minimal valid MP4 bytes.

        Args:
            task_id: Ignored.
            video_url: Ignored.

        Returns:
            Minimal MP4 bytes (ftyp + moov boxes).
        """
        return _MINIMAL_MP4

    def estimate_cost(self, duration_s: float) -> float:
        """Estimate mock generation cost.

        Args:
            duration_s: Requested clip duration.

        Returns:
            Simulated cost in USD.
        """
        return 0.01 * max(duration_s, 1.0)

    async def health_check(self) -> bool:
        return True

    def get_models(self) -> list[str]:
        return ["mock-v1"]

    def get_styles(self) -> list[str]:
        return ["default"]

    def get_max_duration_s(self) -> float | None:
        return 60.0

    def supports_image_to_video(self) -> bool:
        return True
