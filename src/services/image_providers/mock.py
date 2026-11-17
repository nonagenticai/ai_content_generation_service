"""Mock image generation provider for testing and development."""

from __future__ import annotations

import asyncio
import base64

from src.services.image_providers.base import ImageGenProvider, ImageGenResult

_MINIMAL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9VE0cJQAAAAASUVORK5CYII="
)


class MockImageProvider(ImageGenProvider):
    """Mock provider that returns a tiny PNG."""

    name = "mock_images"

    def __init__(self, delay_s: float = 0.0) -> None:
        self._delay_s = delay_s

    async def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        style: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        reference_image_urls: list[str] | None = None,
        extra_params: dict | None = None,
        usage_context: dict | None = None,
        target_usage: str | None = None,
    ) -> ImageGenResult:
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        width, height = 1, 1
        if size and "x" in size:
            raw_w, raw_h = size.split("x", 1)
            width = int(raw_w)
            height = int(raw_h)
        return ImageGenResult(
            success=True,
            image_data=_MINIMAL_PNG,
            provider_job_id="mock-image-job",
            mime_type="image/png",
            width=width,
            height=height,
        )

    def estimate_cost(self, *, size: str | None = None, quality: str | None = None) -> float:
        return 0.01

    async def health_check(self) -> bool:
        return True

    def get_models(self) -> list[str]:
        return ["mock-image-v1"]

    def get_styles(self) -> list[str]:
        return ["default"]
