"""Deterministic stub image provider.

Produces real PNG bytes (parseable by Pillow and any downstream slide /
video pipeline). Output is deterministic per (width, height, prompt) so
tests can assert exact contents.

Activated via ``IMAGE_PROVIDER=stub``. Intended for offline pipelines
(CI, demos, local dev) without paid APIs.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
from threading import Lock

from PIL import Image, ImageDraw

from src.services.image_providers.base import ImageGenProvider, ImageGenResult

logger = logging.getLogger(__name__)

_DEFAULT_WIDTH = 256
_DEFAULT_HEIGHT = 256
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _parse_size(size: str | None) -> tuple[int, int]:
    """Parse a ``WxH`` size string. Falls back to defaults on bad input."""
    if not size or "x" not in size:
        return _DEFAULT_WIDTH, _DEFAULT_HEIGHT
    raw_w, raw_h = size.split("x", 1)
    try:
        w = int(raw_w)
        h = int(raw_h)
    except ValueError:
        return _DEFAULT_WIDTH, _DEFAULT_HEIGHT
    if w <= 0 or h <= 0:
        return _DEFAULT_WIDTH, _DEFAULT_HEIGHT
    return w, h


def _color_for_prompt(prompt: str) -> tuple[int, int, int]:
    """Hash the prompt to a stable RGB tuple."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return digest[0], digest[1], digest[2]


def _render_png(width: int, height: int, prompt: str) -> bytes:
    """Render a real PNG with a prompt-derived background and a text label.

    The label encodes the first 32 chars of the prompt so a human can tell
    which prompt produced which slide when debugging. Falls back to a
    plain background if the image is too small for legible text.
    """
    color = _color_for_prompt(prompt)
    img = Image.new("RGB", (width, height), color=color)
    if width >= 64 and height >= 16:
        draw = ImageDraw.Draw(img)
        # Use the bundled default font -- always available with Pillow.
        label = (prompt[:32] + "...") if len(prompt) > 32 else prompt
        # Pick a contrasting text color (white on dark, black on light).
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)
        draw.text((4, 4), label, fill=text_color)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


class StubImageProvider(ImageGenProvider):
    """Deterministic image provider that emits real PNG bytes."""

    name = "stub"

    def __init__(self, delay_s: float = 0.0) -> None:
        self._delay_s = delay_s
        # In-memory cache keyed by (width, height, prompt).
        self._cache: dict[tuple[int, int, str], bytes] = {}
        self._cache_lock = Lock()

    def _get_or_render(self, width: int, height: int, prompt: str) -> bytes:
        key = (width, height, prompt)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        # Render outside the lock -- expensive op, no shared state mutated.
        png = _render_png(width, height, prompt)
        with self._cache_lock:
            # Another thread may have populated the cache concurrently;
            # both renders are deterministic so either is correct.
            self._cache.setdefault(key, png)
        return png

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
        del negative_prompt, style, quality, reference_image_urls
        del extra_params, usage_context, target_usage
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        width, height = _parse_size(size)
        png = self._get_or_render(width, height, prompt)
        if png[:8] != _PNG_SIGNATURE:  # defensive -- Pillow guarantees this
            raise RuntimeError("stub PNG signature corrupted")
        return ImageGenResult(
            success=True,
            image_data=png,
            provider_job_id=f"stub-{hashlib.sha256(prompt.encode()).hexdigest()[:12]}",
            mime_type="image/png",
            width=width,
            height=height,
        )

    def estimate_cost(self, *, size: str | None = None, quality: str | None = None) -> float:
        del size, quality
        return 0.0

    async def health_check(self) -> bool:
        return True

    def get_models(self) -> list[str]:
        return ["stub-v1"]

    def get_styles(self) -> list[str]:
        return ["default"]
