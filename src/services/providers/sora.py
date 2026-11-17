"""OpenAI Sora video generation provider."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from w2v_common.net import ResponseTooLargeError, SSRFError, safe_get

from src.config import get_settings
from src.constants import (
    HTTPX_DOWNLOAD_TIMEOUT_S,
    HTTPX_HEALTH_CHECK_TIMEOUT_S,
    HTTPX_POLL_TIMEOUT_S,
)
from src.services.http_client import get_client
from src.services.providers.base import VideoGenerationProvider

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.openai.com/v1"


class SoraProvider(VideoGenerationProvider):
    """Generate video clips using OpenAI Sora API."""

    name = "sora"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Platform instance by default; explicit args build a per-job (slot) instance.

        Any explicit argument switches the instance to its own private HTTP
        clients (see ``VideoGenerationProvider._dedicated_client``). Omitted
        arguments keep today's values: the ``SORA_*`` settings and the
        public API base URL.
        """
        settings = get_settings()
        self._dedicated = any(v is not None for v in (api_key, base_url, model))
        self._api_key = api_key if api_key is not None else settings.sora.api_key
        self._base_url = base_url or _BASE_URL
        self._model = model or settings.sora.model
        self._timeout = settings.sora.timeout
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._client = self._dedicated_client(
            "", base_url=self._base_url, timeout=self._timeout, headers=self._headers
        ) or get_client(
            self.name,
            base_url=self._base_url,
            timeout=self._timeout,
            headers=self._headers,
        )

    async def submit(self, prompt: str, **kwargs: Any) -> str:
        """Submit a video generation request to OpenAI Sora API.

        Args:
            prompt: Text prompt for generation.
            **kwargs: Extra params including duration_s, width, height, style, image_url.

        Returns:
            Provider generation ID or 'direct:<url>' for immediate results.
        """
        payload: dict = {
            "model": self._model,
            "prompt": prompt,
        }
        if kwargs.get("duration_s"):
            payload["duration"] = kwargs["duration_s"]
        width, height = kwargs.get("width"), kwargs.get("height")
        if width and height:
            payload["size"] = f"{width}x{height}"
        if kwargs.get("style"):
            payload["style"] = kwargs["style"]
        if kwargs.get("image_url"):
            payload["image_url"] = kwargs["image_url"]
        if kwargs.get("extra_params"):
            payload.update(kwargs["extra_params"])

        response = await self._client.post(
            "/video/generations",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        gen_id = data.get("id")
        if not gen_id:
            video_url = data.get("data", [{}])[0].get("url", "")
            if video_url:
                return f"direct:{video_url}"
            raise RuntimeError("No generation ID returned from Sora API")
        return gen_id

    async def check_status(self, task_id: str) -> dict:
        """Poll OpenAI Sora API for generation status.

        Args:
            task_id: Provider generation ID or 'direct:<url>'.

        Returns:
            Dict with status, video_url, and error fields.
        """
        if task_id.startswith("direct:"):
            return {
                "status": "completed",
                "video_url": task_id.removeprefix("direct:"),
                "error": None,
            }

        poll_client = self._dedicated_client(
            "_poll",
            base_url=self._base_url,
            timeout=HTTPX_POLL_TIMEOUT_S,
            headers={"Authorization": f"Bearer {self._api_key}"},
        ) or get_client(
            f"{self.name}_poll",
            base_url=self._base_url,
            timeout=HTTPX_POLL_TIMEOUT_S,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response = await poll_client.get(f"/video/generations/{task_id}")
        response.raise_for_status()
        data = response.json()
        status = data.get("status", "")
        status_map = {
            "pending": "pending",
            "processing": "processing",
            "completed": "completed",
            "failed": "failed",
        }
        mapped = status_map.get(status, "pending")
        video_url = None
        if mapped == "completed":
            video_url = data.get("data", [{}])[0].get("url", "")
        error = None
        if mapped == "failed":
            error = data.get("error", {}).get("message", "Generation failed")
        return {"status": mapped, "video_url": video_url, "error": error}

    async def download(self, task_id: str, video_url: str) -> bytes:
        """Download the completed video from OpenAI CDN SSRF-safely.

        ``video_url`` is provider-response data, not a value this service chose:
        a bare ``dl_client.get(video_url)`` resolves whatever hostname it carries
        and would happily reach ``169.254.169.254`` / ``10.x``. Any advisory check
        made earlier discards the resolved IP, so :func:`w2v_common.net.safe_get`
        is the enforcement point — it re-validates, pins the connection to the
        validated IP (closing the DNS-rebinding TOCTOU window) and re-checks every
        redirect hop with ``follow_redirects`` forced off.

        Args:
            task_id: Provider task identifier.
            video_url: URL to download the video from.

        Returns:
            Raw video bytes.

        Raises:
            ProviderError: non-retriable, when the target is blocked or oversized.
        """
        from src.services.exceptions import ProviderError, format_error

        dl_client = get_client("downloads", timeout=HTTPX_DOWNLOAD_TIMEOUT_S)
        try:
            response = await safe_get(dl_client, video_url)
        except (SSRFError, ResponseTooLargeError) as exc:
            # Both subclass ValueError, and ``VideoGenerationProvider.generate``
            # catches ``(OSError, RuntimeError, ValueError)`` and DEGRADES it into
            # ``VideoGenerationResult(success=False)`` — indistinguishable from an
            # ordinary provider fault, so the chain falls through to the next
            # (paid) provider and re-attempts the same blocked fetch. ProviderError
            # is not a ValueError, so this escapes that arm as a real,
            # non-retriable refusal.
            raise ProviderError(
                f"Refused to fetch video_url: {format_error(exc)}",
                provider=self.name,
                retriable=False,
                cause=exc,
            ) from exc
        response.raise_for_status()
        return response.content

    def estimate_cost(self, duration_s: float) -> float:
        """Estimate generation cost in USD.

        Args:
            duration_s: Requested clip duration.

        Returns:
            Estimated cost in USD.
        """
        return 0.10 * max(duration_s, 5.0) / 5.0

    async def health_check(self) -> bool:
        """Check provider API health with HTTP request."""
        if not self._api_key:
            return False
        try:
            health_client = get_client(f"{self.name}_health", timeout=HTTPX_HEALTH_CHECK_TIMEOUT_S)
            resp = await health_client.head(_BASE_URL)
            return resp.status_code < 500
        except httpx.HTTPError, OSError:
            return False

    def get_models(self) -> list[str]:
        return ["sora-1"]

    def get_styles(self) -> list[str]:
        return ["natural", "vivid"]

    def get_max_duration_s(self) -> float | None:
        return 20.0

    def supports_image_to_video(self) -> bool:
        return True
