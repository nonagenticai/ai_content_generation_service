"""OpenAI image generation provider."""

from __future__ import annotations

import base64
import logging

from w2v_common.net import ResponseTooLargeError, SSRFError, safe_get

from src.config import get_settings
from src.services.http_client import get_client
from src.services.image_providers.base import ImageGenProvider, ImageGenResult
from src.services.slot_http import slot_http_client

logger = logging.getLogger(__name__)


class OpenAIImageProvider(ImageGenProvider):
    """Generate images with OpenAI's Images API."""

    name = "openai_images"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Platform instance by default; explicit args build a per-job (slot) instance.

        A per-job instance (any explicit argument) gets its OWN httpx client:
        the shared ``get_client`` pool is cached by name with the platform key in
        its default headers, so reusing it would send the wrong credential.
        Omitted arguments keep today's ``DALLE_*`` settings.
        """
        settings = get_settings()
        self._settings = settings.dalle
        self._api_key = api_key if api_key is not None else self._settings.api_key
        self._base_url = base_url or self._settings.base_url
        self._model = model or self._settings.model
        self._dedicated = any(v is not None for v in (api_key, base_url, model))
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._client = (
            # Connect-time SSRF guard (DNS rebinding), see ``slot_http``.
            slot_http_client(
                base_url=self._base_url,
                timeout=self._settings.timeout,
                headers=self._headers,
            )
            if self._dedicated
            else get_client(
                self.name,
                base_url=self._base_url,
                timeout=self._settings.timeout,
                headers=self._headers,
            )
        )

    async def aclose(self) -> None:
        """Close a per-job instance's private client (no-op for the platform one)."""
        if self._dedicated:
            await self._client.aclose()

    @staticmethod
    def _compose_prompt(
        prompt: str,
        negative_prompt: str | None,
        reference_image_urls: list[str] | None,
        usage_context: dict | None,
        target_usage: str | None,
    ) -> str:
        """Build the user-facing prompt by appending optional context blocks."""
        parts = [prompt]
        if negative_prompt:
            parts.append(f"Avoid: {negative_prompt}")
        if reference_image_urls:
            parts.append(f"Reference inspiration URLs: {', '.join(reference_image_urls[:3])}")
        if usage_context:
            usage_parts = [f"{key}: {value}" for key, value in usage_context.items() if value]
            if usage_parts:
                parts.append("Context:\n" + "\n".join(usage_parts[:8]))
        if target_usage:
            parts.append(f"Intended usage: {target_usage}")
        return "\n\n".join(parts)

    def _build_payload(
        self,
        request_prompt: str,
        size: str | None,
        quality: str | None,
        style: str | None,
        extra_params: dict | None,
    ) -> dict[str, object]:
        """Assemble the request payload for /images/generations."""
        resolved_quality = quality or self._settings.quality
        if str(resolved_quality).lower() == "hd":
            resolved_quality = "high"

        payload: dict[str, object] = {
            "model": self._model,
            "prompt": request_prompt,
            "size": size or self._settings.size,
            "quality": resolved_quality,
            "background": self._settings.background,
            "output_format": self._settings.output_format,
        }
        if style:
            payload["style"] = style
        if extra_params:
            payload.update(extra_params)
        return payload

    async def _extract_image(self, item: dict) -> tuple[bytes | None, str | None, str | None]:
        """Return ``(image_data, image_url, error)`` from an OpenAI response item.

        The ``url`` branch re-fetches an address carried in the API response, so a
        bare ``dl_client.get(image_url)`` would re-resolve that hostname and could
        land on ``169.254.169.254`` / ``10.x``. :func:`w2v_common.net.safe_get`
        re-validates, pins the connection to the validated IP (closing the
        DNS-rebinding TOCTOU window) and re-checks every redirect hop with
        ``follow_redirects`` forced off.

        Raises:
            ProviderError: non-retriable, when the target is blocked or oversized.
        """
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"]), None, None
        if item.get("url"):
            image_url = str(item["url"])
            dl_client = get_client("image_downloads", timeout=self._settings.timeout)
            try:
                download = await safe_get(dl_client, image_url)
            except (SSRFError, ResponseTooLargeError) as exc:
                # Do NOT fold this into the ``(None, None, error)`` tuple that this
                # method uses for ordinary failures: that degrades into
                # ``ImageGenResult(success=False)``, which reads as a routine
                # provider fault and lets the caller fall through to the next
                # (paid) provider and re-attempt the same blocked fetch.
                # ProviderError(retriable=False) is what ``is_retriable_error``
                # reads to stop the retry/fallback chain on a security refusal.
                from src.services.exceptions import ProviderError, format_error

                raise ProviderError(
                    f"Refused to fetch image url: {format_error(exc)}",
                    provider=self.name,
                    retriable=False,
                    cause=exc,
                ) from exc
            download.raise_for_status()
            return download.content, image_url, None
        return None, None, "OpenAI returned no image data"

    @staticmethod
    def _resolve_dimensions(size_value: str) -> tuple[int | None, int | None]:
        """Parse a 'WxH' size string into ``(width, height)``."""
        if "x" not in size_value:
            return None, None
        raw_w, raw_h = size_value.split("x", 1)
        return int(raw_w), int(raw_h)

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
        if not self._api_key:
            return ImageGenResult(success=False, error="Missing OpenAI image API key")

        request_prompt = self._compose_prompt(
            prompt, negative_prompt, reference_image_urls, usage_context, target_usage
        )
        payload = self._build_payload(request_prompt, size, quality, style, extra_params)

        response = await self._client.post("/images/generations", json=payload)
        response.raise_for_status()
        data = response.json()
        item = data.get("data", [{}])[0] if isinstance(data.get("data"), list) else {}

        image_data, image_url, err = await self._extract_image(item)
        if err is not None:
            return ImageGenResult(success=False, error=err)

        width, height = self._resolve_dimensions(str(payload["size"]))

        return ImageGenResult(
            success=True,
            image_data=image_data,
            image_url=image_url,
            mime_type=f"image/{self._settings.output_format}",
            width=width,
            height=height,
        )

    def estimate_cost(self, *, size: str | None = None, quality: str | None = None) -> float:
        resolved_size = size or self._settings.size
        resolved_quality = str(quality or self._settings.quality).lower()
        if resolved_quality == "hd":
            resolved_quality = "high"
        size_multiplier = 1.0
        if resolved_size.startswith("1792x"):
            size_multiplier = 1.5
        elif resolved_size.startswith("1024x"):
            size_multiplier = 1.0
        if resolved_quality in {"high", "hd"}:
            size_multiplier *= 1.5
        return round(0.04 * size_multiplier, 4)

    async def health_check(self) -> bool:
        return bool(self._settings.enabled and self._api_key)

    def get_models(self) -> list[str]:
        return [self._model]

    def get_styles(self) -> list[str]:
        return ["clean", "editorial", "product", "presentation"]
