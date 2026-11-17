"""Per-job image providers for a tenant's SELF-HOSTED GPU endpoint (P5-2 slots).

Two ``image`` slot kinds point ACG at a server the tenant runs:

* ``nonagentic_endpoint`` -- the platform protocol: ``POST {base_url}/api/image/generate``
  with the same body/response as :class:`LocalImageProvider` speaks to
  modality-gateway / image-gen (``{"prompt", "negative_prompt", "width", "height",
  "steps", "guidance", "seed"}`` -> ``{"image_base64", "width", "height"}``).
* ``openai_compatible`` -- the OpenAI Images shape: ``POST {base_url}/images/generations``
  -> ``{"data": [{"b64_json"} | {"url"}]}``.

Both are built per job by ``provider_slots.build_image_provider`` and never
registered. They use the SSRF-guarded :func:`slot_http_client` with an EXPLICIT
timeout, send ``Authorization: Bearer <secret>`` only when the slot has a secret,
and never log the secret. HTTP errors are RAISED (``raise_for_status``) so the
slot classifier can name a 401/403 as a credentials problem; a malformed body is
a ``success=False`` result.
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Final

import httpx
from w2v_common.net import ResponseTooLargeError, SSRFError, safe_get

from src.services.http_client import get_client
from src.services.image_providers.base import ImageGenProvider, ImageGenResult
from src.services.image_providers.local_image import build_platform_image_payload
from src.services.slot_http import slot_http_client

logger = logging.getLogger(__name__)

#: Request timeout (seconds) for a tenant self-hosted image endpoint. A single
#: GPU generation can take minutes on a modest card; the platform mflux backend
#: is given 180s, a tenant box gets more headroom but is still bounded.
TENANT_IMAGE_TIMEOUT_S: Final = 300.0
#: Connect timeout: an unreachable tenant host must fail fast, not after 300s.
TENANT_IMAGE_CONNECT_TIMEOUT_S: Final = 10.0


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(TENANT_IMAGE_TIMEOUT_S, connect=TENANT_IMAGE_CONNECT_TIMEOUT_S)


def _headers(secret: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    return headers


def _sniff_mime(data: bytes) -> str:
    """Best-effort mime type from magic bytes (default PNG)."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _first_item(response: httpx.Response) -> dict | None:
    """``data[0]`` of an OpenAI Images response; ``{}`` when empty, None when malformed."""
    try:
        items = response.json().get("data")
    except AttributeError, ValueError:
        return None
    item = items[0] if isinstance(items, list) and items else {}
    return item if isinstance(item, dict) else None


def _parse_wh(size: str | None) -> tuple[int | None, int | None]:
    if size and "x" in size:
        raw_w, raw_h = size.split("x", 1)
        if raw_w.isdigit() and raw_h.isdigit():
            return int(raw_w), int(raw_h)
    return None, None


class _TenantEndpointImageProvider(ImageGenProvider):
    """Shared per-job plumbing: guarded client, explicit timeout, aclose."""

    def __init__(self, *, base_url: str, secret: str, model: str | None) -> None:
        if not base_url:
            # Fail loud: there is no platform default to fall back to.
            raise ValueError(f"{self.name}: a base_url is required")
        self._base_url = base_url.rstrip("/")
        self._model = model or None
        self._client = slot_http_client(
            base_url=self._base_url,
            timeout=_timeout(),
            headers=_headers(secret),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def estimate_cost(self, *, size: str | None = None, quality: str | None = None) -> float:
        # Tenant-funded compute: the platform pays nothing.
        return 0.0

    def get_models(self) -> list[str]:
        return [self._model] if self._model else []


class NonagenticEndpointImageProvider(_TenantEndpointImageProvider):
    """``nonagentic_endpoint``: the platform image protocol on a tenant host."""

    name = "tenant_nonagentic_endpoint"

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
        payload = build_platform_image_payload(prompt, negative_prompt, size, extra_params)
        if self._model:
            payload["model"] = self._model
        response = await self._client.post("/api/image/generate", json=payload)
        response.raise_for_status()
        try:
            data = response.json()
            image_data = base64.b64decode(data["image_base64"], validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            return ImageGenResult(
                success=False,
                error=f"{self.name}: malformed response ({type(exc).__name__})",
            )
        return ImageGenResult(
            success=True,
            image_data=image_data,
            mime_type=_sniff_mime(image_data),
            width=data.get("width"),
            height=data.get("height"),
        )


class OpenAICompatibleImageProvider(_TenantEndpointImageProvider):
    """``openai_compatible``: the OpenAI Images API shape on a tenant host.

    Sends only the standard Images fields (no gpt-image-only ``background`` /
    ``output_format``) and ``model`` only when the slot names one, so a
    self-hosted server is not handed a vendor model it does not serve.
    """

    name = "tenant_openai_compatible_images"

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
        payload = self._build_payload(prompt, negative_prompt, style, size, quality, extra_params)
        response = await self._client.post("/images/generations", json=payload)
        response.raise_for_status()
        item = _first_item(response)
        if item is None:
            return ImageGenResult(success=False, error=f"{self.name}: malformed response")

        image_url: str | None = None
        if item.get("b64_json"):
            try:
                image_data = base64.b64decode(item["b64_json"], validate=True)
            except TypeError, ValueError, binascii.Error:
                return ImageGenResult(success=False, error=f"{self.name}: invalid b64_json")
        elif item.get("url"):
            image_url = str(item["url"])
            image_data = await self._download(image_url)
        else:
            return ImageGenResult(success=False, error=f"{self.name}: no image data in response")

        width, height = _parse_wh(size)
        return ImageGenResult(
            success=True,
            image_data=image_data,
            image_url=image_url,
            mime_type=_sniff_mime(image_data),
            width=width,
            height=height,
        )

    def _build_payload(
        self,
        prompt: str,
        negative_prompt: str | None,
        style: str | None,
        size: str | None,
        quality: str | None,
        extra_params: dict | None,
    ) -> dict[str, object]:
        request_prompt = f"{prompt}\n\nAvoid: {negative_prompt}" if negative_prompt else prompt
        payload: dict[str, object] = {"prompt": request_prompt, "n": 1}
        optional = {"model": self._model, "size": size, "quality": quality, "style": style}
        payload.update({key: value for key, value in optional.items() if value})
        if extra_params:
            payload.update(extra_params)
        return payload

    async def _download(self, image_url: str) -> bytes:
        """Fetch a response ``url`` through the SSRF-safe one-shot fetcher.

        Raises:
            ProviderError: non-retriable, when the target is blocked or oversized.
        """
        dl_client = get_client("image_downloads", timeout=TENANT_IMAGE_TIMEOUT_S)
        try:
            download = await safe_get(dl_client, image_url)
        except (SSRFError, ResponseTooLargeError) as exc:
            from src.services.exceptions import ProviderError, format_error

            raise ProviderError(
                f"Refused to fetch image url: {format_error(exc)}",
                provider=self.name,
                retriable=False,
                cause=exc,
            ) from exc
        download.raise_for_status()
        return download.content
