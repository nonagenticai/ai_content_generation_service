"""Local image generation provider backed by the on-device mflux/FLUX service.

Calls the launchd image-gen backend at ``image-gen.test:9204`` (mini1).
Generation is synchronous: mflux returns the PNG inline as base64.

Outbound requests carry the request-scoped tenant identity (via the w2v_common
``identity_event_hook``) so image-gen's metering attributes GPU usage to the
real customer tenant instead of a default. The identity ContextVars are bound
by the AuthMiddleware (sync paths) and re-bound inside the background image job
before the provider runs.
"""

from __future__ import annotations

import asyncio
import base64
import logging

import httpx
from w2v_common.auth.identity_context import identity_event_hook

from src.config import LocalImageSettings, get_settings
from src.services.http_client import get_client
from src.services.image_providers.base import ImageGenProvider, ImageGenResult

logger = logging.getLogger(__name__)

# HTTP statuses the single-flight mflux backend returns when it is busy
# generating another image; these are retryable (the backend is healthy).
_BUSY_STATUSES = frozenset({429, 503})


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Return the ``Retry-After`` delay in seconds, or None if absent/unparseable.

    Only the delta-seconds form is honoured (mflux emits that); an HTTP-date
    form is ignored in favour of the exponential-backoff schedule.
    """
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _clamp_dim(value: int) -> int:
    """Clamp to mflux's accepted range and round to a multiple of 64."""
    bounded = max(256, min(2048, value))
    return (bounded // 64) * 64 or 256


def _parse_size(size: str | None) -> tuple[int | None, int | None]:
    """Parse a ``"WxH"`` size string into clamped ``(width, height)``."""
    if not size or "x" not in size:
        return None, None
    raw_w, raw_h = size.split("x", 1)
    try:
        return _clamp_dim(int(raw_w)), _clamp_dim(int(raw_h))
    except ValueError:
        return None, None


def build_platform_image_payload(
    prompt: str,
    negative_prompt: str | None,
    size: str | None,
    extra_params: dict | None,
) -> dict[str, object]:
    """Request body of the platform ``POST /api/image/generate`` protocol.

    Shared by :class:`LocalImageProvider` and the tenant ``nonagentic_endpoint``
    slot provider, so both speak exactly the same body.
    """
    payload: dict[str, object] = {"prompt": prompt}
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    width, height = _parse_size(size)
    if width is not None and height is not None:
        payload["width"] = width
        payload["height"] = height
    if extra_params:
        for key in ("steps", "guidance", "seed"):
            if key in extra_params and extra_params[key] is not None:
                payload[key] = extra_params[key]
    return payload


class LocalImageProvider(ImageGenProvider):
    """Generate images via the mflux/FLUX backend on mini1."""

    name = "local_image"

    def __init__(self, settings: LocalImageSettings | None = None) -> None:
        self._settings = settings or get_settings().local_image
        # When a service token is configured we authenticate to an in-cluster
        # modality-gateway (fail-closed) as a delegating service account:
        # X-Tenant-Id carries the SA Bearer token; identity_event_hook preserves
        # it and adds X-Effective-Tenant-Id from the per-request context so the
        # gateway routes + meters per customer tenant. No token => direct-backend
        # behaviour is byte-for-byte unchanged (bare-UUID X-Tenant-Id).
        headers = (
            {"X-Tenant-Id": f"Bearer {self._settings.service_token}"}
            if self._settings.service_token
            else None
        )
        self._client = get_client(
            self.name,
            base_url=self._settings.backend_url,
            timeout=self._settings.timeout,
            headers=headers,
            event_hooks={"request": [identity_event_hook]},
        )
        # The mflux backend is single-flight (503 while busy). Cap in-pod
        # concurrency so a batch's per-child tasks (all created on this pod)
        # serialize instead of hammering the backend with N concurrent requests.
        # asyncio.Semaphore(n) binds lazily to the running loop in 3.14, so
        # constructing it here (outside a loop) is safe.
        self._sem = asyncio.Semaphore(max(1, self._settings.max_concurrency))

    def _build_payload(
        self,
        prompt: str,
        negative_prompt: str | None,
        size: str | None,
        extra_params: dict | None,
    ) -> dict[str, object]:
        return build_platform_image_payload(prompt, negative_prompt, size, extra_params)

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
        payload = self._build_payload(prompt, negative_prompt, size, extra_params)
        # Hold the single in-pod slot across the whole attempt+backoff loop so
        # this pod does exactly one generation at a time; `async with` releases
        # it on return/exception. Cross-pod collisions (other replicas) are the
        # reason each attempt still retries on a busy (503/429) backend.
        async with self._sem:
            return await self._generate_with_retry(payload)

    async def _generate_with_retry(self, payload: dict[str, object]) -> ImageGenResult:
        attempts = max(0, self._settings.busy_retry_attempts)
        base_delay = max(0.0, self._settings.busy_retry_base_delay)
        last_busy_status = 503
        # attempt indices 0..attempts (initial try + `attempts` retries)
        for attempt in range(attempts + 1):
            try:
                response = await self._client.post("/api/image/generate", json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in _BUSY_STATUSES and attempt < attempts:
                    last_busy_status = status
                    retry_after = _parse_retry_after(exc.response)
                    delay = retry_after if retry_after is not None else base_delay * (2**attempt)
                    logger.warning(
                        "local_image: backend busy (%s); retry %d/%d in %.1fs",
                        status,
                        attempt + 1,
                        attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                # Non-busy status, or busy status with retries exhausted.
                return ImageGenResult(
                    success=False,
                    error=f"local_image: backend returned {status}",
                )
            except httpx.RequestError as exc:
                return ImageGenResult(success=False, error=f"local_image: transport error: {exc}")
            return self._parse_success(response)
        # Loop exhausted purely on busy statuses.
        return ImageGenResult(
            success=False,
            error=f"local_image: backend returned {last_busy_status}",
        )

    @staticmethod
    def _parse_success(response: httpx.Response) -> ImageGenResult:
        try:
            data = response.json()
            image_data = base64.b64decode(data["image_base64"])
        except (KeyError, ValueError) as exc:
            return ImageGenResult(success=False, error=f"local_image: malformed response: {exc}")

        return ImageGenResult(
            success=True,
            image_data=image_data,
            mime_type="image/png",
            width=data.get("width"),
            height=data.get("height"),
        )

    def estimate_cost(self, *, size: str | None = None, quality: str | None = None) -> float:
        return 0.0

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/health")
            response.raise_for_status()
            return bool(response.json().get("model_loaded"))
        except httpx.HTTPError, ValueError:
            return False

    def get_models(self) -> list[str]:
        return ["schnell", "dev"]

    def get_styles(self) -> list[str]:
        return []
