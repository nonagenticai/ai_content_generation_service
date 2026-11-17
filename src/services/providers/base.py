"""Abstract base class, result types, and provider registry for AI video generation."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.services.slot_http import slot_http_client

logger = logging.getLogger(__name__)

_POLL_INITIAL_DELAY = 2.0
_POLL_MULTIPLIER = 1.5
_POLL_MAX_DELAY = 30.0

# Called with the provider task id the instant ``submit`` returns it, BEFORE
# any polling. The caller uses this to durably record the id so a reclaim on a
# different pod can re-attach to the task instead of buying the clip again.
OnSubmit = Callable[[str], Awaitable[None]]


@dataclass
class VideoGenerationResult:
    """Result from a video generation request."""

    success: bool
    video_data: bytes | None = None
    video_url: str | None = None
    provider_job_id: str | None = None
    duration_s: float | None = None
    error: str | None = None


def publishable_video_url(locator: str | None, *, provider: str) -> str | None:
    """Return ``locator`` only if a CONSUMER can actually fetch it, else ``None``.

    ``check_status`` returns a provider-side handle whose only guaranteed
    consumer is this provider's own ``download()``. Several providers return a
    handle that is not a URL at all: ``stub`` returns ``stub://video/<id>.mp4``
    and ``wan`` returns a bare on-pod filesystem path. Publishing either one in
    a field named ``video_url`` produced a NON-EMPTY string that satisfied every
    emptiness/`"none"` check downstream and then failed silently at the point of
    consumption -- the orchestrator's ``is_fetchable_url`` dropped the shot, so a
    paid run reported COMPLETED with N videos while every AI clip was absent from
    the finished render.

    The fix is to make the un-fetchable case EXPRESSIBLE and LOUD rather than
    disguised as a URL: an un-fetchable handle becomes ``None``, which the
    service layer replaces with its own stream-through download path (see
    ``api/routes/jobs.get_job_result``, mirroring the image lane's
    ``_preferred_asset_url``), and which -- if no artifact was stored either --
    trips the downstream "no usable video_url" guard into a hard stage failure.

    Allow-list, not deny-list, ON PURPOSE: a future provider that invents its own
    scheme is nulled by default instead of silently leaking a new bad shape.
    """
    if not locator:
        return None
    if locator.startswith(("http://", "https://")):
        return locator
    logger.info(
        "Provider %s returned a non-fetchable locator (%.40s); not publishing it as "
        "video_url -- the stored artifact's download path is used instead",
        provider,
        locator,
    )
    return None


@dataclass
class ProviderInfo:
    """Information about a registered provider."""

    name: str
    available: bool
    models: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    max_duration_s: float | None = None
    supports_image_to_video: bool = False


class VideoGenerationProvider(ABC):
    """Abstract interface for AI video generation providers.

    Implements submit -> poll -> download pattern with a high-level
    generate() method that orchestrates the full lifecycle.
    """

    name: str

    # Per-job (tenant provider-slot) instances carry their OWN credentials, so
    # they must never share the process-wide ``get_client`` pool: those clients
    # are cached by NAME with the platform key baked into their default headers,
    # and the first creator wins. ``True`` routes every authenticated client of
    # this instance through ``_dedicated_client`` instead; ``aclose`` frees them.
    _dedicated: bool = False

    def _dedicated_client(
        self,
        suffix: str,
        *,
        base_url: str,
        timeout: float,
        headers: dict[str, str],
    ) -> httpx.AsyncClient | None:
        """This instance's private client for ``suffix``, or None for a platform instance.

        Callers write ``self._dedicated_client(...) or get_client(...)`` so a
        platform (registry) instance keeps using the shared pool exactly as before.
        """
        if not self._dedicated:
            return None
        clients: dict[str, httpx.AsyncClient] = self.__dict__.setdefault("_own_clients", {})
        client = clients.get(suffix)
        if client is None:
            # Connect-time SSRF guard: the slot's host is re-resolved and
            # validated on EVERY request (DNS rebinding), see ``slot_http``.
            client = slot_http_client(base_url=base_url, timeout=timeout, headers=headers)
            clients[suffix] = client
        return client

    async def aclose(self) -> None:
        """Close this instance's private clients (no-op for a platform instance)."""
        clients: dict[str, httpx.AsyncClient] = self.__dict__.pop("_own_clients", {})
        for client in clients.values():
            await client.aclose()

    @abstractmethod
    async def submit(self, prompt: str, **kwargs: Any) -> str:
        """Submit a generation request. Returns provider task ID."""

    @abstractmethod
    async def check_status(self, task_id: str) -> dict:
        """Check task status. Returns dict with status, video_url, error."""

    @abstractmethod
    async def download(self, task_id: str, video_url: str) -> bytes:
        """Download the completed video."""

    @abstractmethod
    def estimate_cost(self, duration_s: float) -> float:
        """Estimate cost in USD."""

    async def health_check(self) -> bool:
        """Check if provider API is reachable."""
        return True

    def get_models(self) -> list[str]:
        """Return available model names."""
        return []

    def get_styles(self) -> list[str]:
        """Return available style presets."""
        return []

    def get_max_duration_s(self) -> float | None:
        """Return maximum supported clip duration."""
        return None

    def supports_image_to_video(self) -> bool:
        """Whether this provider supports image-to-video."""
        return False

    def can_handle(
        self,
        *,
        duration_s: float | None = None,
        image_url: str | None = None,
    ) -> bool:
        """Check whether this provider can handle the requested parameters.

        Returns ``False`` when the request clearly exceeds the
        provider's capabilities (e.g. duration longer than
        ``max_duration_s``, or image-to-video when unsupported).
        """
        max_dur = self.get_max_duration_s()
        if duration_s is not None and max_dur is not None and duration_s > max_dur:
            return False
        return not (image_url and not self.supports_image_to_video())

    def _classify_http_status_error(self, exc: httpx.HTTPStatusError) -> Exception:
        """Map a definite (response-received) HTTP failure to a ProviderError.

        Used for both submit and poll/download: unlike a timeout, an
        HTTPStatusError means a response DID arrive, so there is no
        ambiguity about whether the request landed -- the existing
        status-based retriable/non-retriable split is unaffected by which
        phase produced it.
        """
        from src.services.exceptions import ProviderError, ProviderRateLimitError, format_error

        status = exc.response.status_code
        if status == 429:
            return ProviderRateLimitError(format_error(exc), provider=self.name, cause=exc)
        retriable = status >= 500
        return ProviderError(format_error(exc), provider=self.name, retriable=retriable, cause=exc)

    async def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        image_url: str | None = None,
        duration_s: float = 5.0,
        width: int = 1920,
        height: int = 1080,
        style: str | None = None,
        seed: int | None = None,
        extra_params: dict | None = None,
        resume_task_id: str | None = None,
        on_submit: OnSubmit | None = None,
    ) -> VideoGenerationResult:
        """Full lifecycle: submit -> poll -> download.

        ``submit`` is where the money is spent: every provider here bills on
        acceptance of the task, not on download. Three things protect that spend,
        and they cover DIFFERENT failures -- none of them subsumes another.

        ``on_submit``
            Awaited with the task id the instant ``submit`` returns, before the
            first poll. The caller persists the id, so a crash at any point
            after acceptance leaves a durable handle on the paid task.
        ``resume_task_id``
            When set, ``submit`` is SKIPPED entirely and the lifecycle re-attaches
            to that already-paid task: poll -> download. This is the cross-pod
            double-billing guard.
        The submit-phase error boundary below
            Covers the one case the other two CANNOT. A timeout while POSTing the
            submit request is AMBIGUOUS: the provider may have already accepted
            and billed the task before the response reached us -- but we never
            received a task id, so ``on_submit`` was never called and there is
            nothing for ``resume_task_id`` to re-attach to. No provider here
            exposes an outbound idempotency key either. Retrying would call
            ``submit()`` again and buy a second billable task for one logical job,
            so a submit-phase timeout is deliberately reclassified NON-retriable:
            failing the job is strictly better than silently double-billing. This
            mirrors ``_store_video_artifacts`` in ``video_generation_service.py``,
            which applies the identical "ambiguous, already possibly paid-for, so
            stop retrying" move one step later in the pipeline for storage faults.

        Polling is a GET against ``check_status`` (idempotent, no side effect) and
        download creates no billable provider state, so a timeout in either of
        those phases keeps its normal retriable classification -- there is no
        double-billing risk from retrying a read.

        Re-attach is available to every provider because ``check_status`` and
        ``download`` are keyed purely by task id and hit the provider's API --
        no pod-local state is required to resume. (``mock``/``stub``, which do
        keep local state, are free and unbilled, so a resumed one merely
        re-renders defaults.)

        Follow-up (out of scope here): where a provider's API supports a client
        idempotency key (worth checking for runway/kling/seedance/sora/veo
        individually), ``submit()`` could replay one instead of failing outright
        on an ambiguous timeout. Not attempted here -- scoped to making the
        failure safe, not to building a multi-provider idempotency scheme.
        """
        if resume_task_id:
            # Do NOT call submit(): this task is already paid for. The
            # submit-phase boundary below is deliberately skipped with it --
            # nothing was submitted, so nothing is ambiguous.
            task_id = resume_task_id
            logger.warning(
                "Provider %s re-attaching to existing task %s instead of re-submitting",
                self.name,
                task_id,
            )
        else:
            # This try wraps EXACTLY the submit call and nothing else. Widening it
            # would sweep unrelated failures into the non-retriable timeout rule.
            try:
                task_id = await self.submit(
                    prompt,
                    negative_prompt=negative_prompt,
                    image_url=image_url,
                    duration_s=duration_s,
                    width=width,
                    height=height,
                    style=style,
                    seed=seed,
                    extra_params=extra_params,
                )
            except httpx.TimeoutException as exc:
                from src.services.exceptions import ProviderError, format_error

                logger.error(
                    "Provider %s SUBMIT timed out -- NOT retrying: the provider task "
                    "may already exist and be billable, and because no task id came "
                    "back there is no on_submit record and nothing to resume by. "
                    "Retrying here would risk a second billable provider task for "
                    "one logical job: %s",
                    self.name,
                    exc,
                )
                raise ProviderError(
                    format_error(exc), provider=self.name, retriable=False, cause=exc
                ) from exc
            except httpx.HTTPStatusError as exc:
                raise self._classify_http_status_error(exc) from exc
            except (OSError, RuntimeError, ValueError) as exc:
                from src.services.exceptions import format_error

                logger.error("Provider %s submit failed: %s", self.name, exc)
                return VideoGenerationResult(success=False, error=format_error(exc))

            logger.info("Provider %s submitted task: %s", self.name, task_id)

        try:
            # Kept inside the poll/download boundary so on_submit keeps the error
            # handling it had on main; it still runs before the first poll, which
            # is the ordering guarantee the caller relies on.
            if on_submit is not None and not resume_task_id:
                await on_submit(task_id)

            # The locator returned by ``check_status`` is a PROVIDER-SIDE handle.
            # It is the correct input to ``download()`` -- but it is NOT
            # necessarily something any other consumer can fetch. Keep the two
            # roles in separate names so the un-fetchable one cannot escape.
            provider_locator = await self._poll_until_complete(task_id)

            video_data = None
            if provider_locator:
                video_data = await self.download(task_id, provider_locator)

            return VideoGenerationResult(
                success=True,
                video_data=video_data,
                video_url=publishable_video_url(provider_locator, provider=self.name),
                provider_job_id=task_id,
                duration_s=duration_s,
            )
        except httpx.TimeoutException as exc:
            from src.services.exceptions import ProviderTimeoutError, format_error

            logger.error("Provider %s timed out during poll/download: %s", self.name, exc)
            raise ProviderTimeoutError(format_error(exc), provider=self.name, cause=exc) from exc
        except httpx.HTTPStatusError as exc:
            raise self._classify_http_status_error(exc) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            from src.services.exceptions import format_error

            logger.error("Provider %s generation failed: %s", self.name, exc)
            return VideoGenerationResult(success=False, error=format_error(exc))

    async def _poll_until_complete(
        self,
        task_id: str,
        poll_interval: float = _POLL_INITIAL_DELAY,
        max_poll_time: float = 600.0,
    ) -> str | None:
        """Poll provider until task completes or times out."""
        elapsed = 0.0
        delay = poll_interval

        while elapsed < max_poll_time:
            status = await self.check_status(task_id)
            current = status.get("status", "pending")

            if current == "completed":
                return status.get("video_url")
            if current == "failed":
                raise RuntimeError(status.get("error") or "Generation failed")

            await asyncio.sleep(delay)
            elapsed += delay
            delay = min(delay * _POLL_MULTIPLIER, _POLL_MAX_DELAY)

        raise TimeoutError(f"Polling timed out after {max_poll_time}s for task {task_id}")


class VideoProviderRegistry:
    """Registry managing available video generation providers."""

    def __init__(self) -> None:
        self._providers: dict[str, VideoGenerationProvider] = {}

    def register(self, provider: VideoGenerationProvider) -> None:
        """Register a video generation provider.

        Args:
            provider: Provider instance to register.
        """
        self._providers[provider.name] = provider
        logger.info("Registered video provider: %s", provider.name)

    def get(self, name: str) -> VideoGenerationProvider | None:
        """Retrieve a provider by name.

        Args:
            name: Provider name.

        Returns:
            Provider instance if registered, None otherwise.
        """
        return self._providers.get(name)

    def list_providers(self) -> list[ProviderInfo]:
        """List all registered providers with their capabilities.

        Returns:
            List of ProviderInfo dataclass instances.
        """
        result = []
        for name, provider in self._providers.items():
            try:
                info = ProviderInfo(
                    name=name,
                    available=True,
                    models=provider.get_models(),
                    styles=provider.get_styles(),
                    max_duration_s=provider.get_max_duration_s(),
                    supports_image_to_video=provider.supports_image_to_video(),
                )
            except OSError, RuntimeError, ValueError:
                info = ProviderInfo(name=name, available=False)
            result.append(info)
        return result

    @property
    def provider_names(self) -> list[str]:
        return list(self._providers.keys())
