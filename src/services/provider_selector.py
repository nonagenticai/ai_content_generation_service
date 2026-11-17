"""Pluggable provider selection strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.services.provider_health import ProviderHealthMonitor
from src.services.providers.base import VideoProviderRegistry

if TYPE_CHECKING:
    from src.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)


class ProviderSelector(ABC):
    """Abstract base class for provider selection strategies."""

    @abstractmethod
    def select(
        self,
        registry: VideoProviderRegistry,
        health_monitor: ProviderHealthMonitor | None,
        duration_s: float | None,
        *,
        image_url: str | None = None,
    ) -> str:
        """Select a provider name. Raises RuntimeError if none available."""


def _provider_can_handle(
    registry: VideoProviderRegistry,
    name: str,
    duration_s: float | None,
    image_url: str | None,
) -> bool:
    """Return True if the named provider can handle the requested params."""
    provider = registry.get(name)
    if provider is None:
        return False
    return provider.can_handle(duration_s=duration_s, image_url=image_url)


def _availability_and_latency(
    health_monitor: ProviderHealthMonitor, name: str
) -> tuple[bool, float | None]:
    """Return ``(available, average_latency_ms)`` for one provider.

    ``average_latency_ms`` is ``None`` when the provider is available but has
    never recorded a success — "no data yet", which is very different from
    "unavailable" and must not be collapsed into it. A failing health check is
    reported as unavailable, matching the previous behaviour.
    """
    try:
        if not health_monitor.is_available(name):
            return False, None
        return True, health_monitor.get_health(name).average_latency_ms
    except (OSError, RuntimeError) as e:
        logger.warning("Health check failed for provider %s: %s", name, e)
        return False, None


def _partition_candidates(
    registry: VideoProviderRegistry,
    health_monitor: ProviderHealthMonitor,
    duration_s: float | None,
    image_url: str | None,
) -> tuple[list[tuple[float, str]], list[str]]:
    """Split capable, available providers into measured and never-measured.

    Both lists are in registry order. A provider only appears in either list
    if it passed the capability check *and* the health check, so an empty
    result really is "no available providers".
    """
    measured: list[tuple[float, str]] = []
    unmeasured: list[str] = []

    for info in registry.list_providers():
        if not _provider_can_handle(registry, info.name, duration_s, image_url):
            continue
        available, latency = _availability_and_latency(health_monitor, info.name)
        if not available:
            continue
        if latency is None:
            unmeasured.append(info.name)
        else:
            measured.append((latency, info.name))

    return measured, unmeasured


class ManualSelector(ProviderSelector):
    """Always returns the default provider (current behavior)."""

    def __init__(self, default_provider: str) -> None:
        self._default = default_provider

    def select(
        self,
        registry: VideoProviderRegistry,
        health_monitor: ProviderHealthMonitor | None,
        duration_s: float | None,
        *,
        image_url: str | None = None,
    ) -> str:
        return self._default


class CheapestSelector(ProviderSelector):
    """Select the cheapest provider based on estimate_cost."""

    def select(
        self,
        registry: VideoProviderRegistry,
        health_monitor: ProviderHealthMonitor | None,
        duration_s: float | None,
        *,
        image_url: str | None = None,
    ) -> str:
        dur = duration_s or 5.0
        best_name = None
        best_cost = float("inf")

        for info in registry.list_providers():
            if health_monitor is not None:
                try:
                    if not health_monitor.is_available(info.name):
                        continue
                except (OSError, RuntimeError) as e:
                    # An UNMEASURABLE provider is not a healthy one. Without this
                    # `continue` the except merely logged and fell through, so a
                    # provider whose availability could not be read stayed in the
                    # candidate list and was selected exactly as if it had been
                    # measured healthy -- then paid. `_availability_and_latency` in
                    # this same module already chose the other direction for the
                    # identical exception tuple (`return False, None`); these three
                    # selectors simply never got it.
                    logger.warning("Health check failed for provider %s: %s", info.name, e)
                    continue

            # Capability check: skip providers that cannot handle request
            if not _provider_can_handle(registry, info.name, duration_s, image_url):
                continue

            provider = registry.get(info.name)
            if provider is None:
                continue
            cost = provider.estimate_cost(dur)
            if cost < best_cost:
                best_cost = cost
                best_name = info.name

        if best_name is None:
            raise RuntimeError("No available providers")
        return best_name


class FastestSelector(ProviderSelector):
    """Select the provider with lowest average latency.

    ``average_latency_ms`` is ``None`` until a provider records its first
    success, which is the state of every pod on cold start, after a restart,
    and after a Redis flush. Ranking alone therefore cannot decide the first
    request: the selector also tracks the first provider that passed the
    capability and health checks and falls back to it when no latency sample
    exists yet. Without that fallback a fully healthy fleet reported
    "No available providers" — a fake outage precisely when traffic resumes.

    A genuine outage (nothing available, nothing capable, empty registry)
    still raises, because the fallback is only ever set by a provider that
    already passed both checks.
    """

    def select(
        self,
        registry: VideoProviderRegistry,
        health_monitor: ProviderHealthMonitor | None,
        duration_s: float | None,
        *,
        image_url: str | None = None,
    ) -> str:
        if health_monitor is None:
            # No health data - just pick the first capable provider.
            for info in registry.list_providers():
                if _provider_can_handle(registry, info.name, duration_s, image_url):
                    return info.name
            raise RuntimeError("No available providers")

        measured, unmeasured = _partition_candidates(
            registry, health_monitor, duration_s, image_url
        )

        if measured:
            # ``min`` keeps the first of equal latencies, i.e. registry order,
            # matching the strict ``<`` comparison this replaced.
            return min(measured, key=lambda candidate: candidate[0])[1]

        if unmeasured:
            logger.info(
                "No provider has a latency sample yet; falling back to %s (cold start)",
                unmeasured[0],
            )
            return unmeasured[0]

        raise RuntimeError("No available providers")


class RoundRobinSelector(ProviderSelector):
    """Round-robin across healthy providers."""

    def __init__(self, redis_client: RedisClient | None = None) -> None:
        self._redis = redis_client
        self._counter = 0

    def select(
        self,
        registry: VideoProviderRegistry,
        health_monitor: ProviderHealthMonitor | None,
        duration_s: float | None,
        *,
        image_url: str | None = None,
    ) -> str:
        available = []
        for info in registry.list_providers():
            if health_monitor is not None:
                try:
                    if not health_monitor.is_available(info.name):
                        continue
                except (OSError, RuntimeError) as e:
                    # An UNMEASURABLE provider is not a healthy one. Without this
                    # `continue` the except merely logged and fell through, so a
                    # provider whose availability could not be read stayed in the
                    # candidate list and was selected exactly as if it had been
                    # measured healthy -- then paid. `_availability_and_latency` in
                    # this same module already chose the other direction for the
                    # identical exception tuple (`return False, None`); these three
                    # selectors simply never got it.
                    logger.warning("Health check failed for provider %s: %s", info.name, e)
                    continue
            # Capability check
            if not _provider_can_handle(registry, info.name, duration_s, image_url):
                continue
            available.append(info.name)

        if not available:
            raise RuntimeError("No available providers")

        idx = self._counter % len(available)
        self._counter += 1
        return available[idx]


class FallbackChainSelector(ProviderSelector):
    """Select from an ordered chain, skipping unhealthy providers."""

    def __init__(self, chain: list[str]) -> None:
        self._chain = chain

    def select(
        self,
        registry: VideoProviderRegistry,
        health_monitor: ProviderHealthMonitor | None,
        duration_s: float | None,
        *,
        image_url: str | None = None,
    ) -> str:
        for name in self._chain:
            if registry.get(name) is None:
                continue
            if health_monitor is not None:
                try:
                    if not health_monitor.is_available(name):
                        continue
                except (OSError, RuntimeError) as e:
                    # See the sibling selectors above: an unmeasurable provider must
                    # be skipped, not silently treated as measured-healthy and paid.
                    logger.warning("Health check failed for provider %s: %s", name, e)
                    continue
            # Capability check
            if not _provider_can_handle(registry, name, duration_s, image_url):
                continue
            return name

        raise RuntimeError("No available providers in fallback chain")


def create_selector(
    strategy: str,
    default_provider: str = "runway",
    fallback_chain: list[str] | None = None,
    redis_client: RedisClient | None = None,
) -> ProviderSelector:
    """Create a provider selector based on strategy name.

    Args:
        strategy: One of 'manual', 'cheapest', 'fastest', 'round_robin', 'fallback_chain'.
        default_provider: Default provider for the manual strategy.
        fallback_chain: Ordered provider names for the fallback_chain strategy.
        redis_client: Optional Redis client for stateful strategies.

    Returns:
        Configured ProviderSelector instance.
    """
    if strategy == "cheapest":
        return CheapestSelector()
    if strategy == "fastest":
        return FastestSelector()
    if strategy == "round_robin":
        return RoundRobinSelector(redis_client=redis_client)
    if strategy == "fallback_chain":
        return FallbackChainSelector(chain=fallback_chain or [])
    return ManualSelector(default_provider=default_provider)
