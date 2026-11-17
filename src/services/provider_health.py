"""Provider health monitoring with auto-disable and cooldown.

Supports optional Redis persistence so health state survives pod restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)

_REDIS_KEY_PREFIX = "content_generation:provider_health:"
_REDIS_TTL_SECONDS = 3600  # 1 hour -- covers cooldown period and then some


@dataclass
class ProviderHealthState:
    """Health state for a single provider."""

    is_healthy: bool = True
    consecutive_failures: int = 0
    total_successes: int = 0
    total_latency_ms: float = 0.0
    disabled_at: float | None = None
    cooldown_until: float | None = None

    @property
    def average_latency_ms(self) -> float | None:
        if self.total_successes == 0:
            return None
        return self.total_latency_ms / self.total_successes

    def to_dict(self) -> dict:
        """Serialize to a dict suitable for JSON / Redis storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ProviderHealthState:
        """Deserialize from a dict (e.g. loaded from Redis)."""
        return cls(
            is_healthy=bool(data.get("is_healthy", True)),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            total_successes=int(data.get("total_successes", 0)),
            total_latency_ms=float(data.get("total_latency_ms", 0.0)),
            disabled_at=data.get("disabled_at"),
            cooldown_until=data.get("cooldown_until"),
        )


class ProviderHealthMonitor:
    """Tracks provider health, auto-disables after consecutive failures.

    When a ``redis_client`` is provided the monitor persists every state
    change to a Redis hash so that health history survives process
    restarts.  On first access of a provider the monitor attempts to
    load existing state from Redis before falling back to a fresh
    ``ProviderHealthState``.
    """

    def __init__(
        self,
        max_consecutive_failures: int = 5,
        cooldown_seconds: int = 300,
        redis_client: RedisClient | None = None,
    ) -> None:
        self._max_failures = max_consecutive_failures
        self._cooldown_seconds = cooldown_seconds
        self._redis = redis_client
        self._states: dict[str, ProviderHealthState] = {}
        # Strong refs to in-flight persist tasks so they aren't GC'd mid-write.
        # The event loop only holds weak refs to tasks from loop.create_task;
        # dropping the ref can vanish the task silently. Same pattern as
        # image_gen_backend/src/routers/image_routes.py.
        self._background_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Redis persistence helpers
    # ------------------------------------------------------------------

    def _redis_key(self, provider: str) -> str:
        return f"{_REDIS_KEY_PREFIX}{provider}"

    async def _persist_state(self, provider: str, state: ProviderHealthState) -> None:
        """Write state to Redis (best-effort, never raises)."""
        if self._redis is None:
            return
        try:
            key = self._redis_key(provider)
            payload = json.dumps(state.to_dict())
            await self._redis.set(key, payload, ttl=_REDIS_TTL_SECONDS)
        except OSError:
            logger.debug("Failed to persist health state for %s", provider, exc_info=True)

    async def _load_state(self, provider: str) -> ProviderHealthState | None:
        """Try to load state from Redis. Returns None on miss or error."""
        if self._redis is None:
            return None
        try:
            key = self._redis_key(provider)
            raw = await self._redis.get(key)
            if raw is not None:
                data = raw if isinstance(raw, dict) else json.loads(raw)
                return ProviderHealthState.from_dict(data)
        except OSError, KeyError, ValueError:
            logger.debug("Failed to load health state for %s", provider, exc_info=True)
        return None

    async def load_all_from_redis(self, provider_names: list[str]) -> None:
        """Bulk-load persisted health state on startup.

        Call this once after constructing the monitor so the in-memory
        cache is warm before the first request arrives.
        """
        for name in provider_names:
            loaded = await self._load_state(name)
            if loaded is not None:
                self._states[name] = loaded
                logger.info(
                    "Loaded health state for %s from Redis (healthy=%s, failures=%d)",
                    name,
                    loaded.is_healthy,
                    loaded.consecutive_failures,
                )

    # ------------------------------------------------------------------
    # Core state access
    # ------------------------------------------------------------------

    def _get_state(self, provider: str) -> ProviderHealthState:
        if provider not in self._states:
            self._states[provider] = ProviderHealthState()
        return self._states[provider]

    def record_success(self, provider: str, latency_ms: float) -> None:
        """Record a successful generation."""
        state = self._get_state(provider)
        state.consecutive_failures = 0
        state.is_healthy = True
        state.disabled_at = None
        state.cooldown_until = None
        state.total_successes += 1
        state.total_latency_ms += latency_ms
        self._schedule_persist(provider, state)

    def record_failure(self, provider: str) -> None:
        """Record a failed generation."""
        state = self._get_state(provider)
        state.consecutive_failures += 1
        if state.consecutive_failures >= self._max_failures:
            now = time.time()
            state.is_healthy = False
            state.disabled_at = now
            state.cooldown_until = now + self._cooldown_seconds
            logger.warning(
                "Provider %s disabled after %d failures, cooldown until %s",
                provider,
                state.consecutive_failures,
                state.cooldown_until,
            )
        self._schedule_persist(provider, state)

    def is_available(self, provider: str) -> bool:
        """Check if a provider is available for use."""
        state = self._get_state(provider)
        if state.is_healthy:
            return True
        # Check cooldown expiry
        if state.cooldown_until and time.time() >= state.cooldown_until:
            state.is_healthy = True
            state.consecutive_failures = 0
            state.disabled_at = None
            state.cooldown_until = None
            logger.info("Provider %s re-enabled after cooldown", provider)
            self._schedule_persist(provider, state)
            return True
        return False

    def get_health(self, provider: str) -> ProviderHealthState:
        """Get health state for a provider."""
        self.check_cooldowns()
        return self._get_state(provider)

    def check_cooldowns(self) -> None:
        """Re-enable providers whose cooldowns have expired."""
        now = time.time()
        for name, state in self._states.items():
            if not state.is_healthy and state.cooldown_until and now >= state.cooldown_until:
                state.is_healthy = True
                state.consecutive_failures = 0
                state.disabled_at = None
                state.cooldown_until = None
                logger.info("Provider %s re-enabled after cooldown", name)
                self._schedule_persist(name, state)

    # ------------------------------------------------------------------
    # Fire-and-forget persistence
    # ------------------------------------------------------------------

    def _schedule_persist(self, provider: str, state: ProviderHealthState) -> None:
        """Schedule an async Redis write without blocking the caller.

        ``record_success`` / ``record_failure`` are sync methods so we
        cannot ``await`` here.  Instead we create a background task.
        If there is no running event loop (e.g. unit tests) we silently
        skip the write -- the in-memory state is still authoritative.
        """
        if self._redis is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop -- skip (common in sync tests)
            return
        task = loop.create_task(self._persist_state(provider, state))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
