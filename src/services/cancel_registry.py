"""Cross-pod job cancellation via a Redis flag.

The old cancel path looked up a pod-local ``_active_tasks`` entry, so a cancel
request that landed on replica B could never reach a task running on replica A.

This registry writes a short-lived Redis key that the job's processor polls at
its checkpoints, so a cancel reaches the running task on ANY pod. It degrades
safely: with no Redis the flag is a no-op and cancellation falls back to the
pod-local task / still-``pending`` DB transition.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)

# TTL bounds the flag's lifetime; it only needs to outlive an in-flight job.
_CANCEL_TTL_S = 3600


class CancelRegistry:
    """Redis-backed cross-pod cancellation flags."""

    def __init__(self, redis_client: RedisClient | None = None) -> None:
        self._redis = redis_client

    def _key(self, job_id: str) -> str:
        return f"content_generation:cancel:{job_id}"

    async def request_cancel(self, job_id: str, ttl: int = _CANCEL_TTL_S) -> bool:
        """Set the cancel flag for ``job_id``. Returns True if the flag was set."""
        if self._redis is None:
            return False
        try:
            return await self._redis.set(self._key(job_id), True, ttl=ttl)
        except OSError:
            logger.debug("Failed to set cancel flag for job %s", job_id, exc_info=True)
            return False

    async def is_cancelled(self, job_id: str) -> bool:
        """Return True if a cancel has been requested for ``job_id`` (any pod)."""
        if self._redis is None:
            return False
        try:
            return bool(await self._redis.get(self._key(job_id)))
        except OSError:
            logger.debug("Failed to read cancel flag for job %s", job_id, exc_info=True)
            return False

    async def clear(self, job_id: str) -> None:
        """Remove the cancel flag (best-effort; the TTL also reaps it)."""
        if self._redis is None:
            return
        try:
            await self._redis.delete(self._key(job_id))
        except OSError:
            logger.debug("Failed to clear cancel flag for job %s", job_id, exc_info=True)
