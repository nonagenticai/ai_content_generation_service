"""Token bucket rate limiting middleware using Redis."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

if TYPE_CHECKING:
    from src.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)

# Redis Lua script for atomic token bucket
_RATE_LIMIT_LUA_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])

local data = redis.call('GET', key)
local tokens, last_time

if data then
    local parts = {}
    for part in data:gmatch("[^:]+") do
        parts[#parts + 1] = part
    end
    tokens = tonumber(parts[1])
    last_time = tonumber(parts[2])
else
    tokens = capacity
    last_time = now
end

local elapsed = now - last_time
local refill = elapsed * rate / 60.0
tokens = math.min(capacity, tokens + refill)

if tokens >= 1 then
    tokens = tokens - 1
    redis.call('SETEX', key, 120, tokens .. ':' .. now)
    return {1, math.floor(tokens)}
else
    redis.call('SETEX', key, 120, tokens .. ':' .. now)
    local wait = (1 - tokens) * 60.0 / rate
    return {0, math.ceil(wait)}
end
"""

_SKIP_PATHS = {
    "/",
    "/health",
    "/health/ready",
    "/health/live",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-tenant token bucket rate limiter using Redis.

    Uses a Redis-backed Lua script for atomic token bucket
    enforcement. Rate limits are configured per endpoint category
    (generate, batch, status).

    Attributes:
        _redis: Redis client for rate limit state.
        _limits: Map of endpoint category to requests-per-minute.
    """

    def __init__(
        self,
        app: ASGIApp,
        redis_client: RedisClient | None = None,
        generate_rpm: int = 10,
        status_rpm: int = 60,
        batch_rpm: int = 5,
    ) -> None:
        super().__init__(app)
        self._redis = redis_client
        self._limits = {
            "generate": generate_rpm,
            "status": status_rpm,
            "batch": batch_rpm,
        }

    def _classify_endpoint(self, path: str, method: str) -> str | None:
        """Classify request to a rate limit bucket."""
        if path in _SKIP_PATHS:
            return None
        if (
            ("/generate/batch" in path) or path.endswith("/concept-images/batches")
        ) and method == "POST":
            return "batch"
        if ("/generate" in path or path == "/api/concept-images") and method == "POST":
            return "generate"
        if "/jobs/" in path or "/batches/" in path or "/api/concept-images/" in path:
            return "status"
        if "/preview" in path and method == "POST":
            return "generate"
        return "status"

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if self._redis is None:
            return await call_next(request)

        bucket = self._classify_endpoint(request.url.path, request.method)
        if bucket is None:
            return await call_next(request)

        tenant_id = getattr(request.state, "tenant_id", "anonymous")
        rpm = self._limits.get(bucket, 60)
        key = f"content_generation:ratelimit:{tenant_id}:{bucket}"

        # ⛔ `call_next` is DELIBERATELY outside this try, and must stay outside it.
        #
        # It used to be inside, together with the two `except ... : return await
        # call_next(request)` fall-open handlers below. Those handlers are written for
        # a failure of the RATE-LIMIT CHECK — a Redis blip on `eval_script` — where
        # re-dispatching is the correct fail-open. With `call_next` inside their reach
        # they also caught anything the ROUTE HANDLER raised, and then ran the entire
        # request A SECOND TIME.
        #
        # On this service that is a money bug, not a retry: `POST /api/content/videos/
        # generate` and `/api/content/images/generate` create a job, register a
        # durable dispatch, take a quota concurrency slot and spawn a task that CALLS
        # A PAID PROVIDER. One client request, one token consumed, two paid
        # generations — and the only trace is a log line reading "Rate limit check
        # failed", which sends the reader to Redis instead of the handler. `ValueError`
        # and `RuntimeError` are ordinary handler-level exceptions here, so the
        # trigger is routine, not exotic.
        #
        # So: the try covers ONLY the measurement. Dispatch happens exactly once on
        # every path through this method.
        try:
            now = time.time()
            result = await self._redis.eval_script(
                _RATE_LIMIT_LUA_SCRIPT, keys=[key], args=[now, rpm, rpm]
            )
            allowed = int(result[0])
            value = int(result[1])
        except RedisError as exc:
            logger.warning("Rate limit Redis error, allowing request: %s", exc)
            return await call_next(request)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("Rate limit check failed, allowing request: %s", exc)
            return await call_next(request)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded", "retry_after": value},
                headers={
                    "Retry-After": str(value),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(value)
        return response
