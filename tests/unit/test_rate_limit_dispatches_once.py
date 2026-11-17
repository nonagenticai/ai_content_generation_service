"""One client request must reach the route handler exactly ONCE.

``RateLimitMiddleware.dispatch`` fails OPEN when its own Redis check breaks: it
logs and re-dispatches. That is correct for a failure of the MEASUREMENT. It used
to be written with ``call_next`` INSIDE the same ``try``, so the two fall-open
handlers also caught whatever the ROUTE HANDLER raised -- and re-ran the whole
request.

On this service the generate routes create a job, register a durable dispatch,
take a quota concurrency slot and spawn a task that calls a PAID provider. A
handler-level ``ValueError``/``RuntimeError`` therefore bought the clip twice off
one request, consuming one rate-limit token, logged as "Rate limit check failed"
-- a line that sends the reader to Redis rather than to the handler.

The provider-selector tests below pin the sibling shape: an unmeasurable health
check must not be scored as a healthy one.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.exceptions import RedisError

from src.middleware.rate_limit import RateLimitMiddleware


def _request(path: str = "/api/content/videos/generate", method: str = "POST") -> MagicMock:
    request = MagicMock()
    request.url.path = path
    request.method = method
    request.state.tenant_id = "t1"
    return request


def _mw(*, allowed: int = 1, remaining: int = 42, redis_raises: bool = False):
    mw = RateLimitMiddleware(app=MagicMock())
    redis = MagicMock()
    if redis_raises:
        redis.eval_script = AsyncMock(side_effect=RedisError("redis down"))
    else:
        redis.eval_script = AsyncMock(return_value=[allowed, remaining])
    mw._redis = redis
    return mw


class TestAHandlerErrorIsNotAReDispatch:
    @pytest.mark.parametrize("exc", [ValueError("bad payload"), RuntimeError("boom"), OSError()])
    async def test_handler_exception_runs_the_route_once_and_propagates(self, exc) -> None:
        calls: list[object] = []

        async def call_next(request):
            calls.append(request)
            raise exc

        mw = _mw()
        with pytest.raises(type(exc)):
            await mw.dispatch(_request(), call_next)

        assert len(calls) == 1, (
            f"the route handler ran {len(calls)} times for ONE request -- a "
            f"{type(exc).__name__} from the handler was caught by the rate limiter's "
            f"fail-open branch and the paid generation was dispatched again"
        )

    async def test_a_redis_error_still_fails_open_and_dispatches_once(self) -> None:
        """Control: the fail-open the handlers were actually written for."""
        calls: list[object] = []

        async def call_next(request):
            calls.append(request)
            return MagicMock(headers={})

        mw = _mw(redis_raises=True)
        await mw.dispatch(_request(), call_next)
        assert len(calls) == 1

    async def test_an_allowed_request_still_dispatches_once_with_the_header(self) -> None:
        """Control: the happy path is unchanged."""
        calls: list[object] = []

        async def call_next(request):
            calls.append(request)
            return MagicMock(headers={})

        mw = _mw(allowed=1, remaining=7)
        response = await mw.dispatch(_request(), call_next)
        assert len(calls) == 1
        assert response.headers["X-RateLimit-Remaining"] == "7"

    async def test_a_throttled_request_never_reaches_the_handler(self) -> None:
        """Control: the 429 path must not dispatch at all."""
        calls: list[object] = []

        async def call_next(request):
            calls.append(request)
            return MagicMock(headers={})

        mw = _mw(allowed=0, remaining=30)
        response = await mw.dispatch(_request(), call_next)
        assert response.status_code == 429
        assert calls == []


# ---------------------------------------------------------------------------
# The sibling shape: an UNMEASURABLE provider must not be scored as healthy.
# ---------------------------------------------------------------------------


def _registry_with(names: list[str]):
    from src.services.providers.base import VideoProviderRegistry

    registry = VideoProviderRegistry()
    for i, name in enumerate(names):
        provider = MagicMock()
        provider.name = name
        provider.estimate_cost.return_value = 1.0 + i
        provider.get_models.return_value = []
        provider.get_styles.return_value = []
        provider.get_max_duration_s.return_value = None
        provider.supports_image_to_video.return_value = True
        provider.can_handle = lambda **_kw: True
        registry.register(provider)
    return registry


def _monitor_raising_for(bad: str) -> MagicMock:
    monitor = MagicMock()

    def _is_available(name: str) -> bool:
        if name == bad:
            raise RuntimeError("health backend unreadable")
        return True

    monitor.is_available = _is_available
    monitor.get_health.return_value = MagicMock(average_latency_ms=10.0)
    return monitor


class TestAnUnmeasurableProviderIsNotAHealthyProvider:
    """`_availability_and_latency` in this same module already returns
    ``(False, None)`` for the identical exception tuple. These three selectors
    merely logged and fell through, so the provider stayed a candidate and was
    then PAID on the strength of a measurement that never happened."""

    def test_cheapest_selector_skips_the_unmeasurable_provider(self) -> None:
        from src.services.provider_selector import CheapestSelector

        # "cheap" is the cheapest, and is the one whose health cannot be read.
        registry = _registry_with(["cheap", "dear"])
        chosen = CheapestSelector().select(registry, _monitor_raising_for("cheap"), 5.0)
        assert chosen == "dear", (
            "the cheapest provider was selected off a health check that RAISED; "
            "an unmeasured provider scored identically to a measured-healthy one"
        )

    def test_round_robin_selector_skips_the_unmeasurable_provider(self) -> None:
        from src.services.provider_selector import RoundRobinSelector

        registry = _registry_with(["a", "b"])
        selector = RoundRobinSelector()
        picks = {selector.select(registry, _monitor_raising_for("a"), 5.0) for _ in range(4)}
        assert picks == {"b"}

    def test_fallback_chain_skips_the_unmeasurable_head_of_the_chain(self) -> None:
        from src.services.provider_selector import FallbackChainSelector

        registry = _registry_with(["first", "second"])
        chosen = FallbackChainSelector(["first", "second"]).select(
            registry, _monitor_raising_for("first"), 5.0
        )
        assert chosen == "second"

    def test_a_readable_monitor_still_selects_the_cheapest(self) -> None:
        """Control: the guard must only fire on an unreadable health check."""
        from src.services.provider_selector import CheapestSelector

        registry = _registry_with(["cheap", "dear"])
        monitor = MagicMock()
        monitor.is_available = lambda _name: True
        monitor.get_health.return_value = MagicMock(average_latency_ms=10.0)
        assert CheapestSelector().select(registry, monitor, 5.0) == "cheap"
