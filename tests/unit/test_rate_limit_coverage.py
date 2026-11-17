"""Tests for rate_limit.py to cover Redis rate limiting dispatch paths.

Missing lines: 99 (fallback to 'status'), 109 (skip None bucket),
121-136 (Redis eval_script: allowed and denied paths).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from starlette.responses import JSONResponse

from src.middleware.rate_limit import RateLimitMiddleware


def _make_request(
    path="/api/content/videos/generate",
    method="POST",
    tenant_id="t1",
):
    request = MagicMock()
    request.url.path = path
    request.method = method
    request.state.tenant_id = tenant_id
    return request


# ============================================================================
# _classify_endpoint edge cases
# ============================================================================


def test_classify_unknown_path_returns_status():
    """An unknown path (not generate, batch, jobs, batches) -> 'status'."""
    mw = RateLimitMiddleware(app=MagicMock())
    result = mw._classify_endpoint("/api/content/videos/something/else", "GET")
    assert result == "status"


def test_classify_batches_path():
    """Batches path classified as 'status'."""
    mw = RateLimitMiddleware(app=MagicMock())
    result = mw._classify_endpoint("/api/content/videos/batches/b1/status", "GET")
    assert result == "status"


# ============================================================================
# dispatch with Redis: allowed and rate limited
# ============================================================================


async def test_dispatch_skip_path():
    """Health check paths bypass rate limiting."""
    redis = AsyncMock()
    mw = RateLimitMiddleware(app=MagicMock(), redis_client=redis)
    request = _make_request(path="/health", method="GET")
    response = MagicMock()
    call_next = AsyncMock(return_value=response)

    result = await mw.dispatch(request, call_next)

    call_next.assert_called_once_with(request)
    assert result is response
    # Redis should NOT have been consulted for skip paths
    redis.eval_script.assert_not_awaited()


async def test_dispatch_redis_allowed():
    """When Redis script returns [1, remaining], request is allowed with header."""
    redis = AsyncMock()
    redis.eval_script = AsyncMock(return_value=[1, 9])

    mw = RateLimitMiddleware(app=MagicMock(), redis_client=redis, generate_rpm=10)
    request = _make_request(path="/api/content/videos/generate", method="POST")

    response = MagicMock()
    response.headers = {}
    call_next = AsyncMock(return_value=response)

    result = await mw.dispatch(request, call_next)

    call_next.assert_called_once_with(request)
    assert result is response
    assert response.headers["X-RateLimit-Remaining"] == "9"
    # Redis eval_script called with correct key pattern and RPM args
    redis.eval_script.assert_awaited_once()
    eval_kwargs = redis.eval_script.call_args
    assert eval_kwargs.kwargs["keys"] == ["content_generation:ratelimit:t1:generate"]
    assert eval_kwargs.kwargs["args"][1] == 10  # generate_rpm
    assert eval_kwargs.kwargs["args"][2] == 10  # capacity = generate_rpm


async def test_dispatch_redis_rate_limited():
    """When Redis script returns [0, wait_seconds], return 429."""
    redis = AsyncMock()
    redis.eval_script = AsyncMock(return_value=[0, 5])

    mw = RateLimitMiddleware(app=MagicMock(), redis_client=redis, generate_rpm=10)
    request = _make_request(path="/api/content/videos/generate", method="POST")
    call_next = AsyncMock(return_value=MagicMock())

    result = await mw.dispatch(request, call_next)

    assert isinstance(result, JSONResponse)
    assert result.status_code == 429
    call_next.assert_not_called()
    # Response headers include Retry-After and zero remaining
    assert result.headers["Retry-After"] == "5"
    assert result.headers["X-RateLimit-Remaining"] == "0"


async def test_dispatch_redis_exception_falls_through():
    """When Redis eval_script raises, request is allowed (fail-open)."""
    redis = AsyncMock()
    redis.eval_script = AsyncMock(side_effect=OSError("Redis connection error"))

    mw = RateLimitMiddleware(app=MagicMock(), redis_client=redis)
    request = _make_request(path="/api/content/videos/generate", method="POST")

    response = MagicMock()
    call_next = AsyncMock(return_value=response)

    result = await mw.dispatch(request, call_next)

    call_next.assert_called_once_with(request)
    # Fail-open: the original response is returned, not a 429
    assert result is response
    # Redis was indeed called (and failed)
    redis.eval_script.assert_awaited_once()


async def test_dispatch_status_endpoint():
    """Status endpoint uses 'status' bucket."""
    redis = AsyncMock()
    redis.eval_script = AsyncMock(return_value=[1, 59])

    mw = RateLimitMiddleware(app=MagicMock(), redis_client=redis, status_rpm=60)
    request = _make_request(path="/api/content/videos/jobs/j1", method="GET")

    response = MagicMock()
    response.headers = {}
    call_next = AsyncMock(return_value=response)

    result = await mw.dispatch(request, call_next)

    call_next.assert_called_once_with(request)
    assert result is response
    assert response.headers["X-RateLimit-Remaining"] == "59"
    # Verify the 'status' bucket key and status_rpm=60 were used
    eval_kwargs = redis.eval_script.call_args
    assert eval_kwargs.kwargs["keys"] == ["content_generation:ratelimit:t1:status"]
    assert eval_kwargs.kwargs["args"][1] == 60  # status_rpm


async def test_dispatch_batch_endpoint():
    """Batch endpoint uses 'batch' bucket."""
    redis = AsyncMock()
    redis.eval_script = AsyncMock(return_value=[1, 4])

    mw = RateLimitMiddleware(app=MagicMock(), redis_client=redis, batch_rpm=5)
    request = _make_request(path="/api/content/videos/generate/batch", method="POST")

    response = MagicMock()
    response.headers = {}
    call_next = AsyncMock(return_value=response)

    result = await mw.dispatch(request, call_next)

    call_next.assert_called_once_with(request)
    assert result is response
    assert response.headers["X-RateLimit-Remaining"] == "4"
    # Verify the 'batch' bucket key and batch_rpm=5 were used
    eval_kwargs = redis.eval_script.call_args
    assert eval_kwargs.kwargs["keys"] == ["content_generation:ratelimit:t1:batch"]
    assert eval_kwargs.kwargs["args"][1] == 5  # batch_rpm
