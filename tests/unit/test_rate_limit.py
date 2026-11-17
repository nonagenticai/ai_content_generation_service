"""Unit tests for rate limiting middleware (Phase 5)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.middleware.rate_limit import _SKIP_PATHS, RateLimitMiddleware


def _make_request(path="/api/content/videos/generate", method="POST", tenant_id="t1"):
    request = MagicMock()
    request.url.path = path
    request.method = method
    request.state.tenant_id = tenant_id
    return request


def test_classify_generate():
    mw = RateLimitMiddleware(app=MagicMock())
    assert mw._classify_endpoint("/api/content/videos/generate", "POST") == "generate"


def test_classify_batch():
    mw = RateLimitMiddleware(app=MagicMock())
    assert mw._classify_endpoint("/api/content/videos/generate/batch", "POST") == "batch"


def test_classify_jobs():
    mw = RateLimitMiddleware(app=MagicMock())
    assert mw._classify_endpoint("/api/content/videos/jobs/123", "GET") == "status"


def test_classify_preview():
    mw = RateLimitMiddleware(app=MagicMock())
    assert mw._classify_endpoint("/api/content/videos/preview", "POST") == "generate"


def test_skip_health():
    mw = RateLimitMiddleware(app=MagicMock())
    assert mw._classify_endpoint("/health", "GET") is None


def test_skip_root():
    mw = RateLimitMiddleware(app=MagicMock())
    assert mw._classify_endpoint("/", "GET") is None


def test_skip_metrics():
    mw = RateLimitMiddleware(app=MagicMock())
    assert mw._classify_endpoint("/metrics", "GET") is None


async def test_no_redis_passes_through():
    mw = RateLimitMiddleware(app=MagicMock(), redis_client=None)
    request = _make_request()
    call_next = AsyncMock(return_value=MagicMock())
    await mw.dispatch(request, call_next)
    call_next.assert_called_once()


async def test_redis_error_passes_through():
    redis = AsyncMock()
    redis.eval_script = AsyncMock(side_effect=OSError("connection error"))
    mw = RateLimitMiddleware(app=MagicMock(), redis_client=redis)
    request = _make_request()
    call_next = AsyncMock(return_value=MagicMock(headers={}))
    await mw.dispatch(request, call_next)
    call_next.assert_called_once()


def test_skip_paths():
    assert "/" in _SKIP_PATHS
    assert "/health" in _SKIP_PATHS
    assert "/docs" in _SKIP_PATHS
    assert "/metrics" in _SKIP_PATHS
