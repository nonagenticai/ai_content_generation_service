"""Unit tests for health check endpoints."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.responses import JSONResponse

import src.api.health as _health_module
from src.api.health import health_check, health_live, health_ready


@pytest.fixture(autouse=True)
def _reset_health_cache():
    """Clear the in-memory TTL cache before each test."""
    _health_module._health_cache = None
    _health_module._health_cache_time = 0.0
    _health_module._ready_cache = None
    _health_module._ready_cache_time = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_engine_healthy() -> AsyncMock:
    """Return an AsyncMock engine whose .connect() is an async context manager."""
    mock_conn = AsyncMock()

    @asynccontextmanager
    async def _connect():
        yield mock_conn

    mock_engine = AsyncMock()
    mock_engine.connect = _connect
    return mock_engine


def _as_dict(result: dict[str, Any] | JSONResponse) -> dict[str, Any]:
    """Assert result is a dict and return it with proper typing for subscript access."""
    assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
    return cast(dict[str, Any], result)


def _registry(names: list[str]) -> Any:
    """A stand-in for ``VideoProviderRegistry`` exposing only ``provider_names``."""
    return SimpleNamespace(provider_names=names)


def _request(
    *, storage_healthy: bool | None = True, video_providers: list[str] | None = None
) -> Any:
    """A Request stand-in carrying the app's INITIALISED storage on ``app.state``.

    Deliberately not a patch of ``StorageService``. The previous helper patched the
    constructor, which is precisely why the defect this file now guards was invisible:
    production built a FRESH ``StorageService()`` per request whose ``_minio_client`` is
    None until ``initialize()`` runs at startup, so the real probe always answered
    "unhealthy" — while the tests, holding a mock in the constructor's place, always saw
    "healthy". The mock proved the mock. Routing through ``app.state.storage_service``
    is the same lookup production performs.

    ``storage_healthy=None`` models startup never having reached ``_init_storage``.
    """
    registry = _registry(["stub"] if video_providers is None else video_providers)
    request = MagicMock()
    if storage_healthy is None:
        request.app.state = SimpleNamespace(provider_registry=registry)
        return request
    storage = AsyncMock()
    storage.health_check.return_value = storage_healthy
    request.app.state = SimpleNamespace(storage_service=storage, provider_registry=registry)
    return request


def _as_error_body(result: dict[str, Any] | JSONResponse, expected_code: int) -> dict[str, Any]:
    """Assert result is a JSONResponse with `expected_code` and return its decoded body."""
    assert isinstance(result, JSONResponse), f"Expected JSONResponse, got {type(result).__name__}"
    assert result.status_code == expected_code
    return cast(dict[str, Any], json.loads(bytes(result.body)))


# ===========================================================================
# /health/live
# ===========================================================================


class TestHealthLive:
    """Liveness probe must always succeed."""

    async def test_returns_ok(self):
        result = await health_live()
        assert result == {"status": "ok"}

    async def test_returns_exact_structure(self):
        """Liveness response contains only the status key."""
        result = await health_live()
        assert list(result.keys()) == ["status"]
        assert result["status"] == "ok"


# ===========================================================================
# /health  (combined health check)
# ===========================================================================


class TestHealthCheck:
    """Tests for the combined health_check endpoint."""

    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_all_healthy(self, mock_get_engine, mock_get_redis):
        # DB
        mock_get_engine.return_value = _mock_engine_healthy()

        # Redis
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_get_redis.return_value = mock_redis

        raw = await health_check(_request())

        result = _as_dict(raw)
        assert result["status"] == "healthy"
        assert result["components"]["database"] == "healthy"
        assert result["components"]["redis"] == "healthy"
        assert result["components"]["api"] == "healthy"
        assert result["components"]["storage"] == "healthy"
        assert result["components"]["video_providers"] == "healthy"
        # Verify all expected component keys are present
        assert set(result["components"].keys()) == {
            "database",
            "redis",
            "storage",
            "video_providers",
            "api",
        }

    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_db_unhealthy(self, mock_get_engine, mock_get_redis):
        # DB raises
        mock_get_engine.side_effect = OSError("connection refused")

        # Redis healthy
        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis

        raw = await health_check(_request())

        # The DB is a fatal dependency: unhealthy + 503, not degraded/200.
        body = _as_error_body(raw, 503)
        assert body["status"] == "unhealthy"
        assert body["components"]["database"].startswith("unhealthy")
        assert "connection refused" in body["components"]["database"]

    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_storage_unhealthy_drives_overall_status(self, mock_get_engine, mock_get_redis):
        """A failing storage check must surface on `.status`, not only in components.

        Regression guard: storage was computed and reported under `components`
        but excluded from `overall`, so MinIO down yielded
        `status: "healthy"` and no alert keyed on `.status` ever fired.
        """
        mock_get_engine.return_value = _mock_engine_healthy()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_get_redis.return_value = mock_redis

        raw = await health_check(_request(storage_healthy=False))

        body = _as_error_body(raw, 503)
        assert body["status"] == "unhealthy"
        assert body["components"]["storage"] == "unhealthy"
        # The other dependencies stay truthful.
        assert body["components"]["database"] == "healthy"
        assert body["components"]["redis"] == "healthy"

    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_storage_raising_is_reported_not_propagated(
        self, mock_get_engine, mock_get_redis
    ):
        """A non-OSError from the storage client is caught and reported as unhealthy."""
        mock_get_engine.return_value = _mock_engine_healthy()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_get_redis.return_value = mock_redis

        request = _request()
        request.app.state.storage_service.health_check.side_effect = ValueError(
            "minio bucket policy denied"
        )
        raw = await health_check(request)

        body = _as_error_body(raw, 503)
        assert body["status"] == "unhealthy"
        assert body["components"]["storage"].startswith("unhealthy")
        assert "minio bucket policy denied" in body["components"]["storage"]

    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_storage_probe_uses_the_apps_initialised_instance(
        self, mock_get_engine, mock_get_redis
    ):
        """The probe must read `app.state.storage_service`, not construct a new one.

        This is the regression guard for a defect that was live in production:
        `health_check` did `StorageService()` per request, and a fresh instance has
        `_minio_client = None` until `initialize()` runs at startup — so
        `check_health()` returned False unconditionally. Verified on the running pod:
        `_minio_client is None` was True while endpoint, bucket and credentials were
        all valid and MinIO was fine.

        The old tests could not see it: they patched the CONSTRUCTOR, so the mock stood
        in for the very object whose initialisation was the bug.
        """
        mock_get_engine.return_value = _mock_engine_healthy()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_get_redis.return_value = mock_redis

        request = _request()
        raw = await health_check(request)

        _as_dict(raw)
        # The app's own instance was consulted...
        request.app.state.storage_service.health_check.assert_awaited_once()
        # ...and no new StorageService was built to answer the probe.
        with patch("src.services.storage_service.StorageService") as ctor:
            _health_module._health_cache = None
            await health_check(_request())
            ctor.assert_not_called()

    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_uninitialised_storage_is_unhealthy_not_healthy(
        self, mock_get_engine, mock_get_redis
    ):
        """Startup never reaching `_init_storage` must not read as healthy."""
        mock_get_engine.return_value = _mock_engine_healthy()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_get_redis.return_value = mock_redis

        raw = await health_check(_request(storage_healthy=None))

        body = _as_error_body(raw, 503)
        assert body["status"] == "unhealthy"
        assert "not initialised" in body["components"]["storage"]

    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_redis_not_configured(self, mock_get_engine, mock_get_redis):
        # DB healthy
        mock_get_engine.return_value = _mock_engine_healthy()

        # Redis returns None
        mock_get_redis.return_value = None

        raw = await health_check(_request())

        result = _as_dict(raw)
        assert result["components"]["redis"] == "not configured"
        # "not configured" != "healthy" so overall status becomes "degraded"
        assert result["status"] == "degraded"

    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_redis_unhealthy(self, mock_get_engine, mock_get_redis):
        # DB healthy
        mock_get_engine.return_value = _mock_engine_healthy()

        # Redis ping raises
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = OSError("redis down")
        mock_get_redis.return_value = mock_redis

        raw = await health_check(_request())

        result = _as_dict(raw)
        assert result["components"]["redis"].startswith("unhealthy")
        assert "redis down" in result["components"]["redis"]
        assert result["status"] == "degraded"

    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    async def test_result_is_cached(self, mock_get_engine, mock_get_redis):
        """Subsequent calls within TTL return cached result."""
        mock_get_engine.return_value = _mock_engine_healthy()
        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis

        result1 = await health_check(_request())
        result2 = await health_check(_request())

        # Same object is returned (cached)
        assert result1 is result2
        # get_engine called only once (second call uses cache)
        mock_get_engine.assert_awaited_once()


# ===========================================================================
# /health/ready  (readiness probe)
# ===========================================================================


class TestHealthReady:
    """Tests for the readiness probe endpoint."""

    @patch("src.api.health.get_engine")
    async def test_db_up(self, mock_get_engine):
        mock_get_engine.return_value = _mock_engine_healthy()

        raw = await health_ready()

        result = _as_dict(raw)
        assert result["ready"] is True
        assert result["status"] == "healthy"
        assert result["checks"]["database"]["status"] == "up"

    @patch("src.api.health.get_engine")
    async def test_db_down_returns_503(self, mock_get_engine):
        mock_get_engine.side_effect = OSError("cannot connect")

        result = await health_ready()

        assert isinstance(result, JSONResponse)
        assert result.status_code == 503

    @patch("src.api.health.get_engine")
    async def test_ready_result_is_cached(self, mock_get_engine):
        """Ready probe results are cached within TTL."""
        mock_get_engine.return_value = _mock_engine_healthy()

        result1 = await health_ready()
        result2 = await health_ready()

        assert result1 is result2
        mock_get_engine.assert_awaited_once()

    @patch("src.api.health.get_engine")
    async def test_failed_probe_is_not_cached_so_recovery_is_immediate(self, mock_get_engine):
        """A recovered pod must report ready on the NEXT probe, not after the TTL.

        The cache used to store failures as well as successes, so a single blip
        pinned the pod to 503 for the whole TTL even though the database was back:
        the second probe short-circuited on the cached failure and never re-ran the
        `SELECT 1`. No time is advanced here -- both calls happen well inside the
        TTL, which is exactly the window the defect lived in.
        """
        mock_get_engine.side_effect = [OSError("cannot connect"), _mock_engine_healthy()]

        first = await health_ready()
        body = _as_error_body(first, 503)
        assert body["ready"] is False

        # Same TTL window, dependency recovered -> must re-probe, not replay the 503.
        second = await health_ready()

        result = _as_dict(second)
        assert result["ready"] is True
        assert result["status"] == "healthy"
        assert result["checks"]["database"]["status"] == "up"
        assert mock_get_engine.await_count == 2

    @patch("src.api.health.get_engine")
    async def test_success_is_still_cached_after_a_failure(self, mock_get_engine):
        """Not caching failures must not disable the success-path TTL."""
        mock_get_engine.side_effect = [OSError("cannot connect"), _mock_engine_healthy()]

        await health_ready()
        second = await health_ready()
        third = await health_ready()

        assert second is third
        assert mock_get_engine.await_count == 2


class TestVideoProvidersHealth:
    """Zero registered video providers must be visible on ``/health``.

    Regression guard for the 2026-08-20 prod defect: the image shipped without
    ffmpeg, the stub provider (``VIDEO_PROVIDER=stub``, every cloud key blank)
    failed its health check, the registry ended up EMPTY -- and both replicas
    reported themselves healthy and Ready anyway. The only signal was a
    start-up WARNING nothing polls.
    """

    @pytest.mark.asyncio
    async def test_no_video_providers_reports_unhealthy_503(self):
        """An empty provider registry must make /health 503, not 'healthy'."""
        request = _request(video_providers=[])
        with (
            patch("src.api.health.get_engine", return_value=_mock_engine_healthy()),
            patch("src.api.health.get_redis_client", return_value=None),
        ):
            result = await health_check(request)

        body = _as_error_body(result, 503)
        assert body["status"] == "unhealthy"
        assert body["components"]["video_providers"] == "unhealthy: no video providers registered"

    @pytest.mark.asyncio
    async def test_uninitialised_registry_reports_unhealthy(self):
        """Start-up never reaching provider init is a fault, not an unknown."""
        request = MagicMock()
        request.app.state = SimpleNamespace(storage_service=AsyncMock())
        request.app.state.storage_service.health_check.return_value = True
        with (
            patch("src.api.health.get_engine", return_value=_mock_engine_healthy()),
            patch("src.api.health.get_redis_client", return_value=None),
        ):
            result = await health_check(request)

        body = _as_error_body(result, 503)
        assert (
            body["components"]["video_providers"]
            == "unhealthy: video provider registry not initialised"
        )

    @pytest.mark.asyncio
    async def test_registered_provider_reports_healthy(self):
        """With a provider registered the component must not drag /health down."""
        request = _request(video_providers=["stub"])
        redis = AsyncMock()
        redis.ping = AsyncMock()
        with (
            patch("src.api.health.get_engine", return_value=_mock_engine_healthy()),
            patch("src.api.health.get_redis_client", return_value=redis),
        ):
            result = await health_check(request)

        body = _as_dict(result)
        assert body["components"]["video_providers"] == "healthy"
        assert body["status"] == "healthy"
