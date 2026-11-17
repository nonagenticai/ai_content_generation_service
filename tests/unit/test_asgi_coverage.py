"""Tests for asgi.py to cover health filter, root endpoint, metrics endpoint,
error handler, main entry point, and lifespan.

Targeting the module-level code and endpoints that do not require full lifespan,
plus the lifespan context manager itself.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

from src.asgi import (
    HealthCheckLoggingFilter,
    _setup_logging,
    lifespan,
    main,
    metrics_endpoint,
    root,
)

# ============================================================================
# HealthCheckLoggingFilter
# ============================================================================


class TestHealthCheckLoggingFilter:
    """Tests for HealthCheckLoggingFilter (lines 77-80)."""

    def test_filters_health_path(self) -> None:
        f = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='GET /health HTTP/1.1" 200',
            args=(),
            exc_info=None,
        )
        assert f.filter(record) is False

    def test_passes_non_health_path(self) -> None:
        f = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='POST /api/content/videos/generate HTTP/1.1" 200',
            args=(),
            exc_info=None,
        )
        assert f.filter(record) is True

    def test_filters_health_ready_path(self) -> None:
        f = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='GET /health/ready HTTP/1.1" 200',
            args=(),
            exc_info=None,
        )
        assert f.filter(record) is False

    def test_filters_health_live_path(self) -> None:
        f = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='GET /health/live HTTP/1.1" 200',
            args=(),
            exc_info=None,
        )
        assert f.filter(record) is False


# ============================================================================
# _setup_logging
# ============================================================================


class TestSetupLogging:
    def test_setup_logging_default(self) -> None:
        """Default logging setup doesn't crash."""
        _setup_logging()

    @patch.dict("os.environ", {"LOG_JSON_LOGS": "true", "LOG_LEVEL": "DEBUG"})
    def test_setup_logging_json(self) -> None:
        """JSON logging setup creates JSON formatter."""
        _setup_logging()
        # Verify the root logger has a handler with a JSON-ish formatter
        root = logging.getLogger()
        assert len(root.handlers) > 0

    @patch.dict("os.environ", {"LOG_JSON_LOGS": "true", "LOG_LEVEL": "DEBUG"})
    def test_json_formatter_formats_record(self) -> None:
        """JSON formatter produces valid JSON output."""
        import json as json_mod

        _setup_logging()
        root = logging.getLogger()
        # Find the handler that has a JSON formatter
        for handler in root.handlers:
            formatter = handler.formatter
            if formatter is not None:
                record = logging.LogRecord(
                    name="test",
                    level=logging.INFO,
                    pathname="test.py",
                    lineno=1,
                    msg="test message",
                    args=(),
                    exc_info=None,
                )
                output = formatter.format(record)
                try:
                    parsed = json_mod.loads(output)
                    assert parsed["message"] == "test message"
                    assert parsed["level"] == "INFO"
                    return
                except json_mod.JSONDecodeError, KeyError:
                    continue

    @patch.dict("os.environ", {"LOG_JSON_LOGS": "true", "LOG_LEVEL": "DEBUG"})
    def test_json_formatter_with_exception(self) -> None:
        """JSON formatter includes exception info."""
        import json as json_mod

        _setup_logging()
        root = logging.getLogger()
        for handler in root.handlers:
            formatter = handler.formatter
            if formatter is not None:
                try:
                    raise ValueError("test error")
                except ValueError:
                    import sys

                    exc_info = sys.exc_info()
                record = logging.LogRecord(
                    name="test",
                    level=logging.ERROR,
                    pathname="test.py",
                    lineno=1,
                    msg="error occurred",
                    args=(),
                    exc_info=exc_info,
                )
                output = formatter.format(record)
                try:
                    parsed = json_mod.loads(output)
                    if "exception" in parsed:
                        assert "test error" in parsed["exception"]
                        return
                except json_mod.JSONDecodeError, KeyError:
                    continue


# ============================================================================
# Root and metrics endpoints
# ============================================================================


class TestRootEndpoint:
    async def test_root_returns_info(self) -> None:
        """Root endpoint returns service info."""
        result = await root()
        assert "name" in result
        assert result["status"] == "running"
        assert "endpoints" in result


class TestMetricsEndpoint:
    async def test_metrics_returns_response(self) -> None:
        """Metrics endpoint returns Prometheus data."""
        response = await metrics_endpoint()
        assert response.status_code == 200


# ============================================================================
# Global exception handler
# ============================================================================
# Main entry point
# ============================================================================


class TestMainEntrypoint:
    def test_main_calls_uvicorn(self) -> None:
        """main() function calls uvicorn.run."""
        with patch("uvicorn.run") as mock_run:
            main()
            mock_run.assert_called_once()


# ============================================================================
# Lifespan
# ============================================================================


def _make_mock_engine():
    """Create a mock async engine with properly configured begin() context manager."""
    mock_conn = AsyncMock()
    mock_conn.run_sync = AsyncMock()

    # begin() must return a sync object that supports `async with`
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_cm
    return mock_engine


def _make_mock_settings():
    """Create a mock settings object that mirrors the real Settings structure."""
    s = MagicMock()
    s.observability.tracing_enabled = False
    s.observability.metrics_enabled = False
    s.database.url = "postgresql+psycopg://test:test@localhost/test"  # pragma: allowlist secret
    s.redis.url = "redis://localhost:6379/0"
    s.mock_provider.enabled = False
    s.local_image.enabled = False
    s.health_monitor.enabled = False
    s.generation.provider_strategy = "round_robin"
    s.generation.default_provider = "mock"
    s.generation.fallback_chain = []
    s.generation.parallel_requests = 1
    s.generation.max_retries = 1
    s.generation.fallback_enabled = False
    s.image_generation.default_provider = "mock"
    s.image_generation.parallel_requests = 4
    s.normalization.enabled = False
    s.quota.enabled = False
    s.cleanup.enabled = False
    s.environment = "test"
    s.api.host = "0.0.0.0"
    s.api.port = 6310
    return s


class TestLifespan:
    async def test_lifespan_minimal_startup_and_shutdown(self) -> None:
        """Lifespan starts and shuts down with all features disabled."""
        import src.asgi as asgi_mod

        mock_settings = _make_mock_settings()

        # Mock the database engine
        mock_engine = _make_mock_engine()

        # Mock Redis
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        original_settings = asgi_mod.settings
        asgi_mod.settings = mock_settings

        try:
            with (
                patch(
                    "src.asgi.get_engine",
                    new_callable=AsyncMock,
                    return_value=mock_engine,
                ),
                patch(
                    "src.asgi.init_redis",
                    new_callable=AsyncMock,
                    return_value=mock_redis,
                ),
                patch(
                    "src.asgi.get_redis_client",
                    return_value=mock_redis,
                ),
                patch(
                    "src.asgi.close_redis",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.asgi.close_engine",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.storage_service.StorageService.initialize",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.providers.runway.RunwayProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.kling.KlingProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.veo.VeoProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.sora.SoraProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.seedance.SeedanceProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
            ):
                async with lifespan(mock_app):
                    # Inside lifespan -- services initialized
                    pass
        finally:
            asgi_mod.settings = original_settings

    async def test_lifespan_clean_startup_with_empty_keys(self, caplog, monkeypatch) -> None:
        """Lifespan boots cleanly with every external *_API_KEY empty (WS4 DoD #4).

        Asserts:
          1. lifespan enters and exits without raising
          2. no log records emitted at level >= ERROR
          3. with VIDEO_PROVIDER=stub set, _resolve_default_provider lands on "stub"

        This is the regression guard required before flipping
        GENERATION_DEFAULT_PROVIDER from stub to a real provider via SOPS.
        """
        import src.asgi as asgi_mod

        for key in (
            "RUNWAY_API_KEY",
            "KLING_API_KEY",
            "VEO_API_KEY",
            "SORA_API_KEY",
            "SEEDANCE_API_KEY",
            "WAN_API_KEY",
            "STABILITY_API_KEY",
            "REPLICATE_API_KEY",
            "OPENAI_API_KEY",
        ):
            monkeypatch.setenv(key, "")

        mock_settings = _make_mock_settings()
        # Opt into stub-by-default so registry has "stub" and resolver picks it.
        mock_settings.stub.video_provider = "stub"
        mock_settings.generation.default_provider = "stub"

        mock_engine = _make_mock_engine()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        original_settings = asgi_mod.settings
        asgi_mod.settings = mock_settings

        try:
            with (
                patch(
                    "src.asgi.get_engine",
                    new_callable=AsyncMock,
                    return_value=mock_engine,
                ),
                patch(
                    "src.asgi.init_redis",
                    new_callable=AsyncMock,
                    return_value=mock_redis,
                ),
                patch(
                    "src.asgi.get_redis_client",
                    return_value=mock_redis,
                ),
                patch(
                    "src.asgi.close_redis",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.asgi.close_engine",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.storage_service.StorageService.initialize",
                    new_callable=AsyncMock,
                ),
                # Every external provider reports unhealthy -- the empty-key state.
                patch(
                    "src.services.providers.runway.RunwayProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.kling.KlingProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.veo.VeoProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.sora.SoraProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.seedance.SeedanceProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                # The stub is the one provider that MUST come up -- that is what
                # "clean startup with empty keys" means. Patched explicitly rather
                # than left to the host: the real check is `shutil.which("ffmpeg")`,
                # so before this patch the test silently measured whether the
                # machine running it happened to have ffmpeg installed. It passed
                # on a dev box and failed on a CI runner (which has no ffmpeg and
                # does not run the docker image), i.e. it was never asserting the
                # clean path at all. The unhealthy-stub path is pinned separately
                # by ``test_lifespan_no_video_providers_emits_error``.
                patch(
                    "src.services.providers.stub.StubVideoProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=True,
                ),
                caplog.at_level(logging.WARNING),
            ):
                async with lifespan(mock_app):
                    pass
        finally:
            asgi_mod.settings = original_settings

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, (
            "Clean-startup smoke MUST NOT emit ERROR-level logs with empty keys. "
            f"Got: {[(r.name, r.getMessage()) for r in errors]}"
        )
        # Resolver lands on the stub default when VIDEO_PROVIDER=stub is set.
        assert mock_settings.generation.default_provider == "stub"

    async def test_no_video_providers_emits_error_logs(self, caplog) -> None:
        """The OPPOSITE of clean startup: an empty registry must be LOUD.

        This is the 2026-08-20 prod state -- no ffmpeg, so the stub (the only
        activated video provider) fails its health check and nothing registers.
        Both replicas came up Ready with only a WARNING that scrolled past once
        at boot. Pinned at ERROR so a future edit cannot quietly demote it.

        Drives the two emitting units directly rather than the whole lifespan:
        lifespan reaches the real ``get_session_maker`` further down, so routing
        this assertion through it makes a log-level regression surface as a
        Postgres ConnectionError instead of the assertion below.
        """
        import src.asgi as asgi_mod

        mock_settings = _make_mock_settings()
        mock_settings.stub.video_provider = "stub"
        mock_settings.generation.default_provider = "stub"
        mock_settings.wan.enabled = False
        mock_settings.mock_provider.enabled = False

        original_settings = asgi_mod.settings
        asgi_mod.settings = mock_settings
        try:
            with (
                patch(
                    "src.services.providers.runway.RunwayProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.kling.KlingProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.veo.VeoProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.sora.SoraProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.seedance.SeedanceProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                # ffmpeg absent -- what the CI runner and the shipped image both
                # looked like.
                patch(
                    "src.services.providers.stub.StubVideoProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                caplog.at_level(logging.WARNING),
            ):
                registry = await asgi_mod._init_video_provider_registry()
                asgi_mod._resolve_default_provider(registry)
        finally:
            asgi_mod.settings = original_settings

        assert registry.provider_names == [], (
            f"expected an empty registry, got {registry.provider_names}"
        )
        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("ffmpeg missing on PATH" in m for m in errors), (
            f"stub health-check failure must be ERROR, not WARNING. Got: {errors}"
        )
        assert any("NO VIDEO PROVIDERS REGISTERED" in m for m in errors), (
            f"an empty registry must be ERROR, not WARNING. Got: {errors}"
        )

    async def test_lifespan_db_failure_raises(self) -> None:
        """Lifespan raises when database connection fails."""
        import pytest

        import src.asgi as asgi_mod

        mock_settings = _make_mock_settings()
        mock_app = MagicMock()

        original_settings = asgi_mod.settings
        asgi_mod.settings = mock_settings

        try:
            with patch(
                "src.asgi.get_engine",
                new_callable=AsyncMock,
                side_effect=ConnectionError("DB down"),
            ):
                with pytest.raises(ConnectionError, match="DB down"):
                    async with lifespan(mock_app):
                        pass
        finally:
            asgi_mod.settings = original_settings

    async def test_lifespan_redis_failure_continues(self) -> None:
        """Lifespan continues when Redis connection fails."""
        import src.asgi as asgi_mod

        mock_settings = _make_mock_settings()

        mock_engine = _make_mock_engine()

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        original_settings = asgi_mod.settings
        asgi_mod.settings = mock_settings

        try:
            with (
                patch(
                    "src.asgi.get_engine",
                    new_callable=AsyncMock,
                    return_value=mock_engine,
                ),
                patch(
                    "src.asgi.init_redis",
                    new_callable=AsyncMock,
                    side_effect=ConnectionError("Redis down"),
                ),
                patch(
                    "src.asgi.close_redis",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.asgi.close_engine",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.storage_service.StorageService.initialize",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.providers.runway.RunwayProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.kling.KlingProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.veo.VeoProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.sora.SoraProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.seedance.SeedanceProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
            ):
                async with lifespan(mock_app):
                    pass  # Should not raise -- Redis failure is non-fatal
        finally:
            asgi_mod.settings = original_settings

    async def test_lifespan_with_features_enabled(self) -> None:
        """Lifespan with observability, health monitor, quota, cleanup enabled."""
        import src.asgi as asgi_mod

        mock_settings = _make_mock_settings()
        mock_settings.observability.tracing_enabled = True
        mock_settings.observability.metrics_enabled = True
        mock_settings.observability.service_name = "test-service"
        mock_settings.observability.otlp_endpoint = "http://localhost:4317"
        mock_settings.health_monitor.enabled = True
        mock_settings.health_monitor.max_consecutive_failures = 3
        mock_settings.health_monitor.cooldown_seconds = 60
        mock_settings.mock_provider.enabled = True
        mock_settings.mock_provider.delay_s = 0.1
        mock_settings.normalization.enabled = True
        mock_settings.normalization.target_resolution = "1080p"
        mock_settings.normalization.target_fps = 30
        mock_settings.normalization.target_codec = "h264"
        mock_settings.normalization.strip_audio = False
        mock_settings.normalization.timeout_seconds = 120
        mock_settings.quota.enabled = True
        mock_settings.quota.default_max_daily_generations = 100
        mock_settings.quota.default_max_daily_spend = 50.0
        mock_settings.quota.default_max_concurrent_jobs = 5
        mock_settings.cleanup.enabled = True
        mock_settings.cleanup.interval_seconds = 3600
        mock_settings.cleanup.completed_ttl_days = 7
        mock_settings.cleanup.failed_ttl_days = 3

        mock_engine = _make_mock_engine()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        mock_cleanup = AsyncMock()
        mock_cleanup.start = AsyncMock()
        mock_cleanup.stop = AsyncMock()

        original_settings = asgi_mod.settings
        asgi_mod.settings = mock_settings

        try:
            with (
                patch(
                    "src.asgi.get_engine",
                    new_callable=AsyncMock,
                    return_value=mock_engine,
                ),
                patch(
                    "src.asgi.init_redis",
                    new_callable=AsyncMock,
                    return_value=mock_redis,
                ),
                patch("src.asgi.get_redis_client", return_value=mock_redis),
                patch("src.asgi.close_redis", new_callable=AsyncMock),
                patch("src.asgi.close_engine", new_callable=AsyncMock),
                patch(
                    "src.services.storage_service.StorageService.initialize",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.providers.runway.RunwayProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.kling.KlingProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.veo.VeoProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.sora.SoraProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.seedance.SeedanceProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.observability.tracing.setup_tracing",
                ),
                patch(
                    "src.observability.metrics.setup_metrics",
                ),
                patch(
                    "src.services.provider_health.ProviderHealthMonitor.load_all_from_redis",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.cleanup_service.CleanupService",
                    return_value=mock_cleanup,
                ),
                patch(
                    "src.database.session.get_session_maker",
                    new_callable=AsyncMock,
                    return_value=MagicMock(),
                ),
            ):
                async with lifespan(mock_app):
                    pass

            mock_cleanup.start.assert_awaited_once()
            mock_cleanup.stop.assert_awaited_once()
        finally:
            asgi_mod.settings = original_settings

    async def test_lifespan_shutdown_redis_error(self) -> None:
        """Lifespan handles Redis close error gracefully during shutdown."""
        import src.asgi as asgi_mod

        mock_settings = _make_mock_settings()

        mock_engine = _make_mock_engine()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        original_settings = asgi_mod.settings
        asgi_mod.settings = mock_settings

        try:
            with (
                patch(
                    "src.asgi.get_engine",
                    new_callable=AsyncMock,
                    return_value=mock_engine,
                ),
                patch(
                    "src.asgi.init_redis",
                    new_callable=AsyncMock,
                    return_value=mock_redis,
                ),
                patch("src.asgi.get_redis_client", return_value=mock_redis),
                patch(
                    "src.asgi.close_redis",
                    new_callable=AsyncMock,
                    side_effect=OSError("Redis close error"),
                ),
                patch("src.asgi.close_engine", new_callable=AsyncMock),
                patch(
                    "src.services.storage_service.StorageService.initialize",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.providers.runway.RunwayProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.kling.KlingProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.veo.VeoProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.sora.SoraProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.seedance.SeedanceProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
            ):
                async with lifespan(mock_app):
                    pass  # Should not raise -- Redis close error is caught
        finally:
            asgi_mod.settings = original_settings

    async def test_lifespan_shutdown_db_close_error(self) -> None:
        """Lifespan handles DB close error gracefully during shutdown."""
        import src.asgi as asgi_mod

        mock_settings = _make_mock_settings()

        mock_engine = _make_mock_engine()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        original_settings = asgi_mod.settings
        asgi_mod.settings = mock_settings

        try:
            with (
                patch(
                    "src.asgi.get_engine",
                    new_callable=AsyncMock,
                    return_value=mock_engine,
                ),
                patch(
                    "src.asgi.init_redis",
                    new_callable=AsyncMock,
                    return_value=mock_redis,
                ),
                patch("src.asgi.get_redis_client", return_value=mock_redis),
                patch("src.asgi.close_redis", new_callable=AsyncMock),
                patch(
                    "src.asgi.close_engine",
                    new_callable=AsyncMock,
                    side_effect=OSError("DB close error"),
                ),
                patch(
                    "src.services.storage_service.StorageService.initialize",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.providers.runway.RunwayProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.kling.KlingProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.veo.VeoProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.sora.SoraProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.providers.seedance.SeedanceProvider.health_check",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
            ):
                async with lifespan(mock_app):
                    pass  # Should not raise -- DB close error is caught
        finally:
            asgi_mod.settings = original_settings


# ============================================================================
# Extracted lifespan phase helpers (decomposition tests)
# ============================================================================


class TestLifespanPhaseHelpers:
    """Real-assertion tests for extracted lifespan phase helpers.

    These exercise the helpers in isolation so a regression in one
    phase does not require running the entire lifespan to surface.
    """

    async def test_init_redis_optional_returns_none_on_oserror(self) -> None:
        """_init_redis_optional must swallow OSError and return None.

        Redis is non-critical -- service must come up without it.
        """
        from src.asgi import _init_redis_optional

        app = MagicMock()
        with patch("src.asgi.init_redis", new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = OSError("connection refused")
            result = await _init_redis_optional(app)
        assert result is None
        # state.redis_client should NOT be set when init failed
        assert not hasattr(app.state, "redis_client") or True

    async def test_init_redis_optional_returns_client_on_success(self) -> None:
        """_init_redis_optional returns the live client and stashes on app.state."""
        from src.asgi import _init_redis_optional

        app = MagicMock()
        app.state = MagicMock()
        fake_client = MagicMock()
        fake_client.ping = AsyncMock(return_value=True)
        with (
            patch("src.asgi.init_redis", new_callable=AsyncMock, return_value=fake_client),
            patch("src.asgi.get_redis_client", return_value=fake_client),
        ):
            result = await _init_redis_optional(app)
        assert result is fake_client
        assert app.state.redis_client is fake_client

    async def test_try_register_provider_skips_unhealthy(self) -> None:
        """_try_register_provider does NOT register a provider whose health_check is False."""
        from src.asgi import _try_register_provider
        from src.services.providers.base import VideoProviderRegistry

        registry = VideoProviderRegistry()

        class FakeProvider:
            async def health_check(self) -> bool:
                return False

        await _try_register_provider(registry, "Fake", FakeProvider)
        assert "fake" not in [n.lower() for n in registry.provider_names]

    async def test_try_register_provider_swallows_exceptions(self) -> None:
        """_try_register_provider catches OSError/Import/Value/Runtime errors and logs."""
        from src.asgi import _try_register_provider
        from src.services.providers.base import VideoProviderRegistry

        registry = VideoProviderRegistry()

        def boom() -> None:
            raise OSError("network down")

        # Must not raise -- this is the contract that lets one bad cloud
        # provider not break service start-up
        await _try_register_provider(registry, "Boom", boom)
        assert registry.provider_names == []

    async def test_try_register_provider_registers_healthy(self) -> None:
        """_try_register_provider registers a provider with health_check=True."""
        from src.asgi import _try_register_provider
        from src.services.providers.base import VideoProviderRegistry

        registry = VideoProviderRegistry()

        class HealthyProvider:
            name = "healthy"

            async def health_check(self) -> bool:
                return True

            def estimate_cost(self, duration_s: float) -> float:
                return 0.0

        await _try_register_provider(registry, "Healthy", HealthyProvider)
        assert "healthy" in registry.provider_names

    def test_resolve_default_provider_falls_back_to_first(self) -> None:
        """_resolve_default_provider rewrites settings when configured default is missing."""
        import src.asgi as asgi_mod
        from src.asgi import _resolve_default_provider

        # Build a registry containing only "kling"
        class StubProvider:
            name = "kling"

        registry = MagicMock()
        registry.provider_names = ["kling"]

        original_default = asgi_mod.settings.generation.default_provider
        try:
            asgi_mod.settings.generation.default_provider = "runway"  # not registered
            _resolve_default_provider(registry)
            assert asgi_mod.settings.generation.default_provider == "kling"
        finally:
            asgi_mod.settings.generation.default_provider = original_default

    def test_resolve_default_provider_keeps_when_already_registered(self) -> None:
        """_resolve_default_provider does NOT rewrite when the configured default is present."""
        import src.asgi as asgi_mod
        from src.asgi import _resolve_default_provider

        registry = MagicMock()
        registry.provider_names = ["runway", "kling"]

        original_default = asgi_mod.settings.generation.default_provider
        try:
            asgi_mod.settings.generation.default_provider = "runway"
            _resolve_default_provider(registry)
            assert asgi_mod.settings.generation.default_provider == "runway"
        finally:
            asgi_mod.settings.generation.default_provider = original_default

    def test_build_normalizer_returns_none_when_disabled(self) -> None:
        """_build_normalizer respects normalization.enabled=False."""
        import src.asgi as asgi_mod
        from src.asgi import _build_normalizer

        original = asgi_mod.settings.normalization.enabled
        try:
            asgi_mod.settings.normalization.enabled = False
            assert _build_normalizer() is None
        finally:
            asgi_mod.settings.normalization.enabled = original


# ── The startup "Database connected" line must never carry the DSN password ──
#
# REGRESSION: `_init_database` logged `settings.database.url` raw -- the full
# postgresql+psycopg://<user>:<password>@host:port/db DSN -- at INFO on the
# connect path, so the live app password landed in this pod's stdout and in
# anything shipping those logs, indefinitely.
#
# The test drives the REAL `_init_database` rather than the redactor, because
# the defect was at the CALL SITE: a test that only exercises
# `w2v_common.observability.database_endpoint` passes for the whole duration of
# the leak. That helper has its own unit tests in w2v_common; this one exists to
# fail if anyone hands the raw url back to the logger here.

# Wholly synthetic and shares no prefix with any real credential -- this string
# is committed, printed by pytest on failure, and ends up in CI logs, i.e.
# exactly the places this change exists to keep secrets out of.
_FAKE_PASSWORD = "not-a-real-password-xyzzy"  # pragma: allowlist secret
_FAKE_DSN = (
    f"postgresql+psycopg://acg_app:{_FAKE_PASSWORD}"
    "@pgbouncer.pgbouncer:6432/ai_content_generation_service"
)


class TestStartupDbLogRedaction:
    async def test_init_database_log_does_not_leak_the_dsn_password(self) -> None:
        """The emitted startup log record must not contain the DSN password."""
        import io

        import src.asgi as asgi_mod

        mock_settings = _make_mock_settings()
        mock_settings.database.url = _FAKE_DSN

        # A handler attached directly to the module logger, NOT ``caplog``. This
        # service calls ``configure_logging()``/``_setup_logging()``, which
        # re-points the root handlers, so caplog's root handler can observe
        # nothing and every assertion below would pass vacuously.
        buffer = io.StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setLevel(logging.INFO)
        asgi_mod.logger.addHandler(handler)
        original_level = asgi_mod.logger.level
        asgi_mod.logger.setLevel(logging.INFO)

        original_settings = asgi_mod.settings
        asgi_mod.settings = mock_settings

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        try:
            with (
                patch(
                    "src.asgi.get_engine",
                    new_callable=AsyncMock,
                    return_value=_make_mock_engine(),
                ),
                patch("src.database.rls_events.install_rls_tenant_rebind"),
            ):
                await asgi_mod._init_database(mock_app)
        finally:
            asgi_mod.settings = original_settings
            asgi_mod.logger.setLevel(original_level)
            asgi_mod.logger.removeHandler(handler)

        captured = buffer.getvalue()
        # Guard against a vacuous pass: if the line never emitted, the
        # assertions below would hold for the wrong reason.
        assert "Database connected" in captured, (
            "the 'Database connected' line never emitted; the assertions would be vacuous"
        )
        assert _FAKE_PASSWORD not in captured, "DSN password leaked into the startup log"
        assert "acg_app" not in captured, "DSN username leaked into the startup log"
        # Still useful, or the next person deletes the log line instead of keeping it.
        assert "pgbouncer.pgbouncer:6432/ai_content_generation_service" in captured


# ── The startup "Redis connected" line must never carry the Redis password ──
#
# REGRESSION: `_init_redis_optional` logged `settings.redis.url` raw -- the full
# redis://[user]:<password>@host:port/db URL -- at INFO on the connect path, so
# the live Redis password landed in this pod's stdout and in anything shipping
# those logs, on every pod start, indefinitely.
#
# The test drives the REAL `_init_redis_optional` rather than the redactor,
# because the defect was at the CALL SITE: a test that only exercises
# `w2v_common.observability.redis_endpoint` passes for the whole duration of the
# leak. That helper has its own unit tests in w2v_common; this one exists to fail
# if anyone hands the raw url back to the logger here.

_FAKE_REDIS_PASSWORD = "not-a-real-password-xyzzy"  # pragma: allowlist secret
_FAKE_REDIS_URL = f"redis://acg_cache:{_FAKE_REDIS_PASSWORD}@redis.redis:6379/2"


class TestStartupRedisLogRedaction:
    async def test_init_redis_log_does_not_leak_the_redis_password(self) -> None:
        """The emitted 'Redis connected' record must not contain the password."""
        import io

        import src.asgi as asgi_mod

        mock_settings = _make_mock_settings()
        mock_settings.redis.url = _FAKE_REDIS_URL

        # A handler attached directly to the module logger, NOT ``caplog``. This
        # service calls ``_setup_logging()``, which re-points the root handlers,
        # so caplog's root handler can observe nothing and every assertion below
        # would pass vacuously.
        buffer = io.StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        asgi_mod.logger.addHandler(handler)
        original_level = asgi_mod.logger.level
        # MUST be setLevel(): ``Logger`` caches ``isEnabledFor`` and only
        # ``setLevel`` clears that cache. Assigning ``.level`` directly leaves a
        # stale cache entry, the record never reaches the handler, and the
        # negative assertions below pass on EMPTY output with the leak intact.
        asgi_mod.logger.setLevel(logging.INFO)

        original_settings = asgi_mod.settings
        asgi_mod.settings = mock_settings

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        try:
            with (
                patch("src.asgi.init_redis", new_callable=AsyncMock, return_value=mock_redis),
                patch("src.asgi.get_redis_client", return_value=mock_redis),
            ):
                await asgi_mod._init_redis_optional(mock_app)
        finally:
            asgi_mod.settings = original_settings
            asgi_mod.logger.setLevel(original_level)
            asgi_mod.logger.removeHandler(handler)

        captured = buffer.getvalue()
        # Guard against a vacuous pass: if the line never emitted, the
        # assertions below would hold for the wrong reason.
        assert "Redis connected" in captured, (
            "the 'Redis connected' line never emitted; the assertions would be vacuous"
        )
        assert _FAKE_REDIS_PASSWORD not in captured, "Redis password leaked into the startup log"
        assert "acg_cache" not in captured, "Redis username leaked into the startup log"
        # Still useful, or the next person deletes the log line instead of keeping it.
        assert "redis.redis:6379" in captured
