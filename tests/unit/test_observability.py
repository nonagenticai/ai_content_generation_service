"""Unit tests for observability: metrics and tracing modules."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

import src.observability.metrics as metrics_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_metrics():
    """Reset module-level metric globals between tests."""
    metrics_mod._jobs_total = None
    metrics_mod._job_duration = None
    metrics_mod._cost_total = None
    metrics_mod._active_jobs = None
    metrics_mod._cache_invalidate_failures = None
    metrics_mod._initialized = False


@pytest.fixture(autouse=True)
def _clean_metrics():
    """Ensure every test starts with a clean metrics state and registry."""
    from prometheus_client import REGISTRY

    # Unregister our metric collectors so setup_metrics can recreate them
    for name in list(REGISTRY._names_to_collectors.keys()):
        if name.startswith("video_generation_") or name.startswith("ai_content_gen_"):
            with contextlib.suppress(KeyError, ValueError):
                REGISTRY.unregister(REGISTRY._names_to_collectors[name])

    _reset_metrics()
    yield
    _reset_metrics()


# ===========================================================================
# Metrics: setup_metrics
# ===========================================================================


async def test_setup_metrics_initializes_instruments():
    """setup_metrics creates Counter, Histogram, Gauge and sets _initialized."""
    from prometheus_client import Counter, Gauge, Histogram

    metrics_mod.setup_metrics()

    assert metrics_mod._initialized is True
    # Validate actual types, not just "not None"
    assert isinstance(metrics_mod._jobs_total, Counter)
    assert isinstance(metrics_mod._job_duration, Histogram)
    assert isinstance(metrics_mod._cost_total, Counter)
    assert isinstance(metrics_mod._active_jobs, Gauge)


async def test_setup_metrics_is_idempotent():
    """Calling setup_metrics twice is a no-op the second time."""
    metrics_mod.setup_metrics()
    first_jobs = metrics_mod._jobs_total

    metrics_mod.setup_metrics()
    assert metrics_mod._jobs_total is first_jobs  # same object, not re-created


async def test_setup_metrics_import_error():
    """setup_metrics handles ImportError gracefully."""
    import builtins

    real_import = builtins.__import__

    def _fail_import(name, *args, **kwargs):
        if name == "prometheus_client":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fail_import):
        metrics_mod.setup_metrics()

    assert metrics_mod._initialized is False
    assert metrics_mod._jobs_total is None


# ===========================================================================
# Metrics: record_* functions AFTER setup
# ===========================================================================


async def test_record_job_completed():
    """record_job_completed updates counter, histogram, and cost values."""
    metrics_mod.setup_metrics()

    assert metrics_mod._jobs_total is not None
    assert metrics_mod._job_duration is not None
    assert metrics_mod._cost_total is not None

    before_count = metrics_mod._jobs_total.labels(
        provider="runway", status="completed"
    )._value.get()
    before_duration = metrics_mod._job_duration.labels(provider="runway")._sum.get()
    before_cost = metrics_mod._cost_total.labels(provider="runway")._value.get()

    metrics_mod.record_job_completed(provider="runway", duration_s=12.5, cost=0.05)

    after_count = metrics_mod._jobs_total.labels(provider="runway", status="completed")._value.get()
    after_duration = metrics_mod._job_duration.labels(provider="runway")._sum.get()
    after_cost = metrics_mod._cost_total.labels(provider="runway")._value.get()

    assert after_count == before_count + 1
    assert abs((after_duration - before_duration) - 12.5) < 0.001
    assert abs((after_cost - before_cost) - 0.05) < 0.001


async def test_record_job_completed_zero_cost():
    """record_job_completed with cost=0 skips cost increment."""
    metrics_mod.setup_metrics()

    assert metrics_mod._cost_total is not None

    cost_before = metrics_mod._cost_total.labels(provider="kling")._value.get()
    metrics_mod.record_job_completed(provider="kling", duration_s=5.0, cost=0.0)
    cost_after = metrics_mod._cost_total.labels(provider="kling")._value.get()

    assert cost_after == cost_before  # No change when cost is 0


async def test_record_job_failed():
    """record_job_failed increments the failed counter."""
    metrics_mod.setup_metrics()

    assert metrics_mod._jobs_total is not None

    before = metrics_mod._jobs_total.labels(provider="veo", status="failed")._value.get()
    metrics_mod.record_job_failed(provider="veo")
    after = metrics_mod._jobs_total.labels(provider="veo", status="failed")._value.get()

    assert after == before + 1


async def test_record_job_started():
    """record_job_started increments started counter and active gauge."""
    metrics_mod.setup_metrics()

    assert metrics_mod._jobs_total is not None
    assert metrics_mod._active_jobs is not None

    before_count = metrics_mod._jobs_total.labels(provider="sora", status="started")._value.get()
    before_active = metrics_mod._active_jobs._value.get()

    metrics_mod.record_job_started(provider="sora")

    after_count = metrics_mod._jobs_total.labels(provider="sora", status="started")._value.get()
    after_active = metrics_mod._active_jobs._value.get()

    assert after_count == before_count + 1
    assert after_active == before_active + 1


async def test_record_job_finished():
    """record_job_finished decrements the active gauge."""
    metrics_mod.setup_metrics()

    assert metrics_mod._active_jobs is not None

    # Start first so gauge is > 0
    metrics_mod.record_job_started(provider="wan")
    before_active = metrics_mod._active_jobs._value.get()

    metrics_mod.record_job_finished()

    after_active = metrics_mod._active_jobs._value.get()
    assert after_active == before_active - 1


# ===========================================================================
# Metrics: record_* functions BEFORE setup (no-ops, must not crash)
# ===========================================================================


async def test_record_functions_before_setup_are_noops():
    """All record_* functions are safe to call before setup_metrics."""
    assert metrics_mod._initialized is False
    assert metrics_mod._jobs_total is None

    # None of these should raise, and no state should change
    metrics_mod.record_job_completed(provider="test", duration_s=1.0, cost=0.1)
    metrics_mod.record_job_failed(provider="test")
    metrics_mod.record_job_started(provider="test")
    metrics_mod.record_job_finished()

    # Still not initialized -- nothing was created
    assert metrics_mod._initialized is False
    assert metrics_mod._jobs_total is None


# ===========================================================================
# Metrics: get_metrics_response
# ===========================================================================


async def test_get_metrics_response_returns_bytes_and_content_type():
    """get_metrics_response returns (bytes, content_type_string)."""
    metrics_mod.setup_metrics()
    body, content_type = metrics_mod.get_metrics_response()

    assert isinstance(body, bytes)
    assert len(body) > 0
    # The body should contain actual metric names from our setup
    assert b"video_generation_jobs_total" in body
    assert b"video_generation_active_jobs" in body
    assert "text/plain" in content_type or "text/plain" in str(content_type)


async def test_get_metrics_response_import_error():
    """get_metrics_response handles ImportError gracefully."""
    import builtins

    real_import = builtins.__import__

    def _fail_import(name, *args, **kwargs):
        if name == "prometheus_client":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fail_import):
        body, content_type = metrics_mod.get_metrics_response()

    assert body == b"# prometheus_client not installed\n"
    assert content_type == "text/plain"


# ===========================================================================
# Tracing: setup_tracing
# ===========================================================================


async def test_setup_tracing_with_endpoint():
    """setup_tracing with an endpoint creates an exporter."""
    with (
        patch("opentelemetry.trace.set_tracer_provider") as mock_set_provider,
        patch(
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
        ) as mock_exporter_cls,
        patch("opentelemetry.sdk.trace.TracerProvider") as mock_tp_cls,
        patch("opentelemetry.sdk.trace.export.BatchSpanProcessor"),
        patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor") as mock_fai,
        patch("opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor") as mock_hci,
        patch("opentelemetry.instrumentation.sqlalchemy.SQLAlchemyInstrumentor") as mock_sai,
        patch("opentelemetry.sdk.resources.Resource.create"),
    ):
        mock_provider = MagicMock()
        mock_tp_cls.return_value = mock_provider

        from src.observability.tracing import setup_tracing

        setup_tracing(service_name="test-svc", otlp_endpoint="http://localhost:4317")

        mock_exporter_cls.assert_called_once_with(endpoint="http://localhost:4317", insecure=True)
        mock_provider.add_span_processor.assert_called_once()
        mock_set_provider.assert_called_once_with(mock_provider)
        mock_fai.return_value.instrument.assert_called_once()
        mock_hci.return_value.instrument.assert_called_once()
        mock_sai.return_value.instrument.assert_called_once()


async def test_setup_tracing_without_endpoint():
    """setup_tracing without an endpoint skips the exporter."""
    with (
        patch("opentelemetry.trace.set_tracer_provider") as mock_set_provider,
        patch(
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
        ) as mock_exporter_cls,
        patch("opentelemetry.sdk.trace.TracerProvider") as mock_tp_cls,
        patch("opentelemetry.sdk.trace.export.BatchSpanProcessor"),
        patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor"),
        patch("opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor"),
        patch("opentelemetry.instrumentation.sqlalchemy.SQLAlchemyInstrumentor"),
        patch("opentelemetry.sdk.resources.Resource.create"),
    ):
        mock_provider = MagicMock()
        mock_tp_cls.return_value = mock_provider

        from src.observability.tracing import setup_tracing

        setup_tracing(service_name="test-svc", otlp_endpoint=None)

        mock_exporter_cls.assert_not_called()
        mock_set_provider.assert_called_once_with(mock_provider)


async def test_setup_tracing_import_error():
    """setup_tracing handles ImportError when otel not installed."""
    import builtins

    real_import = builtins.__import__

    def _fail_import(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fail_import):
        from src.observability.tracing import setup_tracing

        # Should not raise
        setup_tracing()


# ===========================================================================
# Tracing: get_tracer
# ===========================================================================


async def test_get_tracer_returns_tracer():
    """get_tracer returns a tracer with the expected interface."""

    from src.observability.tracing import get_tracer

    tracer = get_tracer("test-tracer")
    # Verify it has the Tracer interface (start_span method)
    assert tracer is not None
    assert hasattr(tracer, "start_span")


async def test_get_tracer_import_error_returns_none():
    """get_tracer returns None when opentelemetry is not installed."""
    import builtins

    real_import = builtins.__import__

    def _fail_import(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fail_import):
        from src.observability.tracing import get_tracer

        result = get_tracer()
        assert result is None
