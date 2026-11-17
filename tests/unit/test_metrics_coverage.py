"""Tests for metrics.py to cover missing lines.

Tests that record_* functions interact with REAL Prometheus metric objects,
not mocks. Validates actual metric values via the Prometheus exposition format.
"""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import patch

import pytest

import src.observability.metrics as metrics_mod


@pytest.fixture(autouse=True)
def _reset():
    """Reset metrics state and unregister collectors before and after each test."""
    from prometheus_client import REGISTRY

    # Unregister our metric collectors so setup_metrics can recreate them
    for name in list(REGISTRY._names_to_collectors.keys()):
        if name.startswith("video_generation_") or name.startswith("ai_content_gen_"):
            with contextlib.suppress(KeyError, ValueError):
                REGISTRY.unregister(REGISTRY._names_to_collectors[name])

    metrics_mod._jobs_total = None
    metrics_mod._job_duration = None
    metrics_mod._cost_total = None
    metrics_mod._active_jobs = None
    metrics_mod._cache_invalidate_failures = None
    metrics_mod._initialized = False
    yield
    metrics_mod._jobs_total = None
    metrics_mod._job_duration = None
    metrics_mod._cost_total = None
    metrics_mod._active_jobs = None
    metrics_mod._cache_invalidate_failures = None
    metrics_mod._initialized = False


def _get_metric_value(metric, labels: dict | None = None) -> float:
    """Extract the current value from a Prometheus metric (Counter, Gauge, Histogram)."""
    from prometheus_client import Histogram

    child = metric.labels(**labels) if labels else metric

    if isinstance(metric, Histogram) or (labels and isinstance(metric, Histogram)):
        # For Histogram, read _sum from the labeled child
        return child._sum.get()
    # Counter / Gauge
    return child._value.get()


def test_setup_metrics_early_return_when_initialized():
    """setup_metrics returns early when _initialized is True (line 22)."""
    metrics_mod._initialized = True
    # Set a sentinel to verify nothing is overwritten
    sentinel: Any = "sentinel"
    metrics_mod._jobs_total = sentinel

    metrics_mod.setup_metrics()

    # The sentinel should still be there -- nothing was re-created
    assert metrics_mod._jobs_total == "sentinel"


def test_setup_metrics_general_exception():
    """setup_metrics handles general exception (not ImportError) gracefully."""
    import builtins

    real_import = builtins.__import__

    def _fail_on_counter(name, *args, **kwargs):
        mod = real_import(name, *args, **kwargs)
        if name == "prometheus_client":
            # Raise a general exception after import succeeds
            raise RuntimeError("Unexpected error initializing metrics")
        return mod

    with patch("builtins.__import__", side_effect=_fail_on_counter):
        metrics_mod.setup_metrics()

    assert metrics_mod._initialized is False


def test_record_job_completed_increments_real_metrics():
    """record_job_completed updates real Prometheus Counter, Histogram, and cost Counter."""
    metrics_mod.setup_metrics()
    assert metrics_mod._initialized is True

    assert metrics_mod._job_duration is not None

    # Record initial baselines
    jobs_before = _get_metric_value(
        metrics_mod._jobs_total, {"provider": "runway", "status": "completed"}
    )
    duration_sum_before = metrics_mod._job_duration.labels(provider="runway")._sum.get()
    cost_before = _get_metric_value(metrics_mod._cost_total, {"provider": "runway"})

    metrics_mod.record_job_completed(provider="runway", duration_s=12.5, cost=0.05)

    # jobs_total counter incremented by 1
    jobs_after = _get_metric_value(
        metrics_mod._jobs_total, {"provider": "runway", "status": "completed"}
    )
    assert jobs_after == jobs_before + 1

    # duration histogram observed 12.5s
    duration_sum_after = metrics_mod._job_duration.labels(provider="runway")._sum.get()
    assert abs((duration_sum_after - duration_sum_before) - 12.5) < 0.001

    # cost counter incremented by 0.05
    cost_after = _get_metric_value(metrics_mod._cost_total, {"provider": "runway"})
    assert abs((cost_after - cost_before) - 0.05) < 0.001


def test_record_job_completed_zero_cost_skips_cost_increment():
    """record_job_completed with cost=0 does NOT increment cost counter."""
    metrics_mod.setup_metrics()

    cost_before = _get_metric_value(metrics_mod._cost_total, {"provider": "kling"})

    metrics_mod.record_job_completed(provider="kling", duration_s=5.0, cost=0.0)

    cost_after = _get_metric_value(metrics_mod._cost_total, {"provider": "kling"})
    assert cost_after == cost_before  # No change


def test_record_job_failed_increments_counter():
    """record_job_failed increments the failed counter on real metric."""
    metrics_mod.setup_metrics()

    before = _get_metric_value(metrics_mod._jobs_total, {"provider": "veo", "status": "failed"})
    metrics_mod.record_job_failed(provider="veo")
    after = _get_metric_value(metrics_mod._jobs_total, {"provider": "veo", "status": "failed"})

    assert after == before + 1


def test_record_job_started_increments_counter_and_gauge():
    """record_job_started increments started counter and active gauge."""
    metrics_mod.setup_metrics()

    started_before = _get_metric_value(
        metrics_mod._jobs_total, {"provider": "sora", "status": "started"}
    )
    active_before = _get_metric_value(metrics_mod._active_jobs)

    metrics_mod.record_job_started(provider="sora")

    started_after = _get_metric_value(
        metrics_mod._jobs_total, {"provider": "sora", "status": "started"}
    )
    active_after = _get_metric_value(metrics_mod._active_jobs)

    assert started_after == started_before + 1
    assert active_after == active_before + 1


def test_record_job_finished_decrements_gauge():
    """record_job_finished decrements the active gauge."""
    metrics_mod.setup_metrics()

    # First start a job so gauge > 0
    metrics_mod.record_job_started(provider="seedance")
    active_before = _get_metric_value(metrics_mod._active_jobs)

    metrics_mod.record_job_finished()

    active_after = _get_metric_value(metrics_mod._active_jobs)
    assert active_after == active_before - 1
