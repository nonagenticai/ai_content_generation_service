"""Prometheus metrics for ai_content_generation_service."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Metric instances (lazy init)
_jobs_total: Counter | None = None
_job_duration: Histogram | None = None
_cost_total: Counter | None = None
_active_jobs: Gauge | None = None
_cache_invalidate_failures: Counter | None = None
_quota_degraded: Counter | None = None
_normalization_degraded: Counter | None = None
_initialized: bool = False


def setup_metrics() -> None:
    """Initialize Prometheus metrics."""
    global _jobs_total, _job_duration, _cost_total, _active_jobs
    global _cache_invalidate_failures, _quota_degraded, _normalization_degraded
    global _initialized

    if _initialized:
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        _jobs_total = Counter(
            "video_generation_jobs_total",
            "Total video generation jobs",
            ["provider", "status"],
        )
        _job_duration = Histogram(
            "video_generation_job_duration_seconds",
            "Video generation job duration",
            ["provider"],
            buckets=[5, 10, 30, 60, 120, 300, 600],
        )
        _cost_total = Counter(
            "video_generation_cost_dollars_total",
            "Total cost of video generation in USD",
            ["provider"],
        )
        _active_jobs = Gauge(
            "video_generation_active_jobs",
            "Number of currently active video generation jobs",
        )
        _cache_invalidate_failures = Counter(
            "ai_content_gen_cache_invalidate_failures_total",
            "Cache invalidation failures during job state transitions",
            ["reason"],
        )

        # A fail-open quota window is unmeasurable without this: the service
        # keeps serving, nothing errors, and paid generations are admitted
        # against a cap nobody is reading. ``outcome`` separates the spend that
        # went through unmetered from the spend the degraded budget refused.
        _quota_degraded = Counter(
            "ai_content_gen_quota_degraded_total",
            "Quota decisions taken while the quota store was unreachable",
            ["phase", "outcome"],
        )

        # A clip that skipped normalization is delivered anyway and its job is
        # green, so nothing else in the system is countable. ``reason`` is what
        # separates a one-off ffmpeg timeout from ``ffmpeg_missing`` -- the
        # latter is an IMAGE defect degrading every clip the pod produces.
        _normalization_degraded = Counter(
            "ai_content_gen_normalization_degraded_total",
            "Clips delivered without normalization while normalization was enabled",
            ["reason"],
        )

        _initialized = True
        logger.info("Prometheus metrics initialized")
    except ImportError:
        logger.warning("prometheus_client not installed -- metrics disabled")
    except (OSError, ValueError, RuntimeError) as e:
        logger.error("Failed to initialize Prometheus metrics: %s", e)


def record_job_completed(provider: str, duration_s: float, cost: float) -> None:
    """Record a completed job."""
    if _jobs_total:
        _jobs_total.labels(provider=provider, status="completed").inc()
    if _job_duration:
        _job_duration.labels(provider=provider).observe(duration_s)
    if _cost_total and cost > 0:
        _cost_total.labels(provider=provider).inc(cost)


def record_normalization_degraded(reason: str) -> None:
    """Count a clip delivered un-normalized while normalization was enabled."""
    if _normalization_degraded:
        _normalization_degraded.labels(reason=reason).inc()


def record_job_failed(provider: str) -> None:
    """Record a failed job."""
    if _jobs_total:
        _jobs_total.labels(provider=provider, status="failed").inc()


def record_job_started(provider: str) -> None:
    """Record a started job."""
    if _jobs_total:
        _jobs_total.labels(provider=provider, status="started").inc()
    if _active_jobs:
        _active_jobs.inc()


def record_job_finished() -> None:
    """Record a finished job (decrement active gauge)."""
    if _active_jobs:
        _active_jobs.dec()


def record_cache_invalidate_failure(reason: str) -> None:
    """Record a cache invalidation failure with a reason label."""
    if _cache_invalidate_failures:
        _cache_invalidate_failures.labels(reason=reason).inc()


def record_quota_degraded(phase: str, outcome: str) -> None:
    """Record one quota decision taken without a reachable quota store.

    ``phase`` is check/start/complete; ``outcome`` is admitted/rejected/dropped.
    """
    if _quota_degraded:
        _quota_degraded.labels(phase=phase, outcome=outcome).inc()


def get_metrics_response() -> tuple[bytes, str]:
    """Generate Prometheus metrics response."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest(), CONTENT_TYPE_LATEST
    except ImportError:
        return b"# prometheus_client not installed\n", "text/plain"
