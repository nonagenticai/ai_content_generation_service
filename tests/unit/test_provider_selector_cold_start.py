"""Cold-start behaviour of latency-ranked provider selection.

``ProviderHealthState.average_latency_ms`` returns ``None`` until a provider
has recorded at least one success (``total_successes == 0``). That is the
normal state of a freshly started pod, a pod that came up after a Redis flush,
and every pod immediately after a rollout.

``FastestSelector`` only assigned a candidate inside ``if latency is not None``,
so with a health monitor attached and no latency samples yet it fell through to
``raise RuntimeError("No available providers")`` — reporting an outage while
every provider was healthy, available, and capable of serving the request.
That is a fake outage, not a real one, and it lands exactly when traffic
resumes after a restart.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.services.provider_health import ProviderHealthMonitor
from src.services.provider_selector import FastestSelector
from src.services.providers.base import VideoProviderRegistry


def _make_registry(
    providers: dict[str, float],
    max_durations: dict[str, float | None] | None = None,
) -> VideoProviderRegistry:
    """Registry of mock providers, mirroring tests/unit/test_provider_selector.py."""
    max_durations = max_durations or {}
    registry = VideoProviderRegistry()
    for name, cost in providers.items():
        provider = MagicMock()
        provider.name = name
        provider.estimate_cost.return_value = cost
        provider.get_models.return_value = []
        provider.get_styles.return_value = []
        max_dur = max_durations.get(name)
        provider.get_max_duration_s.return_value = max_dur
        provider.supports_image_to_video.return_value = False

        def _can_handle(*, duration_s=None, image_url=None, _max=max_dur):
            if duration_s is not None and _max is not None and duration_s > _max:
                return False
            return not image_url

        provider.can_handle = _can_handle
        registry.register(provider)
    return registry


def test_cold_monitor_with_available_providers_does_not_report_an_outage() -> None:
    """The defect: a monitor with no latency samples yet claims no providers.

    Nothing is unhealthy here. ``is_available`` returns True for both
    providers and both can handle the request — there is simply no latency
    sample yet, which is the state of every pod for the first request after a
    restart or a Redis flush.
    """
    registry = _make_registry({"runway": 0.5, "kling": 0.5})
    monitor = ProviderHealthMonitor()

    # Precondition: this is a cold monitor, not a sick one.
    assert monitor.is_available("runway") is True
    assert monitor.is_available("kling") is True
    assert monitor.get_health("runway").average_latency_ms is None
    assert monitor.get_health("kling").average_latency_ms is None

    result = FastestSelector().select(registry, monitor, 5.0)

    assert result in ("runway", "kling")


def test_cold_monitor_single_provider_is_selected() -> None:
    """One healthy, sample-less provider must still be selectable."""
    registry = _make_registry({"veo": 0.5})
    monitor = ProviderHealthMonitor()

    assert FastestSelector().select(registry, monitor, 5.0) == "veo"


def test_partially_warm_monitor_prefers_the_provider_with_a_sample() -> None:
    """Once any sample exists, latency ranking must still drive the choice.

    The cold-start fallback must not degrade into "always pick the first
    provider" — a measured provider beats an unmeasured one.
    """
    registry = _make_registry({"cold": 0.5, "measured": 0.5})
    monitor = ProviderHealthMonitor()
    monitor.record_success("measured", 120.0)

    assert FastestSelector().select(registry, monitor, 5.0) == "measured"


def test_fastest_selection_is_unchanged_once_samples_exist() -> None:
    """The warm path must keep picking the lowest average latency."""
    registry = _make_registry({"slow": 0.5, "fast": 0.5, "mid": 0.5})
    monitor = ProviderHealthMonitor()
    monitor.record_success("slow", 500.0)
    monitor.record_success("fast", 100.0)
    monitor.record_success("mid", 250.0)

    assert FastestSelector().select(registry, monitor, 5.0) == "fast"


def test_cold_fallback_still_respects_health() -> None:
    """A cold-start fallback must not resurrect an unhealthy provider."""
    registry = _make_registry({"broken": 0.5, "ok": 0.5})
    monitor = ProviderHealthMonitor(max_consecutive_failures=1)
    monitor.record_failure("broken")

    assert monitor.is_available("broken") is False
    assert FastestSelector().select(registry, monitor, 5.0) == "ok"


def test_cold_fallback_still_respects_capability() -> None:
    """A cold-start fallback must not pick a provider that cannot serve this."""
    registry = _make_registry(
        providers={"short_only": 0.5, "long_ok": 0.5},
        max_durations={"short_only": 5.0, "long_ok": 30.0},
    )
    monitor = ProviderHealthMonitor()

    assert FastestSelector().select(registry, monitor, 20.0) == "long_ok"


def test_a_real_outage_is_still_reported() -> None:
    """When every provider really is unhealthy, RuntimeError must still raise.

    The fix must not turn a genuine outage into a silent bad selection.
    """
    registry = _make_registry({"a": 0.5, "b": 0.5})
    monitor = ProviderHealthMonitor(max_consecutive_failures=1)
    monitor.record_failure("a")
    monitor.record_failure("b")

    with pytest.raises(RuntimeError, match="No available providers"):
        FastestSelector().select(registry, monitor, 5.0)


def test_no_capable_provider_is_still_an_error() -> None:
    """A cold monitor must not mask "nothing can serve this request"."""
    registry = _make_registry(
        providers={"short_only": 0.5},
        max_durations={"short_only": 5.0},
    )
    monitor = ProviderHealthMonitor()

    with pytest.raises(RuntimeError, match="No available providers"):
        FastestSelector().select(registry, monitor, 60.0)


def test_empty_registry_is_still_an_error() -> None:
    registry = _make_registry({})
    monitor = ProviderHealthMonitor()

    with pytest.raises(RuntimeError, match="No available providers"):
        FastestSelector().select(registry, monitor, 5.0)
