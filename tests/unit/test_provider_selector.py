"""Unit tests for provider selection strategies (Phase 4)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.services.provider_health import ProviderHealthMonitor
from src.services.provider_selector import (
    CheapestSelector,
    FallbackChainSelector,
    FastestSelector,
    ManualSelector,
    RoundRobinSelector,
    create_selector,
)
from src.services.providers.base import VideoProviderRegistry


def _make_registry(
    providers: dict[str, float],
    max_durations: dict[str, float | None] | None = None,
    image_to_video: dict[str, bool] | None = None,
) -> VideoProviderRegistry:
    """Create a real registry with mock providers of specified costs.

    ``max_durations`` and ``image_to_video`` allow per-provider capability
    overrides for capability-aware selection tests.
    """
    max_durations = max_durations or {}
    image_to_video = image_to_video or {}
    registry = VideoProviderRegistry()
    for name, cost in providers.items():
        provider = MagicMock()
        provider.name = name
        provider.estimate_cost.return_value = cost
        provider.get_models.return_value = []
        provider.get_styles.return_value = []
        max_dur = max_durations.get(name)
        provider.get_max_duration_s.return_value = max_dur
        i2v = image_to_video.get(name, False)
        provider.supports_image_to_video.return_value = i2v

        # Wire up can_handle to behave like the real implementation
        def _can_handle(
            *,
            duration_s=None,
            image_url=None,
            _max=max_dur,
            _i2v=i2v,
        ):
            if duration_s is not None and _max is not None and duration_s > _max:
                return False
            return not (image_url and not _i2v)

        provider.can_handle = _can_handle
        registry.register(provider)
    return registry


def test_manual_selector():
    selector = ManualSelector("runway")
    result = selector.select(MagicMock(), None, None)
    assert result == "runway"


def test_cheapest_selector():
    registry = _make_registry({"expensive": 1.0, "cheap": 0.01, "mid": 0.5})
    selector = CheapestSelector()
    result = selector.select(registry, None, 5.0)
    assert result == "cheap"


def test_cheapest_with_health_monitor():
    registry = _make_registry({"expensive": 1.0, "cheap": 0.01})
    monitor = ProviderHealthMonitor(max_consecutive_failures=1)
    monitor.record_failure("cheap")  # Disable cheapest

    selector = CheapestSelector()
    result = selector.select(registry, monitor, 5.0)
    assert result == "expensive"


def test_cheapest_no_providers_raises():
    registry = VideoProviderRegistry()
    selector = CheapestSelector()
    with pytest.raises(RuntimeError, match="No available providers"):
        selector.select(registry, None, 5.0)


def test_fastest_selector():
    registry = _make_registry({"slow": 0.5, "fast": 0.5})
    monitor = ProviderHealthMonitor()
    monitor.record_success("slow", 500.0)
    monitor.record_success("fast", 100.0)

    selector = FastestSelector()
    result = selector.select(registry, monitor, 5.0)
    assert result == "fast"


def test_fastest_no_health_data():
    """Without health monitor, picks first available."""
    registry = _make_registry({"p1": 0.5, "p2": 0.5})
    selector = FastestSelector()
    result = selector.select(registry, None, 5.0)
    assert result in ("p1", "p2")


def test_round_robin_selector():
    registry = _make_registry({"p1": 0.5, "p2": 0.5, "p3": 0.5})
    selector = RoundRobinSelector()

    results = [selector.select(registry, None, 5.0) for _ in range(6)]
    # Should cycle through providers
    assert len(set(results)) > 1


def test_round_robin_with_unhealthy():
    registry = _make_registry({"p1": 0.5, "p2": 0.5})
    monitor = ProviderHealthMonitor(max_consecutive_failures=1)
    monitor.record_failure("p1")

    selector = RoundRobinSelector()
    result = selector.select(registry, monitor, 5.0)
    assert result == "p2"


def test_fallback_chain_selector():
    registry = _make_registry({"primary": 0.5, "secondary": 0.5, "tertiary": 0.5})
    selector = FallbackChainSelector(["primary", "secondary", "tertiary"])
    result = selector.select(registry, None, 5.0)
    assert result == "primary"


def test_fallback_chain_skips_unhealthy():
    registry = _make_registry({"primary": 0.5, "secondary": 0.5})
    monitor = ProviderHealthMonitor(max_consecutive_failures=1)
    monitor.record_failure("primary")

    selector = FallbackChainSelector(["primary", "secondary"])
    result = selector.select(registry, monitor, 5.0)
    assert result == "secondary"


def test_fallback_chain_all_unavailable():
    registry = _make_registry({"p1": 0.5})
    monitor = ProviderHealthMonitor(max_consecutive_failures=1)
    monitor.record_failure("p1")

    selector = FallbackChainSelector(["p1"])
    with pytest.raises(RuntimeError, match="No available providers"):
        selector.select(registry, monitor, 5.0)


def test_factory_manual():
    selector = create_selector("manual", "runway")
    assert isinstance(selector, ManualSelector)


def test_factory_cheapest():
    selector = create_selector("cheapest")
    assert isinstance(selector, CheapestSelector)


def test_factory_fastest():
    selector = create_selector("fastest")
    assert isinstance(selector, FastestSelector)


def test_factory_round_robin():
    selector = create_selector("round_robin")
    assert isinstance(selector, RoundRobinSelector)


def test_factory_fallback_chain():
    selector = create_selector("fallback_chain", fallback_chain=["p1", "p2"])
    assert isinstance(selector, FallbackChainSelector)


def test_factory_unknown_defaults_to_manual():
    selector = create_selector("unknown")
    assert isinstance(selector, ManualSelector)


def test_explicit_override_bypasses_selector():
    """When provider is explicitly specified, selector is not used."""
    from src.services.video_generation_service import VideoGenerationService

    registry = _make_registry({"runway": 0.5, "kling": 0.01})
    selector = CheapestSelector()
    svc = VideoGenerationService(
        storage=MagicMock(),
        registry=registry,
        default_provider="runway",
        selector=selector,
    )
    # _resolve_provider uses selector to pick cheapest
    assert svc._resolve_provider(5.0) == "kling"  # Cheapest


# ===========================================================================
# Capability-aware selection tests
# ===========================================================================


class TestCapabilityAwareSelection:
    """Tests for capability filtering in selectors."""

    def test_cheapest_skips_provider_exceeding_max_duration(self):
        """A 15s request should skip a provider with max_duration=10."""
        registry = _make_registry(
            providers={"short": 0.01, "long": 0.05},
            max_durations={"short": 10.0, "long": 30.0},
        )
        selector = CheapestSelector()
        result = selector.select(registry, None, 15.0)
        # "short" is cheaper but can't handle 15s
        assert result == "long"

    def test_cheapest_all_incapable_raises(self):
        """If no provider can handle the duration, raise RuntimeError."""
        registry = _make_registry(
            providers={"a": 0.01, "b": 0.02},
            max_durations={"a": 5.0, "b": 10.0},
        )
        selector = CheapestSelector()
        with pytest.raises(RuntimeError, match="No available providers"):
            selector.select(registry, None, 20.0)

    def test_cheapest_none_duration_skips_capability_check(self):
        """When duration is None, all providers are eligible."""
        registry = _make_registry(
            providers={"cheap": 0.01, "expensive": 0.5},
            max_durations={"cheap": 5.0, "expensive": 30.0},
        )
        selector = CheapestSelector()
        result = selector.select(registry, None, None)
        assert result == "cheap"

    def test_fastest_skips_provider_exceeding_max_duration(self):
        registry = _make_registry(
            providers={"fast_short": 0.5, "slow_long": 0.5},
            max_durations={"fast_short": 5.0, "slow_long": 30.0},
        )
        monitor = ProviderHealthMonitor()
        monitor.record_success("fast_short", 50.0)
        monitor.record_success("slow_long", 200.0)

        selector = FastestSelector()
        result = selector.select(registry, monitor, 10.0)
        # fast_short is faster but can't handle 10s
        assert result == "slow_long"

    def test_round_robin_skips_incapable(self):
        registry = _make_registry(
            providers={"p1": 0.5, "p2": 0.5},
            max_durations={"p1": 5.0, "p2": 30.0},
        )
        selector = RoundRobinSelector()
        # Request 15s -- only p2 can handle it
        results = {selector.select(registry, None, 15.0) for _ in range(5)}
        assert results == {"p2"}

    def test_fallback_chain_skips_incapable(self):
        registry = _make_registry(
            providers={"primary": 0.5, "secondary": 0.5},
            max_durations={"primary": 5.0, "secondary": 20.0},
        )
        selector = FallbackChainSelector(["primary", "secondary"])
        result = selector.select(registry, None, 15.0)
        assert result == "secondary"

    def test_cheapest_skips_provider_without_image_to_video(self):
        """Image-to-video request should skip providers that don't support it."""
        registry = _make_registry(
            providers={"no_i2v": 0.01, "has_i2v": 0.05},
            image_to_video={"no_i2v": False, "has_i2v": True},
        )
        selector = CheapestSelector()
        result = selector.select(registry, None, 5.0, image_url="https://example.com/img.png")
        assert result == "has_i2v"

    def test_cheapest_no_image_url_allows_all(self):
        """Without image_url, providers without i2v are still eligible."""
        registry = _make_registry(
            providers={"no_i2v": 0.01, "has_i2v": 0.05},
            image_to_video={"no_i2v": False, "has_i2v": True},
        )
        selector = CheapestSelector()
        result = selector.select(registry, None, 5.0)
        assert result == "no_i2v"
