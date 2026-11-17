"""Unit tests for cost tracking and budget enforcement in VideoGenerationService."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.services.providers.base import VideoProviderRegistry
from src.services.video_generation_service import VideoGenerationService


class FakeCostProvider:
    """Minimal provider-like for cost estimation."""

    name = "costprov"

    def estimate_cost(self, duration_s: float) -> float:
        return 0.10 * max(duration_s, 5.0) / 5.0


def _make_service() -> VideoGenerationService:
    registry = VideoProviderRegistry()
    prov = FakeCostProvider()
    registry._providers["costprov"] = prov  # ty: ignore[invalid-assignment]
    return VideoGenerationService(
        storage=MagicMock(),
        registry=registry,
        default_provider="costprov",
    )


def test_estimate_cost_known_provider():
    svc = _make_service()
    cost = svc.estimate_cost("costprov", 5.0)
    assert cost is not None
    assert abs(cost - 0.10) < 0.001


def test_estimate_cost_unknown_provider():
    svc = _make_service()
    cost = svc.estimate_cost("nonexistent", 5.0)
    assert cost is None


def test_estimate_cost_duration_scaling():
    svc = _make_service()
    c5 = svc.estimate_cost("costprov", 5.0)
    c10 = svc.estimate_cost("costprov", 10.0)
    assert c5 is not None
    assert c10 is not None
    assert c10 > c5


def test_estimate_cost_min_duration():
    svc = _make_service()
    c1 = svc.estimate_cost("costprov", 1.0)
    c5 = svc.estimate_cost("costprov", 5.0)
    assert c1 is not None
    assert c5 is not None
    # Both should return the same (min 5s charge)
    assert abs(c1 - c5) < 0.001


async def test_check_budget_allowed():
    svc = _make_service()
    allowed, cost = await svc.check_budget("costprov", 5.0, budget_limit=5.0, spent=0.0)
    assert allowed is True
    assert cost > 0


async def test_check_budget_exceeded():
    svc = _make_service()
    allowed, cost = await svc.check_budget("costprov", 5.0, budget_limit=0.05, spent=0.0)
    assert allowed is False
    assert cost > 0.05


async def test_check_budget_remaining():
    svc = _make_service()
    allowed, _cost = await svc.check_budget("costprov", 5.0, budget_limit=0.15, spent=0.10)
    assert allowed is False


async def test_check_budget_unknown_provider():
    svc = _make_service()
    allowed, cost = await svc.check_budget("unknown", 5.0, budget_limit=1.0, spent=0.0)
    assert allowed is True
    assert cost == 0.0
