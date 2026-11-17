"""Unit tests for usage dashboard (Phase 3)."""

from __future__ import annotations

from uuid import UUID

from src.api.schemas import UsageByProviderItem, UsagePeriodItem, UsageResponse


def test_usage_response_empty():
    resp = UsageResponse(tenant_id=UUID("11111111-1111-1111-1111-111111111111"))
    assert resp.periods == []
    assert resp.by_provider == []
    assert resp.total_cost == 0.0
    assert resp.total_generations == 0


def test_usage_period_item():
    item = UsagePeriodItem(
        date="2024-01-15",
        generations_started=10,
        generations_completed=8,
        generations_failed=2,
        total_duration_generated_s=40.0,
        total_file_size_bytes=1024000,
    )
    assert item.date == "2024-01-15"
    assert item.generations_started == 10
    assert item.total_file_size_bytes == 1024000


def test_usage_by_provider():
    item = UsageByProviderItem(
        provider="runway",
        total_cost=5.50,
        job_count=25,
    )
    assert item.provider == "runway"
    assert item.total_cost == 5.50
    assert item.job_count == 25


def test_usage_response_with_data():
    resp = UsageResponse(
        tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
        date_from="2024-01-01",
        date_to="2024-01-31",
        periods=[
            UsagePeriodItem(
                date="2024-01-15",
                generations_completed=5,
            ),
        ],
        by_provider=[
            UsageByProviderItem(provider="runway", total_cost=2.50, job_count=5),
        ],
        total_cost=2.50,
        total_generations=5,
    )
    assert len(resp.periods) == 1
    assert len(resp.by_provider) == 1
    assert resp.total_cost == 2.50
    assert resp.total_generations == 5


def test_usage_aggregation():
    """Multiple periods aggregate correctly."""
    periods = [
        UsagePeriodItem(date="2024-01-01", generations_completed=3),
        UsagePeriodItem(date="2024-01-02", generations_completed=7),
    ]
    total = sum(p.generations_completed for p in periods)
    assert total == 10
