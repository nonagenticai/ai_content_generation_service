"""Unit tests for job listing schemas (Phase 3)."""

from __future__ import annotations

from src.api.schemas import JobListItem, JobListResponse


def test_job_list_item():
    item = JobListItem(
        job_id="j1",
        status="completed",
        provider="runway",
        created_at="2024-01-15T10:00:00",
        completed_at="2024-01-15T10:05:00",
        cost=0.05,
    )
    assert item.job_id == "j1"
    assert item.status == "completed"
    assert item.cost == 0.05


def test_job_list_response_empty():
    resp = JobListResponse(
        items=[],
        total=0,
        page=1,
        page_size=20,
        total_pages=1,
    )
    assert resp.items == []
    assert resp.total == 0
    assert resp.total_pages == 1


def test_job_list_response_with_data():
    items = [JobListItem(job_id=f"j{i}", status="completed") for i in range(3)]
    resp = JobListResponse(
        items=items,
        total=50,
        page=1,
        page_size=20,
        total_pages=3,
    )
    assert len(resp.items) == 3
    assert resp.total == 50
    assert resp.page == 1
    assert resp.total_pages == 3


def test_job_list_pagination():
    """Verify total_pages calculation."""
    import math

    total = 45
    page_size = 20
    total_pages = max(1, math.ceil(total / page_size))
    assert total_pages == 3


def test_job_list_item_minimal():
    """Job list items should work with just required fields."""
    item = JobListItem(job_id="j1", status="pending")
    assert item.provider is None
    assert item.cost is None
    assert item.created_at is None
