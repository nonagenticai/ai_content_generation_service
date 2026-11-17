"""Unit tests for preview endpoint (Phase 3)."""

from __future__ import annotations

from pydantic import ValidationError

from src.api.schemas import PreviewRequest, PreviewResponse


def test_preview_request_defaults():
    req = PreviewRequest(prompt="test prompt")
    assert req.duration_s == 5.0
    assert req.provider is None


def test_preview_request_duration_cap():
    """Duration is capped at 10s by schema."""
    import pytest

    with pytest.raises(ValidationError):
        PreviewRequest(prompt="test", duration_s=15.0)


def test_preview_response_success():
    resp = PreviewResponse(
        status="completed",
        provider="runway",
        video_base64="dGVzdA==",
        estimated_cost=0.05,
    )
    assert resp.status == "completed"
    assert resp.video_base64 == "dGVzdA=="


def test_preview_response_failure():
    resp = PreviewResponse(
        status="failed",
        provider="runway",
        error="Generation failed",
        estimated_cost=0.05,
    )
    assert resp.status == "failed"
    assert resp.error == "Generation failed"


def test_preview_response_timeout():
    resp = PreviewResponse(
        status="timeout",
        provider="runway",
        error="Preview generation timed out",
    )
    assert resp.status == "timeout"
