"""Unit tests for request validation (Phase 1)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.api.routes import _validate_generate_request
from src.api.schemas import GenerateRequest


def _make_request(registry=None):
    """Create a mock request with a registry."""
    request = MagicMock()
    if registry is None:
        registry = MagicMock()
        registry.get.return_value = None
        registry.provider_names = []
    request.app.state.provider_registry = registry
    return request


def _make_registry(providers=None):
    """Create a mock registry with optional providers."""
    registry = MagicMock()
    if providers:

        def get_side(name):
            return providers.get(name)

        registry.get.side_effect = get_side
        registry.provider_names = list(providers.keys())
    else:
        registry.get.return_value = None
        registry.provider_names = []
    return registry


def _make_provider(max_duration=None, supports_i2v=False):
    """Create a mock provider."""
    provider = MagicMock()
    provider.get_max_duration_s.return_value = max_duration
    provider.supports_image_to_video.return_value = supports_i2v
    return provider


def test_unknown_provider_422():
    registry = _make_registry()
    request = _make_request(registry)
    body = GenerateRequest(prompt="test", provider="nonexistent")
    with pytest.raises(HTTPException) as exc_info:
        _validate_generate_request(request, body)
    assert exc_info.value.status_code == 422
    assert "Unknown provider" in str(exc_info.value.detail)


def test_duration_exceeding_max_422():
    provider = _make_provider(max_duration=10.0)
    registry = _make_registry({"runway": provider})
    request = _make_request(registry)
    body = GenerateRequest(prompt="test", provider="runway", duration_s=15.0)
    with pytest.raises(HTTPException) as exc_info:
        _validate_generate_request(request, body)
    assert exc_info.value.status_code == 422
    assert "exceeds provider max" in str(exc_info.value.detail)


def test_invalid_image_url_422():
    provider = _make_provider(supports_i2v=True)
    registry = _make_registry({"runway": provider})
    request = _make_request(registry)
    body = GenerateRequest(prompt="test", provider="runway", image_url="ftp://bad.com/img.png")
    with pytest.raises(HTTPException) as exc_info:
        _validate_generate_request(request, body)
    assert exc_info.value.status_code == 422
    assert "HTTP(S)" in str(exc_info.value.detail)


def test_i2v_unsupported_422():
    provider = _make_provider(supports_i2v=False)
    registry = _make_registry({"veo": provider})
    request = _make_request(registry)
    body = GenerateRequest(prompt="test", provider="veo", image_url="https://example.com/img.png")
    with pytest.raises(HTTPException) as exc_info:
        _validate_generate_request(request, body)
    assert exc_info.value.status_code == 422
    assert "image-to-video" in str(exc_info.value.detail)


def test_unknown_fallback_provider_422():
    provider = _make_provider()
    registry = _make_registry({"runway": provider})
    request = _make_request(registry)
    body = GenerateRequest(prompt="test", provider="runway", fallback_providers=["nonexistent"])
    with pytest.raises(HTTPException) as exc_info:
        _validate_generate_request(request, body)
    assert exc_info.value.status_code == 422
    assert "Unknown fallback provider" in str(exc_info.value.detail)


def test_valid_request_passes():
    provider = _make_provider(max_duration=30.0, supports_i2v=True)
    registry = _make_registry({"runway": provider})
    request = _make_request(registry)
    body = GenerateRequest(
        prompt="test",
        provider="runway",
        duration_s=5.0,
        image_url="https://example.com/img.png",
        width=1920,
        height=1080,
    )
    _validate_generate_request(request, body)  # Should not raise


def test_no_provider_passes():
    """No provider specified - skip provider validation."""
    registry = _make_registry()
    request = _make_request(registry)
    body = GenerateRequest(prompt="test")
    _validate_generate_request(request, body)  # Should not raise


def test_prompt_min_length():
    """Prompt must be at least 1 character (enforced by Pydantic)."""
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="")


def test_duration_constraints():
    """Duration must be > 0 and <= 300 (enforced by Pydantic)."""
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="test", duration_s=0)
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="test", duration_s=301)


def test_dimension_constraints():
    """Width/height must be 128-4096 (enforced by Pydantic)."""
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="test", width=64)
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="test", height=5000)


def test_invalid_callback_url_422():
    registry = _make_registry()
    request = _make_request(registry)
    body = GenerateRequest(prompt="test", callback_url="not-a-url")
    with pytest.raises(HTTPException) as exc_info:
        _validate_generate_request(request, body)
    assert exc_info.value.status_code == 422
    assert "callback_url" in str(exc_info.value.detail)
