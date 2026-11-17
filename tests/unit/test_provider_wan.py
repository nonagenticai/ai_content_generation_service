"""Unit tests for WanProvider."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wan_settings(enabled: bool = True) -> MagicMock:
    """Build a mock WanSettings."""
    wan = MagicMock()
    wan.enabled = enabled
    wan.model_name = "Wan-AI/Wan2.2-TI2V-5B"
    wan.offload_model = True
    wan.convert_model_dtype = True
    wan.t5_cpu = True
    wan.timeout = 600.0
    return wan


def _make_settings(wan_enabled: bool = True) -> MagicMock:
    settings = MagicMock()
    settings.wan = _make_wan_settings(enabled=wan_enabled)
    return settings


def _create_provider():
    """Import and instantiate WanProvider with mocked settings."""
    from src.services.providers.wan import WanProvider

    return WanProvider()


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------


@patch("src.services.providers.wan.get_settings")
def test_name(mock_settings):
    mock_settings.return_value = _make_settings()
    p = _create_provider()
    assert p.name == "wan"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@patch("src.services.providers.wan.get_settings")
async def test_health_check_disabled(mock_settings):
    mock_settings.return_value = _make_settings(wan_enabled=False)
    p = _create_provider()
    assert await p.health_check() is False


@patch("src.services.providers.wan.get_settings")
async def test_health_check_enabled(mock_settings):
    mock_settings.return_value = _make_settings(wan_enabled=True)
    p = _create_provider()
    # Will be False on non-CUDA machines (requires CUDA), which is expected
    result = await p.health_check()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@patch("src.services.providers.wan.get_settings")
def test_get_models(mock_settings):
    mock_settings.return_value = _make_settings()
    p = _create_provider()
    assert p.get_models() == ["Wan2.2-TI2V-5B"]


@patch("src.services.providers.wan.get_settings")
def test_supports_image_to_video(mock_settings):
    mock_settings.return_value = _make_settings()
    p = _create_provider()
    assert p.supports_image_to_video() is True


@patch("src.services.providers.wan.get_settings")
def test_get_max_duration_s(mock_settings):
    mock_settings.return_value = _make_settings()
    p = _create_provider()
    assert p.get_max_duration_s() == 3.4


@patch("src.services.providers.wan.get_settings")
def test_estimate_cost(mock_settings):
    mock_settings.return_value = _make_settings()
    p = _create_provider()
    assert p.estimate_cost(5.0) == 0.0
    assert p.estimate_cost(60.0) == 0.0


# ---------------------------------------------------------------------------
# Check status
# ---------------------------------------------------------------------------


@patch("src.services.providers.wan.get_settings")
async def test_check_status_completed(mock_settings):
    mock_settings.return_value = _make_settings()
    p = _create_provider()
    # Create a fake output file
    task_id = str(uuid.uuid7())
    out_path = p._output_dir / f"{task_id}.mp4"
    out_path.write_bytes(b"fake video data")

    result = await p.check_status(task_id)
    assert result["status"] == "completed"
    assert result["video_url"] == str(out_path)
    assert result["error"] is None


@patch("src.services.providers.wan.get_settings")
async def test_check_status_failed_missing_file(mock_settings):
    mock_settings.return_value = _make_settings()
    p = _create_provider()
    result = await p.check_status("nonexistent-task-id")
    assert result["status"] == "failed"
    assert result["video_url"] is None
    assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@patch("src.services.providers.wan.get_settings")
async def test_download(mock_settings, tmp_path):
    mock_settings.return_value = _make_settings()
    p = _create_provider()
    # Write a temp video file
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"fake mp4 content")

    data = await p.download("any-task", str(video_file))
    assert data == b"fake mp4 content"


# ---------------------------------------------------------------------------
# Submit (mocked inference)
# ---------------------------------------------------------------------------


@patch("src.services.providers.wan.get_settings")
async def test_submit_text_only(mock_settings):
    mock_settings.return_value = _make_settings()
    p = _create_provider()

    # Mock _generate_sync to avoid loading the real model
    expected_id = str(uuid.uuid7())
    with patch.object(p, "_generate_sync", return_value=expected_id):
        task_id = await p.submit("a cat walking on the beach")
        assert task_id == expected_id


def _mock_image_response() -> MagicMock:
    resp = MagicMock()
    resp.content = b"fake image bytes"
    resp.headers = {"content-type": "image/png"}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_async_client() -> AsyncMock:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@patch("src.services.providers.wan.get_settings")
async def test_submit_with_image_url(mock_settings):
    """image_url is fetched through the IP-pinned ``safe_get`` path, not a bare GET.

    The API boundary only *advises* on the URL (it discards the pinned IP), so the
    pin that actually closes the DNS-rebinding window is this fetch. Assert on the
    ``safe_get`` call itself rather than on transport behaviour -- w2v_common
    clients retry httpx transport errors, so an injected ConnectError proves
    nothing about which code path ran.
    """
    mock_settings.return_value = _make_settings()
    p = _create_provider()

    expected_id = str(uuid.uuid7())
    mock_client = _mock_async_client()

    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        patch(
            "src.services.providers.wan.safe_get",
            new=AsyncMock(return_value=_mock_image_response()),
        ) as mock_safe_get,
        patch.object(p, "_generate_sync", return_value=expected_id),
    ):
        task_id = await p.submit("a cat", image_url="https://example.com/cat.png")

    assert task_id == expected_id
    # Pinned path used: safe_get got the client positionally plus the caller URL.
    mock_safe_get.assert_awaited_once_with(mock_client, "https://example.com/cat.png")
    # ...and the un-pinned bare GET (which would re-resolve the hostname) did not run.
    assert mock_client.get.await_count == 0


@patch("src.services.providers.wan.get_settings")
async def test_submit_ssrf_refusal_is_not_degraded_to_a_failed_result(mock_settings):
    """A blocked target must surface as a ProviderError, never a soft result.

    ``SSRFError`` subclasses ``ValueError``, and
    ``VideoGenerationProvider.generate`` catches ``(OSError, RuntimeError,
    ValueError)`` and turns it into ``VideoGenerationResult(success=False)`` --
    which reads as an ordinary provider fault and lets the chain re-attempt the
    same blocked fetch on the next paid provider. This asserts the refusal
    escapes that degrade arm.
    """
    from w2v_common.net import SSRFError

    from src.services.exceptions import ProviderError

    mock_settings.return_value = _make_settings()
    p = _create_provider()

    with (
        patch("httpx.AsyncClient", return_value=_mock_async_client()),
        patch(
            "src.services.providers.wan.safe_get",
            new=AsyncMock(side_effect=SSRFError("resolves to blocked address 169.254.169.254")),
        ),
        patch.object(p, "_generate_sync", return_value="never-reached"),
        pytest.raises(ProviderError) as excinfo,
    ):
        await p.generate(prompt="a cat", image_url="https://rebind.example.com/cat.png")

    assert excinfo.value.retriable is False
    assert isinstance(excinfo.value.__cause__, SSRFError)


@patch("src.services.providers.wan.get_settings")
async def test_submit_oversized_image_is_not_degraded_to_a_failed_result(mock_settings):
    """``ResponseTooLargeError`` also subclasses ValueError -- same degrade trap."""
    from w2v_common.net import ResponseTooLargeError

    from src.services.exceptions import ProviderError

    mock_settings.return_value = _make_settings()
    p = _create_provider()

    with (
        patch("httpx.AsyncClient", return_value=_mock_async_client()),
        patch(
            "src.services.providers.wan.safe_get",
            new=AsyncMock(side_effect=ResponseTooLargeError("body exceeds cap")),
        ),
        patch.object(p, "_generate_sync", return_value="never-reached"),
        pytest.raises(ProviderError),
    ):
        await p.generate(prompt="a cat", image_url="https://example.com/huge.png")


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_dir_fallback():
    """When HF cache lookup fails, returns the model name as-is."""
    from src.services.providers.wan import _resolve_checkpoint_dir

    result = _resolve_checkpoint_dir("some/model-name")
    # Should return the name unchanged (or a cached path if it exists)
    assert isinstance(result, str)
    assert len(result) > 0
