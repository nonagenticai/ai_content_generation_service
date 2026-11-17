"""Unit tests for SeedanceProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_submit(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"task_id": "seed-task-abc"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = SeedanceProvider()

    task_id = await p.submit(
        "ocean waves at sunset",
        duration_s=5,
        width=1920,
        height=1080,
        style="realistic",
    )
    assert task_id == "seed-task-abc"
    mock_client.post.assert_awaited_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs.args[0] == "/generations"
    payload = call_kwargs.kwargs["json"]
    assert payload["prompt"] == "ocean waves at sunset"
    assert payload["model"] == "seedance-1"
    assert payload["duration"] == 5
    assert payload["width"] == 1920
    assert payload["height"] == 1080
    assert payload["style"] == "realistic"


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_submit_fallback_id_field(mock_settings, mock_get_client):
    """Provider accepts 'id' as fallback when 'task_id' is absent."""
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"id": "seed-fallback-id"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = SeedanceProvider()

    task_id = await p.submit("test prompt")
    assert task_id == "seed-fallback-id"


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_submit_no_task_id_raises(mock_settings, mock_get_client):
    """Provider raises RuntimeError when neither task_id nor id is returned."""
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = SeedanceProvider()

    with pytest.raises(RuntimeError, match="No task ID"):
        await p.submit("prompt")


# ---------------------------------------------------------------------------
# Check status
# ---------------------------------------------------------------------------


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_check_status_completed(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "completed",
        "video_url": "https://cdn.seedance.ai/video.mp4",
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = SeedanceProvider()

    result = await p.check_status("seed-task-abc")
    assert result["status"] == "completed"
    assert result["video_url"] == "https://cdn.seedance.ai/video.mp4"
    assert result["error"] is None


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_check_status_pending(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "pending"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = SeedanceProvider()

    result = await p.check_status("seed-task-abc")
    assert result["status"] == "pending"
    assert result["video_url"] is None


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_check_status_error_maps_to_failed(mock_settings, mock_get_client):
    """The 'error' status should map to 'failed'."""
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "error",
        "error": "Internal server error",
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = SeedanceProvider()

    result = await p.check_status("seed-task-abc")
    assert result["status"] == "failed"
    assert result["error"] == "Internal server error"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_download(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    mock_response = MagicMock()
    mock_response.content = b"seedance-video-bytes"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = SeedanceProvider()

    with patch(
        "src.services.providers.seedance.safe_get",
        new=AsyncMock(return_value=mock_response),
    ) as mock_safe_get:
        data = await p.download("seed-task-abc", "https://cdn.seedance.ai/video.mp4")
    assert data == b"seedance-video-bytes"
    # Pinned path used: safe_get got the download client positionally plus the URL.
    mock_safe_get.assert_awaited_once_with(mock_client, "https://cdn.seedance.ai/video.mp4")
    # ...and the un-pinned bare GET (which would re-resolve the hostname) did not run.
    assert mock_client.get.await_count == 0


# ---------------------------------------------------------------------------
# Estimate cost
# ---------------------------------------------------------------------------


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_estimate_cost(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    p = SeedanceProvider()
    # $0.04 per 5s, min 5s
    assert p.estimate_cost(5.0) == 0.04
    assert abs(p.estimate_cost(10.0) - 0.08) < 1e-9
    # Below min still charges for 5s
    assert p.estimate_cost(1.0) == 0.04
    assert p.estimate_cost(0.0) == 0.04


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_health_check_with_key(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test-key", model="seedance-1", timeout=300.0)
    )
    mock_resp = MagicMock(status_code=200)
    mock_client = AsyncMock()
    mock_client.head = AsyncMock(return_value=mock_resp)
    mock_get_client.return_value = mock_client
    from src.services.providers.seedance import SeedanceProvider

    p = SeedanceProvider()
    assert await p.health_check() is True


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_health_check_without_key(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    p = SeedanceProvider()
    assert await p.health_check() is False


# ---------------------------------------------------------------------------
# Model info / metadata
# ---------------------------------------------------------------------------


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_get_models(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    p = SeedanceProvider()
    assert p.get_models() == ["seedance-1"]


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_get_styles(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    p = SeedanceProvider()
    assert p.get_styles() == ["realistic", "artistic", "animated"]


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_get_max_duration_s(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    p = SeedanceProvider()
    assert p.get_max_duration_s() == 10.0


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_supports_image_to_video(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    p = SeedanceProvider()
    assert p.supports_image_to_video() is True


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_name(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    from src.services.providers.seedance import SeedanceProvider

    p = SeedanceProvider()
    assert p.name == "seedance"


# ---------------------------------------------------------------------------
# Download is IP-pinned: a blocked target must not degrade into a soft failure
# ---------------------------------------------------------------------------


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_download_ssrf_refusal_is_not_degraded_to_a_failed_result(
    mock_settings, mock_get_client
):
    """A blocked download target must surface as a ProviderError, never a soft result.

    ``SSRFError`` subclasses ``ValueError``, and
    ``VideoGenerationProvider.generate`` catches ``(OSError, RuntimeError,
    ValueError)`` and turns it into ``VideoGenerationResult(success=False)`` --
    which reads as an ordinary provider fault and lets the chain re-attempt the
    same blocked fetch on the next paid provider. This asserts the refusal
    escapes that degrade arm.
    """
    from w2v_common.net import SSRFError

    from src.services.exceptions import ProviderError
    from src.services.providers.seedance import SeedanceProvider

    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = SeedanceProvider()

    with (
        patch(
            "src.services.providers.seedance.safe_get",
            new=AsyncMock(side_effect=SSRFError("resolves to blocked address 169.254.169.254")),
        ),
        patch.object(p, "submit", new=AsyncMock(return_value="seed-task-abc")),
        patch.object(
            p,
            "_poll_until_complete",
            new=AsyncMock(return_value="https://rebind.example.com/video.mp4"),
        ),
        pytest.raises(ProviderError) as excinfo,
    ):
        await p.generate(prompt="a cat")

    assert excinfo.value.retriable is False
    assert isinstance(excinfo.value.__cause__, SSRFError)
    # The refusal happened before any un-pinned GET could be issued.
    assert mock_client.get.await_count == 0


@patch("src.services.providers.seedance.get_client")
@patch("src.services.providers.seedance.get_settings")
async def test_download_oversized_is_not_degraded_to_a_failed_result(
    mock_settings, mock_get_client
):
    """``ResponseTooLargeError`` also subclasses ValueError -- same degrade trap."""
    from w2v_common.net import ResponseTooLargeError

    from src.services.exceptions import ProviderError
    from src.services.providers.seedance import SeedanceProvider

    mock_settings.return_value = MagicMock(
        seedance=MagicMock(api_key="test", model="seedance-1", timeout=300.0)
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = SeedanceProvider()

    with (
        patch(
            "src.services.providers.seedance.safe_get",
            new=AsyncMock(side_effect=ResponseTooLargeError("body exceeds cap")),
        ),
        patch.object(p, "submit", new=AsyncMock(return_value="seed-task-abc")),
        patch.object(
            p,
            "_poll_until_complete",
            new=AsyncMock(return_value="https://cdn.seedance.ai/video.mp4"),
        ),
        pytest.raises(ProviderError) as excinfo,
    ):
        await p.generate(prompt="a cat")

    assert excinfo.value.retriable is False
