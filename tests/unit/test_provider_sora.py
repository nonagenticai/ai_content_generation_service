"""Unit tests for SoraProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Submit - returns generation ID
# ---------------------------------------------------------------------------


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_submit_returns_gen_id(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"id": "gen-xyz-123"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = SoraProvider()

    task_id = await p.submit("a dancing robot", duration_s=10, width=1920, height=1080)
    assert task_id == "gen-xyz-123"
    mock_client.post.assert_awaited_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs.args[0] == "/video/generations"
    payload = call_kwargs.kwargs["json"]
    assert payload["prompt"] == "a dancing robot"
    assert payload["model"] == "sora-1"
    assert payload["duration"] == 10
    assert payload["size"] == "1920x1080"


# ---------------------------------------------------------------------------
# Submit - direct URL return (no generation ID)
# ---------------------------------------------------------------------------


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_submit_direct_url(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"url": "https://cdn.openai.com/direct-video.mp4"}]}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = SoraProvider()

    task_id = await p.submit("a cat jumping")
    assert task_id == "direct:https://cdn.openai.com/direct-video.mp4"


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_submit_with_style_and_image(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"id": "gen-styled"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = SoraProvider()

    task_id = await p.submit(
        "a sunset",
        style="vivid",
        image_url="https://example.com/img.jpg",
    )
    assert task_id == "gen-styled"
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["style"] == "vivid"
    assert payload["image_url"] == "https://example.com/img.jpg"


# ---------------------------------------------------------------------------
# Check status - direct URL pattern
# ---------------------------------------------------------------------------


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_check_status_direct_url(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    p = SoraProvider()
    result = await p.check_status("direct:https://cdn.openai.com/video.mp4")
    assert result["status"] == "completed"
    assert result["video_url"] == "https://cdn.openai.com/video.mp4"
    assert result["error"] is None


# ---------------------------------------------------------------------------
# Check status - pending
# ---------------------------------------------------------------------------


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_check_status_pending(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "pending"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = SoraProvider()

    result = await p.check_status("gen-xyz-123")
    assert result["status"] == "pending"
    assert result["video_url"] is None


# ---------------------------------------------------------------------------
# Check status - completed
# ---------------------------------------------------------------------------


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_check_status_completed(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "completed",
        "data": [{"url": "https://cdn.openai.com/result.mp4"}],
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = SoraProvider()

    result = await p.check_status("gen-xyz-123")
    assert result["status"] == "completed"
    assert result["video_url"] == "https://cdn.openai.com/result.mp4"


# ---------------------------------------------------------------------------
# Check status - failed
# ---------------------------------------------------------------------------


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_check_status_failed(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "failed",
        "error": {"message": "Rate limited"},
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = SoraProvider()

    result = await p.check_status("gen-xyz-123")
    assert result["status"] == "failed"
    assert result["error"] == "Rate limited"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_download(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    mock_response = MagicMock()
    mock_response.content = b"sora-video-data"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = SoraProvider()

    with patch(
        "src.services.providers.sora.safe_get",
        new=AsyncMock(return_value=mock_response),
    ) as mock_safe_get:
        data = await p.download("gen-xyz-123", "https://cdn.openai.com/result.mp4")
    assert data == b"sora-video-data"
    # Pinned path used: safe_get got the download client positionally plus the URL.
    mock_safe_get.assert_awaited_once_with(mock_client, "https://cdn.openai.com/result.mp4")
    # ...and the un-pinned bare GET (which would re-resolve the hostname) did not run.
    assert mock_client.get.await_count == 0


# ---------------------------------------------------------------------------
# Estimate cost
# ---------------------------------------------------------------------------


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_estimate_cost(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    p = SoraProvider()
    # $0.10 per 5s, min 5s
    assert p.estimate_cost(5.0) == 0.10
    assert abs(p.estimate_cost(10.0) - 0.20) < 1e-9
    assert abs(p.estimate_cost(20.0) - 0.40) < 1e-9
    # Below min still charges for 5s
    assert p.estimate_cost(1.0) == 0.10
    assert p.estimate_cost(0.0) == 0.10


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_health_check_with_key(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test-key", model="sora-1", timeout=300.0)
    )
    mock_resp = MagicMock(status_code=200)
    mock_client = AsyncMock()
    mock_client.head = AsyncMock(return_value=mock_resp)
    mock_get_client.return_value = mock_client
    from src.services.providers.sora import SoraProvider

    p = SoraProvider()
    assert await p.health_check() is True


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_health_check_without_key(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    p = SoraProvider()
    assert await p.health_check() is False


# ---------------------------------------------------------------------------
# Model info / metadata
# ---------------------------------------------------------------------------


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_get_models(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    p = SoraProvider()
    assert p.get_models() == ["sora-1"]


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_get_styles(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    p = SoraProvider()
    assert p.get_styles() == ["natural", "vivid"]


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_get_max_duration_s(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    p = SoraProvider()
    assert p.get_max_duration_s() == 20.0


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_supports_image_to_video(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    p = SoraProvider()
    assert p.supports_image_to_video() is True


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_name(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    from src.services.providers.sora import SoraProvider

    p = SoraProvider()
    assert p.name == "sora"


# ---------------------------------------------------------------------------
# Download is IP-pinned: a blocked target must not degrade into a soft failure
# ---------------------------------------------------------------------------


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
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
    from src.services.providers.sora import SoraProvider

    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = SoraProvider()

    with (
        patch(
            "src.services.providers.sora.safe_get",
            new=AsyncMock(side_effect=SSRFError("resolves to blocked address 169.254.169.254")),
        ),
        patch.object(p, "submit", new=AsyncMock(return_value="gen-xyz-123")),
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


@patch("src.services.providers.sora.get_client")
@patch("src.services.providers.sora.get_settings")
async def test_download_oversized_is_not_degraded_to_a_failed_result(
    mock_settings, mock_get_client
):
    """``ResponseTooLargeError`` also subclasses ValueError -- same degrade trap."""
    from w2v_common.net import ResponseTooLargeError

    from src.services.exceptions import ProviderError
    from src.services.providers.sora import SoraProvider

    mock_settings.return_value = MagicMock(
        sora=MagicMock(api_key="test", model="sora-1", timeout=300.0)
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = SoraProvider()

    with (
        patch(
            "src.services.providers.sora.safe_get",
            new=AsyncMock(side_effect=ResponseTooLargeError("body exceeds cap")),
        ),
        patch.object(p, "submit", new=AsyncMock(return_value="gen-xyz-123")),
        patch.object(
            p,
            "_poll_until_complete",
            new=AsyncMock(return_value="https://cdn.openai.com/result.mp4"),
        ),
        pytest.raises(ProviderError) as excinfo,
    ):
        await p.generate(prompt="a cat")

    assert excinfo.value.retriable is False
