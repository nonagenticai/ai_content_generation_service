"""Unit tests for RunwayProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_submit(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    from src.services.providers.runway import RunwayProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"id": "task-abc"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = RunwayProvider()

    task_id = await p.submit("test prompt", duration_s=5)
    assert task_id == "task-abc"
    mock_client.post.assert_awaited_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs.args[0] == "/image_to_video"
    payload = call_kwargs.kwargs["json"]
    assert payload["promptText"] == "test prompt"
    assert payload["model"] == "gen-3"
    assert payload["duration"] == 5


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_submit_with_image_url(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    from src.services.providers.runway import RunwayProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"id": "task-img"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = RunwayProvider()

    task_id = await p.submit("a cat", image_url="https://example.com/cat.jpg", duration_s=5)
    assert task_id == "task-img"
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["promptImage"] == "https://example.com/cat.jpg"


# ---------------------------------------------------------------------------
# Check status
# ---------------------------------------------------------------------------


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_check_status_succeeded(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    from src.services.providers.runway import RunwayProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "SUCCEEDED",
        "output": ["https://cdn.runway.com/video.mp4"],
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = RunwayProvider()

    result = await p.check_status("task-abc")
    assert result["status"] == "completed"
    assert result["video_url"] == "https://cdn.runway.com/video.mp4"
    assert result["error"] is None


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_check_status_pending(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    from src.services.providers.runway import RunwayProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "PENDING", "output": None}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = RunwayProvider()

    result = await p.check_status("task-abc")
    assert result["status"] == "pending"
    assert result["video_url"] is None


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_check_status_failed(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    from src.services.providers.runway import RunwayProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "FAILED",
        "failure": "Content policy violation",
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = RunwayProvider()

    result = await p.check_status("task-abc")
    assert result["status"] == "failed"
    assert result["error"] == "Content policy violation"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_download(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    from src.services.providers.runway import RunwayProvider

    mock_response = MagicMock()
    mock_response.content = b"\x00\x00\x00\x1cftyp"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = RunwayProvider()

    with patch(
        "src.services.providers.runway.safe_get",
        new=AsyncMock(return_value=mock_response),
    ) as mock_safe_get:
        data = await p.download("task-abc", "https://cdn.runway.com/video.mp4")
    assert data == b"\x00\x00\x00\x1cftyp"
    # Pinned path used: safe_get got the download client positionally plus the URL.
    mock_safe_get.assert_awaited_once_with(mock_client, "https://cdn.runway.com/video.mp4")
    # ...and the un-pinned bare GET (which would re-resolve the hostname) did not run.
    assert mock_client.get.await_count == 0


# ---------------------------------------------------------------------------
# Estimate cost
# ---------------------------------------------------------------------------


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_estimate_cost(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    from src.services.providers.runway import RunwayProvider

    p = RunwayProvider()
    # $0.05 per 5s, min 5s
    assert p.estimate_cost(5.0) == 0.05
    assert abs(p.estimate_cost(10.0) - 0.10) < 1e-9
    # Below min still charges for 5s
    assert p.estimate_cost(1.0) == 0.05
    assert p.estimate_cost(0.0) == 0.05


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_health_check_with_key(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test-key", model="gen-3", timeout=300.0)
    )
    mock_resp = MagicMock(status_code=200)
    mock_client = AsyncMock()
    mock_client.head = AsyncMock(return_value=mock_resp)
    mock_get_client.return_value = mock_client
    from src.services.providers.runway import RunwayProvider

    p = RunwayProvider()
    assert await p.health_check() is True


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_health_check_without_key(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="", model="gen-3", timeout=300.0)
    )
    from src.services.providers.runway import RunwayProvider

    p = RunwayProvider()
    assert await p.health_check() is False


# ---------------------------------------------------------------------------
# Model info / metadata
# ---------------------------------------------------------------------------


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_get_models(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    from src.services.providers.runway import RunwayProvider

    p = RunwayProvider()
    assert p.get_models() == ["gen-3", "gen-3-turbo"]


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_get_max_duration_s(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    from src.services.providers.runway import RunwayProvider

    p = RunwayProvider()
    assert p.get_max_duration_s() == 10.0


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_supports_image_to_video(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    from src.services.providers.runway import RunwayProvider

    p = RunwayProvider()
    assert p.supports_image_to_video() is True


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_name(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    from src.services.providers.runway import RunwayProvider

    p = RunwayProvider()
    assert p.name == "runway"


# ---------------------------------------------------------------------------
# Download is IP-pinned: a blocked target must not degrade into a soft failure
# ---------------------------------------------------------------------------


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
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
    from src.services.providers.runway import RunwayProvider

    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = RunwayProvider()

    with (
        patch(
            "src.services.providers.runway.safe_get",
            new=AsyncMock(side_effect=SSRFError("resolves to blocked address 169.254.169.254")),
        ),
        patch.object(p, "submit", new=AsyncMock(return_value="task-abc")),
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


@patch("src.services.providers.runway.get_client")
@patch("src.services.providers.runway.get_settings")
async def test_download_oversized_is_not_degraded_to_a_failed_result(
    mock_settings, mock_get_client
):
    """``ResponseTooLargeError`` also subclasses ValueError -- same degrade trap."""
    from w2v_common.net import ResponseTooLargeError

    from src.services.exceptions import ProviderError
    from src.services.providers.runway import RunwayProvider

    mock_settings.return_value = MagicMock(
        runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = RunwayProvider()

    with (
        patch(
            "src.services.providers.runway.safe_get",
            new=AsyncMock(side_effect=ResponseTooLargeError("body exceeds cap")),
        ),
        patch.object(p, "submit", new=AsyncMock(return_value="task-abc")),
        patch.object(
            p,
            "_poll_until_complete",
            new=AsyncMock(return_value="https://cdn.runway.com/video.mp4"),
        ),
        pytest.raises(ProviderError) as excinfo,
    ):
        await p.generate(prompt="a cat")

    assert excinfo.value.retriable is False
