"""Unit tests for KlingProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_submit(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    from src.services.providers.kling import KlingProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"data": {"task_id": "kling-task-1"}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = KlingProvider()

    task_id = await p.submit("cinematic sunset", duration_s=5)
    assert task_id == "kling-task-1"
    mock_client.post.assert_awaited_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs.args[0] == "/videos/text2video"
    payload = call_kwargs.kwargs["json"]
    assert payload["prompt"] == "cinematic sunset"
    assert payload["model_name"] == "kling-v1"
    assert payload["duration"] == "5"


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_submit_with_negative_prompt(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    from src.services.providers.kling import KlingProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"data": {"task_id": "kling-task-2"}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = KlingProvider()

    task_id = await p.submit("a cat", negative_prompt="blurry", duration_s=5)
    assert task_id == "kling-task-2"
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["negative_prompt"] == "blurry"


# ---------------------------------------------------------------------------
# Check status
# ---------------------------------------------------------------------------


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_check_status_succeed(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    from src.services.providers.kling import KlingProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "task_status": "succeed",
            "task_result": {"videos": [{"url": "https://cdn.kling.ai/video.mp4"}]},
        }
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = KlingProvider()

    result = await p.check_status("kling-task-1")
    assert result["status"] == "completed"
    assert result["video_url"] == "https://cdn.kling.ai/video.mp4"


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_check_status_processing(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    from src.services.providers.kling import KlingProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"data": {"task_status": "processing", "task_result": {}}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = KlingProvider()

    result = await p.check_status("kling-task-1")
    assert result["status"] == "processing"
    assert result["video_url"] is None


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_check_status_failed(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    from src.services.providers.kling import KlingProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "task_status": "failed",
            "task_status_msg": "Content rejected",
            "task_result": {},
        }
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = KlingProvider()

    result = await p.check_status("kling-task-1")
    assert result["status"] == "failed"
    assert result["error"] == "Content rejected"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_download(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    from src.services.providers.kling import KlingProvider

    mock_response = MagicMock()
    mock_response.content = b"fake-video-bytes"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = KlingProvider()

    with patch(
        "src.services.providers.kling.safe_get",
        new=AsyncMock(return_value=mock_response),
    ) as mock_safe_get:
        data = await p.download("kling-task-1", "https://cdn.kling.ai/video.mp4")
    assert data == b"fake-video-bytes"
    # Pinned path used: safe_get got the download client positionally plus the URL.
    mock_safe_get.assert_awaited_once_with(mock_client, "https://cdn.kling.ai/video.mp4")
    # ...and the un-pinned bare GET (which would re-resolve the hostname) did not run.
    assert mock_client.get.await_count == 0


# ---------------------------------------------------------------------------
# Estimate cost
# ---------------------------------------------------------------------------


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_estimate_cost(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    from src.services.providers.kling import KlingProvider

    p = KlingProvider()
    # $0.03 per 5s, min 5s
    assert p.estimate_cost(5.0) == 0.03
    assert abs(p.estimate_cost(10.0) - 0.06) < 1e-9
    # Below min still charges for 5s
    assert p.estimate_cost(1.0) == 0.03
    assert p.estimate_cost(0.0) == 0.03


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_health_check_with_key(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test-key", model="kling-v1", timeout=300.0)
    )
    mock_resp = MagicMock(status_code=200)
    mock_client = AsyncMock()
    mock_client.head = AsyncMock(return_value=mock_resp)
    mock_get_client.return_value = mock_client
    from src.services.providers.kling import KlingProvider

    p = KlingProvider()
    assert await p.health_check() is True


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_health_check_without_key(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="", model="kling-v1", timeout=300.0)
    )
    from src.services.providers.kling import KlingProvider

    p = KlingProvider()
    assert await p.health_check() is False


# ---------------------------------------------------------------------------
# Model info / metadata
# ---------------------------------------------------------------------------


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_get_models(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    from src.services.providers.kling import KlingProvider

    p = KlingProvider()
    assert p.get_models() == ["kling-v1", "kling-v1-5"]


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_get_max_duration_s(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    from src.services.providers.kling import KlingProvider

    p = KlingProvider()
    assert p.get_max_duration_s() == 10.0


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_supports_image_to_video(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    from src.services.providers.kling import KlingProvider

    p = KlingProvider()
    assert p.supports_image_to_video() is True


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_name(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    from src.services.providers.kling import KlingProvider

    p = KlingProvider()
    assert p.name == "kling"


# ---------------------------------------------------------------------------
# Download is IP-pinned: a blocked target must not degrade into a soft failure
# ---------------------------------------------------------------------------


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
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
    from src.services.providers.kling import KlingProvider

    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = KlingProvider()

    with (
        patch(
            "src.services.providers.kling.safe_get",
            new=AsyncMock(side_effect=SSRFError("resolves to blocked address 169.254.169.254")),
        ),
        patch.object(p, "submit", new=AsyncMock(return_value="kling-task-1")),
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


@patch("src.services.providers.kling.get_client")
@patch("src.services.providers.kling.get_settings")
async def test_download_oversized_is_not_degraded_to_a_failed_result(
    mock_settings, mock_get_client
):
    """``ResponseTooLargeError`` also subclasses ValueError -- same degrade trap."""
    from w2v_common.net import ResponseTooLargeError

    from src.services.exceptions import ProviderError
    from src.services.providers.kling import KlingProvider

    mock_settings.return_value = MagicMock(
        kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = KlingProvider()

    with (
        patch(
            "src.services.providers.kling.safe_get",
            new=AsyncMock(side_effect=ResponseTooLargeError("body exceeds cap")),
        ),
        patch.object(p, "submit", new=AsyncMock(return_value="kling-task-1")),
        patch.object(
            p, "_poll_until_complete", new=AsyncMock(return_value="https://cdn.kling.ai/video.mp4")
        ),
        pytest.raises(ProviderError) as excinfo,
    ):
        await p.generate(prompt="a cat")

    assert excinfo.value.retriable is False
