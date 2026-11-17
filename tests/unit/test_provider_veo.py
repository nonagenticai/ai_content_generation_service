"""Unit tests for VeoProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_submit(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
    )
    from src.services.providers.veo import VeoProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "operations/veo-op-123"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = VeoProvider()

    task_id = await p.submit("a futuristic city", duration_s=5)
    assert task_id == "operations/veo-op-123"
    mock_client.post.assert_awaited_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs.args[0] == "/models/veo-2:predictLongRunning"
    payload = call_kwargs.kwargs["json"]
    assert payload["instances"][0]["prompt"] == "a futuristic city"
    assert payload["parameters"]["sampleCount"] == 1


# ---------------------------------------------------------------------------
# Check status - LRO done
# ---------------------------------------------------------------------------


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_check_status_done(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
    )
    from src.services.providers.veo import VeoProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "done": True,
        "response": {
            "generatedSamples": [{"video": {"uri": "https://storage.googleapis.com/video.mp4"}}]
        },
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = VeoProvider()

    result = await p.check_status("operations/veo-op-123")
    assert result["status"] == "completed"
    assert result["video_url"] == "https://storage.googleapis.com/video.mp4"


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_check_status_done_with_error(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
    )
    from src.services.providers.veo import VeoProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "done": True,
        "response": {"generatedSamples": []},
        "error": {"message": "Safety filter triggered"},
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = VeoProvider()

    result = await p.check_status("operations/veo-op-123")
    # A finished op carrying an error is a FAILURE, not a billable completion.
    # Previously this returned "completed" with video_url=None, so the base
    # poller treated it as success and the tenant was billed for no video.
    assert result["status"] == "failed"
    assert result["video_url"] is None
    assert result["error"] == "Safety filter triggered"


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_check_status_done_without_video_is_failed(mock_settings, mock_get_client):
    """A finished op with no error but no produced sample must be 'failed' too —
    a 'completed' with video_url=None would bill for a nonexistent video."""
    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
    )
    from src.services.providers.veo import VeoProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"done": True, "response": {"generatedSamples": []}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = VeoProvider()

    result = await p.check_status("operations/veo-op-123")
    assert result["status"] == "failed"
    assert result["video_url"] is None
    assert result["error"]  # a non-empty explanatory message


# ---------------------------------------------------------------------------
# Check status - LRO not done
# ---------------------------------------------------------------------------


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_check_status_not_done(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
    )
    from src.services.providers.veo import VeoProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"done": False}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    p = VeoProvider()

    result = await p.check_status("operations/veo-op-123")
    assert result["status"] == "processing"
    assert result["video_url"] is None
    assert result["error"] is None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_download(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
    )
    from src.services.providers.veo import VeoProvider

    mock_response = MagicMock()
    mock_response.content = b"veo-video-bytes"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = VeoProvider()

    with patch(
        "src.services.providers.veo.safe_get",
        new=AsyncMock(return_value=mock_response),
    ) as mock_safe_get:
        data = await p.download("operations/veo-op-123", "https://storage.googleapis.com/video.mp4")
    assert data == b"veo-video-bytes"
    # Pinned path used: safe_get got the download client positionally plus the URL.
    mock_safe_get.assert_awaited_once_with(mock_client, "https://storage.googleapis.com/video.mp4")
    # ...and the un-pinned bare GET (which would re-resolve the hostname) did not run.
    assert mock_client.get.await_count == 0


# ---------------------------------------------------------------------------
# Estimate cost
# ---------------------------------------------------------------------------


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_estimate_cost(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
    )
    from src.services.providers.veo import VeoProvider

    p = VeoProvider()
    # $0.04 per 5s, min 5s
    assert p.estimate_cost(5.0) == 0.04
    assert abs(p.estimate_cost(10.0) - 0.08) < 1e-9
    # Below min still charges for 5s
    assert p.estimate_cost(1.0) == 0.04
    assert p.estimate_cost(0.0) == 0.04


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_health_check_with_key(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test-key", model="veo-2", timeout=300.0)
    )
    mock_resp = MagicMock(status_code=200)
    mock_client = AsyncMock()
    mock_client.head = AsyncMock(return_value=mock_resp)
    mock_get_client.return_value = mock_client
    from src.services.providers.veo import VeoProvider

    p = VeoProvider()
    assert await p.health_check() is True


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_health_check_without_key(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(veo=MagicMock(api_key="", model="veo-2", timeout=300.0))
    from src.services.providers.veo import VeoProvider

    p = VeoProvider()
    assert await p.health_check() is False


# ---------------------------------------------------------------------------
# Model info / metadata
# ---------------------------------------------------------------------------


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_get_models(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
    )
    from src.services.providers.veo import VeoProvider

    p = VeoProvider()
    assert p.get_models() == ["veo-2"]


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_supports_image_to_video(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
    )
    from src.services.providers.veo import VeoProvider

    p = VeoProvider()
    assert p.supports_image_to_video() is False


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_name(mock_settings, mock_get_client):
    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
    )
    from src.services.providers.veo import VeoProvider

    p = VeoProvider()
    assert p.name == "veo"


# ---------------------------------------------------------------------------
# Download is IP-pinned: a blocked target must not degrade into a soft failure
# ---------------------------------------------------------------------------


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
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
    from src.services.providers.veo import VeoProvider

    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = VeoProvider()

    with (
        patch(
            "src.services.providers.veo.safe_get",
            new=AsyncMock(side_effect=SSRFError("resolves to blocked address 169.254.169.254")),
        ),
        patch.object(p, "submit", new=AsyncMock(return_value="operations/veo-op-123")),
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


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_download_oversized_is_not_degraded_to_a_failed_result(
    mock_settings, mock_get_client
):
    """``ResponseTooLargeError`` also subclasses ValueError -- same degrade trap."""
    from w2v_common.net import ResponseTooLargeError

    from src.services.exceptions import ProviderError
    from src.services.providers.veo import VeoProvider

    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock()
    mock_get_client.return_value = mock_client

    p = VeoProvider()

    with (
        patch(
            "src.services.providers.veo.safe_get",
            new=AsyncMock(side_effect=ResponseTooLargeError("body exceeds cap")),
        ),
        patch.object(p, "submit", new=AsyncMock(return_value="operations/veo-op-123")),
        patch.object(
            p,
            "_poll_until_complete",
            new=AsyncMock(return_value="https://storage.googleapis.com/video.mp4"),
        ),
        pytest.raises(ProviderError) as excinfo,
    ):
        await p.generate(prompt="a cat")

    assert excinfo.value.retriable is False


@patch("src.services.providers.veo.get_client")
@patch("src.services.providers.veo.get_settings")
async def test_submit_targets_the_instance_model(mock_settings, mock_get_client):
    """The predict path names the INSTANCE model (settings or slot), not a literal."""
    mock_settings.return_value = MagicMock(
        veo=MagicMock(api_key="test", model="veo-3.0-generate-001", timeout=300.0)
    )
    from src.services.providers.veo import VeoProvider

    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "operations/x"}
    mock_response.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    await VeoProvider().submit("p")

    assert mock_client.post.call_args.args[0] == "/models/veo-3.0-generate-001:predictLongRunning"
