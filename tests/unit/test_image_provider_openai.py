"""Unit tests for OpenAIImageProvider."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import DalleImageSettings

_MINIMAL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9VE0cJQAAAAASUVORK5CYII="
)
_MINIMAL_PNG_B64 = base64.b64encode(_MINIMAL_PNG_BYTES).decode()


def _make_settings(**overrides: Any) -> MagicMock:
    params: dict[str, Any] = {
        "enabled": True,
        "api_key": "test-key",  # pragma: allowlist secret
        "model": "gpt-image-1",
        "size": "1024x1024",
        "timeout": 60.0,
    }
    params.update(overrides)
    return MagicMock(dalle=DalleImageSettings(**params))


def _create_provider(mock_get_client: MagicMock) -> Any:
    from src.services.image_providers.openai import OpenAIImageProvider

    api_client = AsyncMock()
    mock_get_client.return_value = api_client
    return OpenAIImageProvider()


def _generations_response(item: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"data": [item]}
    response.raise_for_status = MagicMock()
    return response


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


@patch("src.services.image_providers.openai.get_client")
@patch("src.services.image_providers.openai.get_settings")
async def test_generate_from_b64_json_does_not_fetch_anything(mock_settings, mock_get_client):
    """The inline-base64 branch never touches the network."""
    mock_settings.return_value = _make_settings()
    p = _create_provider(mock_get_client)
    p._client.post = AsyncMock(return_value=_generations_response({"b64_json": _MINIMAL_PNG_B64}))

    with patch("src.services.image_providers.openai.safe_get", new=AsyncMock()) as mock_safe_get:
        result = await p.generate(prompt="a cat")

    assert result.success is True
    assert result.image_data == _MINIMAL_PNG_BYTES
    assert result.width == 1024
    assert result.height == 1024
    assert mock_safe_get.await_count == 0


@patch("src.services.image_providers.openai.get_client")
@patch("src.services.image_providers.openai.get_settings")
async def test_generate_from_url_uses_the_pinned_fetch(mock_settings, mock_get_client):
    """A URL-shaped response is re-fetched through ``safe_get``, not a bare GET.

    The URL comes back from the OpenAI response, so nothing this service chose
    constrains where it points. ``safe_get`` re-validates, pins the connection to
    the validated IP (closing the DNS-rebinding TOCTOU window) and re-checks every
    redirect hop. Assert on the ``safe_get`` call itself rather than on transport
    behaviour -- w2v_common clients retry httpx transport errors, so an injected
    ConnectError would prove nothing about which code path ran.
    """
    mock_settings.return_value = _make_settings()
    p = _create_provider(mock_get_client)
    p._client.post = AsyncMock(
        return_value=_generations_response({"url": "https://cdn.openai.com/img.png"})
    )

    download = MagicMock()
    download.content = _MINIMAL_PNG_BYTES
    download.raise_for_status = MagicMock()

    dl_client = AsyncMock()
    dl_client.get = AsyncMock()
    mock_get_client.return_value = dl_client

    with patch(
        "src.services.image_providers.openai.safe_get",
        new=AsyncMock(return_value=download),
    ) as mock_safe_get:
        result = await p.generate(prompt="a cat")

    assert result.success is True
    assert result.image_data == _MINIMAL_PNG_BYTES
    assert result.image_url == "https://cdn.openai.com/img.png"
    # Pinned path used: safe_get got the download client positionally plus the URL.
    mock_safe_get.assert_awaited_once_with(dl_client, "https://cdn.openai.com/img.png")
    # ...and the un-pinned bare GET (which would re-resolve the hostname) did not run.
    assert dl_client.get.await_count == 0


# ---------------------------------------------------------------------------
# A blocked target must not degrade into a soft failure
# ---------------------------------------------------------------------------


@patch("src.services.image_providers.openai.get_client")
@patch("src.services.image_providers.openai.get_settings")
async def test_ssrf_refusal_is_not_degraded_to_a_failed_result(mock_settings, mock_get_client):
    """A blocked target must surface as a ProviderError, never ``success=False``.

    ``_extract_image`` reports ordinary problems as an ``(None, None, error)``
    tuple that becomes ``ImageGenResult(success=False)`` -- indistinguishable from
    a routine provider fault, which lets the caller fall through to the next (paid)
    provider and re-attempt the same blocked fetch. ``ProviderError(retriable=
    False)`` is what ``is_retriable_error`` reads to stop that chain.
    """
    from w2v_common.net import SSRFError

    from src.services.exceptions import ProviderError

    mock_settings.return_value = _make_settings()
    p = _create_provider(mock_get_client)
    p._client.post = AsyncMock(
        return_value=_generations_response({"url": "https://rebind.example.com/img.png"})
    )

    dl_client = AsyncMock()
    dl_client.get = AsyncMock()
    mock_get_client.return_value = dl_client

    with (
        patch(
            "src.services.image_providers.openai.safe_get",
            new=AsyncMock(side_effect=SSRFError("resolves to blocked address 169.254.169.254")),
        ),
        pytest.raises(ProviderError) as excinfo,
    ):
        await p.generate(prompt="a cat")

    assert excinfo.value.retriable is False
    assert isinstance(excinfo.value.__cause__, SSRFError)
    # The refusal happened before any un-pinned GET could be issued.
    assert dl_client.get.await_count == 0


@patch("src.services.image_providers.openai.get_client")
@patch("src.services.image_providers.openai.get_settings")
async def test_oversized_download_is_not_degraded_to_a_failed_result(
    mock_settings, mock_get_client
):
    """``ResponseTooLargeError`` also subclasses ValueError -- same degrade trap."""
    from w2v_common.net import ResponseTooLargeError

    from src.services.exceptions import ProviderError

    mock_settings.return_value = _make_settings()
    p = _create_provider(mock_get_client)
    p._client.post = AsyncMock(
        return_value=_generations_response({"url": "https://cdn.openai.com/huge.png"})
    )
    mock_get_client.return_value = AsyncMock()

    with (
        patch(
            "src.services.image_providers.openai.safe_get",
            new=AsyncMock(side_effect=ResponseTooLargeError("body exceeds cap")),
        ),
        pytest.raises(ProviderError) as excinfo,
    ):
        await p.generate(prompt="a cat")

    assert excinfo.value.retriable is False
    assert excinfo.value.provider == "openai_images"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


@patch("src.services.image_providers.openai.get_client")
@patch("src.services.image_providers.openai.get_settings")
async def test_generate_without_api_key_fails_softly(mock_settings, mock_get_client):
    mock_settings.return_value = _make_settings(api_key="")
    p = _create_provider(mock_get_client)

    result = await p.generate(prompt="a cat")
    assert result.success is False
    assert "api key" in (result.error or "").lower()


@patch("src.services.image_providers.openai.get_client")
@patch("src.services.image_providers.openai.get_settings")
async def test_generate_with_no_image_data_fails_softly(mock_settings, mock_get_client):
    """Missing image payload is an ordinary fault -- it keeps the soft-result shape."""
    mock_settings.return_value = _make_settings()
    p = _create_provider(mock_get_client)
    p._client.post = AsyncMock(return_value=_generations_response({}))

    result = await p.generate(prompt="a cat")
    assert result.success is False
    assert result.error == "OpenAI returned no image data"


@patch("src.services.image_providers.openai.get_client")
@patch("src.services.image_providers.openai.get_settings")
async def test_name_and_estimate_cost(mock_settings, mock_get_client):
    mock_settings.return_value = _make_settings()
    p = _create_provider(mock_get_client)

    assert p.name == "openai_images"
    assert p.estimate_cost(size="1024x1024", quality="standard") == 0.04
    assert p.estimate_cost(size="1792x1024", quality="hd") == 0.09
