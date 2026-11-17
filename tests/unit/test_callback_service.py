"""Unit tests for callback service (Phase 3)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.services.callback_service import _MAX_RETRIES, deliver_callback


async def test_callback_success():
    db = MagicMock()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("src.services.callback_service.get_client", return_value=mock_client):
        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is True


async def test_callback_retry_on_failure():
    db = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=OSError("connection refused"))

    with (
        patch("src.services.callback_service.get_client", return_value=mock_client),
        patch("src.services.callback_service.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is False
    assert mock_client.post.call_count == 3


async def test_callback_connect_error_retries_and_marks_failed():
    """A connection-refused webhook (httpx.ConnectError — a RequestError/HTTPError,
    NOT an OSError) must be retried _MAX_RETRIES times, then set callback_status
    'failed' and return False WITHOUT propagating the exception."""
    db = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    with (
        patch("src.services.callback_service.get_client", return_value=mock_client),
        patch("src.services.callback_service.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "src.services.callback_service._update_callback_status",
            new_callable=AsyncMock,
        ) as mock_status,
    ):
        # Must NOT raise — the ConnectError is caught inside the retry loop.
        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is False
    assert mock_client.post.call_count == _MAX_RETRIES
    mock_status.assert_awaited_once_with(db, "job-1", "failed")


async def test_callback_404_is_not_retried():
    """A permanent 4xx (404) callback must fail on the FIRST attempt without
    burning retries — a dead URL will never succeed."""
    db = MagicMock()

    request = httpx.Request("POST", "https://example.com/webhook")
    response = httpx.Response(404, request=request)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("not found", request=request, response=response)
    )

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with (
        patch("src.services.callback_service.get_client", return_value=mock_client),
        patch("src.services.callback_service.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is False
    assert mock_client.post.call_count == 1  # no retries on a terminal 4xx
    mock_sleep.assert_not_awaited()


async def test_callback_503_is_retried():
    """A transient 5xx (503) callback must be retried the full _MAX_RETRIES."""
    db = MagicMock()

    request = httpx.Request("POST", "https://example.com/webhook")
    response = httpx.Response(503, request=request)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("unavailable", request=request, response=response)
    )

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with (
        patch("src.services.callback_service.get_client", return_value=mock_client),
        patch("src.services.callback_service.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is False
    assert mock_client.post.call_count == _MAX_RETRIES


async def test_callback_all_retries_exhausted():
    db = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=OSError("timeout"))

    with (
        patch("src.services.callback_service.get_client", return_value=mock_client),
        patch("src.services.callback_service.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is False
