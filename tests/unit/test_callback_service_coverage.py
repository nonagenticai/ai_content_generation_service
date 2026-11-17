"""Tests for callback_service.py to cover lines 35-42 and 63-70.

Covers: successful callback updates job.callback_status='sent',
and exhausted retries update job.callback_status='failed'.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.services.callback_service import deliver_callback


async def test_callback_success_updates_job_status_sent():
    """On success, job.callback_status is set to 'sent' via AsyncSession."""
    from sqlalchemy.ext.asyncio import AsyncSession

    db = MagicMock(spec=AsyncSession)
    db.commit = AsyncMock()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    job = MagicMock()
    job.callback_status = "pending"

    with (
        patch("src.services.callback_service.get_client", return_value=mock_client),
        patch("src.services.job_manager.JobManager.get_job", new_callable=AsyncMock) as mock_get,
    ):
        mock_get.return_value = job

        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is True
    assert job.callback_status == "sent"
    db.commit.assert_awaited()


async def test_callback_success_job_not_found():
    """On success, if job is None the status update is skipped."""
    from sqlalchemy.ext.asyncio import AsyncSession

    db = MagicMock(spec=AsyncSession)
    db.commit = AsyncMock()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with (
        patch("src.services.callback_service.get_client", return_value=mock_client),
        patch("src.services.job_manager.JobManager.get_job", new_callable=AsyncMock) as mock_get,
        patch(
            "src.services.image_job_manager.ImageJobManager.get_job",
            new_callable=AsyncMock,
        ) as mock_img_get,
    ):
        mock_get.return_value = None
        mock_img_get.return_value = None

        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is True


async def test_callback_success_db_update_exception():
    """On success, exception during db update is caught gracefully."""
    from sqlalchemy.ext.asyncio import AsyncSession

    db = MagicMock(spec=AsyncSession)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with (
        patch("src.services.callback_service.get_client", return_value=mock_client),
        patch(
            "src.services.job_manager.JobManager.get_job",
            new_callable=AsyncMock,
            side_effect=OSError("db error"),
        ),
        patch(
            "src.services.image_job_manager.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is True  # callback itself succeeded


async def test_callback_all_retries_exhausted_updates_status_failed():
    """When all retries fail, job.callback_status is set to 'failed'."""
    from sqlalchemy.ext.asyncio import AsyncSession

    db = MagicMock(spec=AsyncSession)
    db.commit = AsyncMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=OSError("connection refused"))

    job = MagicMock()
    job.callback_status = "pending"

    with (
        patch("src.services.callback_service.get_client", return_value=mock_client),
        patch("src.services.callback_service.asyncio.sleep", new_callable=AsyncMock),
        patch("src.services.job_manager.JobManager.get_job", new_callable=AsyncMock) as mock_get,
    ):
        mock_get.return_value = job

        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is False
    assert job.callback_status == "failed"
    db.commit.assert_awaited()


async def test_callback_all_retries_exhausted_job_not_found():
    """When retries exhausted but job not found, no crash."""
    from sqlalchemy.ext.asyncio import AsyncSession

    db = MagicMock(spec=AsyncSession)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=OSError("connection refused"))

    with (
        patch("src.services.callback_service.get_client", return_value=mock_client),
        patch("src.services.callback_service.asyncio.sleep", new_callable=AsyncMock),
        patch("src.services.job_manager.JobManager.get_job", new_callable=AsyncMock) as mock_get,
        patch(
            "src.services.image_job_manager.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        mock_get.return_value = None

        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is False


async def test_callback_all_retries_exhausted_db_exception():
    """When retries exhausted and db update fails, exception is caught."""
    from sqlalchemy.ext.asyncio import AsyncSession

    db = MagicMock(spec=AsyncSession)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=OSError("connection refused"))

    with (
        patch("src.services.callback_service.get_client", return_value=mock_client),
        patch("src.services.callback_service.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "src.services.job_manager.JobManager.get_job",
            new_callable=AsyncMock,
            side_effect=OSError("db error"),
        ),
        patch(
            "src.services.image_job_manager.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is False


async def test_callback_non_async_session_skips_db_update():
    """When db is not an AsyncSession, status update is skipped."""
    db = MagicMock()  # plain MagicMock, not spec=AsyncSession

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("src.services.callback_service.get_client", return_value=mock_client):
        result = await deliver_callback(
            db, "job-1", "https://example.com/webhook", {"status": "completed"}
        )

    assert result is True
