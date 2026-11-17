"""Webhook callback delivery service.

Delivers HTTP POST callbacks to notify external systems when
video generation jobs complete, with exponential backoff retries.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
from sqlalchemy.exc import SQLAlchemyError

from src.constants import HTTPX_CALLBACK_TIMEOUT_S
from src.services.errors import is_retriable_error
from src.services.http_client import get_client

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_SEQUENCE = [1, 2, 4]  # seconds


async def _update_callback_status(db: object, job_id: str, status: str) -> None:
    """Best-effort update of callback status for video or image jobs."""
    from sqlalchemy.ext.asyncio import AsyncSession

    if not isinstance(db, AsyncSession):
        return

    try:
        from src.services.job_manager import JobManager

        job = await JobManager.get_job(db, job_id)
        if job:
            job.callback_status = status
            await db.commit()
            return
    except (OSError, SQLAlchemyError) as exc:
        logger.warning("Failed to update video callback status for job %s: %s", job_id, exc)

    try:
        from src.services.image_job_manager import ImageJobManager

        image_job = await ImageJobManager.get_job(db, job_id)
        if image_job:
            image_job.callback_status = status
            await db.commit()
    except (OSError, SQLAlchemyError) as exc:
        logger.warning("Failed to update image callback status for job %s: %s", job_id, exc)


async def deliver_callback(
    db: object,
    job_id: str,
    callback_url: str,
    payload: dict,
) -> bool:
    """Deliver a webhook callback with retries.

    Args:
        db: Database session (used to update callback_status on the job).
        job_id: Job identifier for logging and status updates.
        callback_url: Target URL to POST the payload to.
        payload: JSON-serializable dict with job result data.

    Returns:
        True if callback was delivered successfully, False otherwise.
    """
    logger.info("Starting callback delivery: job_id=%s url=%s", job_id, callback_url)
    start = time.monotonic()
    for attempt in range(_MAX_RETRIES):
        try:
            client = get_client("callbacks", timeout=HTTPX_CALLBACK_TIMEOUT_S)
            resp = await client.post(callback_url, json=payload)
            resp.raise_for_status()

            await _update_callback_status(db, job_id, "sent")

            elapsed = time.monotonic() - start
            logger.info(
                "Callback delivered for job %s to %s (attempt=%d elapsed=%.2fs)",
                job_id,
                callback_url,
                attempt + 1,
                elapsed,
            )
            return True

        except httpx.HTTPStatusError as exc:
            # Only 5xx / 429 are transient; a permanent 4xx (400/401/403/404/
            # 405/410/422) will never succeed, so stop immediately instead of
            # burning the full 1+2+4s backoff on a dead callback URL.
            retriable = is_retriable_error(exc)
            logger.warning(
                "Callback delivery HTTP error for job %s (attempt %d/%d, retriable=%s): %s",
                job_id,
                attempt + 1,
                _MAX_RETRIES,
                retriable,
                exc,
            )
            if not retriable:
                break
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BACKOFF_SEQUENCE[attempt])
        except httpx.TimeoutException as exc:
            logger.warning(
                "Callback delivery timed out for job %s (attempt %d/%d): %s",
                job_id,
                attempt + 1,
                _MAX_RETRIES,
                exc,
            )
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BACKOFF_SEQUENCE[attempt])
        except (httpx.HTTPError, OSError) as exc:
            # httpx.ConnectError / ConnectTimeout / RequestError subclass
            # httpx.HTTPError (NOT OSError), so a connection-refused / DNS-fail
            # webhook lands here — retried with backoff, and on exhaustion the
            # terminal _update_callback_status(..., "failed") below runs instead
            # of the exception propagating to the caller's job handler.
            logger.warning(
                "Callback delivery failed for job %s (attempt %d/%d): %s",
                job_id,
                attempt + 1,
                _MAX_RETRIES,
                exc,
            )
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BACKOFF_SEQUENCE[attempt])

    # All retries exhausted
    await _update_callback_status(db, job_id, "failed")

    logger.error("All callback retries exhausted for job %s", job_id)
    return False
