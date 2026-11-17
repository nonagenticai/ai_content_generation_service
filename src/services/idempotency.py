"""``Idempotency-Key`` support for the job-creating POST routes.

Why this exists: the orchestrator's ``ai_image_gen`` / ``ai_video_gen`` stages
re-submit the SAME generation after a client-side wait timeout and send an
``Idempotency-Key`` so the re-submit gets the ORIGINAL job back. Until this
module, ACG ignored the header and created a second job + background task for
every re-submit. On the single-lane local image backend (~20s per render) that
doubled GPU work (28 renders for 16 delivered images in one window), lengthened
the queue, and caused more timeouts -- a self-feeding loop.

Contract:

* The key is unique per ``(tenant_id, idempotency_key)`` -- a partial unique
  index (migration 016) enforces it, so two tenants may use the same key.
* A repeat with the same key and the same body returns the existing job and
  starts NO background task.
* The same key with a DIFFERENT body is a client bug: 422.
* A FAILED or CANCELLED job releases its key, so a retry after a terminal
  failure (or after the caller cancelled on timeout) gets a fresh attempt
  instead of being handed the dead job forever.
* Race-safe: ``INSERT ... ON CONFLICT DO NOTHING RETURNING``; the loser of a
  concurrent duplicate blocks on the winner's uncommitted row, then sees the
  conflict and reads the winner's committed row (READ COMMITTED re-snapshots
  per statement).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

if TYPE_CHECKING:
    from pydantic import BaseModel
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.database.models import ImageGenJob, VideoGenerationJob

    JobModel = type[ImageGenJob] | type[VideoGenerationJob]
    JobRow = ImageGenJob | VideoGenerationJob

IDEMPOTENCY_HEADER = "Idempotency-Key"
#: Response header set on a replayed (deduplicated) submission.
REPLAYED_HEADER = "Idempotent-Replayed"
MAX_KEY_LENGTH = 255

#: Statuses that RELEASE a key: a retry after these gets a new attempt.
RELEASING_STATUSES = ("failed", "cancelled")


class IdempotentReplayError(Exception):
    """Raised by a job manager's create when the key already names a live job."""

    def __init__(self, job: JobRow) -> None:
        super().__init__(f"idempotency key already bound to job {job.job_id}")
        self.job = job


def read_idempotency_key(request: Request) -> str | None:
    """The request's ``Idempotency-Key``, or None. 422 on an unusable value."""
    raw = request.headers.get(IDEMPOTENCY_HEADER)
    if raw is None:
        return None
    key = raw.strip()
    if not key or len(key) > MAX_KEY_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"{IDEMPOTENCY_HEADER} must be 1-{MAX_KEY_LENGTH} characters",
        )
    return key


def request_fingerprint(body: BaseModel) -> str:
    """sha256 of the canonical JSON body.

    ``SecretStr`` fields (a provider slot's secret) dump as a fixed mask in
    ``mode="json"``, so no secret material reaches the hash.
    """
    canonical = json.dumps(
        body.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def ensure_same_request(job: JobRow, fingerprint: str) -> None:
    """422 when a key is reused with a different body."""
    if job.idempotency_fingerprint != fingerprint:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{IDEMPOTENCY_HEADER} was already used for a different request body; "
                "use a new key for a different request"
            ),
        )


async def find_live_job(
    db: AsyncSession, model: JobModel, *, tenant_id: str, key: str
) -> JobRow | None:
    """The job currently holding ``key`` for ``tenant_id`` (never a released one)."""
    result = await db.execute(
        select(model).where(model.tenant_id == tenant_id, model.idempotency_key == key)
    )
    job = result.scalar_one_or_none()
    if job is None or job.status in RELEASING_STATUSES:
        return None
    return job


async def insert_job_once(
    db: AsyncSession,
    model: JobModel,
    values: dict[str, object],
    *,
    tenant_id: str,
    key: str,
) -> JobRow:
    """Insert a job row bound to ``key``; raise :class:`IdempotentReplayError` on a live dup.

    ``values`` must already carry ``idempotency_key`` / ``idempotency_fingerprint``.
    """
    # A failed/cancelled holder gives the key up, so the insert below can take it.
    await db.execute(
        update(model)
        .where(
            model.tenant_id == tenant_id,
            model.idempotency_key == key,
            model.status.in_(RELEASING_STATUSES),
        )
        .values(idempotency_key=None)
    )
    stmt = (
        pg_insert(model)
        .values(**values)
        .on_conflict_do_nothing(
            index_elements=[model.tenant_id, model.idempotency_key],
            index_where=model.idempotency_key.isnot(None),
        )
        .returning(model.id)
    )
    inserted_id = (await db.execute(stmt)).scalar_one_or_none()
    if inserted_id is not None:
        return (await db.execute(select(model).where(model.id == inserted_id))).scalar_one()
    existing = (
        await db.execute(
            select(model).where(model.tenant_id == tenant_id, model.idempotency_key == key)
        )
    ).scalar_one_or_none()
    if existing is None:  # pragma: no cover -- conflict implies a committed holder
        raise RuntimeError(f"idempotency conflict on {key!r} but no holder row is visible")
    raise IdempotentReplayError(existing)
