"""Retention for ``cancelled`` jobs and for the whole image side.

Two gaps, one sweep. ``CleanupService.run_cleanup`` visited exactly
``[completed, failed]`` on exactly ``video_generation_jobs``, so:

  * a ``cancelled`` job (a real terminal status, written by ``mark_cancelled`` on
    both managers) was never pruned; and
  * ``imagegen_jobs`` / ``imagegen_batches`` had no retention path at all.

The row growth is the small half. ``_delete_job_storage`` runs ONLY on this sweep
and nothing else in the service ever deletes a generated artifact, so a status or
a table the sweep does not visit strands its MinIO object permanently. The
cluster's whole-bucket lifecycle rule is hand-applied, in no git repo, and expires
delivered work too — it is not a backstop this service may lean on. So the
load-bearing assertions here are the ``delete_from_minio`` calls, not the row
counts.

Real Postgres via ``db_session_factory`` (the fixture's ``create_all`` applies the
live SECURITY DEFINER resolvers), real rows, real resolver, real DELETE.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

from sqlalchemy import func, select

from src.database.models import ImageGenBatch, ImageGenJob, VideoGenerationJob
from src.services.cleanup_service import CleanupService
from src.services.image_job_manager import ImageJobManager
from tests.conftest import TENANT_A


async def _seed_video(factory, *, job_id, status, days_ago, video_path=None, minio_key=None):
    now = datetime.now(UTC)
    async with factory() as db:
        db.add(
            VideoGenerationJob(
                job_id=job_id,
                tenant_id=TENANT_A,
                provider="runway",
                prompt="p",
                status=status,
                video_path=video_path,
                minio_key=minio_key,
                created_at=now,
                updated_at=now,
                completed_at=now - timedelta(days=days_ago),
            )
        )
        await db.commit()


async def _seed_image(
    factory, *, job_id, status, days_ago, image_path=None, minio_key=None, batch_id=None
):
    now = datetime.now(UTC)
    async with factory() as db:
        db.add(
            ImageGenJob(
                job_id=job_id,
                asset_id=f"asset-{job_id}",
                batch_id=batch_id,
                tenant_id=TENANT_A,
                provider="openai",
                prompt="p",
                status=status,
                image_path=image_path,
                minio_key=minio_key,
                created_at=now,
                updated_at=now,
                completed_at=now - timedelta(days=days_ago),
            )
        )
        await db.commit()


async def _seed_batch(factory, *, batch_id, status, days_ago, job_count=1):
    now = datetime.now(UTC)
    async with factory() as db:
        db.add(
            ImageGenBatch(
                batch_id=batch_id,
                tenant_id=TENANT_A,
                status=status,
                job_count=job_count,
                completed_jobs=job_count,
                failed_jobs=0,
                created_at=now,
                updated_at=now,
                completed_at=now - timedelta(days=days_ago),
            )
        )
        await db.commit()


async def _count(factory, model) -> int:
    async with factory() as db:
        return (await db.execute(select(func.count()).select_from(model))).scalar() or 0


def _service(factory, storage):
    return CleanupService(
        completed_ttl_days=30,
        failed_ttl_days=7,
        cancelled_ttl_days=7,
        storage=storage,
        session_factory=factory,
    )


def _minio_keys(storage) -> set[str]:
    return {c.args[0] for c in storage.delete_from_minio.await_args_list}


# ---------------------------------------------------------------------------
# Finding 1: cancelled is pruned, and its MinIO object is actually freed
# ---------------------------------------------------------------------------


async def test_expired_cancelled_video_job_frees_its_minio_object(db_session_factory):
    """The one that matters: the object, not just the row."""
    await _seed_video(
        db_session_factory,
        job_id="cancelled-video",
        status="cancelled",
        days_ago=8,
        video_path="/tmp/cancelled-video/clip.mp4",
        minio_key="content_generation/cancelled-video/clip.mp4",
    )
    storage = AsyncMock()

    assert await _service(db_session_factory, storage).run_cleanup() == 1

    assert await _count(db_session_factory, VideoGenerationJob) == 0
    storage.delete_from_minio.assert_awaited_once_with(
        "content_generation/cancelled-video/clip.mp4"
    )
    storage.delete_local.assert_awaited_once_with("/tmp/cancelled-video/clip.mp4")


async def test_expired_cancelled_image_job_frees_its_minio_object(db_session_factory):
    await _seed_image(
        db_session_factory,
        job_id="cancelled-image",
        status="cancelled",
        days_ago=8,
        image_path="/tmp/cancelled-image/i.png",
        minio_key="content_generation/cancelled-image/i.png",
    )
    storage = AsyncMock()

    assert await _service(db_session_factory, storage).run_cleanup() == 1

    assert await _count(db_session_factory, ImageGenJob) == 0
    storage.delete_from_minio.assert_awaited_once_with("content_generation/cancelled-image/i.png")


async def test_recent_cancelled_job_is_not_prematurely_deleted(db_session_factory):
    """Control: the new status must obey its TTL, not delete on sight."""
    await _seed_video(
        db_session_factory,
        job_id="fresh-cancel",
        status="cancelled",
        days_ago=1,
        minio_key="content_generation/fresh-cancel/clip.mp4",
    )
    await _seed_image(
        db_session_factory,
        job_id="fresh-cancel-img",
        status="cancelled",
        days_ago=1,
        minio_key="content_generation/fresh-cancel/i.png",
    )
    storage = AsyncMock()

    assert await _service(db_session_factory, storage).run_cleanup() == 0

    assert await _count(db_session_factory, VideoGenerationJob) == 1
    assert await _count(db_session_factory, ImageGenJob) == 1
    storage.delete_from_minio.assert_not_awaited()


async def test_completed_and_failed_video_jobs_still_prune(db_session_factory):
    """Control: the pre-existing sweep is not broken by the new statuses."""
    await _seed_video(
        db_session_factory,
        job_id="old-completed",
        status="completed",
        days_ago=31,
        minio_key="content_generation/old-completed/clip.mp4",
    )
    await _seed_video(
        db_session_factory,
        job_id="old-failed",
        status="failed",
        days_ago=8,
        minio_key="content_generation/old-failed/clip.mp4",
    )
    await _seed_video(db_session_factory, job_id="recent-completed", status="completed", days_ago=1)
    await _seed_video(
        db_session_factory, job_id="still-processing", status="processing", days_ago=99
    )
    storage = AsyncMock()

    assert await _service(db_session_factory, storage).run_cleanup() == 2

    assert await _count(db_session_factory, VideoGenerationJob) == 2
    assert _minio_keys(storage) == {
        "content_generation/old-completed/clip.mp4",
        "content_generation/old-failed/clip.mp4",
    }


# ---------------------------------------------------------------------------
# Finding 2: the image tables have a retention path at all
# ---------------------------------------------------------------------------


async def test_expired_completed_image_job_is_pruned_and_its_object_freed(db_session_factory):
    await _seed_image(
        db_session_factory,
        job_id="old-image",
        status="completed",
        days_ago=31,
        image_path="/tmp/old-image/i.png",
        minio_key="content_generation/old-image/i.png",
    )
    await _seed_image(db_session_factory, job_id="recent-image", status="completed", days_ago=1)
    storage = AsyncMock()

    assert await _service(db_session_factory, storage).run_cleanup() == 1

    assert await _count(db_session_factory, ImageGenJob) == 1
    storage.delete_from_minio.assert_awaited_once_with("content_generation/old-image/i.png")


async def test_childless_terminal_batch_is_pruned(db_session_factory):
    await _seed_batch(db_session_factory, batch_id="old-batch", status="completed", days_ago=31)
    storage = AsyncMock()

    assert await _service(db_session_factory, storage).run_cleanup() == 1
    assert await _count(db_session_factory, ImageGenBatch) == 0


async def test_failed_batch_retires_on_the_failed_ttl(db_session_factory):
    """Per-status batch TTLs, not one batch clock."""
    await _seed_batch(db_session_factory, batch_id="failed-batch", status="failed", days_ago=8)
    await _seed_batch(db_session_factory, batch_id="fresh-failed", status="failed", days_ago=1)
    storage = AsyncMock()

    assert await _service(db_session_factory, storage).run_cleanup() == 1
    assert await _count(db_session_factory, ImageGenBatch) == 1


async def test_batch_with_an_in_flight_child_is_left_alone(db_session_factory, caplog):
    """The resolver's ``NOT EXISTS``, not the exception handler.

    A ``processing`` child means the batch is genuinely in flight whatever its own
    status says, and the sweep now DELETES a retiring batch's children — so without
    the guard this would destroy running work. Even the FK would not stop it. The
    log is what separates a guarded skip from a crashed cycle: a clean cycle must
    not have logged "Cleanup failed for image batches".
    """
    await _seed_batch(db_session_factory, batch_id="live-batch", status="partial", days_ago=99)
    await _seed_image(
        db_session_factory,
        job_id="live-child",
        status="processing",
        days_ago=99,
        batch_id="live-batch",
    )
    storage = AsyncMock()

    with caplog.at_level(logging.ERROR, logger="src.services.cleanup_service"):
        assert await _service(db_session_factory, storage).run_cleanup() == 0

    assert await _count(db_session_factory, ImageGenBatch) == 1
    assert await _count(db_session_factory, ImageGenJob) == 1
    assert "Cleanup failed for image batches" not in caplog.text


async def test_non_terminal_batch_is_never_pruned(db_session_factory):
    """A ``processing`` batch is in flight however old it is — that is the reaper's job."""
    await _seed_batch(db_session_factory, batch_id="inflight", status="processing", days_ago=99)
    storage = AsyncMock()

    assert await _service(db_session_factory, storage).run_cleanup() == 0
    assert await _count(db_session_factory, ImageGenBatch) == 1


async def test_a_retiring_batch_takes_its_children_and_their_objects(db_session_factory):
    """One unit: children's storage freed, children deleted, parent deleted."""
    await _seed_batch(
        db_session_factory, batch_id="done-batch", status="completed", days_ago=31, job_count=2
    )
    await _seed_image(
        db_session_factory,
        job_id="done-child",
        status="completed",
        days_ago=31,
        batch_id="done-batch",
        minio_key="content_generation/done-child/i.png",
    )
    await _seed_image(
        db_session_factory,
        job_id="cancelled-child",
        status="cancelled",
        days_ago=31,
        batch_id="done-batch",
        minio_key="content_generation/cancelled-child/i.png",
    )
    storage = AsyncMock()

    assert await _service(db_session_factory, storage).run_cleanup() == 3

    assert await _count(db_session_factory, ImageGenJob) == 0
    assert await _count(db_session_factory, ImageGenBatch) == 0
    assert _minio_keys(storage) == {
        "content_generation/done-child/i.png",
        "content_generation/cancelled-child/i.png",
    }


# ---------------------------------------------------------------------------
# The verifier's note: a batch must never disagree with its own results
# ---------------------------------------------------------------------------


async def test_a_batch_never_outlives_a_child_and_reports_stale_totals(db_session_factory):
    """⛔ The regression this pins.

    ``get_concept_image_batch`` renders ``results``/``job_ids`` from surviving child
    rows but ``job_count``/``completed_jobs``/``failed_jobs`` from the batch's stored
    counters. If a child ages out on its own status TTL while the parent survives,
    the endpoint returns a truncated list beside counters claiming the original
    totals — an absent measurement rendered as a real value, and a caller cannot
    tell a reaped row from a lost one.

    A MIXED batch is the sharp case: a ``completed`` child (30d) beside a ``failed``
    one (7d). No choice of batch TTL fixes it — for 23 days the batch is
    inconsistent with itself. The invariant is therefore structural: a batched child
    is never swept on its own TTL, so at day 8 the batch is still WHOLE.
    """
    await _seed_batch(
        db_session_factory, batch_id="mixed", status="partial", days_ago=8, job_count=2
    )
    await _seed_image(
        db_session_factory,
        job_id="mixed-completed",
        status="completed",
        days_ago=8,
        batch_id="mixed",
        minio_key="content_generation/mixed-completed/i.png",
    )
    await _seed_image(
        db_session_factory,
        job_id="mixed-failed",
        status="failed",
        days_ago=8,
        batch_id="mixed",
        minio_key="content_generation/mixed-failed/i.png",
    )
    storage = AsyncMock()

    assert await _service(db_session_factory, storage).run_cleanup() == 0

    # Whole: both children still render, so ``results`` still matches job_count=2.
    assert await _count(db_session_factory, ImageGenJob) == 2
    assert await _count(db_session_factory, ImageGenBatch) == 1
    storage.delete_from_minio.assert_not_awaited()


async def test_a_batched_child_is_never_swept_on_its_own_ttl(db_session_factory):
    """The narrow invariant, isolated from the batch's own eligibility.

    The parent is far too young to retire (1 day). Its ``failed`` child is 99 days
    old — well past the 7-day failed TTL — and must STILL be there, because the
    batch owns the retention decision for everything under it.
    """
    await _seed_batch(db_session_factory, batch_id="young", status="partial", days_ago=1)
    await _seed_image(
        db_session_factory,
        job_id="old-failed-child",
        status="failed",
        days_ago=99,
        batch_id="young",
        minio_key="content_generation/old-failed-child/i.png",
    )
    storage = AsyncMock()

    assert await _service(db_session_factory, storage).run_cleanup() == 0

    assert await _count(db_session_factory, ImageGenJob) == 1
    assert await _count(db_session_factory, ImageGenBatch) == 1
    storage.delete_from_minio.assert_not_awaited()


# ---------------------------------------------------------------------------
# The check-then-act window: the terminal guard lives in the DELETE
# ---------------------------------------------------------------------------


async def test_a_running_child_is_not_deletable_even_when_named(db_session_factory):
    """⛔ The structural guarantee, tested at the statement.

    The batch sweep enumerates eligibility in one session and deletes in another.
    READ COMMITTED, no ``FOR UPDATE``: a row that was terminal at enumeration is
    only *believed* terminal at delete. Nothing can revive one today, but that
    rests on an unstated invariant spread over three call sites
    (``create_job`` writes ``pending`` with its batch in the same request,
    ``mark_processing`` requires ``pending``, ``reclaim_orphaned_image_job`` only
    picks rows already non-terminal) — and the cost of a later refactor breaking
    it is deleting live work AND its storage.

    So the predicate is asserted here, in the statement, by naming a running row
    explicitly and requiring the DELETE to refuse it.
    """
    await _seed_image(db_session_factory, job_id="running", status="processing", days_ago=99)
    await _seed_image(db_session_factory, job_id="done", status="completed", days_ago=99)
    await _seed_image(db_session_factory, job_id="queued", status="pending", days_ago=99)

    async with db_session_factory() as db:
        deleted = await ImageJobManager.delete_terminal_jobs(db, ["running", "done", "queued"])
        await db.commit()

    assert deleted == 1
    async with db_session_factory() as db:
        survivors = set((await db.execute(select(ImageGenJob.job_id))).scalars().all())
    assert survivors == {"running", "queued"}


async def test_the_batch_sweep_deletes_children_through_the_guarded_statement(
    db_session_factory, monkeypatch
):
    """The sweep must go through ``delete_terminal_jobs``, not a raw DELETE.

    A future edit could satisfy every other test in this file by deleting the
    children with an unguarded statement — the guard would still exist, and be
    bypassed. This pins the call site to it.
    """
    calls: list[list[str]] = []
    real = ImageJobManager.delete_terminal_jobs

    async def _spy(db, job_ids):
        calls.append(list(job_ids))
        return await real(db, job_ids)

    monkeypatch.setattr(ImageJobManager, "delete_terminal_jobs", staticmethod(_spy))

    await _seed_batch(db_session_factory, batch_id="spied", status="completed", days_ago=31)
    await _seed_image(
        db_session_factory,
        job_id="spied-child",
        status="completed",
        days_ago=31,
        batch_id="spied",
    )

    assert await _service(db_session_factory, AsyncMock()).run_cleanup() == 2
    assert ["spied-child"] in calls
