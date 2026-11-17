"""Durable multi-replica dispatch: reaper reclaim + lease semantics (Postgres).

These exercise the SECURITY DEFINER reclaim functions and the manager lease
helpers against the real ``ai_content_generation_service_ci`` database. No
SQLite — the whole point is the ``FOR UPDATE SKIP LOCKED`` claim + PG interval
math, which SQLite cannot model.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from sqlalchemy import text

from src.database.models import ImageGenJob, VideoGenerationJob
from src.services.durable_dispatch import KIND_IMAGE_BATCH
from src.services.image_batch_manager import ImageBatchManager
from src.services.image_job_manager import ImageJobManager
from src.services.job_manager import ClaimedJob, JobManager
from src.services.reaper_service import ReaperService
from tests.conftest import TENANT_A, TENANT_B, TENANT_C, TENANT_D, TENANT_E

_PAST = timedelta(seconds=-10)
_FUTURE = timedelta(seconds=600)


def _reaper_with_imagegen_service() -> ReaperService:
    """A ReaperService wired to a real FastAPI app carrying a mock imagegen service."""
    app = FastAPI()
    app.state.imagegen_service = MagicMock()
    return ReaperService(
        app,
        interval_s=1.0,
        lease_seconds=900,
        max_attempts=5,
        heartbeat_enabled=False,
    )


async def _insert_video_job(
    db,
    *,
    tenant_id: str,
    status: str,
    lease_delta: timedelta | None,
    attempts: int = 0,
    job_options: dict | None = None,
    claimed_by: str | None = "pod-a",
    started_at: datetime | None = None,
) -> str:
    job_id = str(uuid.uuid7())
    lease = datetime.now(UTC) + lease_delta if lease_delta is not None else None
    db.add(
        VideoGenerationJob(
            job_id=job_id,
            tenant_id=tenant_id,
            provider="stub",
            prompt="a durable clip",
            status=status,
            attempts=attempts,
            claimed_by=claimed_by,
            lease_expires_at=lease,
            started_at=started_at,
            job_options=job_options if job_options is not None else {"kind": "video_single"},
        )
    )
    await db.commit()
    return job_id


async def _insert_image_job(
    db,
    *,
    tenant_id: str,
    status: str,
    lease_delta: timedelta | None,
    attempts: int = 0,
    claimed_by: str | None = "pod-a",
    started_at: datetime | None = None,
) -> str:
    job_id = str(uuid.uuid7())
    lease = datetime.now(UTC) + lease_delta if lease_delta is not None else None
    db.add(
        ImageGenJob(
            job_id=job_id,
            asset_id=str(uuid.uuid7()),
            tenant_id=tenant_id,
            provider="stub",
            prompt="a durable image",
            status=status,
            attempts=attempts,
            claimed_by=claimed_by,
            lease_expires_at=lease,
            started_at=started_at,
            job_options={"kind": "image_single", "budget_limit": 5.0},
        )
    )
    await db.commit()
    return job_id


async def test_crashed_pod_video_job_reclaimed_by_another_worker(db_session):
    """A processing job whose pod died (expired lease) is reclaimed by any worker."""
    job_id = await _insert_video_job(
        db_session,
        tenant_id=TENANT_B,
        status="processing",
        lease_delta=_PAST,
        job_options={"kind": "video_single", "budget_limit": 3.0, "callback_url": None},
    )

    claimed = await JobManager.reclaim_orphaned(
        db_session, worker_id="pod-b", lease_seconds=900, max_attempts=5
    )

    assert claimed is not None
    assert claimed.job_id == job_id
    assert claimed.tenant_id == TENANT_B
    assert claimed.options["budget_limit"] == 3.0

    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    # Reset to pending under the NEW worker + a fresh lease so the existing
    # per-job processor can re-run it (its mark_processing guard makes it
    # exactly-once even if the dead pod is somehow still alive).
    assert job.status == "pending"
    assert job.claimed_by == "pod-b"
    assert job.attempts == 1
    assert job.lease_expires_at is not None
    assert job.lease_expires_at > datetime.now(UTC)


async def test_reaper_reclaims_stale_but_not_live_heartbeating_job(db_session):
    """Only the expired-lease job is reclaimed; a fresh-lease job is left alone."""
    stale = await _insert_video_job(
        db_session, tenant_id=TENANT_A, status="processing", lease_delta=_PAST
    )
    live = await _insert_video_job(
        db_session, tenant_id=TENANT_A, status="processing", lease_delta=_FUTURE
    )

    first = await JobManager.reclaim_orphaned(
        db_session, worker_id="pod-b", lease_seconds=900, max_attempts=5
    )
    assert first is not None
    assert first.job_id == stale

    # Nothing else is reclaimable — the live, heartbeating job is untouched.
    second = await JobManager.reclaim_orphaned(
        db_session, worker_id="pod-b", lease_seconds=900, max_attempts=5
    )
    assert second is None

    live_job = await JobManager.get_job(db_session, live)
    assert live_job is not None
    assert live_job.status == "processing"
    assert live_job.claimed_by == "pod-a"  # never reassigned


async def test_reaper_fails_poison_job_past_max_attempts(db_session):
    """A job past ``max_attempts`` is failed (not re-dispatched) as poison."""
    job_id = await _insert_video_job(
        db_session,
        tenant_id=TENANT_A,
        status="processing",
        lease_delta=_PAST,
        attempts=5,
    )

    claimed = await JobManager.reclaim_orphaned(
        db_session, worker_id="pod-b", lease_seconds=900, max_attempts=5
    )
    # Poison job is consumed (failed) but not handed back for dispatch.
    assert claimed is None

    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.status == "failed"
    assert "poison" in (job.error_message or "")


async def test_reaper_ignores_unmanaged_and_terminal_rows(db_session):
    """Rows without job_options (not queue-owned) or already terminal are skipped."""
    # A truly unmanaged row — job_options column omitted → SQL NULL, exactly like
    # a legacy row or an image-batch child created without register_dispatch.
    unmanaged_id = str(uuid.uuid7())
    db_session.add(
        VideoGenerationJob(
            job_id=unmanaged_id,
            tenant_id=TENANT_A,
            provider="stub",
            prompt="legacy",
            status="processing",
            lease_expires_at=datetime.now(UTC) + _PAST,
        )
    )
    await db_session.commit()
    # A terminal row with an expired lease — must never be reclaimed.
    terminal_id = await _insert_video_job(
        db_session, tenant_id=TENANT_A, status="completed", lease_delta=_PAST
    )
    # A control row that SHOULD be reclaimable, to prove the sweep is running.
    control_id = await _insert_video_job(
        db_session, tenant_id=TENANT_A, status="processing", lease_delta=_PAST
    )

    reclaimed_ids = []
    while (
        c := await JobManager.reclaim_orphaned(
            db_session, worker_id="pod-b", lease_seconds=900, max_attempts=5
        )
    ) is not None:
        reclaimed_ids.append(c.job_id)

    assert reclaimed_ids == [control_id]
    assert unmanaged_id not in reclaimed_ids
    assert terminal_id not in reclaimed_ids
    unmanaged = await JobManager.get_job(db_session, unmanaged_id)
    assert unmanaged is not None
    assert unmanaged.status == "processing"  # left alone


async def test_reaper_reclaims_orphaned_image_job(db_session):
    """The image reclaim function mirrors the video one on the image table."""
    job_id = await _insert_image_job(
        db_session, tenant_id=TENANT_C, status="processing", lease_delta=_PAST
    )

    claimed = await ImageJobManager.reclaim_orphaned(
        db_session, worker_id="pod-b", lease_seconds=900, max_attempts=5
    )
    assert claimed is not None
    assert claimed.job_id == job_id
    assert claimed.options["kind"] == "image_single"

    job = await ImageJobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.status == "pending"
    assert job.claimed_by == "pod-b"
    assert job.attempts == 1


async def _insert_image_batch_child(
    db,
    *,
    batch_id: str,
    tenant_id: str,
    status: str,
    lease_delta: timedelta | None,
) -> str:
    """Insert an image-batch CHILD job (queue-owned, kind=image_batch)."""
    job_id = str(uuid.uuid7())
    lease = datetime.now(UTC) + lease_delta if lease_delta is not None else None
    db.add(
        ImageGenJob(
            job_id=job_id,
            asset_id=str(uuid.uuid7()),
            batch_id=batch_id,
            tenant_id=tenant_id,
            provider="stub",
            prompt="a durable batch image",
            status=status,
            attempts=0,
            claimed_by="pod-a",
            lease_expires_at=lease,
            job_options={
                "kind": KIND_IMAGE_BATCH,
                "batch_id": batch_id,
                "budget_limit": 5.0,
            },
        )
    )
    await db.commit()
    return job_id


async def test_reaper_reclaims_orphaned_image_batch_child(db_session):
    """A crashed pod's image-BATCH child (expired lease) is reclaimed like a single.

    The child carries ``kind=image_batch`` + its ``batch_id`` in ``job_options`` so
    the reaper can re-dispatch it through the per-child batch processor on any
    replica — closing the old gap where batch children had ``job_options IS NULL``
    and were invisible to the claim (stuck ``pending`` forever after a mid-batch
    crash).
    """
    batch = await ImageBatchManager.create_batch(
        db_session, tenant_id=TENANT_D, budget_limit=5.0, job_count=2
    )
    await db_session.commit()
    child_id = await _insert_image_batch_child(
        db_session,
        batch_id=batch.batch_id,
        tenant_id=TENANT_D,
        status="processing",
        lease_delta=_PAST,
    )

    claimed = await ImageJobManager.reclaim_orphaned(
        db_session, worker_id="pod-b", lease_seconds=900, max_attempts=5
    )
    assert claimed is not None
    assert claimed.job_id == child_id
    assert claimed.options["kind"] == KIND_IMAGE_BATCH
    assert claimed.options["batch_id"] == batch.batch_id

    job = await ImageJobManager.get_job(db_session, child_id)
    assert job is not None
    assert job.status == "pending"
    assert job.claimed_by == "pod-b"
    assert job.attempts == 1
    assert job.lease_expires_at is not None
    assert job.lease_expires_at > datetime.now(UTC)


async def test_concurrent_batch_completions_sum_counters_no_lost_update(db_session_factory):
    """Two children completing concurrently (separate sessions/pods) both count.

    Proves the atomic ``UPDATE ... completed_jobs = completed_jobs + 1``: a
    Python-side read-modify-write would lose one of the two increments
    (last-writer-wins). The SQL increment takes the row lock, so concurrent bumps
    serialise to the correct total.
    """
    async with db_session_factory() as setup:
        batch = await ImageBatchManager.create_batch(
            setup, tenant_id=TENANT_E, budget_limit=10.0, job_count=2
        )
        await setup.commit()
        batch_id = batch.batch_id

    async def _complete(cost: float) -> None:
        async with db_session_factory() as session:
            await ImageBatchManager.record_job_completed(session, batch_id, cost=cost)
            await session.commit()

    # Genuinely concurrent: two independent PG backends contend on the batch row.
    await asyncio.gather(_complete(0.05), _complete(0.10))

    async with db_session_factory() as check:
        fetched = await ImageBatchManager.get_batch(check, batch_id)
        assert fetched is not None
        assert fetched.completed_jobs == 2  # neither increment lost
        assert fetched.total_cost == pytest.approx(0.15)  # both costs summed
        assert fetched.status == "completed"
        assert fetched.completed_at is not None


async def test_reaper_dispatch_image_routes_batch_child_to_batch_processor(db_session):
    """``_dispatch_image`` routes a KIND_IMAGE_BATCH claim to the per-child processor."""
    reaper = _reaper_with_imagegen_service()
    claimed = ClaimedJob(
        job_id="child-1",
        tenant_id=TENANT_D,
        options={"kind": KIND_IMAGE_BATCH, "batch_id": "batch-9", "budget_limit": 3.0},
    )

    with (
        patch("src.services.reaper_service.run_with_lease", new_callable=AsyncMock),
        patch("src.api.content_routes._process_concept_image_batch_job_bg") as mock_batch_proc,
        patch("src.api.content_routes._process_concept_image_job_bg") as mock_single_proc,
    ):
        mock_batch_proc.return_value = MagicMock()  # a stand-in coroutine object
        await reaper._dispatch_image(claimed)

    mock_single_proc.assert_not_called()
    mock_batch_proc.assert_called_once()
    _, kwargs = mock_batch_proc.call_args
    args = mock_batch_proc.call_args.args
    # Signature: (service, job_id, tenant_id, batch_id, *, budget_limit, quota_service)
    assert args[1] == "child-1"
    assert args[2] == TENANT_D
    assert args[3] == "batch-9"
    assert kwargs["budget_limit"] == 3.0


async def test_reaper_dispatch_image_routes_single_to_single_processor(db_session):
    """A KIND_IMAGE_SINGLE claim still routes to the single-image processor."""
    reaper = _reaper_with_imagegen_service()
    claimed = ClaimedJob(
        job_id="solo-1",
        tenant_id=TENANT_D,
        options={"kind": "image_single", "budget_limit": 5.0},
    )

    with (
        patch("src.services.reaper_service.run_with_lease", new_callable=AsyncMock),
        patch("src.api.content_routes._process_concept_image_batch_job_bg") as mock_batch_proc,
        patch("src.api.content_routes._process_concept_image_job_bg") as mock_single_proc,
    ):
        mock_single_proc.return_value = MagicMock()
        await reaper._dispatch_image(claimed)

    mock_batch_proc.assert_not_called()
    mock_single_proc.assert_called_once()


async def test_release_claims_expires_this_pods_leases(db_session):
    """Shutdown release expires only this pod's in-flight leases (fast reclaim)."""
    mine = await _insert_video_job(
        db_session, tenant_id=TENANT_A, status="processing", lease_delta=_FUTURE, claimed_by="pod-a"
    )
    theirs = await _insert_video_job(
        db_session, tenant_id=TENANT_A, status="processing", lease_delta=_FUTURE, claimed_by="pod-z"
    )

    released = await JobManager.release_claims(db_session, "pod-a")
    await db_session.commit()
    assert released == 1

    mine_job = await JobManager.get_job(db_session, mine)
    theirs_job = await JobManager.get_job(db_session, theirs)
    assert mine_job is not None and theirs_job is not None
    assert mine_job.lease_expires_at is not None
    assert theirs_job.lease_expires_at is not None
    # Mine is now immediately reclaimable (lease expired); theirs is untouched.
    assert mine_job.lease_expires_at <= datetime.now(UTC)
    assert theirs_job.lease_expires_at > datetime.now(UTC)


# ===========================================================================
# Poison-refund: a graceful release must REFUND the attempt it consumed
# ===========================================================================
#
# The prod bug: `reclaim_orphaned_*` charges `attempts += 1` on every reclaim and
# hard-fails the job at `attempts >= max_attempts`, but `release_claims` only
# expired the lease — it never refunded. So every k8s rollout / eviction burned
# one of a HEALTHY job's lives, and four of them permanently failed a paid
# provider video generation. These tests prove that dead, on BOTH queues.

_MAX_ATTEMPTS = 5
_MAX_LIFETIME = 3600


async def _reclaim_video(db, worker: str = "pod-b", lifetime: int = 0) -> ClaimedJob | None:
    return await JobManager.reclaim_orphaned(
        db,
        worker_id=worker,
        lease_seconds=900,
        max_attempts=_MAX_ATTEMPTS,
        max_lifetime_seconds=lifetime,
    )


async def _reclaim_image(db, worker: str = "pod-b", lifetime: int = 0) -> ClaimedJob | None:
    return await ImageJobManager.reclaim_orphaned(
        db,
        worker_id=worker,
        lease_seconds=900,
        max_attempts=_MAX_ATTEMPTS,
        max_lifetime_seconds=lifetime,
    )


async def _backdate_started_at(db, table: str, job_id: str, *, seconds: int) -> None:
    """Age a row's started_at into the past (simulate a long-running crash-looper)."""
    await db.execute(
        text(
            f"UPDATE {table} SET started_at = now() - make_interval(secs => :s) "  # noqa: S608
            "WHERE job_id = :j"
        ),
        {"s": seconds, "j": job_id},
    )
    await db.commit()
    db.expunge_all()


async def test_release_claims_refunds_the_attempt_video(db_session):
    """An interrupted claim is refunded: attempts drops back, row stays re-claimable."""
    job_id = await _insert_video_job(
        db_session,
        tenant_id=TENANT_A,
        status="processing",
        lease_delta=_FUTURE,
        attempts=2,
        claimed_by="pod-a",
    )

    released = await JobManager.release_claims(db_session, "pod-a")
    await db_session.commit()
    db_session.expunge_all()
    assert released == 1

    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.attempts == 1  # the attempt this claim consumed is given back
    assert job.lease_expires_at is not None
    assert job.lease_expires_at <= datetime.now(UTC)  # immediately re-claimable

    # ...and it really is re-claimable: the next worker picks it up and re-charges.
    claimed = await _reclaim_video(db_session)
    assert claimed is not None
    assert claimed.job_id == job_id
    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.attempts == 2  # back to where it was — net zero across the churn


async def test_release_claims_refunds_the_attempt_image(db_session):
    """Same refund on the image queue."""
    job_id = await _insert_image_job(
        db_session,
        tenant_id=TENANT_A,
        status="processing",
        lease_delta=_FUTURE,
        attempts=3,
        claimed_by="pod-a",
    )

    released = await ImageJobManager.release_claims(db_session, "pod-a")
    await db_session.commit()
    db_session.expunge_all()
    assert released == 1

    job = await ImageJobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.attempts == 2


async def test_rollout_churn_never_poisons_a_healthy_video_job(db_session):
    """THE PROD BUG. Six rollouts (> max_attempts) must not fail a healthy job.

    Each cycle is exactly what a k8s rolling deploy does to an in-flight job:
    the pod drains (``release_claims``) and a sibling reclaims it. Before the
    refund, cycle 5 tripped ``attempts >= max_attempts`` and the job was hard-failed
    as "poison" — a paid provider video generation, silently dead, on a job that
    never once misbehaved. With the refund the counter oscillates instead of
    ratcheting, so the job survives arbitrary churn.
    """
    job_id = await _insert_video_job(
        db_session,
        tenant_id=TENANT_A,
        status="processing",
        lease_delta=_FUTURE,
        attempts=0,
        claimed_by="pod-0",
    )

    for cycle in range(6):  # 6 > _MAX_ATTEMPTS (5) — would have poisoned it
        holder = f"pod-{cycle}"
        released = await JobManager.release_claims(db_session, holder)
        await db_session.commit()
        db_session.expunge_all()
        assert released == 1, f"cycle {cycle}: the draining pod held the lease"

        claimed = await _reclaim_video(
            db_session, worker=f"pod-{cycle + 1}", lifetime=_MAX_LIFETIME
        )
        assert claimed is not None, f"cycle {cycle}: healthy job was NOT re-dispatched"
        assert claimed.job_id == job_id

        job = await JobManager.get_job(db_session, job_id)
        assert job is not None
        assert job.status == "pending"  # never 'failed'
        assert job.attempts <= 1, f"cycle {cycle}: attempts ratcheted to {job.attempts}"

    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.status != "failed"
    assert "poison" not in (job.error_message or "")


async def test_rollout_churn_never_poisons_a_healthy_image_job(db_session):
    """The same churn loop on the image queue."""
    job_id = await _insert_image_job(
        db_session,
        tenant_id=TENANT_A,
        status="processing",
        lease_delta=_FUTURE,
        attempts=0,
        claimed_by="pod-0",
    )

    for cycle in range(6):
        released = await ImageJobManager.release_claims(db_session, f"pod-{cycle}")
        await db_session.commit()
        db_session.expunge_all()
        assert released == 1

        claimed = await _reclaim_image(
            db_session, worker=f"pod-{cycle + 1}", lifetime=_MAX_LIFETIME
        )
        assert claimed is not None, f"cycle {cycle}: healthy image job was NOT re-dispatched"

    job = await ImageJobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.status == "pending"
    assert job.attempts <= 1


# ===========================================================================
# The bound: a refund without a wall-clock ceiling = infinite re-claim
# ===========================================================================


async def test_crash_looping_video_job_still_dies_via_lifetime_bound(db_session):
    """A genuinely stuck job is failed by the wall-clock bound, refund or not.

    The refund removes the attempt counter as a bound on re-claiming, so
    ``max_lifetime_seconds`` has to replace it — otherwise a job that keeps dying
    would be re-claimed (and re-charged to a paid provider) forever.
    """
    job_id = await _insert_video_job(
        db_session, tenant_id=TENANT_A, status="processing", lease_delta=_PAST, attempts=1
    )
    # It first ran two hours ago and still is not done.
    await _backdate_started_at(
        db_session, "video_generation_jobs", job_id, seconds=_MAX_LIFETIME + 3600
    )

    claimed = await _reclaim_video(db_session, lifetime=_MAX_LIFETIME)
    assert claimed is None  # consumed (failed), never handed back for dispatch

    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.status == "failed"
    assert "exceeded max job lifetime" in (job.error_message or "")
    assert job.completed_at is not None


async def test_crash_looping_image_job_still_dies_via_lifetime_bound(db_session):
    """The image queue enforces the same bound (one templated resolver, two tables)."""
    job_id = await _insert_image_job(
        db_session, tenant_id=TENANT_A, status="processing", lease_delta=_PAST, attempts=1
    )
    await _backdate_started_at(db_session, "imagegen_jobs", job_id, seconds=_MAX_LIFETIME + 60)

    claimed = await _reclaim_image(db_session, lifetime=_MAX_LIFETIME)
    assert claimed is None

    job = await ImageJobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.status == "failed"
    assert "exceeded max job lifetime" in (job.error_message or "")


async def test_young_started_job_is_not_aged_out(db_session):
    """A job inside the lifetime window is reclaimed normally, not failed."""
    job_id = await _insert_video_job(
        db_session,
        tenant_id=TENANT_A,
        status="processing",
        lease_delta=_PAST,
        started_at=datetime.now(UTC) - timedelta(seconds=60),
    )

    claimed = await _reclaim_video(db_session, lifetime=_MAX_LIFETIME)
    assert claimed is not None
    assert claimed.job_id == job_id

    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.status == "pending"


async def test_never_claimed_backlogged_row_never_ages_out(db_session):
    """``started_at IS NULL`` (never ran) is EXEMPT — anchoring on created_at would
    mass-fail a queue that is merely behind.

    The row was CREATED long ago and its lease has lapsed, but it never actually
    ran, so it has consumed nothing and must be dispatched, not aged out.
    """
    job_id = await _insert_video_job(
        db_session,
        tenant_id=TENANT_A,
        status="pending",
        lease_delta=_PAST,
        started_at=None,  # never claimed
    )
    await db_session.execute(
        text(
            "UPDATE video_generation_jobs SET created_at = now() - make_interval(days => 3) "
            "WHERE job_id = :j"
        ),
        {"j": job_id},
    )
    await db_session.commit()
    db_session.expunge_all()

    claimed = await _reclaim_video(db_session, lifetime=_MAX_LIFETIME)

    assert claimed is not None, "a backlogged, never-started job was aged out"
    assert claimed.job_id == job_id

    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.status == "pending"
    assert job.started_at is not None  # the first claim stamps the lifetime anchor


async def test_first_claim_stamps_started_at_and_reclaim_never_resets_it(db_session):
    """``started_at`` is set once and never overwritten — else churn = immortality."""
    job_id = await _insert_video_job(
        db_session, tenant_id=TENANT_A, status="processing", lease_delta=_PAST, started_at=None
    )

    assert await _reclaim_video(db_session, worker="pod-b") is not None
    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    first_stamp = job.started_at
    assert first_stamp is not None

    # Churn it again; the anchor must not move (a moving anchor = an unbounded job).
    await JobManager.release_claims(db_session, "pod-b")
    await db_session.commit()
    db_session.expunge_all()
    assert await _reclaim_video(db_session, worker="pod-c") is not None

    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.started_at == first_stamp


# ===========================================================================
# Release is safe: idempotent, ownership-scoped, never resurrects
# ===========================================================================


async def test_release_claims_is_a_noop_after_the_job_completed(db_session):
    """A completed job is never refunded (and never resurrected) by a late release."""
    job_id = await _insert_video_job(
        db_session,
        tenant_id=TENANT_A,
        status="completed",
        lease_delta=_FUTURE,
        attempts=2,
        claimed_by="pod-a",
    )

    released = await JobManager.release_claims(db_session, "pod-a")
    await db_session.commit()
    db_session.expunge_all()
    assert released == 0

    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.status == "completed"
    assert job.attempts == 2  # untouched


async def test_release_claims_does_not_refund_a_job_another_worker_reclaimed(db_session):
    """Once pod-b holds the job, pod-a's late release must not touch its counter."""
    job_id = await _insert_video_job(
        db_session, tenant_id=TENANT_A, status="processing", lease_delta=_PAST, attempts=1
    )

    claimed = await _reclaim_video(db_session, worker="pod-b")
    assert claimed is not None
    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.claimed_by == "pod-b"
    attempts_under_pod_b = job.attempts

    # pod-a drains late — it no longer owns the row.
    released = await JobManager.release_claims(db_session, "pod-a")
    await db_session.commit()
    db_session.expunge_all()
    assert released == 0

    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.claimed_by == "pod-b"
    assert job.attempts == attempts_under_pod_b  # pod-b's claim is intact
    assert job.lease_expires_at is not None
    assert job.lease_expires_at > datetime.now(UTC)  # pod-b's lease not stolen


async def test_repeated_release_is_idempotent_and_never_refunds_below_zero(db_session):
    """Releasing twice refunds once; the counter floors at 0."""
    job_id = await _insert_video_job(
        db_session,
        tenant_id=TENANT_A,
        status="processing",
        lease_delta=_FUTURE,
        attempts=1,
        claimed_by="pod-a",
    )

    assert await JobManager.release_claims(db_session, "pod-a") == 1
    await db_session.commit()
    db_session.expunge_all()

    # Second release: the lease is already expired, so this pod no longer holds it.
    assert await JobManager.release_claims(db_session, "pod-a") == 0
    await db_session.commit()
    db_session.expunge_all()

    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.attempts == 0  # refunded exactly once, never negative


async def test_release_of_a_zero_attempt_job_floors_at_zero(db_session):
    """A never-reclaimed job (attempts=0) released at shutdown stays at 0, not -1."""
    job_id = await _insert_image_job(
        db_session,
        tenant_id=TENANT_A,
        status="pending",
        lease_delta=_FUTURE,
        attempts=0,
        claimed_by="pod-a",
    )

    assert await ImageJobManager.release_claims(db_session, "pod-a") == 1
    await db_session.commit()
    db_session.expunge_all()

    job = await ImageJobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.attempts == 0


# ===========================================================================
# Rollout compatibility: BOTH function signatures must stay callable
# ===========================================================================
#
# This service is a 2-replica RollingUpdate and alembic runs in an initContainer
# BEFORE the new pods take traffic, so for the length of a deploy the OLD image is
# live and still calling the 3-arg `reclaim_orphaned_*(text,int,int)`. Migration 010
# therefore does NOT drop it — it replaces it with a shim that delegates to the new
# 4-arg function with the bound disabled. These tests PIN that contract: if someone
# "cleans up" the shim before the fleet is on the new image, reclaim breaks on every
# deploy, and this is what stops them.


@pytest.mark.parametrize(
    ("fn", "table"),
    [
        ("reclaim_orphaned_video_job", "video_generation_jobs"),
        ("reclaim_orphaned_image_job", "imagegen_jobs"),
    ],
)
async def test_both_reclaim_signatures_exist(db_session, fn: str, table: str):
    """The 4-arg fn AND the 3-arg rollout shim are both present, for both queues."""
    rows = (
        await db_session.execute(
            text(
                "SELECT p.pronargs AS nargs, "
                "       pg_get_function_identity_arguments(p.oid) AS args "
                "FROM pg_proc p "
                "WHERE p.proname = :fn AND p.pronamespace = 'public'::regnamespace"
            ),
            {"fn": fn},
        )
    ).all()
    arities = {r.nargs for r in rows}
    sigs = {r.args for r in rows}

    assert 4 in arities, f"{fn}: the 4-arg (bound-aware) function is missing — got {sigs}"
    assert 3 in arities, (
        f"{fn}: the 3-arg rollout shim is GONE (got {sigs}). Old pods call this "
        "during every rolling deploy — dropping it breaks reclaim for the length of "
        "the rollout. It may only be removed once the whole fleet is on the new image."
    )
    # The bound param must NOT carry a DEFAULT: a defaulted 4-arg plus the 3-arg
    # shim makes a 3-arg call ambiguous ("function is not unique") — which is
    # exactly the compatibility the shim exists to provide.
    four_arg = next(r for r in rows if r.nargs == 4)
    assert "DEFAULT" not in four_arg.args.upper(), (
        f"{fn}: p_max_lifetime_seconds must have no DEFAULT while the 3-arg shim exists"
    )


async def test_legacy_3arg_shim_reclaims_with_the_bound_disabled(db_session):
    """An OLD pod's 3-arg call still reclaims — and never ages a job out.

    The shim passes ``p_max_lifetime_seconds => 0``, so even a job that is far past
    the lifetime ceiling is re-dispatched rather than failed. That is exactly the
    pre-010 behaviour an in-flight old pod expects.
    """
    job_id = await _insert_video_job(
        db_session, tenant_id=TENANT_A, status="processing", lease_delta=_PAST, attempts=1
    )
    await _backdate_started_at(
        db_session, "video_generation_jobs", job_id, seconds=_MAX_LIFETIME * 10
    )

    row = (
        await db_session.execute(
            text(
                "SELECT job_id, tenant_id, job_options "
                "FROM reclaim_orphaned_video_job(:w, :lease, :maxa)"
            ),
            {"w": "old-pod", "lease": 900, "maxa": _MAX_ATTEMPTS},
        )
    ).first()
    await db_session.commit()
    db_session.expunge_all()

    assert row is not None, "the 3-arg shim did not reclaim (old pods would break)"
    assert row.job_id == job_id

    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.status == "pending"  # NOT failed — the shim disables the bound
    assert job.claimed_by == "old-pod"
    assert job.attempts == 2


async def test_new_4arg_signature_enforces_the_bound_on_the_same_row(db_session):
    """The same aged row IS failed through the 4-arg signature — shim vs. bound-aware."""
    job_id = await _insert_image_job(
        db_session, tenant_id=TENANT_A, status="processing", lease_delta=_PAST, attempts=1
    )
    await _backdate_started_at(db_session, "imagegen_jobs", job_id, seconds=_MAX_LIFETIME * 10)

    # 3-arg shim: bound disabled → reclaimed.
    row = (
        await db_session.execute(
            text("SELECT job_id FROM reclaim_orphaned_image_job(:w, :lease, :maxa)"),
            {"w": "old-pod", "lease": 900, "maxa": _MAX_ATTEMPTS},
        )
    ).first()
    await db_session.commit()
    db_session.expunge_all()
    assert row is not None

    # Re-expire the lease, then let a NEW pod claim it with the bound switched on.
    await db_session.execute(
        text(
            "UPDATE imagegen_jobs SET lease_expires_at = now() - interval '10 s' WHERE job_id = :j"
        ),
        {"j": job_id},
    )
    await db_session.commit()
    db_session.expunge_all()

    claimed = await _reclaim_image(db_session, worker="new-pod", lifetime=_MAX_LIFETIME)
    assert claimed is None  # aged out

    job = await ImageJobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.status == "failed"
    assert "exceeded max job lifetime" in (job.error_message or "")
