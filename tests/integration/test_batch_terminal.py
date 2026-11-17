"""An imagegen batch must be able to reach a terminal status (Postgres).

Two defects kept ``imagegen_batches`` rows IMMORTAL, and each needs its own
row-state proof:

1. ``reclaim_orphaned_image_job`` fails a poison / aged-out child ENTIRELY IN SQL
   and never enters the Python task that owns ``record_job_failed``. One such
   child made ``completed_jobs + failed_jobs >= job_count`` permanently
   unreachable, so the batch could never close no matter what the reaper did.
2. The reaper reclaimed video and image JOBS and left the parent aggregate out
   entirely — and batches are never pruned either.

Every assertion here reads the BATCH ROW back out of Postgres. None of them
counts calls on a mock: a mock call-count would go green against a function that
wrote nothing, which is exactly the failure mode being fixed.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from src.database.models import ImageGenBatch, ImageGenJob
from src.database.rls_resolvers import (
    RECLAIM_IMAGE_JOB_SQL,
    reconcile_upgrade_statements,
)
from src.services.image_batch_manager import ImageBatchManager
from src.services.image_job_manager import ImageJobManager
from tests.conftest import TENANT_A, TENANT_B

_PAST = timedelta(seconds=-10)


@pytest.fixture(autouse=True)
async def _resolvers(db_engine):
    """Apply the CURRENT reclaim + reconcile DDL before each test.

    ``Base.metadata.create_all`` only fires its ``after_create`` hooks when it
    actually creates the tables, so on a database that already exists (a dev box
    running the compose postgres across sessions) the functions under test would
    be whatever an earlier run left behind. Re-applying is idempotent
    (``CREATE OR REPLACE``) and makes the test assert against the resolver source
    in this working tree rather than against DB residue.
    """
    async with db_engine.begin() as conn:
        for stmt in (RECLAIM_IMAGE_JOB_SQL, *reconcile_upgrade_statements()):
            await conn.execute(text(stmt))


async def _insert_batch(
    db,
    *,
    tenant_id: str,
    job_count: int,
    status: str = "processing",
    completed_jobs: int = 0,
    failed_jobs: int = 0,
    age_seconds: int = 3600,
) -> str:
    """A batch row whose bookkeeping stopped ``age_seconds`` ago."""
    batch_id = str(uuid.uuid7())
    stamp = datetime.now(UTC) - timedelta(seconds=age_seconds)
    db.add(
        ImageGenBatch(
            batch_id=batch_id,
            tenant_id=tenant_id,
            budget_limit=5.0,
            job_count=job_count,
            status=status,
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            created_at=stamp,
            updated_at=stamp,
        )
    )
    await db.commit()
    return batch_id


async def _insert_child(
    db,
    *,
    batch_id: str | None,
    tenant_id: str,
    status: str,
    lease_delta: timedelta | None = None,
    attempts: int = 0,
    age_seconds: int = 3600,
    started_at: datetime | None = None,
) -> str:
    job_id = str(uuid.uuid7())
    stamp = datetime.now(UTC) - timedelta(seconds=age_seconds)
    terminal = status in {"completed", "failed", "cancelled"}
    db.add(
        ImageGenJob(
            job_id=job_id,
            asset_id=str(uuid.uuid7()),
            batch_id=batch_id,
            tenant_id=tenant_id,
            provider="stub",
            prompt="a batch child",
            status=status,
            attempts=attempts,
            claimed_by="pod-a",
            lease_expires_at=(datetime.now(UTC) + lease_delta if lease_delta is not None else None),
            started_at=started_at,
            job_options={"kind": "image_batch", "batch_id": batch_id, "budget_limit": 5.0},
            created_at=stamp,
            updated_at=stamp,
            completed_at=stamp if terminal else None,
        )
    )
    await db.commit()
    return job_id


async def _read_batch(db, batch_id: str) -> ImageGenBatch:
    db.expunge_all()
    batch = await ImageBatchManager.get_batch(db, batch_id)
    assert batch is not None
    return batch


# ---------------------------------------------------------------------------
# Defect 1a — the SQL fail path must bump the parent's counters
# ---------------------------------------------------------------------------


async def test_poison_failed_child_bumps_the_batch_and_closes_it(db_session):
    """A child failed as poison IN SQL must move its parent to terminal.

    This is the unreachable-terminal-condition bug in its smallest form: one
    child, failed by the reclaim function itself. Before the fix the child went
    ``failed`` and the batch stayed ``processing`` with ``failed_jobs == 0``
    forever — no later event could ever have closed it.
    """
    batch_id = await _insert_batch(db_session, tenant_id=TENANT_A, job_count=1)
    job_id = await _insert_child(
        db_session,
        batch_id=batch_id,
        tenant_id=TENANT_A,
        status="processing",
        lease_delta=_PAST,
        attempts=5,
    )

    claimed = await ImageJobManager.reclaim_orphaned(
        db_session, worker_id="pod-b", lease_seconds=900, max_attempts=5
    )
    assert claimed is None, "a poison job must not be handed back for re-dispatch"

    child = await ImageJobManager.get_job(db_session, job_id)
    assert child is not None and child.status == "failed"

    batch = await _read_batch(db_session, batch_id)
    assert batch.failed_jobs == 1, (
        "the SQL fail path did not bump the parent — completed+failed can never "
        "reach job_count and the batch is immortal"
    )
    assert batch.status == "failed"
    assert batch.completed_at is not None


async def test_aged_out_child_closes_a_partially_completed_batch(db_session):
    """The lifetime-bound fail path bumps too, and derives ``partial`` correctly."""
    batch_id = await _insert_batch(db_session, tenant_id=TENANT_A, job_count=2, completed_jobs=1)
    await _insert_child(db_session, batch_id=batch_id, tenant_id=TENANT_A, status="completed")
    await _insert_child(
        db_session,
        batch_id=batch_id,
        tenant_id=TENANT_A,
        status="processing",
        lease_delta=_PAST,
        attempts=1,
        # The lifetime bound is measured from the FIRST claim; a row that was
        # never claimed (started_at IS NULL) is exempt by design.
        started_at=datetime.now(UTC) - timedelta(hours=2),
    )

    claimed = await ImageJobManager.reclaim_orphaned(
        db_session,
        worker_id="pod-b",
        lease_seconds=900,
        max_attempts=5,
        max_lifetime_seconds=1,
    )
    assert claimed is None, "the aged-out child must be failed, not re-dispatched"

    batch = await _read_batch(db_session, batch_id)
    assert (batch.completed_jobs, batch.failed_jobs) == (1, 1)
    assert batch.status == "partial"
    assert batch.completed_at is not None


async def test_a_reclaimed_child_does_not_touch_the_parent(db_session):
    """Only the FAIL branches bump. A healthy reclaim is a re-dispatch, not an outcome.

    Bumping here would count a child twice — once now and once when it actually
    finishes — and push the batch past ``job_count`` into a bogus ``partial``.
    """
    batch_id = await _insert_batch(db_session, tenant_id=TENANT_A, job_count=2)
    await _insert_child(
        db_session,
        batch_id=batch_id,
        tenant_id=TENANT_A,
        status="processing",
        lease_delta=_PAST,
        attempts=0,
    )

    claimed = await ImageJobManager.reclaim_orphaned(
        db_session, worker_id="pod-b", lease_seconds=900, max_attempts=5
    )
    assert claimed is not None

    batch = await _read_batch(db_session, batch_id)
    assert (batch.completed_jobs, batch.failed_jobs) == (0, 0)
    assert batch.status == "processing"


# ---------------------------------------------------------------------------
# Defect 1b — the batch row needs a reaper of its own
# ---------------------------------------------------------------------------


async def test_batch_reaper_closes_a_batch_whose_counter_bumps_were_lost(db_session):
    """Children all terminal, counters stuck at zero — the reaper derives from rows.

    This is the pod-died-between-the-two-writes case, and it is also how every
    row already stranded by the old code gets closed after deploy.
    """
    batch_id = await _insert_batch(db_session, tenant_id=TENANT_A, job_count=2)
    await _insert_child(db_session, batch_id=batch_id, tenant_id=TENANT_A, status="completed")
    await _insert_child(db_session, batch_id=batch_id, tenant_id=TENANT_A, status="failed")

    reconciled = await ImageBatchManager.reap_stale_batches(db_session, stale_seconds=60)

    assert [r.batch_id for r in reconciled] == [batch_id]
    batch = await _read_batch(db_session, batch_id)
    assert (batch.completed_jobs, batch.failed_jobs) == (1, 1)
    assert batch.status == "partial"
    assert batch.completed_at is not None


async def test_batch_reaper_marks_an_all_completed_batch_completed(db_session):
    batch_id = await _insert_batch(db_session, tenant_id=TENANT_A, job_count=2)
    for _ in range(2):
        await _insert_child(db_session, batch_id=batch_id, tenant_id=TENANT_A, status="completed")

    await ImageBatchManager.reap_stale_batches(db_session, stale_seconds=60)

    batch = await _read_batch(db_session, batch_id)
    assert batch.status == "completed"
    assert (batch.completed_jobs, batch.failed_jobs) == (2, 0)


async def test_children_that_were_never_written_are_imputed_failed(db_session):
    """A crash mid-enqueue promises ``job_count`` children and writes fewer.

    The missing ones can never complete, so counting them as anything but failed
    would either invent a success or leave the terminal condition unreachable
    all over again.
    """
    batch_id = await _insert_batch(db_session, tenant_id=TENANT_A, job_count=3)
    await _insert_child(db_session, batch_id=batch_id, tenant_id=TENANT_A, status="completed")
    await _insert_child(db_session, batch_id=batch_id, tenant_id=TENANT_A, status="completed")

    await ImageBatchManager.reap_stale_batches(db_session, stale_seconds=60)

    batch = await _read_batch(db_session, batch_id)
    assert (batch.completed_jobs, batch.failed_jobs) == (2, 1)
    assert batch.status == "partial"


async def test_batch_with_a_live_child_is_left_alone(db_session):
    """A child the job reaper can still reclaim means the batch is in flight.

    The guard that matters most: over-reaping would close a batch out from under
    a running generation and report a failure that never happened.
    """
    batch_id = await _insert_batch(db_session, tenant_id=TENANT_B, job_count=2)
    await _insert_child(db_session, batch_id=batch_id, tenant_id=TENANT_B, status="completed")
    await _insert_child(
        db_session,
        batch_id=batch_id,
        tenant_id=TENANT_B,
        status="processing",
        lease_delta=_PAST,
    )

    reconciled = await ImageBatchManager.reap_stale_batches(db_session, stale_seconds=60)

    assert reconciled == []
    batch = await _read_batch(db_session, batch_id)
    assert batch.status == "processing"
    assert batch.completed_at is None


async def test_a_freshly_touched_batch_is_inside_the_quiet_period(db_session):
    """The staleness window keeps the reconcile from racing the LAST child's bump."""
    batch_id = await _insert_batch(db_session, tenant_id=TENANT_B, job_count=1, age_seconds=0)
    await _insert_child(
        db_session,
        batch_id=batch_id,
        tenant_id=TENANT_B,
        status="completed",
        age_seconds=0,
    )

    reconciled = await ImageBatchManager.reap_stale_batches(db_session, stale_seconds=300)

    assert reconciled == []
    batch = await _read_batch(db_session, batch_id)
    assert batch.status == "processing"


async def test_terminal_batches_are_never_re_reconciled(db_session):
    """An already-closed batch must not be rewritten by every later sweep."""
    batch_id = await _insert_batch(
        db_session,
        tenant_id=TENANT_B,
        job_count=1,
        status="completed",
        completed_jobs=1,
    )
    await _insert_child(db_session, batch_id=batch_id, tenant_id=TENANT_B, status="completed")

    reconciled = await ImageBatchManager.reap_stale_batches(db_session, stale_seconds=60)

    assert reconciled == []


async def test_reap_batches_is_wired_into_the_reaper_sweep(db_session, monkeypatch):
    """``reap_once`` must actually run the batch sweep, not just own the method."""
    from contextlib import asynccontextmanager

    from src.services import reaper_service as reaper_mod

    batch_id = await _insert_batch(db_session, tenant_id=TENANT_A, job_count=1)
    await _insert_child(db_session, batch_id=batch_id, tenant_id=TENANT_A, status="failed")

    @asynccontextmanager
    async def _session():
        yield db_session

    monkeypatch.setattr(reaper_mod, "get_db_session", _session)

    from fastapi import FastAPI

    app = FastAPI()
    reaper = reaper_mod.ReaperService(
        app,
        interval_s=1.0,
        lease_seconds=900,
        max_attempts=5,
        batch_stale_seconds=60,
        heartbeat_enabled=False,
    )
    await reaper.reap_once()

    batch = await _read_batch(db_session, batch_id)
    assert batch.status == "failed", "reap_once() did not run the batch sweep"
