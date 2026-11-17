"""Every non-terminal ``imagegen_jobs`` row must be REACHABLE by the reaper.

The defect these pin, measured on the prod database on 2026-09-08: **70 rows**
(30 ``pending`` + 40 ``processing``, created 2026-06-08 → 2026-06-18) sat
non-terminal forever. They carry ``job_options IS NULL`` and
``lease_expires_at IS NULL`` because they were enqueued BEFORE migration
``009_durable_dispatch`` existed, and the reclaim predicate
(``src/database/rls_resolvers.py`` ``_RECLAIM_FN_TEMPLATE``, the WHERE clause at
lines 249-252) requires BOTH to be NOT NULL::

    WHERE j.job_options IS NOT NULL
      AND j.status IN ('pending', 'processing')
      AND j.lease_expires_at IS NOT NULL
      AND j.lease_expires_at < now()

so ``reclaim_orphaned_image_job`` cannot SELECT them at all. They never
reclaim, never re-dispatch, never terminate. The status says "work is coming"
and no code path can ever reach the row — so every count, every dashboard and
every retention sweep derived from that status is wrong. ``014``'s
``expired_imagegen_batches`` is a second victim: it refuses a batch while ANY
child is ``pending``/``processing``, so one stranded child makes its parent
immortal too.

Today's enqueue path CANNOT produce such a row — ``content_routes`` calls
``ImageJobManager.create_job`` and ``ImageJobManager.register_dispatch`` and
only THEN ``db.commit()``, so the NULL-``job_options`` window is never visible
outside the inserting transaction. Nothing asserted that, which is the whole
point of this module: it makes the coupling between the status vocabulary and
the reaper's predicate a test, so a future migration that adds a non-terminal
state, or narrows the predicate, or an enqueue path that commits before it
registers dispatch, goes RED instead of silently stranding rows.

Real Postgres: the ``db_engine`` fixture's ``create_all`` fires the
``after_create`` event that installs the LIVE ``rls_resolvers`` DDL, so the
function these call is byte-for-byte what a migration ships.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from src.database.models import ImageGenJob
from src.database.rls_resolvers import _RECLAIM_FN_TEMPLATE
from src.services.image_job_manager import ImageJobManager
from tests.conftest import TENANT_A

_LEASE_SECONDS = 900
_MAX_ATTEMPTS = 5
_MAX_LIFETIME = 86_400

# The invariant, as one query: a non-terminal row the reclaim predicate cannot
# match. Kept verbatim-parallel to the WHERE clause in ``_RECLAIM_FN_TEMPLATE``
# so a reader can diff the two by eye.
_UNREACHABLE_ROWS_SQL = """
SELECT job_id, status, (job_options IS NULL) AS opts_null,
       (lease_expires_at IS NULL) AS lease_null
FROM imagegen_jobs
WHERE status = ANY(:non_terminal)
  AND NOT (job_options IS NOT NULL AND lease_expires_at IS NOT NULL)
"""


def _predicate_status_list() -> set[str]:
    """The statuses the live reclaim predicate is willing to look at."""
    match = re.search(r"j\.status IN \(([^)]*)\)", _RECLAIM_FN_TEMPLATE)
    assert match is not None, (
        "the reclaim predicate no longer contains a literal `j.status IN (...)` "
        "list — this guard can no longer read it, so it must be rewritten "
        "rather than left silently passing"
    )
    return {s.strip().strip("'") for s in match.group(1).split(",")}


def test_reclaim_predicate_covers_every_non_terminal_status() -> None:
    """Vocabulary and predicate must agree — the static half of the invariant.

    ``ImageJobManager.NON_TERMINAL_STATUSES`` is the service's own definition of
    "work is still coming". Any status in it that the reaper's ``status IN (...)``
    list omits is a state whose rows are born unreachable.
    """
    declared = set(ImageJobManager.NON_TERMINAL_STATUSES)
    in_predicate = _predicate_status_list()

    assert declared <= in_predicate, (
        f"non-terminal status(es) {sorted(declared - in_predicate)} are NOT in the "
        f"reclaim predicate's status list {sorted(in_predicate)} — a row written in "
        "that status can never be selected by reclaim_orphaned_image_job, so it "
        "stays non-terminal forever (see this module's docstring: 70 such rows)"
    )
    assert in_predicate <= declared, (
        f"the reclaim predicate names status(es) {sorted(in_predicate - declared)} "
        "that ImageJobManager.NON_TERMINAL_STATUSES does not — the two definitions "
        "of 'non-terminal' have drifted apart"
    )


@pytest.mark.parametrize("status", list(ImageJobManager.NON_TERMINAL_STATUSES))
async def test_a_row_in_each_non_terminal_status_is_actually_reached(
    db_session, status: str
) -> None:
    """The behavioural half: the real SQL function returns a row in THIS status."""
    job_id = str(uuid.uuid7())
    db_session.add(
        ImageGenJob(
            job_id=job_id,
            asset_id=str(uuid.uuid7()),
            tenant_id=TENANT_A,
            provider="stub",
            prompt="a reachable image",
            status=status,
            attempts=0,
            claimed_by="pod-dead",
            lease_expires_at=datetime.now(UTC) - timedelta(seconds=10),
            job_options={"kind": "image_single", "budget_limit": 5.0},
        )
    )
    await db_session.commit()

    row = (
        await db_session.execute(
            text("SELECT job_id FROM reclaim_orphaned_image_job(:w, :lease, :maxa, :maxlife)"),
            {
                "w": "pod-live",
                "lease": _LEASE_SECONDS,
                "maxa": _MAX_ATTEMPTS,
                "maxlife": _MAX_LIFETIME,
            },
        )
    ).first()
    await db_session.commit()

    assert row is not None and row.job_id == job_id, (
        f"a '{status}' row with an expired lease was NOT reached by "
        "reclaim_orphaned_image_job — rows in this status strand forever"
    )


async def test_the_committed_enqueue_path_never_leaves_an_unreachable_row(
    db_session,
) -> None:
    """create_job + register_dispatch + ONE commit — the window is not observable.

    This is the forward guard for the mechanism that produced the 70 rows. If an
    enqueue path is ever changed to commit between ``create_job`` and
    ``register_dispatch``, the committed row is ``pending`` with NULL
    ``job_options``, outside the reclaim predicate, and immortal.
    """
    job = await ImageJobManager.create_job(
        db_session,
        tenant_id=TENANT_A,
        provider="stub",
        prompt="an enqueued image",
    )
    await ImageJobManager.register_dispatch(
        db_session,
        job.job_id,
        options={"kind": "image_single", "budget_limit": 5.0},
        worker_id="pod-a",
        lease_seconds=_LEASE_SECONDS,
    )
    await db_session.commit()
    db_session.expunge_all()

    stranded = (
        await db_session.execute(
            text(_UNREACHABLE_ROWS_SQL),
            {"non_terminal": list(ImageJobManager.NON_TERMINAL_STATUSES)},
        )
    ).all()

    assert stranded == [], (
        "the enqueue path committed row(s) that the reclaim predicate cannot "
        f"match: {[(r.job_id, r.status, r.opts_null, r.lease_null) for r in stranded]}"
        " — such a row is non-terminal forever"
    )
