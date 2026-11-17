"""``release_claims`` must stamp the lease with the SERVER clock, not the app's.

THE DEFECT. ``release_claims`` expires this pod's lease so a sibling can reclaim
the job immediately on a rolling deploy. The reclaim predicate is
``lease_expires_at < now()``, evaluated by Postgres (``src/database/rls_resolvers.py``
and migrations 009/010). The release wrote ``datetime.now(UTC)`` — the APP's clock —
so a single statement compared two different clocks; note it already used
``func.now()`` in its own WHERE.

Whenever the app clock leads the database's, the lease this release just "expired"
is still in the database's FUTURE, the sibling reaper cannot see the row, and the
drain handoff silently degrades into waiting out the whole lease. Nothing crashes
and nothing logs; a rollout just stops handing over in-flight paid generations.

WHY THIS TEST IS STRUCTURAL RATHER THAN BEHAVIOURAL. The end-to-end symptom is a
RACE — it appears only when the reclaim's round trip is shorter than the clock
skew. Measured on one dev box 2026-08-28: a client-stamped value was invisible to
``< now()`` in 40/40 trials at the instant of stamping (server trailing ~0.72 ms),
yet the churn tests still passed 5/5 with the defect reinstated, because the round
trip happened to exceed the skew. A behavioural test therefore cannot be trusted to
fail when the bug is present — it is exactly the kind of test that goes green on a
broken build. Asserting the emitted SQL is deterministic and cannot race.

CI never caught the original because a Linux runner and its postgres container
share one kernel clock, so the skew there is zero.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from sqlalchemy import update
from sqlalchemy.dialects import postgresql

from src.database.models import VideoGenerationJob
from src.services.image_job_manager import ImageJobManager
from src.services.job_manager import JobManager

_MANAGERS = [
    pytest.param(JobManager, id="video"),
    pytest.param(ImageJobManager, id="image"),
]


def _release_sql(manager: Any) -> str:
    """Compile the UPDATE that ``release_claims`` issues, with literals inlined."""
    captured: list[str] = []

    class _Recorder:
        async def execute(self, stmt: Any) -> object:
            captured.append(
                str(
                    stmt.compile(
                        dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
                    )
                )
            )

            class _R:
                rowcount = 0

            return _R()

    import asyncio

    asyncio.run(manager.release_claims(_Recorder(), "pod-a"))
    assert captured, f"{manager.__name__}.release_claims issued no statement"
    return captured[0]


@pytest.mark.parametrize("manager", _MANAGERS)
def test_release_claims_stamps_the_lease_with_the_server_clock(manager: Any) -> None:
    """The SET must be `now()`, never a client-side timestamp literal."""
    sql = _release_sql(manager)

    set_clause = re.search(r"SET\s+(.*?)\s+WHERE", sql, re.S | re.I)
    assert set_clause, f"could not locate the SET clause in:\n{sql}"
    lease_assignment = re.search(r"lease_expires_at\s*=\s*([^,]+)", set_clause.group(1), re.I)
    assert lease_assignment, f"release_claims no longer assigns lease_expires_at:\n{sql}"

    value = lease_assignment.group(1).strip()
    assert "now()" in value.lower(), (
        f"{manager.__name__}.release_claims stamps lease_expires_at as {value!r} — a "
        "CLIENT-side timestamp. The reclaim predicate is `lease_expires_at < now()` "
        "evaluated by Postgres, so a client value races clock skew and can leave the "
        "released job invisible to the sibling reaper for the whole lease. Use func.now()."
    )
    # A literal timestamp is the exact defect shape: 'YYYY-MM-DD HH:MM:SS...'
    assert not re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", value), (
        f"{manager.__name__}.release_claims inlined a client timestamp: {value!r}"
    )


@pytest.mark.parametrize("manager", _MANAGERS)
def test_the_release_predicate_and_the_written_value_use_the_same_clock(manager: Any) -> None:
    """Non-vacuity + the real invariant: one statement, one clock.

    The WHERE already used `func.now()`. This pins that BOTH sides do, so the
    asymmetry cannot reappear by someone 'fixing' the WHERE instead.
    """
    sql = _release_sql(manager)
    where = re.search(r"WHERE\s+(.*)$", sql, re.S | re.I)
    assert where, f"could not locate the WHERE clause in:\n{sql}"
    assert "now()" in where.group(1).lower(), (
        "the release WHERE no longer uses the server clock — this guard assumed it did, "
        f"and is now measuring the wrong thing:\n{sql}"
    )


def test_the_reclaim_predicate_this_release_unblocks_still_compares_to_now() -> None:
    """Anchor: if the reclaim SQL stops using `< now()`, these guards are moot.

    Reading the shipped SQL rather than restating it, so the anchor cannot rot.
    """
    import pathlib

    resolvers = (
        pathlib.Path(__file__).resolve().parents[2] / "src" / "database" / "rls_resolvers.py"
    ).read_text()
    assert "lease_expires_at < now()" in resolvers, (
        "the reclaim predicate changed; re-derive which clock release_claims must use "
        "before trusting the guards in this file"
    )


def test_update_compiles_with_literal_binds_at_all() -> None:
    """Positive control for the harness itself.

    If `literal_binds` silently stopped inlining, every assertion above would be
    scanning an empty or parameterised string and would pass on anything.
    """
    stmt = update(VideoGenerationJob).where(VideoGenerationJob.job_id == "abc").values(status="x")
    sql = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    assert "'abc'" in sql and "'x'" in sql, f"literal_binds is not inlining: {sql}"
