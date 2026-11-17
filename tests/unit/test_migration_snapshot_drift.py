"""The HEAD migration's DDL snapshots must still equal the live resolver source.

Migrations in this service inline an ERA SNAPSHOT of the SQL rather than importing
``src.database.rls_resolvers`` — the pinning that 008, 010 and 011 all carry, and
for good reason: a mid-chain revision that replays TODAY's function bodies emits
new-era DDL against an old-era database.

That pinning has a blind side, and it is the one this file closes. While a
revision is still the HEAD, its snapshot is supposed to BE the live source. Once
they silently diverge, every existing database stays green (it already ran the
migration), every test stays green (``create_all``'s ``after_create`` applies the
LIVE version, not the snapshot), ``alembic heads`` stays green, and ruff and ty
have nothing to say — because nothing static ever executes the snapshot. Only a
FRESH bootstrap diverges, and it diverges in production, on the day someone
installs the service from zero.

So: exactly while 012 is HEAD, its snapshots are asserted equal to the live
constants they were cut from.

⚠️ WHEN YOU ADD A REVISION, THIS FILE IS A DECISION POINT, NOT A CHORE.
``test_the_pinned_revision_is_still_head`` fails the moment the pinned revision
stops being the head, deliberately. Do NOT relax it. The two knobs are separate
on purpose:

  * ``_HEAD_REVISION`` — always the chain's actual head. Re-point it every time.
  * the revision column of ``_SNAPSHOTS`` — PER FUNCTION, the LATEST revision that
    emits that function's DDL, i.e. the one whose frozen text must equal live. If
    a new revision changes one of these bodies, the old one becomes HISTORICAL and
    MUST keep its old snapshot: move that ENTRY to the new revision. If the new
    revision does not touch it — as 013, a plain ADD COLUMN, does not — leave the
    entry where it is; that pair is still the live-matching one. A revision that
    adds a BRAND-NEW function (as 014 does) gets its own entries and leaves every
    existing one alone.

Getting this wrong in the "just make it pass" direction re-arms the exact hazard
the file exists to disarm.
"""

from __future__ import annotations

import difflib
import importlib.util
import pathlib
from types import ModuleType

import pytest

from src.database import rls_resolvers

# The chain's actual head. Re-point on every new revision.
_HEAD_REVISION = "016_idempotency_key"

# ``_guarded_grant`` is re-snapshotted by every revision that emits DDL, so it is
# checked against the newest such revision. The per-function DDL pairings live in
# ``_SNAPSHOTS`` below, one revision each.
_GUARD_REVISION = "014_imagegen_retention"

_VERSIONS_DIR = pathlib.Path(rls_resolvers.__file__).resolve().parent / "migrations" / "versions"


def _load_revision(stem: str) -> ModuleType:
    """Import a revision module BY PATH.

    Alembic revision filenames start with a digit, so they are not legal Python
    identifiers and cannot be imported by name — which is exactly why 010's own
    downgrade had to re-inline the 009 body instead of importing it.
    """
    path = _VERSIONS_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_rev_{stem}", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _explain(revision: str, name: str, snapshot: str, live: str) -> str:
    """A diagnosis naming the drifted snapshot, not an assertion dump."""
    diff = "\n".join(
        difflib.unified_diff(
            snapshot.splitlines(),
            live.splitlines(),
            fromfile=f"{revision}.py (frozen snapshot)",
            tofile="src/database/rls_resolvers.py (live)",
            lineterm="",
        )
    )
    return (
        f"\nSNAPSHOT DRIFT in {name}.\n\n"
        f"The frozen DDL in migrations/versions/{revision}.py no longer matches "
        f"the live source in src/database/rls_resolvers.py. {revision} is the "
        "latest revision that emits this DDL, so those two are required to be the "
        "same text.\n\n"
        "Nothing else will catch this: existing databases already ran the migration, "
        "the test suite applies the LIVE version via create_all's after_create hook, "
        "and no static check ever executes the snapshot. Only a FRESH bootstrap "
        "diverges — in production, on install day.\n\n"
        "Fix by choosing ONE, deliberately:\n"
        f"  (a) the live edit is intended and {revision} is unreleased -> copy "
        "the live text into the snapshot;\n"
        "  (b) the live edit belongs to a NEW revision -> write that revision, leave "
        f"{revision} frozen, and re-point this entry in _SNAPSHOTS.\n\n"
        f"{diff}\n"
    )


def test_the_pinned_revision_is_still_head() -> None:
    """Guard the guard: it only means anything while it points at HEAD."""
    heads = {_load_revision(p.stem).down_revision for p in _VERSIONS_DIR.glob("[0-9]*.py")}
    revisions = {_load_revision(p.stem).revision for p in _VERSIONS_DIR.glob("[0-9]*.py")}
    actual_head = revisions - heads

    assert actual_head == {_HEAD_REVISION}, (
        f"_HEAD_REVISION is {_HEAD_REVISION!r} but the chain's head is {actual_head}. "
        "Read the ⚠️ block at the top of this file before changing anything — the "
        "answer is to re-point _HEAD_REVISION (and possibly the snapshot map) at the "
        "new head, NOT to delete this assertion."
    )


# (revision stem, function name, attribute on the revision module, attribute on rls_resolvers).
# The live constant is referenced by NAME, not by value: passing the SQL itself
# would paste the whole function body into the parametrised test id.
_SNAPSHOTS = [
    (
        "012_batch_terminal",
        "reclaim_orphaned_image_job",
        "_RECLAIM_IMAGE_FN_V012",
        "RECLAIM_IMAGE_JOB_SQL",
    ),
    (
        "012_batch_terminal",
        "reconcile_stale_image_batches",
        "_RECONCILE_FN_V012",
        "RECONCILE_IMAGE_BATCHES_SQL",
    ),
    (
        "014_imagegen_retention",
        "expired_imagegen_jobs",
        "_EXPIRED_IMAGEGEN_JOBS_FN_V014",
        "EXPIRED_IMAGEGEN_JOBS_SQL",
    ),
    (
        "014_imagegen_retention",
        "expired_imagegen_batches",
        "_EXPIRED_IMAGEGEN_BATCHES_FN_V014",
        "EXPIRED_IMAGEGEN_BATCHES_SQL",
    ),
    # 015 adds a brand-new trigger function + two triggers and emits no grant,
    # so _GUARD_REVISION stays on 014 (the newest revision that snapshots it).
    (
        "015_provider_slot",
        "purge_provider_secret_on_terminal",
        "_PURGE_SEALED_SLOT_FN_V015",
        "PURGE_SEALED_SLOT_FN_SQL",
    ),
    (
        "015_provider_slot",
        "trg_video_generation_jobs_purge_slot_secret",
        "_PURGE_VIDEO_TRIGGER_V015",
        "PURGE_SLOT_SECRET_VIDEO_TRIGGER_SQL",
    ),
    (
        "015_provider_slot",
        "trg_imagegen_jobs_purge_slot_secret",
        "_PURGE_IMAGE_TRIGGER_V015",
        "PURGE_SLOT_SECRET_IMAGE_TRIGGER_SQL",
    ),
]


@pytest.mark.parametrize(
    ("revision", "name", "attr", "live_attr"), _SNAPSHOTS, ids=[s[1] for s in _SNAPSHOTS]
)
def test_head_migration_ddl_matches_the_live_resolver(
    revision: str, name: str, attr: str, live_attr: str
) -> None:
    snapshot = getattr(_load_revision(revision), attr)
    live = getattr(rls_resolvers, live_attr)
    assert snapshot == live, _explain(revision, name, snapshot, live)


def test_head_migration_guarded_grant_matches_the_live_helper() -> None:
    """The pg_roles guard too — it is the statement CI can never execute.

    CI's ephemeral postgres has no ``w2v_ai_content_app`` role, so this branch is
    skipped there and a drifted guard ships green. content_service took 85
    init-container restarts on exactly that.
    """
    grants = ["GRANT EXECUTE ON FUNCTION some_fn(int, int) TO w2v_ai_content_app;"]
    snapshot = _load_revision(_GUARD_REVISION)._guarded_grant(grants)
    live = rls_resolvers._guarded_grant(grants)
    assert snapshot == live, _explain(_GUARD_REVISION, "guarded grant", snapshot, live)
