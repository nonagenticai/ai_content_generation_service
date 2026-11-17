"""Give the image tables a retention path — and with it, storage reclamation.

``expired_video_generation_jobs`` reads ``FROM video_generation_jobs`` and nothing
else, and the cleanup sweep had no other read. So ``imagegen_jobs`` and
``imagegen_batches`` were never pruned by anything: every row ever written is
immortal. The row growth is the small half. ``CleanupService._delete_job_storage``
runs ONLY on that sweep, so an image job's MinIO object had no deletion path at
all — the objects accumulate forever. The whole-bucket lifecycle rule that exists
in the cluster is hand-applied, in no git repo, and expires DELIVERED work too, so
it is not a backstop this service may rely on.

Two new resolvers, mirroring the video one's shape (SECURITY DEFINER + REVOKE
PUBLIC + ``pg_roles``-guarded GRANT, because the sweep is cross-tenant and runs
with no ``app.current_tenant`` bound under Phase D enforce):

  * ``expired_imagegen_jobs(text, timestamptz, int)`` — a terminal STANDALONE job
    (``batch_id IS NULL``) past its ``completed_at`` TTL, returning the two storage
    handles the sweep must free.
  * ``expired_imagegen_batches(text, timestamptz, int)`` — a TERMINAL batch past
    its own status TTL with no child still ``pending``/``processing``. The sweep
    deletes that batch's children AND the parent together, in one transaction.

A batched child is deliberately NOT swept on its own TTL: ``get_concept_image_batch``
renders ``results`` from surviving child rows while reporting ``job_count`` /
``completed_jobs`` from the batch's stored counters, so a child that ages out before
its parent makes the endpoint return a silently truncated list beside counters
claiming the original totals. A mixed batch (a 30d ``completed`` child beside a 7d
``failed`` one) makes that unavoidable under ANY choice of batch TTL. Retiring the
unit as a whole is what keeps counts and results from ever disagreeing.

⚠️ FULLY ADDITIVE, deliberately. Both function NAMES are new, so nothing existing
is dropped, no signature changes arity, and no ACL is destroyed. This service is a
RollingUpdate whose alembic initContainer runs BEFORE the new pods take traffic —
an OLD pod is live for the length of the rollout and simply never calls these.
Every ``GRANT EXECUTE`` below names a signature this same revision CREATEs and
leaves resident (the failure mode being avoided: a grant on a dropped signature is
a hard error in the initContainer, which presents as a pod stuck ``Pending``).

⚠️ PINNED, not derived. The DDL is inlined verbatim rather than imported from
``src.database.rls_resolvers`` — the same pinning 008/010/011/012 carry, for the
same reason: a migration that replays TODAY's function bodies emits new-era DDL
against an old-era database. ``tests/unit/test_migration_snapshot_drift.py``
asserts this snapshot still equals the live source while this revision is head.

Revision ID: 014_imagegen_retention
Revises: 013_job_degradation
Create Date: 2026-09-07 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "014_imagegen_retention"
down_revision: str | None = "013_job_degradation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_APP_ROLE = "w2v_ai_content_app"
_JOBS_SIG = "expired_imagegen_jobs(text, timestamptz, int)"
_BATCHES_SIG = "expired_imagegen_batches(text, timestamptz, int)"
_SIGS = (_JOBS_SIG, _BATCHES_SIG)

# Era-correct AS OF THIS REVISION.
_EXPIRED_IMAGEGEN_JOBS_FN_V014 = """
CREATE OR REPLACE FUNCTION expired_imagegen_jobs(
    p_status text, p_before timestamptz, p_limit int
) RETURNS TABLE(job_id varchar, tenant_id uuid, image_path varchar, minio_key varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT ig.job_id, ig.tenant_id, ig.image_path, ig.minio_key FROM imagegen_jobs ig
    WHERE ig.status = p_status
      AND ig.completed_at <= p_before
      AND ig.batch_id IS NULL
    LIMIT p_limit;
  $fn$
"""

_EXPIRED_IMAGEGEN_BATCHES_FN_V014 = """
CREATE OR REPLACE FUNCTION expired_imagegen_batches(
    p_status text, p_before timestamptz, p_limit int
) RETURNS TABLE(batch_id varchar, tenant_id uuid)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT b.batch_id, b.tenant_id FROM imagegen_batches b
    WHERE b.status = p_status
      AND b.completed_at IS NOT NULL
      AND b.completed_at <= p_before
      AND NOT EXISTS (
        SELECT 1 FROM imagegen_jobs j
        WHERE j.batch_id = b.batch_id
          AND j.status IN ('pending', 'processing')
      )
    LIMIT p_limit;
  $fn$
"""


def _guarded_grant(grants: list[str]) -> str:
    """``pg_roles``-guarded EXECUTE grants: real in prod, clean no-op in CI.

    Snapshot of ``rls_resolvers._guarded_grant`` at this revision. Pinned, not
    derived — CI's ephemeral postgres has no ``w2v_ai_content_app`` role, so this
    branch never executes there and a drifted guard would ship green.
    """
    body = "\n    ".join(grants)
    return (
        "DO $$\nBEGIN\n"  # nosec B608  # noqa: S608 -- role/fn names are hardcoded constants
        f"  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_APP_ROLE}') THEN\n"
        f"    {body}\n"
        "  END IF;\nEND $$"
    )


def upgrade() -> None:
    op.execute(_EXPIRED_IMAGEGEN_JOBS_FN_V014)
    op.execute(_EXPIRED_IMAGEGEN_BATCHES_FN_V014)
    for sig in _SIGS:
        op.execute(f"REVOKE EXECUTE ON FUNCTION {sig} FROM PUBLIC")
    op.execute(
        _guarded_grant([f"GRANT EXECUTE ON FUNCTION {sig} TO {_APP_ROLE};" for sig in _SIGS])
    )


def downgrade() -> None:
    for sig in _SIGS:
        op.execute(f"DROP FUNCTION IF EXISTS {sig}")
