"""Durable multi-replica dispatch: claim-lease columns + reaper functions.

Adds the columns the Postgres claim-lease queue needs on BOTH job tables
(``video_generation_jobs`` and ``imagegen_jobs``):

- ``claimed_by``       which pod holds the job (for fast shutdown release)
- ``lease_expires_at`` when the claim expires; a ``pending``/``processing`` row
                       past this is reclaimable by any replica's reaper
- ``attempts``         reclaim count, for the poison-job cap
- ``job_options``      JSONB dispatch bundle (kind + budget + batch) so any
                       pod/restart can reconstruct the work. Also the
                       reclaim-ownership flag: NULL = not managed by the queue.

Plus the SECURITY DEFINER ``reclaim_orphaned_video_job`` /
``reclaim_orphaned_image_job`` functions (single-sourced from
``src.database.rls_resolvers``) that the reaper uses to atomically claim an
orphaned row across tenants under RLS enforce.

Idempotent raw SQL: a no-op on a DB that already has the columns, corrective on
a fresh CI DB. The service role owns both tables (created in 001/005), so the
``ALTER ... IF NOT EXISTS`` ownership check passes.

Revision ID: 009_durable_dispatch
Revises: 008_rls
Create Date: 2026-07-09 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "009_durable_dispatch"
down_revision: str | None = "008_rls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("video_generation_jobs", "imagegen_jobs")

# FROZEN SNAPSHOT of the reclaim functions as of this revision (3-arg signature,
# no ``started_at``). This used to import ``reclaim_upgrade_statements()`` from
# ``src.database.rls_resolvers``, but a migration that calls live application code
# replays whatever that code says TODAY, not what it said when the revision was
# written. Migration 010 adds ``started_at`` and a 4th ``p_max_lifetime_seconds``
# arg to those functions; had 009 kept importing, a fresh DB would run 009 and
# create a function body referencing a ``started_at`` column that does not exist
# until 010. A migration is a historical snapshot — it is inlined here on purpose.
# 010 drops these 3-arg functions and recreates them with the new signature.
_APP_ROLE = "w2v_ai_content_app"

_RECLAIM_FN_V009 = """
CREATE OR REPLACE FUNCTION {fn}(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id varchar, job_options jsonb)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id bigint; v_job_id varchar; v_tenant varchar; v_options jsonb; v_attempts int;
  BEGIN
    SELECT j.id, j.job_id, j.tenant_id, j.job_options, j.attempts
      INTO v_id, v_job_id, v_tenant, v_options, v_attempts
    FROM {table} j
    WHERE j.job_options IS NOT NULL
      AND j.status IN ('pending', 'processing')
      AND j.lease_expires_at IS NOT NULL
      AND j.lease_expires_at < now()
    ORDER BY j.created_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED;

    IF v_id IS NULL THEN
      RETURN;
    END IF;

    IF COALESCE(v_attempts, 0) >= p_max_attempts THEN
      UPDATE {table}
        SET status = 'failed',
            error_message = 'exceeded max reclaim attempts (' || p_max_attempts
                            || ') — failed as poison job',
            completed_at = now(),
            lease_expires_at = NULL
      WHERE id = v_id;
      RETURN;
    END IF;

    UPDATE {table}
      SET status = 'pending',
          claimed_by = p_worker_id,
          lease_expires_at = now() + make_interval(secs => p_lease_seconds),
          attempts = COALESCE(attempts, 0) + 1
    WHERE id = v_id;

    RETURN QUERY SELECT v_job_id, v_tenant, v_options;
  END
  $fn$
"""

_FNS_V009 = (
    ("reclaim_orphaned_video_job", "video_generation_jobs"),
    ("reclaim_orphaned_image_job", "imagegen_jobs"),
)


def _grants_v009() -> str:
    body = "\n    ".join(
        f"GRANT EXECUTE ON FUNCTION {fn}(text, int, int) TO {_APP_ROLE};" for fn, _ in _FNS_V009
    )
    return (
        "DO $$\nBEGIN\n"  # nosec B608  # noqa: S608 -- role/fn names are hardcoded constants
        f"  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_APP_ROLE}') THEN\n"
        f"    {body}\n"
        "  END IF;\nEND $$"
    )


def reclaim_upgrade_statements() -> list[str]:
    stmts: list[str] = [_RECLAIM_FN_V009.format(fn=fn, table=table) for fn, table in _FNS_V009]
    stmts += [f"REVOKE EXECUTE ON FUNCTION {fn}(text, int, int) FROM PUBLIC" for fn, _ in _FNS_V009]
    stmts.append(_grants_v009())
    return stmts


def reclaim_downgrade_statements() -> list[str]:
    return [f"DROP FUNCTION IF EXISTS {fn}(text, int, int)" for fn, _ in _FNS_V009]


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS claimed_by VARCHAR(64)")
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ")
        op.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0"
        )
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS job_options JSONB")
        # Partial index over only queue-owned rows keeps the reaper's
        # status/lease scan cheap as completed jobs accumulate.
        op.execute(
            f"CREATE INDEX IF NOT EXISTS ix_{table}_claimable "
            f"ON {table} (status, lease_expires_at) "
            "WHERE job_options IS NOT NULL"
        )
    for stmt in reclaim_upgrade_statements():
        op.execute(stmt)


def downgrade() -> None:
    for stmt in reclaim_downgrade_statements():
        op.execute(stmt)
    for table in _TABLES:
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_claimable")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS job_options")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS attempts")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS lease_expires_at")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS claimed_by")
