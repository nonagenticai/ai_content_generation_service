"""Make an imagegen batch able to reach a terminal status.

``imagegen_batches`` rows were IMMORTAL. Two independent defects held them open:

1. **The counter bookkeeping had a hole.** A batch closes on
   ``completed_jobs + failed_jobs >= job_count``, and those counters are bumped
   only by ``ImageBatchManager.record_job_completed`` / ``record_job_failed`` —
   Python writers that run inside a child's own processing task. But
   ``reclaim_orphaned_image_job`` fails a child ENTIRELY IN SQL on its poison and
   aged-out branches, and that path never enters the task. One SQL-failed child
   therefore made the batch's terminal condition permanently UNREACHABLE.
2. **The batch row had no reaper at all.** The reaper reclaimed video and image
   *jobs*; the parent aggregate was left out, and batches are never pruned. A pod
   that died between a child's terminal write and its parent bump left the batch
   ``processing`` forever with nothing that could ever close it.

Fixing only (2) would not work: a batch reaper would find a batch whose children
are all terminal and whose counters disagree. Fixing only (1) would not work
either: it does nothing for the rows already stuck, or for a bump lost with the
pod that owned it. So this revision ships both halves.

* ``reclaim_orphaned_image_job`` (4-arg) is CREATE OR REPLACE'd — same signature,
  so no shim churn and no ACL loss — with the parent-batch bump added to both SQL
  fail branches. ``reclaim_orphaned_video_job`` is untouched:
  ``video_generation_jobs`` has no ``batch_id`` column (video batches live in the
  redis-backed ``BatchTracker``, not a DB aggregate).
* ``reconcile_stale_image_batches(int, int)`` is NEW: the batch reaper. It derives
  the counters from the CHILD ROWS rather than trusting the incremental ones, so
  it is the backstop for every way a bump can be lost — including the rows this
  bug already stranded, which it closes on the first sweep after deploy.

⚠️ ZERO-DOWNTIME — nothing is dropped, and no signature changes.
``reclaim_orphaned_image_job``'s argument list is identical to 010's, so the 3-arg
rollout shim still resolves and prod's ``w2v_ai_content_app`` grant survives (a
``DROP FUNCTION`` would destroy that ACL — see 010). An OLD pod running against
the new function simply gets the batch bump for free; it makes no call it did not
already make. The new reconcile function is additive and is called only by the new
image's reaper, so an old pod never touches it.

⚠️ PINNED — DO NOT REVERT TO THE HELPERS. The DDL below is an inlined snapshot of
``src.database.rls_resolvers`` AS OF THIS REVISION, not an import of it. Rendering
a migration from live application code is the defect 008/010/011 are pinned for: a
later edit to the function body silently rewrites what this historical revision
emits, so a FRESH bootstrap breaks here while every existing DB stays green — and
``alembic heads``, ruff and ty all stay green too, because nothing static runs it.

⚠️ The GRANT below is ``pg_roles``-guarded, and a role-guarded statement is an
UNTESTED statement — CI's ephemeral postgres has no ``w2v_ai_content_app`` role, so
the branch is skipped and a signature typo would ship green. Audited by hand: the
upgrade grants EXECUTE on ``reconcile_stale_image_batches(int, int)``, CREATEd two
statements earlier in this same revision and dropped by nothing. The reclaim
function's existing grants are NOT re-emitted — its signature is unchanged and
CREATE OR REPLACE preserves the ACL, so there is nothing to restore.

Revision ID: 012_batch_terminal
Revises: 011_tenant_id_uuid
Create Date: 2026-09-06 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "012_batch_terminal"
down_revision: str | None = "011_tenant_id_uuid"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_APP_ROLE = "w2v_ai_content_app"
_RECONCILE_SIG = "(int, int)"

# Era-correct ``reclaim_orphaned_image_job`` AS OF THIS REVISION: the 010 body,
# plus ``v_batch_id`` and the parent-batch bump on both SQL fail branches. The
# status/completed_at derivation mirrors ``ImageBatchManager._RECORD_SQL`` so a
# batch closed here is indistinguishable from one closed by a child's own writer.
_RECLAIM_IMAGE_FN_V012 = """
CREATE OR REPLACE FUNCTION reclaim_orphaned_image_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int,
    p_max_lifetime_seconds int
) RETURNS TABLE(job_id varchar, tenant_id varchar, job_options jsonb)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id bigint; v_job_id varchar; v_tenant varchar; v_options jsonb; v_attempts int;
    v_started timestamptz; v_batch_id varchar;
  BEGIN
    SELECT j.id, j.job_id, j.tenant_id, j.job_options, j.attempts, j.started_at, j.batch_id
      INTO v_id, v_job_id, v_tenant, v_options, v_attempts, v_started, v_batch_id
    FROM imagegen_jobs j
    WHERE j.job_options IS NOT NULL
      AND j.status IN ('pending', 'processing')
      AND j.lease_expires_at IS NOT NULL
      AND j.lease_expires_at < now()
    ORDER BY j.created_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED;

    IF v_id IS NULL THEN
      RETURN;  -- no orphaned jobs
    END IF;

    IF COALESCE(v_attempts, 0) >= p_max_attempts THEN
      UPDATE imagegen_jobs
        SET status = 'failed',
            error_message = 'exceeded max reclaim attempts (' || p_max_attempts
                            || ') — failed as poison job',
            completed_at = now(),
            lease_expires_at = NULL
      WHERE id = v_id;

      IF v_batch_id IS NOT NULL THEN
        UPDATE imagegen_batches
          SET failed_jobs = failed_jobs + 1,
              updated_at = now(),
              status = CASE
                  WHEN completed_jobs + failed_jobs + 1 < job_count THEN 'processing'
                  WHEN completed_jobs = job_count THEN 'completed'
                  WHEN failed_jobs + 1 = job_count THEN 'failed'
                  ELSE 'partial'
              END,
              completed_at = CASE
                  WHEN completed_jobs + failed_jobs + 1 >= job_count THEN now()
                  ELSE completed_at
              END
        WHERE batch_id = v_batch_id;
      END IF;

      RETURN;  -- poison job skipped
    END IF;

    -- Wall-clock ceiling on total RUNNING life. ``started_at IS NULL`` (never
    -- claimed → backlogged) is EXEMPT: ageing out a job that never ran would
    -- mass-fail a queue that is merely behind.
    IF p_max_lifetime_seconds > 0
       AND v_started IS NOT NULL
       AND v_started < now() - make_interval(secs => p_max_lifetime_seconds) THEN
      UPDATE imagegen_jobs
        SET status = 'failed',
            error_message = 'exceeded max job lifetime (' || p_max_lifetime_seconds
                            || 's) — never completed across '
                            || COALESCE(v_attempts, 0) || ' reclaim attempt(s)',
            completed_at = now(),
            lease_expires_at = NULL
      WHERE id = v_id;

      IF v_batch_id IS NOT NULL THEN
        UPDATE imagegen_batches
          SET failed_jobs = failed_jobs + 1,
              updated_at = now(),
              status = CASE
                  WHEN completed_jobs + failed_jobs + 1 < job_count THEN 'processing'
                  WHEN completed_jobs = job_count THEN 'completed'
                  WHEN failed_jobs + 1 = job_count THEN 'failed'
                  ELSE 'partial'
              END,
              completed_at = CASE
                  WHEN completed_jobs + failed_jobs + 1 >= job_count THEN now()
                  ELSE completed_at
              END
        WHERE batch_id = v_batch_id;
      END IF;

      RETURN;  -- aged-out job skipped
    END IF;

    UPDATE imagegen_jobs
      SET status = 'pending',
          claimed_by = p_worker_id,
          lease_expires_at = now() + make_interval(secs => p_lease_seconds),
          attempts = COALESCE(attempts, 0) + 1,
          -- First claim wins; a reclaim must never reset the lifetime clock.
          started_at = COALESCE(started_at, now())
    WHERE id = v_id;

    RETURN QUERY SELECT v_job_id, v_tenant, v_options;
  END
  $fn$
"""

# The frozen 010 body, WITHOUT the batch bump — restored by ``downgrade()``.
_RECLAIM_IMAGE_FN_V010 = """
CREATE OR REPLACE FUNCTION reclaim_orphaned_image_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int,
    p_max_lifetime_seconds int
) RETURNS TABLE(job_id varchar, tenant_id varchar, job_options jsonb)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id bigint; v_job_id varchar; v_tenant varchar; v_options jsonb; v_attempts int;
    v_started timestamptz;
  BEGIN
    SELECT j.id, j.job_id, j.tenant_id, j.job_options, j.attempts, j.started_at
      INTO v_id, v_job_id, v_tenant, v_options, v_attempts, v_started
    FROM imagegen_jobs j
    WHERE j.job_options IS NOT NULL
      AND j.status IN ('pending', 'processing')
      AND j.lease_expires_at IS NOT NULL
      AND j.lease_expires_at < now()
    ORDER BY j.created_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED;

    IF v_id IS NULL THEN
      RETURN;  -- no orphaned jobs
    END IF;

    IF COALESCE(v_attempts, 0) >= p_max_attempts THEN
      UPDATE imagegen_jobs
        SET status = 'failed',
            error_message = 'exceeded max reclaim attempts (' || p_max_attempts
                            || ') — failed as poison job',
            completed_at = now(),
            lease_expires_at = NULL
      WHERE id = v_id;
      RETURN;  -- poison job skipped
    END IF;

    IF p_max_lifetime_seconds > 0
       AND v_started IS NOT NULL
       AND v_started < now() - make_interval(secs => p_max_lifetime_seconds) THEN
      UPDATE imagegen_jobs
        SET status = 'failed',
            error_message = 'exceeded max job lifetime (' || p_max_lifetime_seconds
                            || 's) — never completed across '
                            || COALESCE(v_attempts, 0) || ' reclaim attempt(s)',
            completed_at = now(),
            lease_expires_at = NULL
      WHERE id = v_id;
      RETURN;  -- aged-out job skipped
    END IF;

    UPDATE imagegen_jobs
      SET status = 'pending',
          claimed_by = p_worker_id,
          lease_expires_at = now() + make_interval(secs => p_lease_seconds),
          attempts = COALESCE(attempts, 0) + 1,
          started_at = COALESCE(started_at, now())
    WHERE id = v_id;

    RETURN QUERY SELECT v_job_id, v_tenant, v_options;
  END
  $fn$
"""

# The batch reaper. SECURITY DEFINER for the same reason as the reclaim functions:
# the reaper spans tenants with no ``app.current_tenant`` bound, so under RLS
# enforce a plain UPDATE would fail closed to 0 rows and the sweep would report
# success while the batch stayed immortal.
_RECONCILE_FN_V012 = """
CREATE OR REPLACE FUNCTION reconcile_stale_image_batches(
    p_stale_seconds int, p_limit int
) RETURNS TABLE(batch_id varchar, status varchar, completed_jobs int, failed_jobs int)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    r record; v_completed int; v_failed int; v_status varchar;
  BEGIN
    FOR r IN
      SELECT b.id AS row_id, b.batch_id AS bid, b.job_count AS job_count,
             COALESCE(c.completed, 0) AS c_completed,
             COALESCE(c.failed, 0) AS c_failed,
             COALESCE(c.total, 0) AS c_total
      FROM imagegen_batches b
      LEFT JOIN (
        SELECT j.batch_id AS bid,
               count(*) FILTER (WHERE j.status = 'completed') AS completed,
               count(*) FILTER (WHERE j.status IN ('failed', 'cancelled')) AS failed,
               count(*) AS total,
               max(COALESCE(j.completed_at, j.updated_at)) AS last_terminal
        FROM imagegen_jobs j
        WHERE j.batch_id IS NOT NULL
        GROUP BY j.batch_id
      ) c ON c.bid = b.batch_id
      WHERE b.status IN ('pending', 'processing')
        AND b.updated_at < now() - make_interval(secs => p_stale_seconds)
        AND COALESCE(c.last_terminal, b.updated_at)
            < now() - make_interval(secs => p_stale_seconds)
        AND NOT EXISTS (
          SELECT 1 FROM imagegen_jobs j2
          WHERE j2.batch_id = b.batch_id
            AND j2.status IN ('pending', 'processing')
        )
      ORDER BY b.updated_at
      LIMIT p_limit
      FOR UPDATE OF b SKIP LOCKED
    LOOP
      v_completed := r.c_completed;
      -- Promised-but-never-written children can never complete; impute as failed.
      v_failed := r.c_failed + GREATEST(r.job_count - r.c_total, 0);

      IF v_completed >= r.job_count THEN
        v_status := 'completed';
      ELSIF v_failed >= r.job_count THEN
        v_status := 'failed';
      ELSE
        v_status := 'partial';
      END IF;

      UPDATE imagegen_batches
        SET completed_jobs = v_completed,
            failed_jobs = v_failed,
            status = v_status,
            updated_at = now(),
            completed_at = COALESCE(completed_at, now())
      WHERE id = r.row_id;

      batch_id := r.bid; status := v_status;
      completed_jobs := v_completed; failed_jobs := v_failed;
      RETURN NEXT;
    END LOOP;
  END
  $fn$
"""


def _guarded_grant(grants: list[str]) -> str:
    """``pg_roles``-guarded EXECUTE grants: real in prod, clean no-op in CI.

    Snapshot of ``rls_resolvers._guarded_grant`` at this revision. Pinned, not
    derived — see the module docstring on why a guarded statement is untested.
    """
    body = "\n    ".join(grants)
    return (
        "DO $$\nBEGIN\n"  # nosec B608  # noqa: S608 -- role/fn names are hardcoded constants
        f"  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_APP_ROLE}') THEN\n"
        f"    {body}\n"
        "  END IF;\nEND $$"
    )


def upgrade() -> None:
    # 1. Half one: the reclaim function's SQL fail paths now bump the parent batch.
    #    Same signature as 010 ⇒ CREATE OR REPLACE keeps the ACL and the 3-arg shim.
    op.execute(_RECLAIM_IMAGE_FN_V012)

    # 2. Half two: the batch reaper itself, plus the REVOKE that a brand-new
    #    function needs on a fresh DB (Postgres grants EXECUTE to PUBLIC by
    #    default, and this one reads and writes across ALL tenants).
    op.execute(_RECONCILE_FN_V012)
    op.execute(
        f"REVOKE EXECUTE ON FUNCTION reconcile_stale_image_batches{_RECONCILE_SIG} FROM PUBLIC"
    )
    op.execute(
        _guarded_grant(
            [
                f"GRANT EXECUTE ON FUNCTION reconcile_stale_image_batches{_RECONCILE_SIG} "
                f"TO {_APP_ROLE};"
            ]
        )
    )


def downgrade() -> None:
    # Restore the 010 reclaim body (no batch bump), then drop the batch reaper.
    # Order is irrelevant here — the two functions do not reference each other —
    # but the reclaim is restored first so a failure mid-downgrade leaves the
    # money-relevant path, not the reaper, in a known state.
    op.execute(_RECLAIM_IMAGE_FN_V010)
    op.execute(f"DROP FUNCTION IF EXISTS reconcile_stale_image_batches{_RECONCILE_SIG}")
