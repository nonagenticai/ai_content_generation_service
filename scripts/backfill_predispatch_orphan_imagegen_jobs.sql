-- Terminate imagegen_jobs rows that predate durable dispatch and can never run.
--
-- WHAT
-- ----
-- Measured on the prod database 2026-09-08: 70 rows (30 `pending`, 40
-- `processing`, created 2026-06-08 -> 2026-06-18, all `batch_id IS NULL`, one
-- tenant) carry `job_options IS NULL` AND `lease_expires_at IS NULL`. The
-- reclaim predicate (`src/database/rls_resolvers.py` `_RECLAIM_FN_TEMPLATE`,
-- WHERE clause at lines 249-252, shipped by migrations 009/010/012) requires
-- both to be NOT NULL, so `reclaim_orphaned_image_job` cannot SELECT them at
-- all. They never reclaim, never re-dispatch, never terminate.
--
-- A row with no `job_options` has nothing to execute -- there is no prompt
-- envelope, no budget, no callback to replay. It cannot be re-dispatched, only
-- terminated. It is `failed`, not `cancelled`: nobody cancelled these, the
-- service lost them. `failed` is the repo's existing terminal vocabulary for
-- "the service could not produce this" (see the poison / aged-out branches of
-- `_RECLAIM_FN_TEMPLATE`, `ImageJobManager.mark_failed`); `cancelled` is
-- written only by `mark_cancelled` on an actual user cancel and reusing it here
-- would put a lie in the tenant's history.
--
-- WHY THIS IS NOT AN ALEMBIC MIGRATION
-- ------------------------------------
-- The `run-migrations` initContainer connects as `w2v_ai_content_app`. Verified
-- on the primary 2026-09-08:
--
--   has_schema_privilege('w2v_ai_content_app','public','CREATE') -> f
--   pg_has_role('w2v_ai_content_app','w2v_service','MEMBER')     -> f
--   imagegen_jobs: owner w2v_service, relrowsecurity t, policy
--     tenant_isolation: tenant_id = current_setting('app.current_tenant')::uuid
--
-- so that role cannot create the SECURITY DEFINER helper a cross-tenant repair
-- needs, and a plain UPDATE from it -- with no `app.current_tenant` bound --
-- matches ZERO rows and reports success. That is a silent no-op, the exact
-- failure mode this repair exists to fix. (Every object 009-014 created is owned
-- by `w2v_service`, i.e. those revisions were applied by the owner, not by the
-- initContainer.)
--
-- So this runs as the TABLE OWNER, deliberately and visibly:
--
--   psql -h <primary> -U w2v_service -d ai_content_generation_service \
--        -f scripts/backfill_predispatch_orphan_imagegen_jobs.sql
--
-- or, on the primary host, `sudo -u postgres psql -d ai_content_generation_service -f -`.
--
-- NEVER `SET row_security = off` to make a repair like this "work" from the app
-- role: that setting leaks across the pgbouncer pool and has crash-looped a
-- reconciler in this fleet.
--
-- PROPERTIES
-- ----------
-- * Idempotent / re-runnable: the predicate only matches NON-terminal rows, and
--   every row it touches becomes terminal. A second run updates 0 rows.
-- * Batched: 500 rows per statement in a loop, not one giant UPDATE.
-- * Tenant provenance comes from the PERSISTED ROW and nowhere else -- the whole
--   repair is SQL, per row, so no ambient identity ContextVar is read and no
--   vouch these rows never had is fabricated. `tenant_id` is not written.
-- * `attempts` is NOT incremented: these jobs were never attempted.
-- * A batched child (`batch_id IS NOT NULL`) also bumps its parent's
--   `failed_jobs`, mirroring `_BATCH_FAIL_BOOKKEEPING` in `rls_resolvers.py` --
--   without it the parent's terminal condition
--   (`completed_jobs + failed_jobs >= job_count`) stays unreachable and the batch
--   is immortal in turn. Zero such rows exist today; the branch is here so the
--   script is correct if one ever does.
--
-- The forward guard is
-- `tests/integration/test_nonterminal_rows_are_always_reclaimable.py`.

\set ON_ERROR_STOP on

DO $repair$
DECLARE
  v_reason  constant text :=
    'enqueued before durable dispatch existed; no job_options to execute';
  v_batch   constant int := 500;
  v_touched int;
  v_total   int := 0;
  r         record;
BEGIN
  LOOP
    v_touched := 0;

    FOR r IN
      SELECT id, batch_id
      FROM imagegen_jobs
      WHERE status IN ('pending', 'processing')
        AND job_options IS NULL
      ORDER BY id
      LIMIT v_batch
      FOR UPDATE SKIP LOCKED
    LOOP
      UPDATE imagegen_jobs
        SET status           = 'failed',
            error_message    = v_reason,
            completed_at     = now(),
            updated_at       = now(),
            lease_expires_at = NULL
      WHERE id = r.id;

      IF r.batch_id IS NOT NULL THEN
        UPDATE imagegen_batches
          SET failed_jobs = failed_jobs + 1,
              updated_at  = now(),
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
        WHERE batch_id = r.batch_id;
      END IF;

      v_touched := v_touched + 1;
    END LOOP;

    v_total := v_total + v_touched;
    EXIT WHEN v_touched = 0;
  END LOOP;

  RAISE NOTICE 'terminated % pre-durable-dispatch imagegen job(s)', v_total;
END
$repair$;

-- Loud verification. A repair that silently matched nothing is the defect it is
-- fixing, so refuse to exit clean while any unreachable row survives.
DO $verify$
DECLARE v_left int;
BEGIN
  SELECT count(*) INTO v_left FROM imagegen_jobs
  WHERE status IN ('pending', 'processing')
    AND NOT (job_options IS NOT NULL AND lease_expires_at IS NOT NULL);

  IF v_left > 0 THEN
    RAISE EXCEPTION
      '% imagegen job(s) are still non-terminal AND outside the reclaim '
      'predicate -- the repair did not take (RLS no-op? wrong role?)', v_left;
  END IF;
  RAISE NOTICE 'verified: 0 unreachable non-terminal imagegen jobs remain';
END
$verify$;
