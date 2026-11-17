"""SECURITY DEFINER cross-tenant READ resolver for the cleanup loop (single source).

The cleanup service (default-disabled, but enableable) sweeps expired
``video_generation_jobs`` across ALL tenants at boot with no tenant context. Under
Phase D RLS enforce the runtime is a non-bypass role with no ``app.current_tenant``
bound, so its SELECT would fail-closed to 0 rows. This resolver does the cross-tenant
read — owned by the table owner so it bypasses ENABLE — returning ``(job_id,
tenant_id)``; the cleanup service then deletes per tenant via the existing JobManager.

EXECUTE is REVOKEd from PUBLIC and then GRANTed to the restricted app role
``w2v_ai_content_app`` via a ``pg_roles``-guarded DO block, so the runtime (which
runs as that role under enforce) can actually call the function in prod, while CI —
where the role is absent — skips the grant cleanly. Single-sourced so BOTH the
migration (prod) and the ``create_all`` ``after_create`` event (tests) run an
identical CREATE + REVOKE + guarded-GRANT sequence.
"""

from __future__ import annotations

from sqlalchemy import Connection, MetaData, event, text

# The restricted runtime role the app connects as under TENANT_RLS_MODE=enforce.
APP_ROLE = "w2v_ai_content_app"

EXPIRED_VIDEO_JOBS_SQL = """
CREATE OR REPLACE FUNCTION expired_video_generation_jobs(
    p_status text, p_before timestamptz, p_limit int
) RETURNS TABLE(job_id varchar, tenant_id uuid, video_path varchar, minio_key varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT vg.job_id, vg.tenant_id, vg.video_path, vg.minio_key FROM video_generation_jobs vg
    WHERE vg.status = p_status AND vg.completed_at <= p_before
    LIMIT p_limit;
  $fn$
"""

_DROP = "DROP FUNCTION IF EXISTS expired_video_generation_jobs(text, timestamptz, int)"
_REVOKE = (
    "REVOKE EXECUTE ON FUNCTION expired_video_generation_jobs(text, timestamptz, int) FROM PUBLIC"
)


def _guarded_grant(grants: list[str]) -> str:
    """Wrap EXECUTE grants in a ``pg_roles``-guarded DO block.

    Grants to ``APP_ROLE`` only when the role exists — so the statement runs in
    prod (role present, runtime can call the SECURITY DEFINER fn) and is a clean
    no-op in CI/tests (role absent). Mirrors slide_service's shipped pattern; no
    separate gitops Stage-A grant file is needed.
    """
    body = "\n    ".join(grants)
    return (
        "DO $$\nBEGIN\n"  # nosec B608  # noqa: S608 -- APP_ROLE/grants are hardcoded constants, not request data
        f"  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN\n"
        f"    {body}\n"
        "  END IF;\nEND $$"
    )


_EXPIRED_GRANT = _guarded_grant(
    [
        f"GRANT EXECUTE ON FUNCTION expired_video_generation_jobs(text, timestamptz, int) TO {APP_ROLE};"
    ]
)


def upgrade_statements() -> list[str]:
    # DROP first: CREATE OR REPLACE cannot change a function's RETURN type, and the
    # tenant_id column flipped varchar -> uuid (migration 011).
    return [_DROP, EXPIRED_VIDEO_JOBS_SQL, _REVOKE, _EXPIRED_GRANT]


def downgrade_statements() -> list[str]:
    return [_DROP]


# ---------------------------------------------------------------------------
# Retention for the IMAGE side: imagegen_jobs and imagegen_batches
# ---------------------------------------------------------------------------
#
# ``expired_video_generation_jobs`` selects ``FROM video_generation_jobs`` and
# nothing else, so the image tables had NO retention path of any kind: every
# imagegen job row and every batch row ever written is immortal, and — because
# the cleanup sweep is also the only caller of ``_delete_job_storage`` — so is
# every image object in MinIO. The bucket lifecycle is NOT a backstop for this
# (it is hand-applied, unversioned, and whole-bucket), so the sweep has to do it.
#
# ⛔ A BATCH RETIRES AS ONE UNIT, WITH ITS CHILDREN — never on a clock of its own.
#
# The obvious design (age each job out on its own status TTL, age the batch out on
# a batch TTL, guard the parent with ``NOT EXISTS`` so the FK is never violated) is
# WRONG, and wrong in the fleet's worst-known way. ``get_concept_image_batch``
# renders ``job_ids`` and ``results`` from the SURVIVING child rows while reporting
# ``job_count`` / ``completed_jobs`` / ``failed_jobs`` from the batch's own stored
# counters. The moment one child ages out before its parent, the endpoint returns a
# silently truncated ``results`` list beside counters that still claim the original
# totals — a caller sees 5 promised, 5 completed, 2 returned, and has no way to
# learn that three rows were reaped rather than lost. That is an absent measurement
# rendered as a real value, with green tests. It bites hardest on a MIXED batch
# (one ``completed`` child at 30d beside one ``failed`` child at 7d), which no
# single choice of batch TTL can fix: for 23 days the batch is necessarily
# inconsistent with itself.
#
# So the batch, not the job, owns the retention decision for everything under it:
#
#   * ``expired_imagegen_jobs`` selects only ``batch_id IS NULL`` — a STANDALONE
#     job. A batched child is never swept on its own status TTL.
#   * ``expired_imagegen_batches`` selects a TERMINAL batch past its own status
#     TTL, and the sweep then deletes that batch's children (freeing their storage)
#     and the parent together, in one transaction. Counts and results can never
#     disagree, because the whole unit is either present or gone.
#
# The guard is now ``NOT EXISTS a child in pending/processing``: a batch with live
# children is genuinely in flight whatever its own status says, and deleting those
# rows would destroy running work. It is NOT a guard against the FK — the sweep
# satisfies that by deleting the children itself, first, in the same transaction.
# (``imagegen_jobs.batch_id`` is ``NO ACTION``, so a child racing in after the
# enumeration raises IntegrityError, which the sweep catches and retries next
# cycle. Were it ever changed to ``ON DELETE CASCADE`` this sweep would still be
# correct — it deletes children explicitly and frees their storage first — but the
# cascade would silently strand objects for any OTHER parent delete, so leave it
# ``NO ACTION``.)
#
# Cost of the rule: a cancelled child of a 30-day batch holds its MinIO object for
# 30 days rather than 7. That is the right trade — it is one coherent deliverable,
# and the alternative is an API that lies.
EXPIRED_IMAGEGEN_JOBS_SQL = """
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

EXPIRED_IMAGEGEN_BATCHES_SQL = """
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

_IMAGEGEN_JOBS_SIG = "expired_imagegen_jobs(text, timestamptz, int)"
_IMAGEGEN_BATCHES_SIG = "expired_imagegen_batches(text, timestamptz, int)"
_IMAGEGEN_SIGS = (_IMAGEGEN_JOBS_SIG, _IMAGEGEN_BATCHES_SIG)

# Brand-new functions on every database, so Postgres' default EXECUTE-to-PUBLIC
# applies and must be revoked: these read across ALL tenants.
_IMAGEGEN_REVOKE = [f"REVOKE EXECUTE ON FUNCTION {sig} FROM PUBLIC" for sig in _IMAGEGEN_SIGS]
_IMAGEGEN_GRANT = _guarded_grant(
    [f"GRANT EXECUTE ON FUNCTION {sig} TO {APP_ROLE};" for sig in _IMAGEGEN_SIGS]
)
_IMAGEGEN_DROP = [f"DROP FUNCTION IF EXISTS {sig}" for sig in _IMAGEGEN_SIGS]


def imagegen_upgrade_statements() -> list[str]:
    """CREATE the image-side retention resolvers, REVOKE PUBLIC, guarded-GRANT the app role."""
    return [
        EXPIRED_IMAGEGEN_JOBS_SQL,
        EXPIRED_IMAGEGEN_BATCHES_SQL,
        *_IMAGEGEN_REVOKE,
        _IMAGEGEN_GRANT,
    ]


def imagegen_downgrade_statements() -> list[str]:
    return list(_IMAGEGEN_DROP)


def register_resolvers(metadata: MetaData) -> None:
    """Create + grant the read resolvers after the tables exist (test create_all path)."""

    def _apply(target: object, connection: Connection, **kw: object) -> None:
        for stmt in (*upgrade_statements(), *imagegen_upgrade_statements()):
            connection.execute(text(stmt))

    event.listen(metadata, "after_create", _apply)


# ---------------------------------------------------------------------------
# Durable claim-queue: SECURITY DEFINER reclaim functions for the reaper
# ---------------------------------------------------------------------------
#
# The reaper drains orphaned jobs across ALL tenants (a crashed pod left a
# ``processing`` row whose lease lapsed, or a ``pending`` row whose owning pod
# died right after enqueue). Under Phase D RLS enforce the runtime is a
# non-bypass role with no ``app.current_tenant`` bound, so a plain claim SELECT
# would fail-closed to 0 rows. The claim therefore runs through these SECURITY
# DEFINER functions (owned by the table owner → bypass ENABLE), which atomically:
# pick one queue-owned (``job_options IS NOT NULL``) row whose lease has lapsed
# via ``FOR UPDATE SKIP LOCKED`` (so concurrent reapers/pods never grab the same
# job), fail poison jobs past ``max_attempts``, else RESET it to ``pending`` with
# a fresh lease + incremented attempt and return ``(job_id, tenant_id,
# job_options)``. Resetting to ``pending`` (not leaving it ``processing``) lets
# the existing per-job processor re-run unchanged. The atomic ``mark_processing``
# (pending→processing) + ``mark_completed`` (processing→completed) guards make the
# terminal DB status exactly-once. They do NOT make SIDE EFFECTS exactly-once: a
# reaper only reclaims after the lease has lapsed, but a stale-yet-still-alive
# original pod is already past ``mark_processing`` and mid-generation, so the
# reclaiming pod can generate concurrently → at-least-once provider spend / storage
# write (same job_id key, so overwrite not orphan) before its late
# ``mark_completed`` loses the race. This is inherent to any lease/reaper queue and
# is strictly better than orphan-forever; the drain window is minimised by
# cancelling this pod's in-flight tasks before releasing leases on shutdown.
#
# Single-sourced so BOTH the migration (prod) and the ``create_all``
# ``after_create`` event (tests) build identical functions.


# ``attempts`` is a CRASH counter, not a claim counter. ``release_claims`` refunds
# the attempt a graceful handoff consumed (see the managers), otherwise every k8s
# rollout / eviction would burn one of a healthy job's lives and 4 deploys would
# permanently fail a paid provider video generation. The refund alone, though,
# removes the only bound on re-claiming: a job that keeps genuinely crash-looping
# would cycle forever, and each cycle costs real provider spend. ``p_max_lifetime_
# seconds`` is the bound that replaces it — a wall-clock ceiling on the job's total
# RUNNING life, measured from ``started_at`` (stamped on the first claim and never
# overwritten). It is deliberately NOT measured from ``created_at``: a job that sat
# in a backlog for hours has consumed nothing, and ageing it out unstarted would
# mass-fail a queue that is merely behind. A row that was never claimed has
# ``started_at IS NULL`` and is exempt.
_RECLAIM_FN_TEMPLATE = """
CREATE OR REPLACE FUNCTION {fn}(
    p_worker_id text, p_lease_seconds int, p_max_attempts int,
    p_max_lifetime_seconds int
) RETURNS TABLE(job_id varchar, tenant_id varchar, job_options jsonb)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id bigint; v_job_id varchar; v_tenant varchar; v_options jsonb; v_attempts int;
    v_started timestamptz;{batch_decl}
  BEGIN
    SELECT j.id, j.job_id, j.tenant_id, j.job_options, j.attempts, j.started_at{batch_cols}
      INTO v_id, v_job_id, v_tenant, v_options, v_attempts, v_started{batch_into}
    FROM {table} j
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
      UPDATE {table}
        SET status = 'failed',
            error_message = 'exceeded max reclaim attempts (' || p_max_attempts
                            || ') — failed as poison job',
            completed_at = now(),
            lease_expires_at = NULL
      WHERE id = v_id;
{batch_bookkeeping}
      RETURN;  -- poison job skipped
    END IF;

    -- Wall-clock ceiling on total RUNNING life. ``started_at IS NULL`` (never
    -- claimed → backlogged) is EXEMPT: ageing out a job that never ran would
    -- mass-fail a queue that is merely behind.
    IF p_max_lifetime_seconds > 0
       AND v_started IS NOT NULL
       AND v_started < now() - make_interval(secs => p_max_lifetime_seconds) THEN
      UPDATE {table}
        SET status = 'failed',
            error_message = 'exceeded max job lifetime (' || p_max_lifetime_seconds
                            || 's) — never completed across '
                            || COALESCE(v_attempts, 0) || ' reclaim attempt(s)',
            completed_at = now(),
            lease_expires_at = NULL
      WHERE id = v_id;
{batch_bookkeeping}
      RETURN;  -- aged-out job skipped
    END IF;

    UPDATE {table}
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


# ---------------------------------------------------------------------------
# Batch counter bookkeeping on the SQL fail paths
# ---------------------------------------------------------------------------
#
# ``imagegen_batches`` is an AGGREGATE row whose ``completed_jobs`` /
# ``failed_jobs`` counters are maintained exclusively by
# ``ImageBatchManager.record_job_completed`` / ``record_job_failed`` — Python
# writers that only ever run inside a child's own processing task. The two
# branches above fail a child ENTIRELY IN SQL (poison, aged-out) and never enter
# that task, so the parent's counters were never bumped for those children.
#
# That is not a cosmetic drift: the batch's terminal condition is
# ``completed_jobs + failed_jobs >= job_count``, so a single SQL-failed child made
# the condition UNREACHABLE and the batch immortal — permanently ``processing``,
# with no reaper and no prune. Adding a batch reaper without this bump would not
# help either: it would find a batch whose children are all terminal but whose
# counters say otherwise. The counters must be bumped by whoever writes the
# child's terminal status, and here that writer is this function.
#
# The status/`completed_at` derivation mirrors ``ImageBatchManager._RECORD_SQL``
# byte-for-byte in meaning (post-increment values, single statement, row lock) so
# a batch closed on this path is indistinguishable from one closed by a child.
_BATCH_FAIL_BOOKKEEPING = """
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
"""


def _reclaim_fn_sql(fn_name: str, table: str, *, batch_table: bool = False) -> str:
    # fn_name/table are internal hardcoded constants (never request data), so the
    # interpolation is not an injection vector — this mirrors html_render's claim fn.
    #
    # ``batch_table`` is image-only: ``video_generation_jobs`` has no ``batch_id``
    # column at all (video batches live in the redis-backed ``BatchTracker``, not a
    # DB aggregate), so selecting it there would not compile.
    return _RECLAIM_FN_TEMPLATE.format(
        fn=fn_name,
        table=table,
        batch_decl=" v_batch_id varchar;" if batch_table else "",
        batch_cols=", j.batch_id" if batch_table else "",
        batch_into=", v_batch_id" if batch_table else "",
        batch_bookkeeping=_BATCH_FAIL_BOOKKEEPING if batch_table else "",
    )


# ---------------------------------------------------------------------------
# Rollout compatibility: the 3-arg signature is kept as a SHIM, never dropped
# ---------------------------------------------------------------------------
#
# ``CREATE OR REPLACE FUNCTION`` cannot change a function's argument list, so
# adding ``p_max_lifetime_seconds`` means a NEW signature. The tempting move is to
# DROP the old 3-arg one — but this service is a **2-replica RollingUpdate** and
# alembic runs in an initContainer BEFORE the new pods take traffic. Between
# "migration applied" and "last old pod terminated", the OLD image is live and
# still calling the 3-arg function. Dropping it would break every reclaim for the
# length of the rollout — a reliability fix that breaks reclaim on deploy defeats
# itself. (DROP FUNCTION would also destroy the function's ACL: prod's live grant
# to ``w2v_ai_content_app`` is a real, non-default one.)
#
# So this is purely ADDITIVE: Postgres overloads on the argument list, so the new
# 4-arg function is created alongside, and the 3-arg signature is CREATE OR
# REPLACE'd into a thin shim that delegates with the bound DISABLED (0) — i.e.
# byte-for-byte today's behaviour for an old pod. Nothing is dropped, so the ACL
# survives; the grants below are re-emitted anyway (harmless, and required on a
# fresh DB where the functions are new).
#
# ⚠️ The 4-arg function deliberately has NO DEFAULT on ``p_max_lifetime_seconds``.
# A ``DEFAULT 0`` there would make a 3-arg call AMBIGUOUS between the shim and the
# defaulted 4-arg ("function is not unique") — the exact opposite of the
# compatibility this shim exists to provide.
#
# The shim is ONLY for in-flight old pods; new code always binds the 4-arg
# explicitly. Drop it in a FOLLOW-UP migration once the whole fleet is on the new
# image — never in this one.
_RECLAIM_SHIM_TEMPLATE = """
CREATE OR REPLACE FUNCTION {fn}(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id varchar, job_options jsonb)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT * FROM {fn}(p_worker_id, p_lease_seconds, p_max_attempts, 0);
  $fn$
"""

RECLAIM_VIDEO_JOB_SQL = _reclaim_fn_sql("reclaim_orphaned_video_job", "video_generation_jobs")
RECLAIM_IMAGE_JOB_SQL = _reclaim_fn_sql(
    "reclaim_orphaned_image_job", "imagegen_jobs", batch_table=True
)
RECLAIM_VIDEO_SHIM_SQL = _RECLAIM_SHIM_TEMPLATE.format(fn="reclaim_orphaned_video_job")
RECLAIM_IMAGE_SHIM_SQL = _RECLAIM_SHIM_TEMPLATE.format(fn="reclaim_orphaned_image_job")

_RECLAIM_FNS = ("reclaim_orphaned_video_job", "reclaim_orphaned_image_job")
# The CURRENT signature (bound-aware) and the legacy one kept alive as a shim.
_RECLAIM_SIG = "(text, int, int, int)"
_RECLAIM_SHIM_SIG = "(text, int, int)"

# REVOKE from PUBLIC — NOT from the app role. On a FRESH database (CI, tests, a new
# install) both functions are brand new, and Postgres grants EXECUTE to PUBLIC by
# default: a SECURITY DEFINER function that reads/writes across ALL tenants must not
# be callable by PUBLIC. On prod this is a no-op (PUBLIC was already revoked in 009)
# and it does not touch ``w2v_ai_content_app``'s explicit grant.
_RECLAIM_REVOKE = [
    f"REVOKE EXECUTE ON FUNCTION {fn}{sig} FROM PUBLIC"
    for fn in _RECLAIM_FNS
    for sig in (_RECLAIM_SIG, _RECLAIM_SHIM_SIG)
]
# Without this GRANT the runtime (restricted ``w2v_ai_content_app`` under
# TENANT_RLS_MODE=enforce) hits "permission denied for function" on the reclaim call
# and the reaper silently no-ops in prod — CI never sees it, because CI runs as the
# DB owner. Granted on BOTH signatures: the 4-arg (what new pods call) and the 3-arg
# shim (what in-flight old pods call during the rollout). pg_roles-guarded so it
# grants in prod and cleanly no-ops in CI, where the role does not exist.
_RECLAIM_GRANT = _guarded_grant(
    [
        f"GRANT EXECUTE ON FUNCTION {fn}{sig} TO {APP_ROLE};"
        for fn in _RECLAIM_FNS
        for sig in (_RECLAIM_SIG, _RECLAIM_SHIM_SIG)
    ]
)
# Downgrade drops ONLY the new 4-arg functions; the 3-arg signature is restored to
# its standalone 009 body by the migration (a shim whose target is gone is broken).
_RECLAIM_DROP = [f"DROP FUNCTION IF EXISTS {fn}{_RECLAIM_SIG}" for fn in _RECLAIM_FNS]


def reclaim_upgrade_statements() -> list[str]:
    """CREATE 4-arg + 3-arg compat shim, REVOKE from PUBLIC, guarded-GRANT to the app role.

    Order matters: the shims delegate to the 4-arg functions, so those must exist
    first. Nothing is dropped — see the note above on the rolling-update hazard.
    """
    return [
        RECLAIM_VIDEO_JOB_SQL,
        RECLAIM_IMAGE_JOB_SQL,
        RECLAIM_VIDEO_SHIM_SQL,
        RECLAIM_IMAGE_SHIM_SQL,
        *_RECLAIM_REVOKE,
        _RECLAIM_GRANT,
    ]


def reclaim_downgrade_statements() -> list[str]:
    return list(_RECLAIM_DROP)


# ---------------------------------------------------------------------------
# Batch reaper: close out imagegen_batches whose children are all terminal
# ---------------------------------------------------------------------------
#
# The child-job reaper above only ever touches ``imagegen_jobs``. The parent
# ``imagegen_batches`` row had NO reaper of any kind: a batch whose owning pod
# died between "children written" and "last child recorded" stayed ``pending`` /
# ``processing`` forever, and a batch is never pruned either — so the row is
# immortal, its tenant's dashboard shows a permanently in-flight batch, and
# ``get_batch`` never reports a terminal status.
#
# This function is the missing reaper. It derives the counters from the CHILD
# ROWS rather than trusting the incremental ones, which makes it the backstop for
# every way a counter bump can be lost (a pod that died after the child's terminal
# write but before ``record_job_*``, a pre-existing row from before the SQL-path
# bump above, a manual DB fix).
#
# Three guards keep it from closing a batch that is still alive:
#
# 1. ``NOT EXISTS`` a child in ``pending``/``processing`` — a child the job reaper
#    can still reclaim means the batch is genuinely in flight, whatever its
#    counters say.
# 2. A STALENESS window on both the batch row and the newest child terminal write.
#    The last child marks itself terminal and bumps the parent in two separate
#    transactions milliseconds apart; reconciling inside that window would set the
#    counters and then let the child's ``+ 1`` push them PAST ``job_count``, which
#    the status CASE reads as ``partial``. The window makes that interleaving
#    impossible in practice rather than merely unlikely.
# 3. ``FOR UPDATE OF b SKIP LOCKED`` so concurrent reapers on different pods never
#    reconcile the same batch twice.
#
# Children that were never persisted at all (a crash mid-enqueue leaves
# ``job_count`` rows promised and fewer written) are imputed as FAILED, not
# ignored: they cannot ever complete, and leaving them uncounted is exactly the
# unreachable-terminal-condition bug this function exists to end. Counting them as
# completed would be inventing successes — and the batch's ``total_cost`` is
# deliberately NOT touched here, because a job that never ran spent nothing.
RECONCILE_IMAGE_BATCHES_SQL = """
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

_RECONCILE_SIG = "(int, int)"
_RECONCILE_FN = "reconcile_stale_image_batches"
_RECONCILE_REVOKE = f"REVOKE EXECUTE ON FUNCTION {_RECONCILE_FN}{_RECONCILE_SIG} FROM PUBLIC"
_RECONCILE_GRANT = _guarded_grant(
    [f"GRANT EXECUTE ON FUNCTION {_RECONCILE_FN}{_RECONCILE_SIG} TO {APP_ROLE};"]
)
_RECONCILE_DROP = f"DROP FUNCTION IF EXISTS {_RECONCILE_FN}{_RECONCILE_SIG}"


def reconcile_upgrade_statements() -> list[str]:
    """CREATE the batch reaper resolver, REVOKE from PUBLIC, guarded-GRANT to the app role."""
    return [RECONCILE_IMAGE_BATCHES_SQL, _RECONCILE_REVOKE, _RECONCILE_GRANT]


def reconcile_downgrade_statements() -> list[str]:
    return [_RECONCILE_DROP]


def register_reclaim_fns(metadata: MetaData) -> None:
    """Create + grant the reclaim resolvers after the tables exist (test create_all path)."""

    def _apply(target: object, connection: Connection, **kw: object) -> None:
        for stmt in (*reclaim_upgrade_statements(), *reconcile_upgrade_statements()):
            connection.execute(text(stmt))

    event.listen(metadata, "after_create", _apply)


# ---------------------------------------------------------------------------
# Tenant provider-slot secrets: purge on ANY terminal transition (P5-2)
# ---------------------------------------------------------------------------
#
# A slot job carries its tenant's provider secret sealed on the row
# (``provider_secret_sealed``) only so the reaper can resume it after a crash.
# Once the job is terminal nothing may ever need it again, so it must not stay at
# rest. A Python-side purge in ``mark_completed`` / ``mark_failed`` /
# ``mark_cancelled`` would MISS the writers that never enter Python: the reclaim
# functions above fail poison and aged-out jobs entirely in SQL. A row trigger is
# the one place every terminal writer passes through, whoever it is.
#
# Not SECURITY DEFINER and no grant needed: it only rewrites NEW of the row being
# written, and Postgres checks EXECUTE on a trigger function at CREATE TRIGGER
# time, not at fire time. ``CREATE OR REPLACE TRIGGER`` (PG14+) keeps it
# idempotent for create_all's after_create re-runs.
PURGE_SEALED_SLOT_FN_SQL = """
CREATE OR REPLACE FUNCTION purge_provider_secret_on_terminal() RETURNS trigger
  LANGUAGE plpgsql AS $fn$
  BEGIN
    NEW.provider_secret_sealed := NULL;
    RETURN NEW;
  END
  $fn$
"""

_PURGE_TRIGGER_TEMPLATE = """
CREATE OR REPLACE TRIGGER trg_{table}_purge_slot_secret
  BEFORE INSERT OR UPDATE ON {table}
  FOR EACH ROW
  WHEN (NEW.status IN ('completed', 'failed', 'cancelled')
        AND NEW.provider_secret_sealed IS NOT NULL)
  EXECUTE FUNCTION purge_provider_secret_on_terminal()
"""

_SLOT_TABLES = ("video_generation_jobs", "imagegen_jobs")
PURGE_SLOT_SECRET_VIDEO_TRIGGER_SQL = _PURGE_TRIGGER_TEMPLATE.format(table=_SLOT_TABLES[0])
PURGE_SLOT_SECRET_IMAGE_TRIGGER_SQL = _PURGE_TRIGGER_TEMPLATE.format(table=_SLOT_TABLES[1])


def slot_secret_purge_upgrade_statements() -> list[str]:
    """CREATE the purge function, then one BEFORE INSERT OR UPDATE trigger per job table."""
    return [
        PURGE_SEALED_SLOT_FN_SQL,
        PURGE_SLOT_SECRET_VIDEO_TRIGGER_SQL,
        PURGE_SLOT_SECRET_IMAGE_TRIGGER_SQL,
    ]


def slot_secret_purge_downgrade_statements() -> list[str]:
    return [
        *(f"DROP TRIGGER IF EXISTS trg_{t}_purge_slot_secret ON {t}" for t in _SLOT_TABLES),
        "DROP FUNCTION IF EXISTS purge_provider_secret_on_terminal()",
    ]


def register_slot_secret_purge(metadata: MetaData) -> None:
    """Create the purge trigger after the tables exist (test create_all path)."""

    def _apply(target: object, connection: Connection, **kw: object) -> None:
        for stmt in slot_secret_purge_upgrade_statements():
            connection.execute(text(stmt))

    event.listen(metadata, "after_create", _apply)
