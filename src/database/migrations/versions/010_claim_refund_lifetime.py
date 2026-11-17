"""Claim-attempt refund + wall-clock job-lifetime bound.

Fixes the poison bug: ``attempts`` is incremented on every reclaim and the job is
hard-failed at ``attempts >= max_attempts``, but ``release_claims`` (the graceful
shutdown path) only expired the lease — it never REFUNDED the attempt. So every
k8s rollout / pod eviction burned one of a healthy job's lives, and four of them
permanently failed a *paid provider* video generation. The refund itself lives in
the managers; this revision supplies the DB half:

1. ``started_at`` (nullable, tz-aware) on BOTH queue tables — stamped on the first
   claim and never overwritten. NULL = never claimed.
2. ``p_max_lifetime_seconds`` on both reclaim functions: a wall-clock ceiling on a
   job's total RUNNING life, so refunding an attempt cannot let a genuinely
   crash-looping job re-claim (and re-spend on a provider) forever.

The bound is anchored on ``started_at``, NOT ``created_at``: a job legitimately
queued for hours has consumed nothing and must not age out before it ever ran.
Rows with ``started_at IS NULL`` are exempt for the same reason — otherwise a
backlog mass-fails.

⚠️ ZERO-DOWNTIME NOTE — the 3-arg functions are NOT dropped.

``CREATE OR REPLACE`` cannot change a function's argument list, so adding
``p_max_lifetime_seconds`` means a new signature. Dropping the old 3-arg one would
be an outage: this service is a **2-replica RollingUpdate** and alembic runs in an
initContainer BEFORE the new pods take traffic, so between "migration applied" and
"last old pod terminated" the OLD image is live and still calling the 3-arg
function. Every reclaim would fail for the length of the rollout. (``DROP FUNCTION``
would also destroy the function's ACL — prod's live grant to ``w2v_ai_content_app``,
``w2v_service=X/w2v_service, w2v_ai_content_app=X/w2v_service``, is a real,
non-default one, and losing it would silently stop the reaper reclaiming ANYTHING.)

So this migration is purely ADDITIVE: Postgres overloads on the argument list, so
the 4-arg function is created ALONGSIDE, and the 3-arg signature is replaced by a
thin backwards-compat SHIM that delegates to it with the bound disabled (0) —
byte-for-byte today's behaviour for an old pod. Nothing is dropped, so the ACL
survives; EXECUTE is re-granted on BOTH signatures anyway (harmless in prod,
required on a fresh DB). ``expired_video_generation_jobs`` is untouched.

🔜 FOLLOW-UP: the 3-arg shims exist ONLY for rollout compatibility. Drop them in a
LATER migration once the whole fleet is on the new image — never in this one.
``tests/integration/test_durable_dispatch.py`` pins BOTH signatures so that the
cleanup cannot silently re-break the rollout.

The ``started_at`` column is additive and nullable with no backfill ⇒ also safe.

⚠️ PINNED — DO NOT REVERT TO THE HELPERS. This revision used to render its DDL
from LIVE application code: ``src.database.rls_resolvers.reclaim_upgrade_
statements()`` / ``reclaim_downgrade_statements()``, whose function bodies,
signature list and guarded GRANT are TODAY's, not this era's. That is the defect
008 and 011 were already pinned for: a mid-chain migration that replays today's
function bodies emits new-era DDL against an old-era database. Add a column to
the reclaim body, change the argument list, or drop the 3-arg shim upstream and a
FRESH bootstrap breaks HERE while every existing DB stays green — and ``alembic
heads``/ruff/ty stay green too, because nothing static executes it. A migration is
a historical record, not a view of today's code, so the era-correct DDL is
inlined below verbatim.

⚠️ The GRANTs below are ``pg_roles``-guarded, and a role-guarded statement is an
UNTESTED statement: CI's ephemeral postgres has no ``w2v_ai_content_app`` role, so
the branch is skipped and a signature typo ships green. That is how content_service
took 85 init-container restarts and ~6 un-rollable hours — migration 007 granted
EXECUTE on a signature it had dropped three statements earlier. Audited here: the
upgrade grants ``(text, int, int, int)`` and ``(text, int, int)`` on both reclaim
functions, all four CREATEd earlier in this same revision and none dropped; the
downgrade grants only ``(text, int, int)``, re-CREATEd two statements earlier and
dropped by nothing (only the 4-arg signature is dropped, afterwards).

Revision ID: 010_claim_refund_lifetime
Revises: 009_durable_dispatch
Create Date: 2026-07-14 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "010_claim_refund_lifetime"
down_revision: str | None = "009_durable_dispatch"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("video_generation_jobs", "imagegen_jobs")
_APP_ROLE = "w2v_ai_content_app"
_FNS = (
    ("reclaim_orphaned_video_job", "video_generation_jobs"),
    ("reclaim_orphaned_image_job", "imagegen_jobs"),
)
# The CURRENT (bound-aware) signature at this revision and the legacy one kept
# alive as a rollout shim. Frozen: a later argument-list change upstream must not
# retroactively rewrite what this historical revision granted.
_SIG = "(text, int, int, int)"
_SHIM_SIG = "(text, int, int)"

# Era-correct 4-arg reclaim body AS OF THIS REVISION (adds p_max_lifetime_seconds
# + started_at to the 009 body). Snapshot of ``rls_resolvers._RECLAIM_FN_TEMPLATE``.
_RECLAIM_FN_V010 = """
CREATE OR REPLACE FUNCTION {fn}(
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

# The 3-arg signature is NOT dropped — it is CREATE OR REPLACE'd into a thin shim
# delegating with the bound disabled (0). Snapshot of ``_RECLAIM_SHIM_TEMPLATE``.
_RECLAIM_SHIM_V010 = """
CREATE OR REPLACE FUNCTION {fn}(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id varchar, job_options jsonb)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT * FROM {fn}(p_worker_id, p_lease_seconds, p_max_attempts, 0);
  $fn$
"""


def _guarded_grant(grants: list[str]) -> str:
    """``pg_roles``-guarded EXECUTE grants: real in prod, clean no-op in CI.

    Snapshot of ``rls_resolvers._guarded_grant`` at this revision. Pinned, not
    derived — see the docstring on why a guarded statement is an untested one.
    """
    body = "\n    ".join(grants)
    return (
        "DO $$\nBEGIN\n"  # nosec B608  # noqa: S608 -- role/fn names are hardcoded constants
        f"  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_APP_ROLE}') THEN\n"
        f"    {body}\n"
        "  END IF;\nEND $$"
    )


def _reclaim_upgrade_statements() -> list[str]:
    """Exactly what ``rls_resolvers.reclaim_upgrade_statements()`` emitted at 010.

    Order matters: the shims delegate to the 4-arg functions, so those must exist
    first. Nothing is dropped — see the rolling-update hazard above.
    """
    stmts: list[str] = [_RECLAIM_FN_V010.format(fn=fn, table=table) for fn, table in _FNS]
    stmts += [_RECLAIM_SHIM_V010.format(fn=fn) for fn, _ in _FNS]
    stmts += [
        f"REVOKE EXECUTE ON FUNCTION {fn}{sig} FROM PUBLIC"
        for fn, _ in _FNS
        for sig in (_SIG, _SHIM_SIG)
    ]
    stmts.append(
        _guarded_grant(
            [
                f"GRANT EXECUTE ON FUNCTION {fn}{sig} TO {_APP_ROLE};"
                for fn, _ in _FNS
                for sig in (_SIG, _SHIM_SIG)
            ]
        )
    )
    return stmts


def _reclaim_downgrade_statements() -> list[str]:
    """Drops ONLY the new 4-arg functions; the 3-arg body is restored separately."""
    return [f"DROP FUNCTION IF EXISTS {fn}{_SIG}" for fn, _ in _FNS]


# Frozen 009 (3-arg, no started_at) body, inlined so ``downgrade()`` can restore it
# without importing the 009 revision module (its name starts with a digit and is
# not a legal Python identifier — alembic loads it by path, not by import).
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


def _restore_v009_statements() -> list[str]:
    """Restore the 3-arg functions to their STANDALONE 009 body + role-guarded re-GRANT.

    After ``upgrade`` the 3-arg signature is a shim that delegates to the 4-arg
    function. ``downgrade`` drops the 4-arg one, so the shim would be left pointing
    at nothing — it must be replaced by the original self-contained 009 body first.
    """
    stmts: list[str] = [_RECLAIM_FN_V009.format(fn=fn, table=table) for fn, table in _FNS]
    stmts += [f"REVOKE EXECUTE ON FUNCTION {fn}{_SHIM_SIG} FROM PUBLIC" for fn, _ in _FNS]
    stmts.append(
        _guarded_grant(
            [f"GRANT EXECUTE ON FUNCTION {fn}{_SHIM_SIG} TO {_APP_ROLE};" for fn, _ in _FNS]
        )
    )
    return stmts


def upgrade() -> None:
    # 1. The lifetime anchor. Nullable + NO backfill: every pre-existing row is
    #    treated as "never claimed" and is exempt from the bound until its next
    #    claim stamps it. Backfilling from created_at would retro-age the current
    #    backlog and mass-fail it on the first reaper sweep after deploy.
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ")

    # 2. Create the 4-arg (bound-aware) functions ALONGSIDE the existing 3-arg ones,
    #    replace the 3-arg bodies with compat shims that delegate with the bound
    #    disabled, then REVOKE from PUBLIC and GRANT EXECUTE to the restricted
    #    runtime role on BOTH signatures. Nothing is dropped — see the module
    #    docstring on the rolling-update hazard. PINNED to the era snapshot above
    #    rather than imported from rls_resolvers — see the docstring.
    for stmt in _reclaim_upgrade_statements():
        op.execute(stmt)


def downgrade() -> None:
    # Restore the 3-arg functions to their standalone 009 body FIRST (they are shims
    # over the 4-arg ones at this point), re-granting EXECUTE under the same pg_roles
    # guard, and only THEN drop the 4-arg functions. A downgrade that strips the grant
    # leaves the reaper unable to reclaim anything — just as broken as a bad upgrade.
    for stmt in _restore_v009_statements():
        op.execute(stmt)
    for stmt in _reclaim_downgrade_statements():
        op.execute(stmt)
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS started_at")
