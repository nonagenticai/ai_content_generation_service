"""Fleet ruling: ``tenant_id`` is a real ``uuid``, never text and never a sentinel.

Live measurement on the standby (2026-08-26) confirmed 0 dirty rows across all four
tenant_id columns (imagegen_batches 2, imagegen_jobs 1849, video_generation_jobs 0,
video_generation_usage 0), so ``USING tenant_id::uuid`` cannot fail on real data.

⚠️ RLS ORDERING IS LOAD-BEARING. Postgres refuses ``ALTER COLUMN ... TYPE`` on a
column referenced by a policy ("cannot alter type of a column used in a policy
definition"), and the shipped 008 policies compare ``tenant_id`` to the TEXT GUC
``NULLIF(current_setting('app.current_tenant', true), '')``. Left in place, a uuid
column against a text GUC raises ``operator does not exist: uuid = text`` — under
enforce that is a hard failure on every scoped statement, not a graceful 0 rows.
So this migration DROPs the policies, ALTERs, and re-CREATEs them in ONE
transaction.

⚠️ The ``expired_video_generation_jobs`` SECURITY DEFINER resolver declares
``RETURNS TABLE(... tenant_id varchar ...)``. Function bodies are NOT revalidated
by ALTER TABLE, so it would survive the migration and then fail at RUNTIME with
"structure of query does not match function result type". It is re-emitted here
with ``tenant_id uuid`` (DROP first — CREATE OR REPLACE cannot change a return
type), together with its REVOKE-from-PUBLIC and pg_roles-guarded GRANT. The
GRANT names ``expired_video_generation_jobs(text, timestamptz, int)``, the exact
signature CREATEd two statements earlier in this same revision — the guard is
``pg_roles``-scoped, so on CI (role absent) the GRANT never executes and a
signature typo would ship green. It is pinned here, not derived.

NOT touched: the ``reclaim_orphaned_*_job`` functions. They read tenant_id into a
plpgsql ``varchar`` local, which plpgsql resolves by I/O conversion, and return
that local — so they keep working unchanged and their prod ACLs (and the 3-arg
rollout shim from 010) are left alone deliberately.

⚠️ PINNED — DO NOT REVERT TO THE HELPERS. This revision used to render its DDL
from LIVE application code: ``w2v_common.database.rls.{upgrade,downgrade}_
statements(Base.metadata)`` (which walks TODAY's ``Base.metadata`` for the set of
``tenant_id`` tables, their nullability, and the column TYPE that picks the
predicate) and ``src.database.rls_resolvers.upgrade_statements()`` /
``rls_resolvers._EXPIRED_GRANT`` (whose resolver body and guarded GRANT are
today's, not this era's). That is the defect 008 was already pinned for: a
mid-chain migration that reads today's models emits new-era DDL against an
old-era database. Add a fifth ``tenant_id`` table, make one nullable, or change
the resolver's columns and a FRESH bootstrap breaks HERE while every existing DB
stays green — and ``alembic heads``/ruff/ty stay green too, because nothing
static executes it. A migration is a historical record, not a view of today's
code, so the era-correct DDL is inlined below verbatim.

Revision ID: 011_tenant_id_uuid
Revises: 010_claim_refund_lifetime
"""

from alembic import op

revision: str = "011_tenant_id_uuid"
down_revision: str | None = "010_claim_refund_lifetime"
branch_labels = None
depends_on = None

TENANT_TABLES = (
    "video_generation_jobs",
    "video_generation_usage",
    "imagegen_jobs",
    "imagegen_batches",
)

# The 4 tenant_id tables in the ORDER the w2v_common helper emitted them at this
# revision (``metadata.sorted_tables``). Frozen so a later model addition cannot
# change what this historical revision does.
_POLICY_TABLES = (
    "imagegen_batches",
    "video_generation_jobs",
    "video_generation_usage",
    "imagegen_jobs",
)

_POLICY = "tenant_isolation"
# Every policy name the helper could ever have created (strict + the nullable
# "global-or-own" quartet), dropped so the reversal stays exact. All 4 tables are
# NOT NULL at this revision → only the strict policy is ever created.
_ALL_POLICIES = (
    _POLICY,
    f"{_POLICY}_read",
    f"{_POLICY}_ins",
    f"{_POLICY}_upd",
    f"{_POLICY}_del",
)

_GUC = "NULLIF(current_setting('app.current_tenant', true), '')"
# Post-011 predicate: uuid column ⇒ the TEXT GUC is cast to uuid.
_UUID_PREDICATE = f"tenant_id = {_GUC}::uuid"
# The pre-011 text-comparison predicate, for downgrade only.
_LEGACY_PREDICATE = f"tenant_id = {_GUC}"

_APP_ROLE = "w2v_ai_content_app"

_EXPIRED_SIG = "expired_video_generation_jobs(text, timestamptz, int)"
_EXPIRED_DROP = f"DROP FUNCTION IF EXISTS {_EXPIRED_SIG}"
_EXPIRED_REVOKE = f"REVOKE EXECUTE ON FUNCTION {_EXPIRED_SIG} FROM PUBLIC"
# pg_roles-guarded: grants in prod (role present), clean no-op in CI (role absent).
_EXPIRED_GRANT = (
    "DO $$\nBEGIN\n"  # nosec B608  # noqa: S608 -- role/fn names are hardcoded constants
    f"  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_APP_ROLE}') THEN\n"
    f"    GRANT EXECUTE ON FUNCTION {_EXPIRED_SIG} TO {_APP_ROLE};\n"
    "  END IF;\nEND $$"
)

# Era-correct resolver AS OF THIS REVISION: ``tenant_id uuid``.
_EXPIRED_FN_V011 = """
CREATE OR REPLACE FUNCTION expired_video_generation_jobs(
    p_status text, p_before timestamptz, p_limit int
) RETURNS TABLE(job_id varchar, tenant_id uuid, video_path varchar, minio_key varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT vg.job_id, vg.tenant_id, vg.video_path, vg.minio_key FROM video_generation_jobs vg
    WHERE vg.status = p_status AND vg.completed_at <= p_before
    LIMIT p_limit;
  $fn$
"""

# Pre-011 resolver body (varchar tenant_id), restored on downgrade.
_LEGACY_EXPIRED_FN = """
CREATE OR REPLACE FUNCTION expired_video_generation_jobs(
    p_status text, p_before timestamptz, p_limit int
) RETURNS TABLE(job_id varchar, tenant_id varchar, video_path varchar, minio_key varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT vg.job_id, vg.tenant_id, vg.video_path, vg.minio_key FROM video_generation_jobs vg
    WHERE vg.status = p_status AND vg.completed_at <= p_before
    LIMIT p_limit;
  $fn$
"""


def _drop_policies() -> None:
    """Exactly what ``downgrade_statements(Base.metadata)`` emitted at this revision."""
    for table in _POLICY_TABLES:
        for policy in _ALL_POLICIES:
            op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")


def upgrade() -> None:
    # 1. Policies reference tenant_id → they block the ALTER. Drop them first.
    _drop_policies()

    # 2. The type flip itself. 0 dirty rows measured live, so USING cannot fail.
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid")

    # 3. Re-create RLS with the uuid predicate (the GUC is cast, not the column).
    for table in _POLICY_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS {_POLICY} ON {table}")
        op.execute(
            f"CREATE POLICY {_POLICY} ON {table} "
            f"USING ({_UUID_PREDICATE}) WITH CHECK ({_UUID_PREDICATE})"
        )

    # 4. Re-emit the SECURITY DEFINER resolver with a uuid tenant_id column.
    #    DROP first: CREATE OR REPLACE cannot change a function's return type.
    op.execute(_EXPIRED_DROP)
    op.execute(_EXPIRED_FN_V011)
    op.execute(_EXPIRED_REVOKE)
    op.execute(_EXPIRED_GRANT)


def downgrade() -> None:
    _drop_policies()

    for table in TENANT_TABLES:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN tenant_id TYPE varchar(128) USING tenant_id::text"
        )

    # Restore the text-comparison policies exactly as 008 shipped them.
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS {_POLICY} ON {table}")
        op.execute(
            f"CREATE POLICY {_POLICY} ON {table} "
            f"USING ({_LEGACY_PREDICATE}) WITH CHECK ({_LEGACY_PREDICATE})"
        )

    op.execute(_EXPIRED_DROP)
    op.execute(_LEGACY_EXPIRED_FN)
    op.execute(_EXPIRED_REVOKE)
    op.execute(_EXPIRED_GRANT)
