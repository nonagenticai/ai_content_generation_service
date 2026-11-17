"""Phase D RLS — enable row-level security on the tenant-scoped tables.

(1) Strict per-table policies (w2v_common) on all 4 ``tenant_id`` tables:
    video_generation_jobs, video_generation_usage, imagegen_jobs, imagegen_batches.
(2) SECURITY DEFINER ``expired_video_generation_jobs`` read resolver so the
    boot-time cleanup loop can still find expired jobs across all tenants under
    enforce (a non-bypass role with no GUC would see 0 rows); the cleanup then
    deletes per-tenant via the existing JobManager. EXECUTE is REVOKEd from PUBLIC
    here and GRANTed to the app role out-of-band in Stage A.

ENABLE-not-FORCE → inert until the runtime connects as the restricted
``w2v_ai_content_app`` role with ``TENANT_RLS_MODE=enforce``. Fully reversible.

⚠️ PINNED — DO NOT REVERT TO THE HELPERS. This revision used to render its DDL
from ``w2v_common.database.rls.upgrade_statements(Base.metadata)`` and from
``src.database.rls_resolvers.upgrade_statements()`` — i.e. from LIVE application
code, which replays whatever the model says TODAY rather than what it said at
this revision's era. Migration 011 flips ``tenant_id`` varchar(128) → uuid, so
the helper now emits ``tenant_id = NULLIF(...)::uuid`` and the resolver now
declares ``RETURNS TABLE(... tenant_id uuid ...)``. Replayed here — against the
varchar column that exists at revision 008 — that fails a FRESH database at THIS
revision with ``operator does not exist: character varying = uuid`` (and a
return-type mismatch on the resolver), while ``alembic heads``/ruff/ty stay
green because nothing static executes it. A migration is a historical snapshot:
the era-correct text-comparison DDL is inlined below on purpose. 011 drops these
policies, ALTERs the column, and re-creates them from the (uuid) metadata.

Revision ID: 008_rls
Revises: 007
"""

from alembic import op

revision = "008_rls"
down_revision = "007"
branch_labels = None
depends_on = None

# The 4 NOT NULL ``tenant_id`` tables as of this revision → strict policies.
_TABLES = (
    "imagegen_batches",
    "imagegen_jobs",
    "video_generation_jobs",
    "video_generation_usage",
)

_POLICY = "tenant_isolation"
# Era-correct predicate: varchar tenant_id compared text-to-text. Postgres stores
# it as ((tenant_id)::text = NULLIF(current_setting('app.current_tenant'::text,
# true), ''::text)) — that is the same expression after parse/deparse.
_PREDICATE = "tenant_id = NULLIF(current_setting('app.current_tenant', true), '')"

# Policy names the helper could ever have created, dropped on downgrade so the
# reversal stays exact even though only the strict one is created here.
_ALL_POLICIES = (
    _POLICY,
    f"{_POLICY}_read",
    f"{_POLICY}_ins",
    f"{_POLICY}_upd",
    f"{_POLICY}_del",
)

_APP_ROLE = "w2v_ai_content_app"

# Era-correct resolver: ``tenant_id varchar``. 011 re-emits it as uuid.
_EXPIRED_FN_V008 = """
CREATE OR REPLACE FUNCTION expired_video_generation_jobs(
    p_status text, p_before timestamptz, p_limit int
) RETURNS TABLE(job_id varchar, tenant_id varchar, video_path varchar, minio_key varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT vg.job_id, vg.tenant_id, vg.video_path, vg.minio_key FROM video_generation_jobs vg
    WHERE vg.status = p_status AND vg.completed_at <= p_before
    LIMIT p_limit;
  $fn$
"""

_EXPIRED_DROP = "DROP FUNCTION IF EXISTS expired_video_generation_jobs(text, timestamptz, int)"
_EXPIRED_REVOKE = (
    "REVOKE EXECUTE ON FUNCTION expired_video_generation_jobs(text, timestamptz, int) FROM PUBLIC"
)
_EXPIRED_GRANT = (
    "DO $$\nBEGIN\n"  # nosec B608  # noqa: S608 -- role/fn names are hardcoded constants
    f"  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_APP_ROLE}') THEN\n"
    "    GRANT EXECUTE ON FUNCTION expired_video_generation_jobs(text, timestamptz, int) "
    f"TO {_APP_ROLE};\n"
    "  END IF;\nEND $$"
)


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS {_POLICY} ON {table}")
        op.execute(
            f"CREATE POLICY {_POLICY} ON {table} USING ({_PREDICATE}) WITH CHECK ({_PREDICATE})"
        )
    op.execute(_EXPIRED_FN_V008)
    op.execute(_EXPIRED_REVOKE)
    op.execute(_EXPIRED_GRANT)


def downgrade() -> None:
    op.execute(_EXPIRED_DROP)
    for table in _TABLES:
        for policy in _ALL_POLICIES:
            op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
