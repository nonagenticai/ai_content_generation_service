"""Tenant provider slots on the job rows (P5-2): slot metadata + sealed secret.

The orchestrator now sends a tenant-sourced ``image`` / ``video_gen`` slot on the
generation request (``provider_slot``). The job must survive a pod crash, and the
reaper resumes it on ANOTHER pod from the row alone -- so the slot has to be on
the row:

  * ``provider_slot`` (jsonb, nullable) -- the NON-secret slot fields
    (slot/kind/base_url/model/options/ref/has_secret). NULL = a platform job,
    which is every existing row and keeps today's behaviour.
  * ``provider_secret_sealed`` (bytea, nullable) -- the slot secret as an
    AES-256-GCM envelope ``v1|kid|nonce|ct`` bound to the job id, sealed with
    ACG's own ``PROVIDER_SECRET_KEYS`` keyring.

Plus a row trigger on both tables that NULLs ``provider_secret_sealed`` the moment
a row turns terminal (``completed`` / ``failed`` / ``cancelled``), whoever writes
that status -- including the SQL-only poison / aged-out paths of the reclaim
functions, which never enter Python.

⚠️ ZERO-DOWNTIME. Nullable ADD COLUMN with no default (no rewrite). An OLD pod
names its columns explicitly and never reads or writes these, so old and new
images run side by side through the rollout. The trigger only fires when
``provider_secret_sealed`` is non-NULL, which an old pod can never cause.

⚠️ DRIFT-TOLERANT. ``ADD COLUMN IF NOT EXISTS`` / ``CREATE OR REPLACE`` /
``DROP ... IF EXISTS`` so a re-run against a partially-applied database converges.

⚠️ PINNED, not derived: the trigger DDL is inlined verbatim (see 014's note);
``tests/unit/test_migration_snapshot_drift.py`` asserts it equals the live
``rls_resolvers`` source while this revision is head.

No RLS change: the tables' policies are row-scoped on ``tenant_id``.

Revision ID: 015_provider_slot
Revises: 014_imagegen_retention
Create Date: 2026-10-09 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "015_provider_slot"
down_revision: str | None = "014_imagegen_retention"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("video_generation_jobs", "imagegen_jobs")

# Era-correct AS OF THIS REVISION.
_PURGE_SEALED_SLOT_FN_V015 = """
CREATE OR REPLACE FUNCTION purge_provider_secret_on_terminal() RETURNS trigger
  LANGUAGE plpgsql AS $fn$
  BEGIN
    NEW.provider_secret_sealed := NULL;
    RETURN NEW;
  END
  $fn$
"""

_PURGE_TRIGGER_TEMPLATE_V015 = """
CREATE OR REPLACE TRIGGER trg_{table}_purge_slot_secret
  BEFORE INSERT OR UPDATE ON {table}
  FOR EACH ROW
  WHEN (NEW.status IN ('completed', 'failed', 'cancelled')
        AND NEW.provider_secret_sealed IS NOT NULL)
  EXECUTE FUNCTION purge_provider_secret_on_terminal()
"""

_PURGE_VIDEO_TRIGGER_V015 = _PURGE_TRIGGER_TEMPLATE_V015.format(table=_TABLES[0])
_PURGE_IMAGE_TRIGGER_V015 = _PURGE_TRIGGER_TEMPLATE_V015.format(table=_TABLES[1])


# Bound the ACCESS EXCLUSIVE wait: ADD COLUMN / CREATE TRIGGER queue behind a
# long-running transaction on these hot job tables, and every later query queues
# behind THEM -- a stuck migration would stall the whole service. Fail fast and
# let the rollout retry instead. SET LOCAL scopes it to the migration transaction.
_LOCK_TIMEOUT = "SET LOCAL lock_timeout = '5s'"


def upgrade() -> None:
    op.execute(_LOCK_TIMEOUT)
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS provider_slot jsonb")
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS provider_secret_sealed bytea")
    op.execute(_PURGE_SEALED_SLOT_FN_V015)
    op.execute(_PURGE_VIDEO_TRIGGER_V015)
    op.execute(_PURGE_IMAGE_TRIGGER_V015)


def downgrade() -> None:
    op.execute(_LOCK_TIMEOUT)
    for table in _TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_purge_slot_secret ON {table}")
    op.execute("DROP FUNCTION IF EXISTS purge_provider_secret_on_terminal()")
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS provider_secret_sealed")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS provider_slot")
