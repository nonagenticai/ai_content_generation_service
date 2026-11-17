"""Inbound Idempotency-Key on both job tables.

The orchestrator re-submits a generation after a client-side wait timeout and
sends ``Idempotency-Key`` expecting the ORIGINAL job back. ACG ignored it and
created a new job + background task per re-submit, doubling GPU work on the
single-lane image backend. This revision stores the key on the job row:

  * ``idempotency_key`` (varchar(255), nullable) -- NULL = no key sent, which is
    every existing row and every batch child.
  * ``idempotency_fingerprint`` (varchar(64), nullable) -- sha256 of the request
    body, so a reused key with a different body is refused.
  * a PARTIAL UNIQUE index on ``(tenant_id, idempotency_key) WHERE
    idempotency_key IS NOT NULL`` -- the race-safety anchor for
    ``INSERT ... ON CONFLICT DO NOTHING``. Per tenant: two tenants may send the
    same key and get two jobs.

⚠️ ZERO-DOWNTIME. Nullable ADD COLUMN with no default (no rewrite). An OLD pod
names its columns explicitly and never reads or writes these. The new index is
built over an all-NULL column, so it is empty and the build is one scan.

⚠️ DRIFT-TOLERANT. ``IF NOT EXISTS`` / ``IF EXISTS`` throughout.

No RLS change: the tables' policies are row-scoped on ``tenant_id``, and the
lookup by key is always tenant-qualified.

Revision ID: 016_idempotency_key
Revises: 015_provider_slot
Create Date: 2026-10-21 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "016_idempotency_key"
down_revision: str | None = "015_provider_slot"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("video_generation_jobs", "imagegen_jobs")

# See 015: bound the ACCESS EXCLUSIVE wait on these hot tables; fail fast and let
# the rollout retry rather than stall every query behind a queued ALTER.
_LOCK_TIMEOUT = "SET LOCAL lock_timeout = '5s'"


def upgrade() -> None:
    op.execute(_LOCK_TIMEOUT)
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS idempotency_key varchar(255)")
        op.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS idempotency_fingerprint varchar(64)"
        )
        op.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS uq_{table}_tenant_idempotency_key "
            f"ON {table} (tenant_id, idempotency_key) WHERE idempotency_key IS NOT NULL"
        )


def downgrade() -> None:
    op.execute(_LOCK_TIMEOUT)
    for table in _TABLES:
        op.execute(f"DROP INDEX IF EXISTS uq_{table}_tenant_idempotency_key")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS idempotency_fingerprint")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS idempotency_key")
