"""Add persisted concept-image batches.

Revision ID: 007
Revises: 006
"""

import sqlalchemy as sa
from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "imagegen_batches",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("batch_id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("budget_limit", sa.Float(), nullable=False, server_default="5.0"),
        sa.Column("total_cost", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("job_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_jobs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_jobs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("batch_id"),
    )
    op.create_index("ix_imagegen_batches_tenant_id", "imagegen_batches", ["tenant_id"])
    op.create_index("ix_imagegen_batches_status", "imagegen_batches", ["status"])
    op.create_index(
        "ix_imagegen_batches_tenant_status",
        "imagegen_batches",
        ["tenant_id", "status"],
    )

    op.add_column("imagegen_jobs", sa.Column("batch_id", sa.String(length=64), nullable=True))
    op.create_index("ix_imagegen_jobs_batch_id", "imagegen_jobs", ["batch_id"])
    op.create_foreign_key(
        "fk_imagegen_jobs_batch_id_imagegen_batches",
        "imagegen_jobs",
        "imagegen_batches",
        ["batch_id"],
        ["batch_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_imagegen_jobs_batch_id_imagegen_batches",
        "imagegen_jobs",
        type_="foreignkey",
    )
    op.drop_index("ix_imagegen_jobs_batch_id", table_name="imagegen_jobs")
    op.drop_column("imagegen_jobs", "batch_id")

    op.drop_index("ix_imagegen_batches_tenant_status", table_name="imagegen_batches")
    op.drop_index("ix_imagegen_batches_status", table_name="imagegen_batches")
    op.drop_index("ix_imagegen_batches_tenant_id", table_name="imagegen_batches")
    op.drop_table("imagegen_batches")
