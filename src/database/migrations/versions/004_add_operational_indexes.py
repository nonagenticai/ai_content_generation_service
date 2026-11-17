"""Add operational indexes for cleanup queries.

Revision ID: 004
Revises: 003
"""

from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_videogen_jobs_completed_at",
        "videogen_jobs",
        ["completed_at"],
    )
    op.create_index(
        "ix_videogen_jobs_status_completed_at",
        "videogen_jobs",
        ["status", "completed_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_videogen_jobs_status_completed_at", table_name="videogen_jobs")
    op.drop_index("ix_videogen_jobs_completed_at", table_name="videogen_jobs")
