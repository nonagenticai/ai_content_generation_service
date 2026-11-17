"""Add fallback, retry, and callback columns.

Revision ID: 003
Revises: 002
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("videogen_jobs", sa.Column("fallback_providers", JSONB, nullable=True))
    op.add_column(
        "videogen_jobs", sa.Column("retry_count", sa.Integer(), server_default="0", nullable=True)
    )
    op.add_column("videogen_jobs", sa.Column("callback_url", sa.Text(), nullable=True))
    op.add_column("videogen_jobs", sa.Column("callback_status", sa.String(32), nullable=True))


def downgrade() -> None:
    op.drop_column("videogen_jobs", "callback_status")
    op.drop_column("videogen_jobs", "callback_url")
    op.drop_column("videogen_jobs", "retry_count")
    op.drop_column("videogen_jobs", "fallback_providers")
