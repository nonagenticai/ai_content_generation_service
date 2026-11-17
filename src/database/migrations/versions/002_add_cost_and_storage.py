"""Add cost and minio_key columns to videogen_jobs.

Revision ID: 002
Revises: 001
Create Date: 2026-02-23
"""

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("videogen_jobs", sa.Column("cost", sa.Float(), nullable=True))
    op.add_column("videogen_jobs", sa.Column("minio_key", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("videogen_jobs", "minio_key")
    op.drop_column("videogen_jobs", "cost")
