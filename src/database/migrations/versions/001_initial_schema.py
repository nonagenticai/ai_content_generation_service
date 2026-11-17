"""Initial schema for videogen_jobs and videogen_usage tables.

Revision ID: 001
Revises: None
Create Date: 2026-02-23
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "videogen_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        # Generation parameters
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("negative_prompt", sa.Text(), nullable=True),
        sa.Column("image_url", sa.Text(), nullable=True),
        sa.Column("duration_s", sa.Float(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("style", sa.String(128), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("extra_params", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        # Provider-side tracking
        sa.Column("provider_job_id", sa.String(256), nullable=True),
        # Results
        sa.Column("video_path", sa.Text(), nullable=True),
        sa.Column("video_url", sa.Text(), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("actual_duration_s", sa.Float(), nullable=True),
        sa.Column("generation_time_seconds", sa.Float(), nullable=True),
        # Metadata
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index("ix_videogen_jobs_tenant_id", "videogen_jobs", ["tenant_id"])
    op.create_index("ix_videogen_jobs_tenant_status", "videogen_jobs", ["tenant_id", "status"])
    op.create_index("ix_videogen_jobs_created_at", "videogen_jobs", ["created_at"])

    op.create_table(
        "videogen_usage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("date", sa.String(10), nullable=False),
        sa.Column("generations_started", sa.Integer(), server_default="0"),
        sa.Column("generations_completed", sa.Integer(), server_default="0"),
        sa.Column("generations_failed", sa.Integer(), server_default="0"),
        sa.Column("total_duration_generated_s", sa.Float(), server_default="0.0"),
        sa.Column("total_file_size_bytes", sa.Integer(), server_default="0"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "date", name="uq_videogen_usage_tenant_date"),
    )


def downgrade() -> None:
    op.drop_table("videogen_usage")
    op.drop_index("ix_videogen_jobs_created_at", table_name="videogen_jobs")
    op.drop_index("ix_videogen_jobs_tenant_status", table_name="videogen_jobs")
    op.drop_index("ix_videogen_jobs_tenant_id", table_name="videogen_jobs")
    op.drop_table("videogen_jobs")
