"""Add image generation jobs table.

Revision ID: 005
Revises: 004
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "imagegen_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(64), nullable=False),
        sa.Column("asset_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("negative_prompt", sa.Text(), nullable=True),
        sa.Column("style", sa.String(128), nullable=True),
        sa.Column("size", sa.String(32), nullable=True),
        sa.Column("quality", sa.String(32), nullable=True),
        sa.Column("reference_image_urls", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("usage_context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("target_usage", sa.String(64), nullable=True),
        sa.Column("extra_params", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("provider_job_id", sa.String(256), nullable=True),
        sa.Column("image_path", sa.Text(), nullable=True),
        sa.Column("image_url", sa.Text(), nullable=True),
        sa.Column("mime_type", sa.String(128), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("cost", sa.Float(), nullable=True),
        sa.Column("minio_key", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("callback_url", sa.Text(), nullable=True),
        sa.Column("callback_status", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("asset_id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index("ix_imagegen_jobs_tenant_id", "imagegen_jobs", ["tenant_id"])
    op.create_index("ix_imagegen_jobs_tenant_status", "imagegen_jobs", ["tenant_id", "status"])
    op.create_index("ix_imagegen_jobs_completed_at", "imagegen_jobs", ["completed_at"])
    op.create_index(
        "ix_imagegen_jobs_status_completed_at", "imagegen_jobs", ["status", "completed_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_imagegen_jobs_status_completed_at", table_name="imagegen_jobs")
    op.drop_index("ix_imagegen_jobs_completed_at", table_name="imagegen_jobs")
    op.drop_index("ix_imagegen_jobs_tenant_status", table_name="imagegen_jobs")
    op.drop_index("ix_imagegen_jobs_tenant_id", table_name="imagegen_jobs")
    op.drop_table("imagegen_jobs")
