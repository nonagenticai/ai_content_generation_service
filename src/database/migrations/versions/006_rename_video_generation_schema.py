"""Rename video generation tables and indexes to match content-generation naming.

Revision ID: 006
Revises: 005
"""

from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.rename_table("videogen_jobs", "video_generation_jobs")
    op.execute(
        "ALTER INDEX IF EXISTS ix_videogen_jobs_tenant_id "
        "RENAME TO ix_video_generation_jobs_tenant_id"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_videogen_jobs_tenant_status "
        "RENAME TO ix_video_generation_jobs_tenant_status"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_videogen_jobs_created_at "
        "RENAME TO ix_video_generation_jobs_created_at"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_videogen_jobs_completed_at "
        "RENAME TO ix_video_generation_jobs_completed_at"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_videogen_jobs_status_completed_at "
        "RENAME TO ix_video_generation_jobs_status_completed_at"
    )

    op.rename_table("videogen_usage", "video_generation_usage")
    op.execute(
        "ALTER TABLE video_generation_usage "
        "RENAME CONSTRAINT uq_videogen_usage_tenant_date "
        "TO uq_video_generation_usage_tenant_date"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE video_generation_usage "
        "RENAME CONSTRAINT uq_video_generation_usage_tenant_date "
        "TO uq_videogen_usage_tenant_date"
    )
    op.rename_table("video_generation_usage", "videogen_usage")

    op.execute(
        "ALTER INDEX IF EXISTS ix_video_generation_jobs_status_completed_at "
        "RENAME TO ix_videogen_jobs_status_completed_at"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_video_generation_jobs_completed_at "
        "RENAME TO ix_videogen_jobs_completed_at"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_video_generation_jobs_created_at "
        "RENAME TO ix_videogen_jobs_created_at"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_video_generation_jobs_tenant_status "
        "RENAME TO ix_videogen_jobs_tenant_status"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_video_generation_jobs_tenant_id "
        "RENAME TO ix_videogen_jobs_tenant_id"
    )
    op.rename_table("video_generation_jobs", "videogen_jobs")
