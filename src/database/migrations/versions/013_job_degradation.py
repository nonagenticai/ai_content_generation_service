"""Record a completion that is not what the pipeline promised.

``video_generation_jobs`` had exactly two terminal vocabularies: ``completed``
(clean) and ``failed`` (``error_message``). A clip that finished but skipped
video normalization -- ffmpeg returned non-zero, timed out, wrote no output, or
was not installed on the pod at all -- was written as ``completed`` with no
marker whatsoever, then counted by the batch tracker and concatenated by
video_compiler under the assumption that every clip shares a resolution, a
framerate, and a stripped audio track. The delivered video jumps format
mid-way and every job in the run is green.

``degradation`` is that missing third state: the job SUCCEEDED, the artifact is
real and still bills, but it is not the artifact that was promised. NULL is a
clean completion, so every existing row keeps its current meaning.

Deliberately NOT ``error_message``: that column is read as "this job failed"
(``JobStatusResponse.error``), and reusing it would make a degraded but usable
clip look like a lost one to every existing consumer.

⚠️ ZERO-DOWNTIME. A nullable ADD COLUMN with no default and no rewrite. An OLD
pod never selects or writes the column -- SQLAlchemy names its columns
explicitly -- so old and new images run side by side through the rollout; the
old one simply leaves the column NULL, which is exactly "no degradation
recorded", not a false clean.

No RLS change: ``video_generation_jobs``'s policies are row-scoped on
``tenant_id`` and apply to whatever columns the row has.

Revision ID: 013_job_degradation
Revises: 012_batch_terminal
Create Date: 2026-09-06 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "013_job_degradation"
down_revision: str | None = "012_batch_terminal"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "video_generation_jobs",
        sa.Column("degradation", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("video_generation_jobs", "degradation")
