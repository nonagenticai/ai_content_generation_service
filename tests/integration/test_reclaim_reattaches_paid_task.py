"""A reclaimed job must RE-ATTACH to its paid provider task, never re-submit.

Defect (cross-pod double billing). ``provider_job_id`` was written only inside
``VideoGenerationService._persist_success``, i.e. after the clip had been
downloaded and stored. Every step before that -- and generation is the slow
part -- left the row with no record of the provider task the tenant had
already been charged for.

The rest of the machinery then does exactly the wrong thing with that gap:

* the service runs ``replicas: 1`` with ``maxSurge: 0``, so a rollout tears the
  pod down before a replacement exists;
* ``migrations/010_claim_refund_lifetime`` resets any ``processing`` row whose
  lease has lapsed back to ``pending``;
* ``reaper_service._dispatch_video`` re-runs the full ``_process_job_bg``;
* ``VideoGenerationProvider.generate`` always started at ``submit()``, and no
  provider sends an outbound idempotency key.

So one logical job bought one clip per reclaim, up to ``max_attempts``.

The fix has two halves, both exercised here against the real Postgres:
``on_submit`` persists the task id the instant the provider accepts it, and
``_AdoptableTask`` feeds that id back into ``generate(resume_task_id=...)`` on
the reclaimed run, which skips ``submit()`` entirely.

Scope note: this covers the CROSS-POD path. The in-process variant -- an
ambiguous client timeout ON the submit request itself -- is PR #118's, in
``tests/integration/test_submit_timeout_no_rebill.py``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

from sqlalchemy import update

from src.database.models import VideoGenerationJob
from src.services.job_manager import JobManager
from src.services.providers.base import VideoGenerationProvider, VideoProviderRegistry
from src.services.video_generation_service import VideoGenerationService
from tests.conftest import TENANT_A


class _CountingProvider(VideoGenerationProvider):
    """Counts submits. ``fail_polls`` makes the first N polls blow up."""

    name = "counting"

    def __init__(self, fail_polls: int = 0) -> None:
        self.submits: list[str] = []
        self.polled: list[str] = []
        self._fail_polls = fail_polls

    async def submit(self, prompt: str, **kwargs: Any) -> str:
        task_id = f"paid-task-{len(self.submits) + 1}"
        self.submits.append(task_id)
        return task_id

    async def check_status(self, task_id: str) -> dict:
        self.polled.append(task_id)
        if self._fail_polls > 0:
            self._fail_polls -= 1
            # A crash/eviction is not an exception the process gets to see, but
            # every observable precondition for the reclaim is the same: the
            # row is left non-terminal-in-spirit with the money already spent.
            # The test drives the row back to `processing` explicitly below.
            raise RuntimeError("pod lost the poll connection")
        return {"status": "completed", "video_url": "https://cdn.example/v.mp4", "error": None}

    async def download(self, task_id: str, video_url: str) -> bytes:
        del video_url
        return b"clip-bytes"

    def estimate_cost(self, duration_s: float) -> float:
        return 0.5 * duration_s


def _service(provider: _CountingProvider) -> VideoGenerationService:
    registry = VideoProviderRegistry()
    registry.register(provider)
    storage = AsyncMock()
    storage.store_video.return_value = "/tmp/clip.mp4"
    storage.upload_to_minio.return_value = "minio/clip.mp4"
    return VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider=provider.name,
        # No in-process retries and no fallback chain: the ONLY way a second
        # submit can happen in this test is the cross-pod reclaim path.
        max_retries=0,
        fallback_enabled=False,
    )


async def _crash_the_pod(db: Any, job_id: str) -> None:
    """Put the row in the state a killed pod leaves: processing, lease lapsed.

    A real pod death writes nothing at all; this test cannot kill its own
    process, so it restores that state by hand. The reclaim itself is then done
    by the REAL ``reclaim_orphaned`` SQL, not simulated.
    """
    await db.execute(
        update(VideoGenerationJob)
        .where(VideoGenerationJob.job_id == job_id)
        .values(
            status="processing",
            claimed_by="pod-a",
            lease_expires_at=datetime.now(UTC) - timedelta(seconds=30),
            error_message=None,
        )
    )
    await db.commit()
    db.expunge_all()


async def _create_dispatched_job(db: Any, svc: VideoGenerationService) -> str:
    job_id = await svc.create_job(
        db,
        tenant_id=TENANT_A,
        prompt="a paid clip",
        provider="counting",
        duration_s=4.0,
    )
    await JobManager.register_dispatch(
        db,
        job_id,
        options={"kind": "video_single", "budget_limit": 100.0},
        worker_id="pod-a",
        lease_seconds=900,
    )
    await db.commit()
    return job_id


async def test_paid_task_id_is_persisted_before_generation_completes(db_session):
    """Half one: the id is on the row even though the job never completed."""
    provider = _CountingProvider(fail_polls=1)
    svc = _service(provider)
    job_id = await _create_dispatched_job(db_session, svc)

    cost = await svc.process_job(db_session, job_id, TENANT_A, budget_limit=100.0)

    assert cost == 0.0
    assert provider.submits == ["paid-task-1"]
    db_session.expunge_all()
    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    # Pre-fix this is None: _persist_success never ran, so nothing recorded the
    # task the provider had already been paid for.
    assert job.provider_job_id == "paid-task-1"


async def test_reclaimed_job_reattaches_and_never_submits_twice(db_session):
    """Half two: the reclaimed run polls the paid task instead of buying another."""
    provider = _CountingProvider(fail_polls=1)
    svc = _service(provider)
    job_id = await _create_dispatched_job(db_session, svc)

    # Run 1: the pod is generating, then dies.
    await svc.process_job(db_session, job_id, TENANT_A, budget_limit=100.0)
    assert provider.submits == ["paid-task-1"]
    await _crash_the_pod(db_session, job_id)

    # The REAL reclaim SQL: lapsed lease -> back to `pending`, attempt charged.
    claimed = await JobManager.reclaim_orphaned(
        db_session, worker_id="pod-b", lease_seconds=900, max_attempts=5
    )
    assert claimed is not None
    assert claimed.job_id == job_id
    db_session.expunge_all()

    # Run 2 is what the reaper does: the very same _process_job_bg entrypoint.
    cost = await svc.process_job(db_session, job_id, TENANT_A, budget_limit=100.0)

    # THE ASSERTION. One logical job, one provider submit, one charge.
    assert provider.submits == ["paid-task-1"], (
        f"reclaim issued a SECOND paid provider submit: {provider.submits}"
    )
    # ...and it got there by polling the adopted task, not by skipping work.
    assert provider.polled[-1] == "paid-task-1"
    assert cost == 2.0  # 0.5 * 4.0s

    db_session.expunge_all()
    job = await JobManager.get_job(db_session, job_id)
    assert job is not None
    assert job.status == "completed"


async def test_a_fallback_provider_never_adopts_another_providers_task(db_session):
    """A task id is only meaningful to the provider that issued it."""
    primary = _CountingProvider()
    primary.name = "counting"
    secondary = _CountingProvider()
    secondary.name = "other"

    registry = VideoProviderRegistry()
    registry.register(primary)
    registry.register(secondary)
    storage = AsyncMock()
    storage.store_video.return_value = "/tmp/clip.mp4"
    storage.upload_to_minio.return_value = "minio/clip.mp4"
    svc = VideoGenerationService(
        storage=storage, registry=registry, default_provider="counting", max_retries=0
    )

    job_id = str(uuid.uuid7())
    db_session.add(
        VideoGenerationJob(
            job_id=job_id,
            tenant_id=TENANT_A,
            provider="counting",
            prompt="a paid clip",
            status="pending",
            duration_s=4.0,
            # Set as if a previous run had already paid `counting` for this id.
            provider_job_id="paid-task-1",
            fallback_providers=["other"],
            job_options={"kind": "video_single"},
        )
    )
    await db_session.commit()
    db_session.expunge_all()

    resumed = svc._resume_task_id(job_id, "other", 0, None)
    assert resumed is None

    from src.services.video_generation_service import _AdoptableTask

    adoptable = _AdoptableTask(provider="counting", task_id="paid-task-1")
    assert svc._resume_task_id(job_id, "other", 0, adoptable) is None
    assert svc._resume_task_id(job_id, "counting", 0, adoptable) == "paid-task-1"
    # A retry only happens because the adopted task was unusable; re-adopting
    # it would just re-fail forever.
    assert svc._resume_task_id(job_id, "counting", 1, adoptable) is None
