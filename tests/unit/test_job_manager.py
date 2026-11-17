"""Unit tests for JobManager — mock-only paths.

Real-db state-transition coverage lives in ``test_job_managers_real_db.py``.
This file targets paths that don't depend on atomic UPDATE semantics
(``create_job``, ``get_job``, ``list_jobs``, ``get_usage``,
``find_expired_jobs``, ``delete_jobs``, ``_invalidate_cache``) plus the
postgres-specific upsert in ``record_usage`` exercised via mocks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.database.models import VideoGenerationJob, VideoGenerationUsage
from src.services.job_manager import JobManager


def _mock_db(job: MagicMock | None = None) -> AsyncMock:
    """Return an AsyncMock db session whose execute() yields *job*."""
    db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = job
    db.execute = AsyncMock(return_value=mock_result)
    return db


def _make_job(**overrides) -> MagicMock:
    """Return a MagicMock that looks like a VideoGenerationJob."""
    defaults = dict(
        job_id="job-001",
        tenant_id="tenant-a",
        status="pending",
        provider="fake",
        prompt="a sunset",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        completed_at=None,
    )
    defaults.update(overrides)
    mock_job = MagicMock(spec=VideoGenerationJob)
    for k, v in defaults.items():
        setattr(mock_job, k, v)
    return mock_job


# ============================================================================
# create_job
# ============================================================================


@patch("src.services.job_manager.uuid.uuid7", return_value="fixed-uuid-1234")
async def test_create_job(mock_uuid):
    """create_job builds a VideoGenerationJob, calls db.add and db.flush."""
    del mock_uuid
    db = AsyncMock()
    db.add = MagicMock()

    job = await JobManager.create_job(
        db,
        tenant_id="tenant-a",
        provider="runway",
        prompt="a sunset over the ocean",
        negative_prompt="blurry",
        image_url="https://img.example.com/photo.png",
        duration_s=10.0,
        width=1920,
        height=1080,
        style="cinematic",
        seed=42,
        extra_params={"fps": 24},
        fallback_providers=["kling"],
        callback_url="https://callback.example.com/done",
    )

    assert isinstance(job, VideoGenerationJob)
    assert job.job_id == "fixed-uuid-1234"
    assert job.tenant_id == "tenant-a"
    assert job.provider == "runway"
    assert job.prompt == "a sunset over the ocean"
    assert job.negative_prompt == "blurry"
    assert job.image_url == "https://img.example.com/photo.png"
    assert job.duration_s == 10.0
    assert job.width == 1920
    assert job.height == 1080
    assert job.style == "cinematic"
    assert job.seed == 42
    assert job.extra_params == {"fps": 24}
    assert job.fallback_providers == ["kling"]
    assert job.callback_url == "https://callback.example.com/done"
    assert job.status == "pending"

    db.add.assert_called_once_with(job)
    db.flush.assert_awaited_once()


# ============================================================================
# get_job
# ============================================================================


async def test_get_job_found():
    """get_job returns the job when the query finds a row."""
    mock_job = _make_job()
    db = _mock_db(mock_job)

    result = await JobManager.get_job(db, "job-001")

    assert result is mock_job
    db.execute.assert_awaited_once()


async def test_get_job_not_found():
    """get_job returns None when no row matches."""
    db = _mock_db(None)

    result = await JobManager.get_job(db, "nonexistent")

    assert result is None
    db.execute.assert_awaited_once()


# ============================================================================
# list_jobs
# ============================================================================


async def test_list_jobs_basic():
    """list_jobs returns jobs list and total count."""
    job1 = _make_job(job_id="j1")
    job2 = _make_job(job_id="j2")

    db = AsyncMock()

    count_result = MagicMock()
    count_result.scalar.return_value = 2

    jobs_result = MagicMock()
    jobs_result.scalars.return_value.all.return_value = [job1, job2]

    db.execute = AsyncMock(side_effect=[count_result, jobs_result])

    jobs, total = await JobManager.list_jobs(db, "tenant-a")

    assert total == 2
    assert len(jobs) == 2
    assert jobs[0] is job1
    assert jobs[1] is job2
    assert db.execute.await_count == 2


async def test_list_jobs_with_filters():
    """list_jobs applies status, provider, date filters."""
    db = AsyncMock()
    count_result = MagicMock()
    count_result.scalar.return_value = 0
    jobs_result = MagicMock()
    jobs_result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(side_effect=[count_result, jobs_result])

    jobs, total = await JobManager.list_jobs(
        db,
        "tenant-a",
        status="completed",
        provider="runway",
        created_after="2025-01-01",
        created_before="2025-12-31",
    )

    assert total == 0
    assert jobs == []
    assert db.execute.await_count == 2


async def test_list_jobs_invalid_sort_by():
    """Invalid sort_by falls back to 'created_at'."""
    db = AsyncMock()
    count_result = MagicMock()
    count_result.scalar.return_value = 0
    jobs_result = MagicMock()
    jobs_result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(side_effect=[count_result, jobs_result])

    _jobs, total = await JobManager.list_jobs(db, "tenant-a", sort_by="hacker_field")

    assert total == 0
    assert db.execute.await_count == 2


async def test_list_jobs_sort_order_asc():
    """list_jobs with sort_order='asc' executes without error."""
    db = AsyncMock()
    count_result = MagicMock()
    count_result.scalar.return_value = 0
    jobs_result = MagicMock()
    jobs_result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(side_effect=[count_result, jobs_result])

    _jobs, total = await JobManager.list_jobs(db, "tenant-a", sort_order="asc")

    assert total == 0
    assert db.execute.await_count == 2


async def test_list_jobs_pagination():
    """list_jobs respects page and page_size params."""
    db = AsyncMock()
    count_result = MagicMock()
    count_result.scalar.return_value = 50
    jobs_result = MagicMock()
    jobs_result.scalars.return_value.all.return_value = [_make_job()] * 10
    db.execute = AsyncMock(side_effect=[count_result, jobs_result])

    jobs, total = await JobManager.list_jobs(db, "tenant-a", page=3, page_size=10)

    assert total == 50
    assert len(jobs) == 10


async def test_list_jobs_count_returns_none():
    """When count query returns None, total defaults to 0."""
    db = AsyncMock()
    count_result = MagicMock()
    count_result.scalar.return_value = None
    jobs_result = MagicMock()
    jobs_result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(side_effect=[count_result, jobs_result])

    _jobs, total = await JobManager.list_jobs(db, "tenant-a")

    assert total == 0


# ============================================================================
# get_usage
# ============================================================================


async def test_get_usage():
    """get_usage returns usage rows and provider breakdown."""
    usage_row = MagicMock(spec=VideoGenerationUsage)
    provider_row = ("runway", 1.50, 10)

    db = AsyncMock()

    usage_result = MagicMock()
    usage_result.scalars.return_value.all.return_value = [usage_row]

    provider_result = MagicMock()
    provider_result.all.return_value = [provider_row]

    db.execute = AsyncMock(side_effect=[usage_result, provider_result])

    usage_rows, breakdown = await JobManager.get_usage(db, "tenant-a")

    assert len(usage_rows) == 1
    assert usage_rows[0] is usage_row
    assert len(breakdown) == 1
    assert breakdown[0] == ("runway", 1.50, 10)
    assert db.execute.await_count == 2


async def test_get_usage_with_date_filters():
    """get_usage applies date_from and date_to filters."""
    db = AsyncMock()
    usage_result = MagicMock()
    usage_result.scalars.return_value.all.return_value = []
    provider_result = MagicMock()
    provider_result.all.return_value = []
    db.execute = AsyncMock(side_effect=[usage_result, provider_result])

    usage_rows, breakdown = await JobManager.get_usage(
        db, "tenant-a", date_from="2025-01-01", date_to="2025-06-30"
    )

    assert usage_rows == []
    assert breakdown == []
    assert db.execute.await_count == 2


# ============================================================================
# find_expired_jobs
# ============================================================================


async def test_find_expired_jobs():
    """find_expired_jobs returns matching jobs."""
    expired_job = _make_job(status="completed", completed_at=datetime(2024, 1, 1, tzinfo=UTC))
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [expired_job]
    db.execute = AsyncMock(return_value=result)

    before = datetime(2025, 1, 1, tzinfo=UTC)
    jobs = await JobManager.find_expired_jobs(db, status="completed", before=before, limit=50)

    assert len(jobs) == 1
    assert jobs[0] is expired_job
    db.execute.assert_awaited_once()


async def test_find_expired_jobs_empty():
    """find_expired_jobs returns empty list when none match."""
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)

    before = datetime(2025, 1, 1, tzinfo=UTC)
    jobs = await JobManager.find_expired_jobs(db, status="completed", before=before)

    assert jobs == []


# ============================================================================
# delete_jobs
# ============================================================================


async def test_delete_jobs():
    """delete_jobs executes delete and returns rowcount."""
    db = AsyncMock()
    cursor_result = MagicMock()
    cursor_result.rowcount = 3
    db.execute = AsyncMock(return_value=cursor_result)

    count = await JobManager.delete_jobs(db, ["j1", "j2", "j3"])

    assert count == 3
    db.execute.assert_awaited_once()


async def test_delete_jobs_empty_list():
    """delete_jobs returns 0 immediately for empty list without querying."""
    db = AsyncMock()

    count = await JobManager.delete_jobs(db, [])

    assert count == 0
    db.execute.assert_not_awaited()


async def test_delete_jobs_rowcount_none():
    """delete_jobs returns 0 when rowcount is None."""
    db = AsyncMock()
    cursor_result = MagicMock()
    cursor_result.rowcount = None
    db.execute = AsyncMock(return_value=cursor_result)

    count = await JobManager.delete_jobs(db, ["j1"])

    assert count == 0


# ============================================================================
# record_usage — postgres-specific upsert (mocked here)
# ============================================================================


async def test_record_usage():
    """record_usage executes an upsert statement."""
    db = AsyncMock()

    await JobManager.record_usage(
        db,
        tenant_id="tenant-a",
        generations_completed=1,
        is_error=False,
        duration_generated=9.5,
        file_size=5_000_000,
    )

    db.execute.assert_awaited_once()


async def test_record_usage_error():
    """record_usage with is_error=True includes failure increment."""
    db = AsyncMock()

    await JobManager.record_usage(
        db,
        tenant_id="tenant-a",
        generations_completed=0,
        is_error=True,
        duration_generated=0.0,
        file_size=0,
    )

    db.execute.assert_awaited_once()


# ============================================================================
# _invalidate_cache
# ============================================================================


@patch("src.services.job_manager.get_redis_client", create=True)
async def test_invalidate_cache_success(mock_get_redis):
    """_invalidate_cache calls redis.delete on the job's status key."""
    mock_redis = AsyncMock()
    mock_get_redis.return_value = mock_redis

    with patch("src.cache.redis_client.get_redis_client", return_value=mock_redis):
        await JobManager._invalidate_cache("job-001")

    mock_redis.delete.assert_awaited_once_with("content_generation:job:job-001:status")


@patch("src.cache.redis_client.get_redis_client")
async def test_invalidate_cache_redis_none(mock_get_redis):
    """_invalidate_cache does nothing when redis client is None."""
    mock_get_redis.return_value = None

    await JobManager._invalidate_cache("job-001")


async def test_invalidate_cache_handles_exception(caplog):
    """_invalidate_cache logs a warning AND increments the failure counter.

    Previously this path silently swallowed the failure (`pass`). The
    counter and the warning together make degradation observable.
    """
    import logging

    from src.observability import metrics as metrics_mod

    metrics_mod.setup_metrics()

    # Capture counter value before
    before = 0.0
    if metrics_mod._cache_invalidate_failures is not None:
        before = metrics_mod._cache_invalidate_failures.labels(reason="network")._value.get()

    with (
        patch(
            "src.cache.redis_client.get_redis_client",
            side_effect=OSError("redis down"),
        ),
        caplog.at_level(logging.WARNING, logger="src.services.job_manager"),
    ):
        await JobManager._invalidate_cache("job-001")

    # Warning was logged
    assert any(
        "Cache invalidation failed" in rec.getMessage() and "job-001" in rec.getMessage()
        for rec in caplog.records
    )

    # Counter was incremented for the network reason
    assert metrics_mod._cache_invalidate_failures is not None
    after = metrics_mod._cache_invalidate_failures.labels(reason="network")._value.get()
    assert after == before + 1.0


async def test_invalidate_cache_handles_runtime_error(caplog):
    """RuntimeError is also captured separately as reason='runtime'."""
    import logging

    from src.observability import metrics as metrics_mod

    metrics_mod.setup_metrics()

    before = 0.0
    if metrics_mod._cache_invalidate_failures is not None:
        before = metrics_mod._cache_invalidate_failures.labels(reason="runtime")._value.get()

    with (
        patch(
            "src.cache.redis_client.get_redis_client",
            side_effect=RuntimeError("redis client misconfigured"),
        ),
        caplog.at_level(logging.WARNING, logger="src.services.job_manager"),
    ):
        await JobManager._invalidate_cache("job-002")

    assert any(
        "runtime" in rec.getMessage() or "job-002" in rec.getMessage() for rec in caplog.records
    )
    assert metrics_mod._cache_invalidate_failures is not None
    after = metrics_mod._cache_invalidate_failures.labels(reason="runtime")._value.get()
    assert after == before + 1.0
