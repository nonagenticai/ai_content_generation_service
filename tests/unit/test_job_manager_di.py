"""Constructor-injection tests for ``JobManager``.

These exercise the new instance interface end-to-end using:

* a real postgres+psycopg ``AsyncSession`` (via the ``db_session``
  / ``db_session_factory`` fixtures), and
* a real ``fakeredis``-backed ``RedisClient`` (via the
  ``real_redis_client`` fixture).

No ``unittest.mock.patch`` is used here — the goal is to prove the
DI path works without monkeypatching internals.
"""

from __future__ import annotations

import pytest

from src.cache.redis_client import RedisClient
from src.services.job_manager import (
    JobManager,
    JobStateConflictError,
    get_default_job_manager,
    set_default_job_manager,
)
from tests.conftest import TENANT_A, TENANT_B


@pytest.fixture
def jm_with_redis(db_session_factory, real_redis_client: RedisClient) -> JobManager:
    """JobManager wired with real (fake) Redis + real session factory."""
    return JobManager(
        session_factory=db_session_factory,
        redis_client=real_redis_client,
    )


class TestJobManagerConstruction:
    """Verify constructor wires collaborators correctly."""

    def test_construct_with_no_args_yields_empty_collaborators(self) -> None:
        jm = JobManager()
        assert jm.session_factory is None
        assert jm.redis_client is None

    def test_construct_with_redis_client(self, real_redis_client: RedisClient) -> None:
        jm = JobManager(redis_client=real_redis_client)
        assert jm.redis_client is real_redis_client
        assert jm.session_factory is None

    def test_construct_with_session_factory(self, db_session_factory) -> None:
        jm = JobManager(session_factory=db_session_factory)
        assert jm.session_factory is db_session_factory
        assert jm.redis_client is None

    def test_construct_with_both(self, db_session_factory, real_redis_client: RedisClient) -> None:
        jm = JobManager(session_factory=db_session_factory, redis_client=real_redis_client)
        assert jm.session_factory is db_session_factory
        assert jm.redis_client is real_redis_client


class TestJobManagerSingleton:
    """Verify the module-level singleton accessor."""

    def test_default_singleton_lazy_creates(self) -> None:
        # Reset to ensure lazy-create path is exercised
        set_default_job_manager(None)
        jm = get_default_job_manager()
        assert isinstance(jm, JobManager)
        # Calling again returns the same instance
        assert get_default_job_manager() is jm
        set_default_job_manager(None)

    def test_set_and_get_default(self, real_redis_client: RedisClient) -> None:
        custom = JobManager(redis_client=real_redis_client)
        set_default_job_manager(custom)
        try:
            assert get_default_job_manager() is custom
            assert get_default_job_manager().redis_client is real_redis_client
        finally:
            set_default_job_manager(None)


class TestJobManagerInstanceCacheInvalidation:
    """``invalidate_cache`` should use the injected Redis client."""

    async def test_invalidate_cache_with_injected_redis_deletes_key(
        self, jm_with_redis: JobManager, real_redis_client: RedisClient
    ) -> None:
        # Pre-seed a cache entry that should be cleared by invalidate_cache.
        key = "content_generation:job:job-abc:status"
        await real_redis_client.set(key, "cached-status")
        assert await real_redis_client.get(key) == "cached-status"

        await jm_with_redis.invalidate_cache("job-abc")

        assert await real_redis_client.get(key) is None

    async def test_invalidate_cache_no_redis_is_noop(self) -> None:
        # When constructed without a redis_client AND no module-level
        # client is configured, the call should silently no-op.
        jm = JobManager()
        # Should not raise -- this is the legacy fallback path.
        await jm.invalidate_cache("nonexistent")


class TestJobManagerInstanceLifecycleAgainstRealDB:
    """Drive the static-method job lifecycle via ``JobManager.create_job``
    while having a fully-constructed instance present.

    Even though the static methods do not currently consume ``self``,
    these tests prove that having an instance around with injected
    deps does not interfere with normal CRUD on a real DB session.
    """

    async def test_create_and_complete_lifecycle(
        self, jm_with_redis: JobManager, db_session
    ) -> None:
        # Use the static interface that production currently uses.
        job = await JobManager.create_job(
            db_session,
            tenant_id=TENANT_A,
            provider="runway",
            prompt="DI lifecycle test",
        )
        assert job.status == "pending"

        await JobManager.mark_processing(db_session, job.job_id)
        await JobManager.mark_completed(
            db_session,
            job.job_id,
            cost=0.42,
        )
        # Then invalidate cache via the instance method (DI path).
        await jm_with_redis.invalidate_cache(job.job_id)

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.cost == 0.42

    async def test_state_conflict_via_real_db(self, jm_with_redis: JobManager, db_session) -> None:
        # Verify that the conflict path raises the expected typed error
        # when driven through the instance-aware setup.
        job = await JobManager.create_job(
            db_session, tenant_id=TENANT_B, provider="kling", prompt="conflict"
        )
        await JobManager.mark_failed(db_session, job.job_id, "boom")

        with pytest.raises(JobStateConflictError):
            await JobManager.mark_completed(db_session, job.job_id)
