"""Unit tests for API routes (src/api/routes.py).

Tests the route handlers through FastAPI's TestClient, verifying correct
status codes, response shapes, error paths, quota enforcement, cache
behaviour, and request validation.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.routes import router
from src.api.schemas import GenerateRequest

# ============================================================================
# Helpers
# ============================================================================


# A real registered tenant, forwarded the way a live request forwards one.
# There is no default/sentinel tenant any more: a request that carries none is a
# 400 (see ``src.api.routes.dependencies._get_tenant_id``), so the harness must
# supply one exactly like ``AuthMiddleware`` does in production.
HARNESS_TENANT = "0195f48b-1234-7000-8000-0000000000aa"


def _build_app(
    *,
    tenant_id: str | None = HARNESS_TENANT,
    video_generation_service: Any = None,
    provider_registry: Any = None,
    storage_service: Any = None,
    batch_tracker: Any = None,
    redis_client: Any = None,
    quota_service: Any = None,
    health_monitor: Any = None,
) -> FastAPI:
    """Create a minimal FastAPI app with mocked app.state dependencies."""
    app = FastAPI()

    if tenant_id is not None:

        @app.middleware("http")
        async def _set_tenant(request: Any, call_next: Any) -> Any:  # pragma: no cover - harness
            request.state.tenant_id = tenant_id
            return await call_next(request)

    app.include_router(router)

    app.state.video_generation_service = video_generation_service or MagicMock()
    app.state.provider_registry = provider_registry or MagicMock()
    app.state.storage_service = storage_service or MagicMock()
    app.state.batch_tracker = batch_tracker or MagicMock()
    if redis_client is not None:
        app.state.redis_client = redis_client
    if quota_service is not None:
        app.state.quota_service = quota_service
    if health_monitor is not None:
        app.state.health_monitor = health_monitor

    return app


def _mock_db() -> AsyncMock:
    """Return an AsyncMock that stands in for AsyncSession."""
    db = AsyncMock()
    db.commit = AsyncMock()
    return db


@dataclass
class _FakeJob:
    """Lightweight stand-in for VideoGenerationJob ORM model."""

    job_id: str = "job-1"
    status: str = "completed"
    provider: str | None = "fake"
    prompt: str | None = "a cat"
    negative_prompt: str | None = None
    image_url: str | None = None
    duration_s: float | None = 5.0
    video_path: str | None = "/tmp/video.mp4"
    video_url: str | None = "https://cdn.example.com/video.mp4"
    minio_key: str | None = None
    file_size_bytes: int | None = 1024
    actual_duration_s: float | None = 5.0
    generation_time_seconds: float | None = 2.5
    cost: float | None = 0.05
    error_message: str | None = None
    degradation: str | None = None
    created_at: datetime | None = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    completed_at: datetime | None = datetime(2024, 1, 15, 12, 0, 3, tzinfo=UTC)
    updated_at: datetime | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    style: str | None = None
    extra_params: dict | None = None
    fallback_providers: list[str] | None = None
    callback_url: str | None = None
    retry_count: int = 0


def _make_provider_mock(
    *,
    name: str = "fake",
    max_duration: float | None = 30.0,
    supports_i2v: bool = True,
) -> MagicMock:
    provider = MagicMock()
    provider.name = name
    provider.get_max_duration_s.return_value = max_duration
    provider.supports_image_to_video.return_value = supports_i2v
    provider.estimate_cost.return_value = 0.05
    provider.generate = AsyncMock()
    return provider


def _make_registry_mock(providers: dict[str, Any] | None = None) -> MagicMock:
    """Build a registry mock. ``providers`` maps name -> provider mock."""
    registry = MagicMock()
    providers = providers or {}

    def _get(name: str) -> Any:
        return providers.get(name)

    registry.get.side_effect = _get
    registry.provider_names = list(providers.keys())
    registry.list_providers.return_value = []
    return registry


# ============================================================================
# POST /generate
# ============================================================================


class TestGenerateVideo:
    """Tests for POST /api/content/videos/generate."""

    async def test_generate_returns_pending_job(self) -> None:
        """Successful generate returns 200 with job_id, status, poll_url."""
        service = MagicMock()
        service.create_job = AsyncMock(return_value="job-abc")
        service.register_task = MagicMock()

        provider = _make_provider_mock()
        registry = _make_registry_mock({"fake": provider})

        app = _build_app(video_generation_service=service, provider_registry=registry)

        with patch("src.api.routes.generation.get_db") as mock_get_db:
            db = _mock_db()
            mock_get_db.return_value = db
            app.dependency_overrides[
                __import__("src.database.session", fromlist=["get_db"]).get_db
            ] = lambda: db

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/content/videos/generate",
                    json={"prompt": "a cat dancing"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert "job_id" in data
        assert "poll_url" in data

    async def test_generate_quota_exceeded_429(self) -> None:
        """When quota service rejects, return 429."""
        quota_service = AsyncMock()
        quota_result = MagicMock()
        quota_result.allowed = False
        quota_result.reason = "Daily limit reached"
        quota_service.check_quota = AsyncMock(return_value=quota_result)

        provider = _make_provider_mock()
        registry = _make_registry_mock({"fake": provider})
        service = MagicMock()

        app = _build_app(
            video_generation_service=service,
            provider_registry=registry,
            quota_service=quota_service,
        )

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/content/videos/generate",
                json={"prompt": "a cat dancing"},
            )

        assert resp.status_code == 429
        assert "Daily limit" in resp.json()["detail"]

    async def test_generate_unknown_provider_422(self) -> None:
        """Unknown provider in request body returns 422."""
        registry = _make_registry_mock()  # no providers
        app = _build_app(provider_registry=registry)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/content/videos/generate",
                json={"prompt": "test", "provider": "nonexistent"},
            )

        assert resp.status_code == 422
        assert "Unknown provider" in resp.json()["detail"]

    async def test_generate_empty_prompt_422(self) -> None:
        """Empty prompt is rejected by Pydantic schema validation."""
        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/content/videos/generate",
                json={"prompt": ""},
            )

        assert resp.status_code == 422

    async def test_generate_records_quota_start(self) -> None:
        """When quota service is configured, record_generation_start is called."""
        quota_service = AsyncMock()
        quota_result = MagicMock()
        quota_result.allowed = True
        quota_service.check_quota = AsyncMock(return_value=quota_result)
        quota_service.record_generation_start = AsyncMock()

        service = MagicMock()
        service.create_job = AsyncMock(return_value="job-1")
        service.register_task = MagicMock()

        provider = _make_provider_mock()
        registry = _make_registry_mock({"fake": provider})

        app = _build_app(
            video_generation_service=service,
            provider_registry=registry,
            quota_service=quota_service,
        )

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/content/videos/generate",
                json={"prompt": "test"},
            )

        assert resp.status_code == 200
        quota_service.record_generation_start.assert_awaited_once()


# ============================================================================
# POST /generate/batch
# ============================================================================


class TestGenerateBatch:
    """Tests for POST /api/content/videos/generate/batch."""

    async def test_batch_returns_batch_id_and_job_ids(self) -> None:
        service = MagicMock()
        service.create_job = AsyncMock(side_effect=["job-1", "job-2"])
        service.register_task = MagicMock()

        tracker = AsyncMock()
        tracker.create_batch = AsyncMock()

        provider = _make_provider_mock()
        registry = _make_registry_mock({"fake": provider})

        app = _build_app(
            video_generation_service=service,
            provider_registry=registry,
            batch_tracker=tracker,
        )

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/content/videos/generate/batch",
                json={
                    "items": [
                        {"prompt": "cat"},
                        {"prompt": "dog"},
                    ],
                    "provider": "fake",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert len(data["job_ids"]) == 2
        assert "batch_id" in data

    async def test_batch_validates_each_item(self) -> None:
        """If one item references an unknown provider, entire batch fails with 422."""
        registry = _make_registry_mock()  # no providers

        app = _build_app(provider_registry=registry)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/content/videos/generate/batch",
                json={
                    "items": [
                        {"prompt": "cat", "provider": "bad_provider"},
                    ],
                },
            )

        assert resp.status_code == 422


# ============================================================================
# POST /jobs/{job_id}/cancel
# ============================================================================


class TestCancelJob:
    """Tests for POST /api/content/videos/jobs/{job_id}/cancel."""

    async def test_cancel_success(self) -> None:
        service = MagicMock()
        service.cancel_job = AsyncMock(return_value=True)

        app = _build_app(video_generation_service=service)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/content/videos/jobs/job-1/cancel")

        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
        assert resp.json()["job_id"] == "job-1"
        # The route must resolve the caller's tenant and hand it down — the
        # service cannot scope the cancel if the route never passes it.
        service.cancel_job.assert_awaited_once_with(db, "job-1", HARNESS_TENANT)

    async def test_cancel_returns_409_when_not_cancellable(self) -> None:
        service = MagicMock()
        service.cancel_job = AsyncMock(return_value=False)

        app = _build_app(video_generation_service=service)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/content/videos/jobs/job-1/cancel")

        assert resp.status_code == 409
        assert "cannot be cancelled" in resp.json()["detail"]


# ============================================================================
# GET /providers
# ============================================================================


class TestListProviders:
    """Tests for GET /api/content/videos/providers."""

    @patch("src.config.get_settings")
    async def test_list_providers_no_cache(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value.generation.default_provider = "fake"

        from src.services.providers.base import ProviderInfo

        registry = MagicMock()
        registry.list_providers.return_value = [
            ProviderInfo(name="fake", available=True, max_duration_s=30.0),
        ]

        app = _build_app(provider_registry=registry)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/providers")

        assert resp.status_code == 200
        data = resp.json()
        assert data["default_provider"] == "fake"
        assert len(data["providers"]) == 1
        assert data["providers"][0]["name"] == "fake"

    @patch("src.config.get_settings")
    async def test_list_providers_uses_redis_cache(self, mock_settings: MagicMock) -> None:
        """When Redis returns cached data, registry.list_providers is NOT called."""
        mock_settings.return_value.generation.default_provider = "fake"

        redis = AsyncMock()
        cached_payload = {
            "providers": [{"name": "cached", "available": True}],
            "default_provider": "cached-default",
        }
        redis.get = AsyncMock(return_value=cached_payload)

        registry = MagicMock()
        registry.list_providers.return_value = []

        app = _build_app(provider_registry=registry, redis_client=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/providers")

        assert resp.status_code == 200
        data = resp.json()
        assert data["default_provider"] == "cached-default"
        # Registry should not have been called -- served from cache
        registry.list_providers.assert_not_called()

    @patch("src.config.get_settings")
    async def test_list_providers_writes_cache(self, mock_settings: MagicMock) -> None:
        """When Redis is available but has no cache, result is written to Redis."""
        mock_settings.return_value.generation.default_provider = "fake"

        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()

        registry = MagicMock()
        registry.list_providers.return_value = []

        app = _build_app(provider_registry=registry, redis_client=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/providers")

        assert resp.status_code == 200
        redis.set.assert_awaited_once()


# ============================================================================
# GET /providers/{name}/health
# ============================================================================


class TestProviderHealth:
    """Tests for GET /api/content/videos/providers/{name}/health."""

    async def test_provider_health_success(self) -> None:
        health_monitor = MagicMock()
        state = MagicMock()
        state.is_healthy = True
        state.consecutive_failures = 0
        state.average_latency_ms = 120.5
        state.disabled_at = None
        state.cooldown_until = None
        health_monitor.get_health.return_value = state

        app = _build_app(health_monitor=health_monitor)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/providers/fake/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "fake"
        assert data["is_healthy"] is True
        assert data["consecutive_failures"] == 0
        assert data["average_latency_ms"] == 120.5

    async def test_provider_health_no_monitor_503(self) -> None:
        """If health monitoring is not configured, return 503."""
        app = _build_app()  # no health_monitor on app.state

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/providers/fake/health")

        assert resp.status_code == 503
        assert "not enabled" in resp.json()["detail"]

    async def test_provider_health_unknown_provider_404(self) -> None:
        """Unknown provider returns 404 before reaching health monitor."""
        registry = MagicMock()
        registry.get.return_value = None
        registry.provider_names = ["runway", "kling"]

        health_monitor = MagicMock()
        app = _build_app(provider_registry=registry, health_monitor=health_monitor)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/providers/nonexistent/health")

        assert resp.status_code == 404
        assert "Unknown provider" in resp.json()["detail"]
        assert "nonexistent" in resp.json()["detail"]
        # Health monitor should never be consulted
        health_monitor.get_health.assert_not_called()


# ============================================================================
# GET /jobs
# ============================================================================


class TestListJobs:
    """Tests for GET /api/content/videos/jobs."""

    @patch("src.api.routes.jobs.JobManager")
    async def test_list_jobs_empty(self, mock_jm: MagicMock) -> None:
        mock_jm.list_jobs = AsyncMock(return_value=([], 0))

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs")

        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["total_pages"] == 1

    @patch("src.api.routes.jobs.JobManager")
    async def test_list_jobs_with_results(self, mock_jm: MagicMock) -> None:
        job = _FakeJob(job_id="j1", status="completed", provider="fake", cost=0.05)
        mock_jm.list_jobs = AsyncMock(return_value=([job], 1))

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["job_id"] == "j1"
        assert data["items"][0]["status"] == "completed"
        assert data["total"] == 1

    @patch("src.api.routes.jobs.JobManager")
    async def test_list_jobs_pagination(self, mock_jm: MagicMock) -> None:
        """total_pages is computed correctly from total and page_size."""
        mock_jm.list_jobs = AsyncMock(return_value=([], 55))

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs?page=2&page_size=20")

        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 2
        assert data["page_size"] == 20
        assert data["total_pages"] == 3  # ceil(55/20)


# ============================================================================
# GET /jobs/{job_id}
# ============================================================================


class TestGetJobStatus:
    """Tests for GET /api/content/videos/jobs/{job_id}."""

    @patch("src.api.routes.jobs.JobManager")
    async def test_get_status_completed(self, mock_jm: MagicMock) -> None:
        job = _FakeJob(job_id="j1", status="completed")
        mock_jm.get_job = AsyncMock(return_value=job)

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "j1"
        assert data["status"] == "completed"
        assert data["result_url"] is not None
        assert "j1" in data["result_url"]

    @patch("src.api.routes.jobs.JobManager")
    async def test_get_status_pending_no_result_url(self, mock_jm: MagicMock) -> None:
        """Pending jobs have no result_url."""
        job = _FakeJob(job_id="j1", status="pending")
        mock_jm.get_job = AsyncMock(return_value=job)

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1")

        assert resp.status_code == 200
        assert resp.json()["result_url"] is None

    @patch("src.api.routes.jobs.JobManager")
    async def test_get_status_not_found_404(self, mock_jm: MagicMock) -> None:
        mock_jm.get_job = AsyncMock(return_value=None)

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/nope")

        assert resp.status_code == 404

    @patch("src.api.routes.jobs.JobManager")
    async def test_get_status_uses_redis_cache(self, mock_jm: MagicMock) -> None:
        """When Redis cache hit, JobManager.get_job is NOT called."""
        redis = AsyncMock()
        redis.get = AsyncMock(
            return_value={
                # The harness forwards a real tenant; the cache value must be
                # tagged with the same value for a hit.
                "tenant_id": HARNESS_TENANT,
                "response": {
                    "job_id": "j1",
                    "status": "processing",
                    "provider": "fake",
                },
            }
        )

        app = _build_app(redis_client=redis)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1")

        assert resp.status_code == 200
        assert resp.json()["status"] == "processing"
        mock_jm.get_job.assert_not_called()


# ============================================================================
# GET /jobs/{job_id}/result
# ============================================================================


class TestGetJobResult:
    """Tests for GET /api/content/videos/jobs/{job_id}/result."""

    @patch("src.api.routes.jobs.JobManager")
    async def test_result_success(self, mock_jm: MagicMock) -> None:
        job = _FakeJob(job_id="j1", status="completed")
        mock_jm.get_job = AsyncMock(return_value=job)

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1/result")

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "j1"
        assert data["status"] == "completed"
        assert data["video_path"] is not None
        assert data["cost"] == 0.05

    @patch("src.api.routes.jobs.JobManager")
    async def test_result_not_completed_409(self, mock_jm: MagicMock) -> None:
        job = _FakeJob(job_id="j1", status="processing")
        mock_jm.get_job = AsyncMock(return_value=job)

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1/result")

        assert resp.status_code == 409
        assert "not completed" in resp.json()["detail"]

    @patch("src.api.routes.jobs.JobManager")
    async def test_result_not_found_404(self, mock_jm: MagicMock) -> None:
        mock_jm.get_job = AsyncMock(return_value=None)

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1/result")

        assert resp.status_code == 404


# ============================================================================
# GET /jobs/{job_id}/costs
# ============================================================================


class TestGetJobCosts:
    """Tests for GET /api/content/videos/jobs/{job_id}/costs."""

    @patch("src.api.routes.jobs.JobManager")
    async def test_costs_completed_job(self, mock_jm: MagicMock) -> None:
        job = _FakeJob(job_id="j1", status="completed", cost=0.10, duration_s=5.0)
        mock_jm.get_job = AsyncMock(return_value=job)

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1/costs")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_cost"] == 0.10
        assert data["clips_generated"] == 1
        assert data["clips_failed"] == 0
        assert len(data["per_clip_costs"]) == 1
        assert data["per_clip_costs"][0]["job_id"] == "j1"

    @patch("src.api.routes.jobs.JobManager")
    async def test_costs_failed_job(self, mock_jm: MagicMock) -> None:
        job = _FakeJob(job_id="j1", status="failed", cost=None)
        mock_jm.get_job = AsyncMock(return_value=job)

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1/costs")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_cost"] == 0.0
        assert data["clips_generated"] == 0
        assert data["clips_failed"] == 1
        assert data["per_clip_costs"] == []

    @patch("src.api.routes.jobs.JobManager")
    async def test_costs_not_found_404(self, mock_jm: MagicMock) -> None:
        mock_jm.get_job = AsyncMock(return_value=None)

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1/costs")

        assert resp.status_code == 404


# ============================================================================
# GET /jobs/{job_id}/video
# ============================================================================


class TestGetJobVideo:
    """Tests for GET /api/content/videos/jobs/{job_id}/video."""

    @patch("src.api.routes.jobs.JobManager")
    async def test_video_minio_redirect(self, mock_jm: MagicMock) -> None:
        """When minio_key is set, return 307 redirect to presigned URL."""
        job = _FakeJob(job_id="j1", minio_key="videos/j1.mp4")
        mock_jm.get_job = AsyncMock(return_value=job)

        storage = MagicMock()
        storage.presign_url = AsyncMock(return_value="https://minio.example.com/presigned")

        app = _build_app(storage_service=storage)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            follow_redirects=False,
        ) as ac:
            resp = await ac.get("/api/content/videos/jobs/j1/video")

        assert resp.status_code == 307
        assert resp.headers["location"] == "https://minio.example.com/presigned"

    @patch("src.api.routes.jobs.JobManager")
    async def test_video_local_file(self, mock_jm: MagicMock) -> None:
        """When no minio_key, serve local file content."""
        job = _FakeJob(job_id="j1", minio_key=None, video_path="/tmp/v.mp4")
        mock_jm.get_job = AsyncMock(return_value=job)

        storage = MagicMock()
        storage.presign_url.return_value = None
        storage.retrieve = AsyncMock(return_value=b"fake-video-bytes")

        app = _build_app(storage_service=storage)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1/video")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"
        assert b"fake-video-bytes" in resp.content

    @patch("src.api.routes.jobs.JobManager")
    async def test_video_no_path_404(self, mock_jm: MagicMock) -> None:
        """No minio_key and no video_path -> 404."""
        job = _FakeJob(job_id="j1", minio_key=None, video_path=None)
        mock_jm.get_job = AsyncMock(return_value=job)

        storage = MagicMock()
        storage.presign_url.return_value = None

        app = _build_app(storage_service=storage)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1/video")

        assert resp.status_code == 404
        assert "No video available" in resp.json()["detail"]

    @patch("src.api.routes.jobs.JobManager")
    async def test_video_file_missing_on_disk_404(self, mock_jm: MagicMock) -> None:
        """video_path set but storage returns None -> 404."""
        job = _FakeJob(job_id="j1", minio_key=None, video_path="/tmp/gone.mp4")
        mock_jm.get_job = AsyncMock(return_value=job)

        storage = MagicMock()
        storage.presign_url.return_value = None
        storage.retrieve = AsyncMock(return_value=None)

        app = _build_app(storage_service=storage)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1/video")

        assert resp.status_code == 404
        assert "not found on disk" in resp.json()["detail"]

    @patch("src.api.routes.jobs.JobManager")
    async def test_video_job_not_found_404(self, mock_jm: MagicMock) -> None:
        mock_jm.get_job = AsyncMock(return_value=None)

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs/j1/video")

        assert resp.status_code == 404


# ============================================================================
# POST /preview
# ============================================================================


class TestPreviewVideo:
    """Tests for POST /api/content/videos/preview."""

    @patch("src.config.get_settings")
    async def test_preview_success(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value.generation.default_provider = "fake"
        mock_settings.return_value.generation.default_width = 1920
        mock_settings.return_value.generation.default_height = 1080

        from src.services.providers.base import VideoGenerationResult

        provider = _make_provider_mock()
        provider.generate = AsyncMock(
            return_value=VideoGenerationResult(success=True, video_data=b"video")
        )
        registry = _make_registry_mock({"fake": provider})

        app = _build_app(provider_registry=registry)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/content/videos/preview",
                json={"prompt": "sunset over ocean"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["provider"] == "fake"
        assert data["video_base64"] is not None

    @patch("src.config.get_settings")
    async def test_preview_unknown_provider_422(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value.generation.default_provider = "fake"

        registry = _make_registry_mock()  # empty
        app = _build_app(provider_registry=registry)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/content/videos/preview",
                json={"prompt": "test", "provider": "nonexistent"},
            )

        assert resp.status_code == 422
        assert "Unknown provider" in resp.json()["detail"]

    @patch("src.config.get_settings")
    async def test_preview_generation_failed(self, mock_settings: MagicMock) -> None:
        """Provider returns success=False -> status 'failed' in response (not HTTP error)."""
        mock_settings.return_value.generation.default_provider = "fake"
        mock_settings.return_value.generation.default_width = 1920
        mock_settings.return_value.generation.default_height = 1080

        from src.services.providers.base import VideoGenerationResult

        provider = _make_provider_mock()
        provider.generate = AsyncMock(
            return_value=VideoGenerationResult(success=False, error="Provider error")
        )
        registry = _make_registry_mock({"fake": provider})

        app = _build_app(provider_registry=registry)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/content/videos/preview",
                json={"prompt": "test"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["error"] == "Provider error"

    @patch("src.config.get_settings")
    async def test_preview_timeout(self, mock_settings: MagicMock) -> None:
        """Timeout returns status 'timeout' in response body."""
        mock_settings.return_value.generation.default_provider = "fake"
        mock_settings.return_value.generation.default_width = 1920
        mock_settings.return_value.generation.default_height = 1080

        provider = _make_provider_mock()
        provider.generate = AsyncMock(side_effect=TimeoutError("timed out"))
        registry = _make_registry_mock({"fake": provider})

        app = _build_app(provider_registry=registry)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/content/videos/preview",
                json={"prompt": "test"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "timeout"

    @patch("src.config.get_settings")
    async def test_preview_exception(self, mock_settings: MagicMock) -> None:
        """Unexpected exception returns status 'failed' with error message."""
        mock_settings.return_value.generation.default_provider = "fake"
        mock_settings.return_value.generation.default_width = 1920
        mock_settings.return_value.generation.default_height = 1080

        provider = _make_provider_mock()
        provider.generate = AsyncMock(side_effect=RuntimeError("boom"))
        registry = _make_registry_mock({"fake": provider})

        app = _build_app(provider_registry=registry)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/content/videos/preview",
                json={"prompt": "test"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "boom" in data["error"]


# ============================================================================
# GET /usage
# ============================================================================


class TestGetUsage:
    """Tests for GET /api/content/videos/usage."""

    @patch("src.api.routes.usage.JobManager")
    async def test_usage_empty(self, mock_jm: MagicMock) -> None:
        mock_jm.get_usage = AsyncMock(return_value=([], []))

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/usage")

        assert resp.status_code == 200
        data = resp.json()
        assert data["periods"] == []
        assert data["by_provider"] == []
        assert data["total_cost"] == 0.0
        assert data["total_generations"] == 0

    @patch("src.api.routes.usage.JobManager")
    async def test_usage_with_data(self, mock_jm: MagicMock) -> None:
        usage_row = MagicMock()
        usage_row.date = "2024-01-15"
        usage_row.generations_started = 10
        usage_row.generations_completed = 8
        usage_row.generations_failed = 2
        usage_row.total_duration_generated_s = 40.0
        usage_row.total_file_size_bytes = 1024000

        provider_breakdown = [("runway", 5.50, 25)]

        mock_jm.get_usage = AsyncMock(return_value=([usage_row], provider_breakdown))

        app = _build_app()

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/usage")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["periods"]) == 1
        assert data["periods"][0]["date"] == "2024-01-15"
        assert len(data["by_provider"]) == 1
        assert data["by_provider"][0]["provider"] == "runway"
        assert data["total_cost"] == 5.50
        assert data["total_generations"] == 8


# ============================================================================
# GET /quotas/{tenant_id}
# ============================================================================


class TestGetQuotaStatus:
    """Tests for GET /api/content/videos/quotas/{tenant_id}.

    The harness forwards ``HARNESS_TENANT`` as the caller's real tenant -- that
    is the only tenant these requests are allowed to name in the path.
    """

    async def test_quota_status_success(self) -> None:
        quota_service = AsyncMock()
        quota_service.get_quota_status = AsyncMock(
            return_value={
                "tenant_id": HARNESS_TENANT,
                "daily_generations_used": 5,
                "daily_generations_limit": 100,
                "daily_spend_used": 1.50,
                "daily_spend_limit": 50.0,
                "concurrent_jobs": 2,
                "concurrent_jobs_limit": 10,
            }
        )

        app = _build_app(quota_service=quota_service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get(f"/api/content/videos/quotas/{HARNESS_TENANT}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == HARNESS_TENANT
        assert data["daily_generations_used"] == 5
        assert data["daily_spend_limit"] == 50.0
        quota_service.get_quota_status.assert_awaited_once_with(HARNESS_TENANT)

    async def test_quota_service_disabled_501(self) -> None:
        """When quota_service is None, return 501."""
        app = _build_app()  # no quota_service

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get(f"/api/content/videos/quotas/{HARNESS_TENANT}")

        assert resp.status_code == 501
        assert "not enabled" in resp.json()["detail"]

    async def test_foreign_tenant_403_before_501(self) -> None:
        """The ownership check runs FIRST: a foreign id is refused even when
        the quota service is switched off, so the route never doubles as a
        probe for whether quota is enabled on this deployment."""
        app = _build_app()  # no quota_service

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/quotas/some-other-tenant")

        assert resp.status_code == 403
        assert resp.json()["detail"] == "Quota is readable only for your own tenant"


# ============================================================================
# GET /batches/{batch_id}/status
# ============================================================================


class TestGetBatchStatus:
    """Tests for GET /api/content/videos/batches/{batch_id}/status."""

    @patch("src.api.routes.batches.JobManager")
    async def test_batch_status_success(self, mock_jm: MagicMock) -> None:
        from src.services.batch_tracker import BatchState

        batch = BatchState(
            batch_id="b1",
            job_ids=["j1", "j2"],
            clips_submitted=2,
            clips_completed=1,
            clips_failed=0,
            clips_skipped=0,
            total_cost=0.05,
            budget_limit=5.0,
        )

        tracker = AsyncMock()
        tracker.get_batch = AsyncMock(return_value=batch)

        job1 = _FakeJob(job_id="j1", status="completed", cost=0.05)
        job2 = _FakeJob(job_id="j2", status="processing")
        mock_jm.get_job = AsyncMock(side_effect=[job1, job2])

        app = _build_app(batch_tracker=tracker)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/batches/b1/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"] == "b1"
        assert data["clips_submitted"] == 2
        assert data["clips_completed"] == 1
        assert len(data["clips"]) == 2
        assert data["cost_report"] is not None
        assert data["cost_report"]["total_cost"] == 0.05

    async def test_batch_status_not_found_404(self) -> None:
        tracker = AsyncMock()
        tracker.get_batch = AsyncMock(return_value=None)

        app = _build_app(batch_tracker=tracker)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/batches/nope/status")

        assert resp.status_code == 404


# ============================================================================
# GET /batches/{batch_id}/result
# ============================================================================


class TestGetBatchResult:
    """Tests for GET /api/content/videos/batches/{batch_id}/result."""

    async def test_batch_result_completed(self) -> None:
        from src.services.batch_tracker import BatchState

        batch = BatchState(
            batch_id="b1",
            job_ids=["j1"],
            clips_submitted=1,
            clips_completed=1,
            total_cost=0.10,
            budget_limit=5.0,
        )

        tracker = AsyncMock()
        tracker.get_batch = AsyncMock(return_value=batch)

        app = _build_app(batch_tracker=tracker)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/batches/b1/result")

        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"] == "b1"
        assert data["status"] == "completed"
        assert data["total_cost"] == 0.10

    async def test_batch_result_still_processing_409(self) -> None:
        from src.services.batch_tracker import BatchState

        batch = BatchState(
            batch_id="b1",
            job_ids=["j1", "j2"],
            clips_submitted=2,
            clips_completed=1,
        )

        tracker = AsyncMock()
        tracker.get_batch = AsyncMock(return_value=batch)

        app = _build_app(batch_tracker=tracker)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/batches/b1/result")

        assert resp.status_code == 409
        assert "still processing" in resp.json()["detail"]

    async def test_batch_result_not_found_404(self) -> None:
        tracker = AsyncMock()
        tracker.get_batch = AsyncMock(return_value=None)

        app = _build_app(batch_tracker=tracker)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/batches/nope/result")

        assert resp.status_code == 404

    async def test_batch_result_pending_409(self) -> None:
        """Batch with no completed clips (pending) returns 409."""
        from src.services.batch_tracker import BatchState

        batch = BatchState(
            batch_id="b1",
            job_ids=["j1"],
            clips_submitted=1,
        )

        tracker = AsyncMock()
        tracker.get_batch = AsyncMock(return_value=batch)

        app = _build_app(batch_tracker=tracker)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/batches/b1/result")

        assert resp.status_code == 409


# ============================================================================
# _validate_generate_request  (supplementary tests beyond test_request_validation.py)
# ============================================================================


class TestValidateGenerateRequest:
    """Additional validation edge cases tested at the function level."""

    # The resolution now happens inside w2v_common.net.safe_fetch (the shared
    # guard), not in this module — so that is where getaddrinfo must be patched.
    @patch(
        "w2v_common.net.safe_fetch.socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )
    def test_valid_callback_url_passes(self, _mock_dns: MagicMock) -> None:
        provider = _make_provider_mock()
        registry = _make_registry_mock({"fake": provider})
        request = MagicMock()
        request.app.state.provider_registry = registry

        body = GenerateRequest(
            prompt="test",
            provider="fake",
            callback_url="https://hooks.example.com/done",
        )
        from src.api.routes import _validate_generate_request

        _validate_generate_request(request, body)  # should not raise

    def test_duration_within_max_passes(self) -> None:
        provider = _make_provider_mock(max_duration=30.0)
        registry = _make_registry_mock({"fake": provider})
        request = MagicMock()
        request.app.state.provider_registry = registry

        body = GenerateRequest(prompt="test", provider="fake", duration_s=25.0)
        from src.api.routes import _validate_generate_request

        _validate_generate_request(request, body)  # should not raise

    def test_i2v_supported_and_valid_url_passes(self) -> None:
        provider = _make_provider_mock(supports_i2v=True)
        registry = _make_registry_mock({"fake": provider})
        request = MagicMock()
        request.app.state.provider_registry = registry

        body = GenerateRequest(
            prompt="test",
            provider="fake",
            image_url="https://example.com/photo.jpg",
        )
        from src.api.routes import _validate_generate_request

        _validate_generate_request(request, body)  # should not raise

    def test_no_provider_no_duration_passes(self) -> None:
        """When provider is None, skip all provider-specific validation."""
        registry = _make_registry_mock()
        request = MagicMock()
        request.app.state.provider_registry = registry

        body = GenerateRequest(prompt="test")
        from src.api.routes import _validate_generate_request

        _validate_generate_request(request, body)  # should not raise

    def test_valid_fallback_providers_pass(self) -> None:
        provider1 = _make_provider_mock(name="fake")
        provider2 = _make_provider_mock(name="other")
        registry = _make_registry_mock({"fake": provider1, "other": provider2})
        request = MagicMock()
        request.app.state.provider_registry = registry

        body = GenerateRequest(prompt="test", fallback_providers=["fake", "other"])
        from src.api.routes import _validate_generate_request

        _validate_generate_request(request, body)  # should not raise

    def test_provider_no_max_duration_skips_check(self) -> None:
        """When provider max_duration is None, any duration_s is accepted."""
        provider = _make_provider_mock(max_duration=None)
        registry = _make_registry_mock({"fake": provider})
        request = MagicMock()
        request.app.state.provider_registry = registry

        body = GenerateRequest(prompt="test", provider="fake", duration_s=200.0)
        from src.api.routes import _validate_generate_request

        _validate_generate_request(request, body)  # should not raise


class TestProcessJobBgCallbackNonFatal:
    """A successfully-generated video must stay completed even if the
    customer's webhook is unreachable (httpx.ConnectError). BUG 1 regression."""

    async def test_unreachable_callback_does_not_mark_failed(self) -> None:
        import httpx

        from src.api.routes.generation import _process_job_bg

        service = MagicMock()
        service.process_job = AsyncMock(return_value=0.42)
        service.unregister_task = MagicMock()

        completed_job = MagicMock()
        completed_job.status = "completed"

        db_mock = AsyncMock()

        with (
            patch("src.api.routes.generation.get_db_session") as mock_ctx,
            patch(
                "src.api.routes.generation.JobManager.get_job",
                new_callable=AsyncMock,
                return_value=completed_job,
            ),
            patch(
                "src.api.routes.generation.JobManager.mark_failed",
                new_callable=AsyncMock,
            ) as mock_mark_failed,
            patch(
                "src.services.callback_service.deliver_callback",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("connection refused"),
            ) as mock_deliver,
        ):
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db_mock)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

            # Must NOT raise despite the callback ConnectError.
            await _process_job_bg(
                service,
                "vid-job-1",
                "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
                callback_url="https://example.com/webhook",
            )

        mock_deliver.assert_awaited_once()
        # The video was generated + stored: the job must NOT be marked failed
        # just because the callback webhook was down.
        mock_mark_failed.assert_not_awaited()
        service.unregister_task.assert_called_once_with("vid-job-1")


# ============================================================================
# Tenant contract: an absent tenant is an ERROR, never a substituted default
# ============================================================================


class TestTenantIsRequired:
    """Pins the product-owner ruling: a ``tenant_id`` may NEVER hold a sentinel.

    This service used to answer ``DEFAULT_TENANT_ID`` (the nil UUID) for a
    request that forwarded no tenant. That is the most dangerous spelling of the
    lie -- a valid uuid that passes every cast, which billing then buckets as
    unattributed and NEVER CHARGES. Both halves below matter: a 4xx alone would
    still pass if the route emitted a sentinel-tenant cost event on the way out.
    """

    @staticmethod
    def _no_tenant_app(**kwargs: Any) -> FastAPI:
        """The harness with NO tenant middleware — i.e. nothing forwarded one."""
        return _build_app(tenant_id=None, **kwargs)

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("post", "/api/content/videos/generate"),
            ("get", "/api/content/videos/jobs"),
            ("post", "/api/content/videos/jobs/job-1/cancel"),
            ("get", f"/api/content/videos/quotas/{HARNESS_TENANT}"),
        ],
    )
    async def test_request_without_tenant_is_refused(self, method: str, path: str) -> None:
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(return_value=b"id")
        service = MagicMock()
        service.cancel_job = AsyncMock(return_value=True)
        service.generate_video = AsyncMock()
        app = self._no_tenant_app(
            video_generation_service=service,
            quota_service=AsyncMock(),
            redis_client=redis_mock,
        )

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            if method == "post":
                resp = await ac.post(path, json={"prompt": "a cat dancing"})
            else:
                resp = await ac.get(path)

        assert resp.status_code == 400, resp.text
        assert "tenant" in resp.json()["detail"].lower()
        # No sentinel-tenant cost event may be written on the way out.
        redis_mock.xadd.assert_not_awaited()

    @pytest.mark.parametrize(
        "sentinel_tenant",
        ["", "   ", "00000000-0000-0000-0000-000000000000"],
    )
    async def test_sentinel_tenant_on_request_state_is_refused(self, sentinel_tenant: str) -> None:
        """A blank OR nil-UUID tenant on ``request.state`` is not a tenant.

        w2v_common 0.66.1's AuthMiddleware can still put the nil UUID there on
        the legacy trusted-network opt-in path, so rejecting only ``None`` would
        leave the exact unbilled-usage hole this change exists to close.
        """
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(return_value=b"id")
        app = self._no_tenant_app(redis_client=redis_mock)

        @app.middleware("http")
        async def _set_sentinel(request: Any, call_next: Any) -> Any:
            request.state.tenant_id = sentinel_tenant
            return await call_next(request)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/content/videos/jobs")

        assert resp.status_code == 400, resp.text
        redis_mock.xadd.assert_not_awaited()
