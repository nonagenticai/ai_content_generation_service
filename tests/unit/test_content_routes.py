"""Unit tests for concept-image routes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.content_routes import router

# A real registered tenant, forwarded the way a live request forwards one. There
# is no default/sentinel tenant any more: a request that carries none is a 400
# (see ``src.api.routes.dependencies._get_tenant_id``), so the harness must
# supply one exactly like ``AuthMiddleware`` does in production.
HARNESS_TENANT = "0195f48b-1234-7000-8000-0000000000aa"


def _build_app(
    *,
    tenant_id: str | None = HARNESS_TENANT,
    imagegen_service=None,
    image_provider_registry=None,
    storage_service=None,
    redis_client=None,
    quota_service=None,
) -> FastAPI:
    app = FastAPI()

    if tenant_id is not None:

        @app.middleware("http")
        async def _set_tenant(request, call_next):  # pragma: no cover - harness
            request.state.tenant_id = tenant_id
            return await call_next(request)

    app.include_router(router)
    app.state.imagegen_service = imagegen_service or MagicMock()
    app.state.image_provider_registry = image_provider_registry or MagicMock()
    app.state.storage_service = storage_service or MagicMock()
    if redis_client is not None:
        app.state.redis_client = redis_client
    if quota_service is not None:
        app.state.quota_service = quota_service
    return app


def _mock_db() -> AsyncMock:
    db = AsyncMock()
    db.commit = AsyncMock()
    return db


@dataclass
class _FakeImageJob:
    job_id: str = "img-job-1"
    asset_id: str = "asset-1"
    batch_id: str | None = "batch-1"
    status: str = "completed"
    provider: str | None = "openai_images"
    prompt: str | None = "a hero image"
    image_path: str | None = "/tmp/image.png"
    image_url: str | None = "https://cdn.example.com/image.png"
    minio_key: str | None = None
    mime_type: str | None = "image/png"
    width: int | None = 1024
    height: int | None = 768
    file_size_bytes: int | None = 2048
    cost: float | None = 0.04
    usage_context: dict | None = None
    target_usage: str | None = "redesign_concept"
    error_message: str | None = None
    provider_job_id: str | None = "provider-job-1"
    created_at: datetime | None = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    completed_at: datetime | None = datetime(2024, 1, 15, 12, 0, 3, tzinfo=UTC)


@dataclass
class _FakeImageBatch:
    batch_id: str = "batch-1"
    status: str = "completed"
    budget_limit: float = 5.0
    total_cost: float = 0.04
    job_count: int = 1
    completed_jobs: int = 1
    failed_jobs: int = 0
    created_at: datetime | None = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    completed_at: datetime | None = datetime(2024, 1, 15, 12, 0, 3, tzinfo=UTC)


def _make_image_registry(providers: dict[str, object] | None = None) -> MagicMock:
    registry = MagicMock()
    providers = providers or {}

    def _get(name: str):
        return providers.get(name)

    registry.get.side_effect = _get
    registry.provider_names = list(providers.keys())
    return registry


class TestCreateConceptImageRoute:
    async def test_create_concept_image_returns_pending_job(self) -> None:
        service = MagicMock()
        service.create_job = AsyncMock(return_value="img-job-1")
        service.register_task = MagicMock()
        registry = _make_image_registry({"openai_images": MagicMock()})
        app = _build_app(imagegen_service=service, image_provider_registry=registry)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        def _fake_create_task(coro):
            coro.close()
            return MagicMock()

        with patch("src.api.content_routes.asyncio.create_task", side_effect=_fake_create_task):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/concept-images",
                    json={"prompt": "homepage redesign hero", "provider": "openai_images"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "img-job-1"
        assert data["poll_url"] == "/api/concept-images/jobs/img-job-1"
        assert data["result_url"] == "/api/concept-images/jobs/img-job-1/result"

    async def test_create_concept_image_batch_returns_pending_batch(self) -> None:
        service = MagicMock()
        service.create_job = AsyncMock(side_effect=["img-job-1", "img-job-2"])
        service.register_task = MagicMock()
        registry = _make_image_registry({"openai_images": MagicMock()})
        app = _build_app(imagegen_service=service, image_provider_registry=registry)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        fake_batch = _FakeImageBatch(batch_id="batch-1", job_count=2)

        def _fake_create_task(coro):
            coro.close()
            return MagicMock()

        with (
            patch(
                "src.api.content_routes.ImageBatchManager.create_batch",
                new_callable=AsyncMock,
                return_value=fake_batch,
            ),
            patch(
                "src.api.content_routes.ImageBatchManager.mark_processing",
                new_callable=AsyncMock,
            ),
            patch(
                "src.api.content_routes.ImageJobManager.register_dispatch",
                new_callable=AsyncMock,
            ),
            patch("src.api.content_routes.asyncio.create_task", side_effect=_fake_create_task),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/concept-images/batches",
                    json={
                        "items": [
                            {"prompt": "hero 1", "provider": "openai_images"},
                            {"prompt": "hero 2", "provider": "openai_images"},
                        ]
                    },
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"] == "batch-1"
        assert data["job_ids"] == ["img-job-1", "img-job-2"]
        assert data["status_url"] == "/api/concept-images/batches/batch-1"
        assert data["result_url"] == "/api/concept-images/batches/batch-1/result"

    async def test_create_concept_image_rejects_unknown_provider(self) -> None:
        app = _build_app(image_provider_registry=_make_image_registry())
        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/concept-images",
                json={"prompt": "homepage redesign hero", "provider": "missing"},
            )

        assert resp.status_code == 422
        assert "Unknown image provider" in resp.json()["detail"]


class TestConceptImageStatusRoutes:
    async def test_get_concept_image_job_status(self) -> None:
        storage = MagicMock()
        storage.presign_url.return_value = None
        app = _build_app(
            image_provider_registry=_make_image_registry({"openai_images": MagicMock()}),
            storage_service=storage,
        )

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=_FakeImageJob(),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/jobs/img-job-1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["asset_id"] == "asset-1"
        assert data["download_url"] == "/api/concept-images/assets/asset-1/download"
        assert data["url"] == "/api/concept-images/assets/asset-1/download"

    async def test_get_concept_image_asset(self) -> None:
        storage = MagicMock()
        storage.presign_url.return_value = None
        app = _build_app(storage_service=storage)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job_by_asset_id",
            new_callable=AsyncMock,
            return_value=_FakeImageJob(),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/assets/asset-1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["asset_id"] == "asset-1"
        assert data["download_url"] == "/api/concept-images/assets/asset-1/download"
        assert data["metadata"]["provider_job_id"] == "provider-job-1"

    async def test_get_concept_image_batch_status(self) -> None:
        storage = MagicMock()
        storage.presign_url.return_value = None
        app = _build_app(storage_service=storage)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with (
            patch(
                "src.api.content_routes.ImageBatchManager.get_batch",
                new_callable=AsyncMock,
                return_value=_FakeImageBatch(),
            ),
            patch(
                "src.api.content_routes.ImageBatchManager.get_batch_jobs",
                new_callable=AsyncMock,
                return_value=[_FakeImageJob()],
            ),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/batches/batch-1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"] == "batch-1"
        assert data["completed_jobs"] == 1
        assert data["results"][0]["asset_id"] == "asset-1"

    async def test_download_concept_image_asset_streams_local_file(self) -> None:
        storage = MagicMock()
        storage.retrieve = AsyncMock(return_value=b"png-data")
        app = _build_app(storage_service=storage)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job_by_asset_id",
            new_callable=AsyncMock,
            return_value=_FakeImageJob(),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/assets/asset-1/download")

        assert resp.status_code == 200
        assert resp.content == b"png-data"
