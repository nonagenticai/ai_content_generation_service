"""Extended unit tests for concept-image content_routes to improve coverage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.content_routes import (
    _batch_result_path,
    _batch_status_path,
    _deliver_concept_image_callback,
    _job_result_path,
    _job_status_path,
    _preferred_asset_url,
    _process_concept_image_batch_job_bg,
    _process_concept_image_job_bg,
    _serialize_concept_image_asset,
    _serialize_concept_image_job,
    router,
)

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
    batch_id: str | None = None
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
    provider_job_id: str | None = None
    callback_url: str | None = None
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


class TestHelperFunctions:
    def test_job_status_path(self) -> None:
        assert _job_status_path("j1") == "/api/concept-images/jobs/j1"

    def test_job_result_path(self) -> None:
        assert _job_result_path("j1") == "/api/concept-images/jobs/j1/result"

    def test_batch_status_path(self) -> None:
        assert _batch_status_path("b1") == "/api/concept-images/batches/b1"

    def test_batch_result_path(self) -> None:
        assert _batch_result_path("b1") == "/api/concept-images/batches/b1/result"

    def test_preferred_asset_url_download_preferred(self) -> None:
        # download_url (stream-through via this service) is canonical now;
        # browsers can't reach the MinIO presigned URL so we never return it.
        url = _preferred_asset_url(
            download_url="/download",
            image_url="https://cdn/img",
        )
        assert url == "/download"

    def test_preferred_asset_url_image_url_fallback(self) -> None:
        url = _preferred_asset_url(
            download_url=None,
            image_url="https://cdn/img",
        )
        assert url == "https://cdn/img"

    def test_preferred_asset_url_all_none(self) -> None:
        url = _preferred_asset_url(
            download_url=None,
            image_url=None,
        )
        assert url is None


class TestSerializeConceptImageJob:
    def test_completed_job_with_minio_key(self) -> None:
        # presigned_url is always None now — browsers can't reach minio.test.
        storage = MagicMock()
        job = _FakeImageJob(minio_key="content_generation/img.png")

        result = _serialize_concept_image_job(storage, job)
        assert result.presigned_url is None
        assert result.download_url == "/api/concept-images/assets/asset-1/download"

    def test_pending_job(self) -> None:
        storage = MagicMock()
        storage.presign_url.return_value = None
        job = _FakeImageJob(status="pending", asset_id="asset-1")

        result = _serialize_concept_image_job(storage, job)
        assert result.download_url is None

    def test_completed_job_no_minio(self) -> None:
        storage = MagicMock()
        storage.presign_url.return_value = None
        job = _FakeImageJob(minio_key=None)

        result = _serialize_concept_image_job(storage, job)
        assert result.presigned_url is None
        assert result.download_url == "/api/concept-images/assets/asset-1/download"


class TestSerializeConceptImageAsset:
    def test_with_provider_job_id(self) -> None:
        storage = MagicMock()
        storage.presign_url.return_value = None
        job = _FakeImageJob(provider_job_id="pj-1")

        result = _serialize_concept_image_asset(storage, job)
        assert result.metadata == {"provider_job_id": "pj-1"}

    def test_without_provider_job_id(self) -> None:
        storage = MagicMock()
        storage.presign_url.return_value = None
        job = _FakeImageJob(provider_job_id=None)

        result = _serialize_concept_image_asset(storage, job)
        assert result.metadata is None

    def test_includes_all_fields(self) -> None:
        storage = MagicMock()
        storage.presign_url.return_value = None
        job = _FakeImageJob()

        result = _serialize_concept_image_asset(storage, job)
        assert result.job_id == "img-job-1"
        assert result.asset_id == "asset-1"
        assert result.prompt == "a hero image"
        assert result.mime_type == "image/png"
        assert result.cost == 0.04


class TestGetConceptImageJobResult:
    async def test_job_result_completed(self) -> None:
        storage = MagicMock()
        storage.presign_url.return_value = None
        app = _build_app(storage_service=storage)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=_FakeImageJob(),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/jobs/img-job-1/result")

        assert resp.status_code == 200
        assert resp.json()["asset_id"] == "asset-1"

    async def test_job_result_not_found(self) -> None:
        app = _build_app()
        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/jobs/missing/result")

        assert resp.status_code == 404

    async def test_job_result_not_completed(self) -> None:
        app = _build_app()
        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=_FakeImageJob(status="processing"),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/jobs/j1/result")

        assert resp.status_code == 409


class TestGetConceptImageBatchResult:
    async def test_batch_result_completed(self) -> None:
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
                resp = await ac.get("/api/concept-images/batches/batch-1/result")

        assert resp.status_code == 200

    async def test_batch_result_not_finished(self) -> None:
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
                return_value=_FakeImageBatch(status="processing"),
            ),
            patch(
                "src.api.content_routes.ImageBatchManager.get_batch_jobs",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/batches/batch-1/result")

        assert resp.status_code == 409

    async def test_batch_not_found(self) -> None:
        app = _build_app()
        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageBatchManager.get_batch",
            new_callable=AsyncMock,
            return_value=None,
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/batches/missing")

        assert resp.status_code == 404


class TestGetConceptImageAsset:
    async def test_asset_not_found(self) -> None:
        app = _build_app()
        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job_by_asset_id",
            new_callable=AsyncMock,
            return_value=None,
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/assets/missing")

        assert resp.status_code == 404


class TestDownloadConceptImageAsset:
    async def test_download_not_found(self) -> None:
        app = _build_app()
        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job_by_asset_id",
            new_callable=AsyncMock,
            return_value=None,
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/assets/missing/download")

        assert resp.status_code == 404

    async def test_download_not_ready(self) -> None:
        app = _build_app()
        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job_by_asset_id",
            new_callable=AsyncMock,
            return_value=_FakeImageJob(status="processing"),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/assets/asset-1/download")

        assert resp.status_code == 409

    async def test_download_streams_minio_bytes(self) -> None:
        """When the asset is stored in MinIO, stream the bytes through this
        service (no 307 to minio.test which browsers can't reach)."""
        storage = MagicMock()
        storage.retrieve = AsyncMock(return_value=b"png-bytes")
        app = _build_app(storage_service=storage)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job_by_asset_id",
            new_callable=AsyncMock,
            return_value=_FakeImageJob(minio_key="content_generation/img.png"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
                follow_redirects=False,
            ) as ac:
                resp = await ac.get("/api/concept-images/assets/asset-1/download")

        assert resp.status_code == 200
        assert resp.content == b"png-bytes"
        storage.retrieve.assert_awaited_once_with("content_generation/img.png")

    async def test_download_no_image_path(self) -> None:
        storage = MagicMock()
        storage.presign_url.return_value = None
        app = _build_app(storage_service=storage)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job_by_asset_id",
            new_callable=AsyncMock,
            return_value=_FakeImageJob(minio_key=None, image_path=None),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/assets/asset-1/download")

        assert resp.status_code == 404

    async def test_download_file_not_on_disk(self) -> None:
        storage = MagicMock()
        storage.presign_url.return_value = None
        storage.retrieve = AsyncMock(return_value=None)
        app = _build_app(storage_service=storage)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job_by_asset_id",
            new_callable=AsyncMock,
            return_value=_FakeImageJob(minio_key=None),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/assets/asset-1/download")

        assert resp.status_code == 404


class TestConceptImageJobStatus:
    async def test_job_status_not_found(self) -> None:
        app = _build_app()
        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/jobs/missing")

        assert resp.status_code == 404

    async def test_job_status_with_redis_cache_hit(self) -> None:
        redis = AsyncMock()
        redis.get = AsyncMock(
            return_value={
                # No tenant resolves on the request -> nil-UUID sentinel; the
                # cache value must be tagged with the same value for a hit.
                "tenant_id": HARNESS_TENANT,
                "response": {
                    "job_id": "j1",
                    "status": "completed",
                    "asset_id": "a1",
                    "provider": "openai_images",
                },
            }
        )

        storage = MagicMock()
        storage.presign_url.return_value = None
        app = _build_app(storage_service=storage, redis_client=redis)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/concept-images/jobs/j1")

        assert resp.status_code == 200
        assert resp.json()["job_id"] == "j1"

    async def test_job_status_redis_cache_exception(self) -> None:
        redis = AsyncMock()
        redis.get = AsyncMock(side_effect=OSError("redis down"))
        redis.set = AsyncMock(side_effect=OSError("redis down"))

        storage = MagicMock()
        storage.presign_url.return_value = None
        app = _build_app(storage_service=storage, redis_client=redis)

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        with patch(
            "src.api.content_routes.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=_FakeImageJob(),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.get("/api/concept-images/jobs/j1")

        assert resp.status_code == 200


class TestDeliverConceptImageCallback:
    async def test_no_callback_url(self) -> None:
        db = AsyncMock()
        await _deliver_concept_image_callback(
            db=db,
            job_id="j1",
            callback_url=None,
            storage=MagicMock(),
        )
        # Should simply return without doing anything

    async def test_callback_delivery_success(self) -> None:
        db = AsyncMock()
        storage = MagicMock()
        storage.presign_url.return_value = None
        job = _FakeImageJob()

        with (
            patch(
                "src.api.content_routes.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=job,
            ),
            patch(
                "src.services.callback_service.deliver_callback",
                new_callable=AsyncMock,
            ) as mock_deliver,
        ):
            await _deliver_concept_image_callback(
                db=db,
                job_id="img-job-1",
                callback_url="https://example.com/cb",
                storage=storage,
            )
            mock_deliver.assert_called_once()

    async def test_callback_delivery_job_not_found(self) -> None:
        db = AsyncMock()
        with patch(
            "src.api.content_routes.ImageJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await _deliver_concept_image_callback(
                db=db,
                job_id="missing",
                callback_url="https://example.com/cb",
                storage=MagicMock(),
            )

    async def test_callback_delivery_exception_handled(self) -> None:
        db = AsyncMock()
        with patch(
            "src.api.content_routes.ImageJobManager.get_job",
            new_callable=AsyncMock,
            side_effect=OSError("DB error"),
        ):
            # Should not raise
            await _deliver_concept_image_callback(
                db=db,
                job_id="j1",
                callback_url="https://example.com/cb",
                storage=MagicMock(),
            )


class TestProcessConceptImageJobBg:
    async def test_successful_processing(self) -> None:
        service = MagicMock()
        service.process_job = AsyncMock(return_value=0.04)
        service._storage = MagicMock()
        service._storage.presign_url = MagicMock(return_value=None)
        service.unregister_task = MagicMock()

        fake_job = _FakeImageJob(callback_url=None)
        db_mock = AsyncMock()
        db_mock.commit = AsyncMock()

        with (
            patch(
                "src.api.content_routes.get_db_session",
            ) as mock_ctx,
            patch(
                "src.api.content_routes.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=fake_job,
            ),
            patch(
                "src.api.content_routes._deliver_concept_image_callback",
                new_callable=AsyncMock,
            ),
        ):
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db_mock)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await _process_concept_image_job_bg(
                service,
                "img-job-1",
                "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
                budget_limit=5.0,
                quota_service=None,
            )

        service.unregister_task.assert_called_once_with("img-job-1")

    async def test_processing_with_quota_service(self) -> None:
        service = MagicMock()
        service.process_job = AsyncMock(return_value=0.04)
        service._storage = MagicMock()
        service._storage.presign_url = MagicMock(return_value=None)
        service.unregister_task = MagicMock()

        fake_job = _FakeImageJob(callback_url=None)
        db_mock = AsyncMock()
        db_mock.commit = AsyncMock()
        quota_service = AsyncMock()
        quota_service.record_generation_complete = AsyncMock()

        with (
            patch(
                "src.api.content_routes.get_db_session",
            ) as mock_ctx,
            patch(
                "src.api.content_routes.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=fake_job,
            ),
            patch(
                "src.api.content_routes._deliver_concept_image_callback",
                new_callable=AsyncMock,
            ),
        ):
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db_mock)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await _process_concept_image_job_bg(
                service,
                "img-job-1",
                "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
                budget_limit=5.0,
                quota_service=quota_service,
            )

        quota_service.record_generation_complete.assert_called_once_with(
            "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa", "img-job-1", cost=0.04
        )

    async def test_processing_exception_marks_failed(self) -> None:
        service = MagicMock()
        service.process_job = AsyncMock(side_effect=RuntimeError("boom"))
        service._storage = MagicMock()
        service.unregister_task = MagicMock()

        fake_job = _FakeImageJob()
        db_mock = AsyncMock()
        db_mock.commit = AsyncMock()

        with (
            patch("src.api.content_routes.get_db_session") as mock_ctx,
            patch(
                "src.api.content_routes.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=fake_job,
            ),
            patch(
                "src.api.content_routes.ImageJobManager.mark_failed",
                new_callable=AsyncMock,
            ),
        ):
            # First call (for the main processing) raises, second call (for marking failed) succeeds
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db_mock)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await _process_concept_image_job_bg(
                service,
                "img-job-1",
                "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
                budget_limit=5.0,
                quota_service=None,
            )

        service.unregister_task.assert_called_once_with("img-job-1")


class TestProcessConceptImageBatchJobBg:
    """The per-child batch processor (one leased job, reclaimable cross-pod)."""

    async def test_successful_batch_child_processing(self) -> None:
        service = MagicMock()
        service.process_job = AsyncMock(return_value=0.04)
        service.estimate_cost = MagicMock(return_value=0.04)
        service._storage = MagicMock()
        service._storage.presign_url = MagicMock(return_value=None)
        service.unregister_task = MagicMock()

        fake_job = _FakeImageJob(callback_url=None)
        fake_batch = _FakeImageBatch(total_cost=0.0)
        db_mock = AsyncMock()
        db_mock.commit = AsyncMock()

        with (
            patch("src.api.content_routes.get_db_session") as mock_ctx,
            patch(
                "src.api.content_routes.ImageBatchManager.get_batch",
                new_callable=AsyncMock,
                return_value=fake_batch,
            ),
            patch(
                "src.api.content_routes.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=fake_job,
            ),
            patch(
                "src.api.content_routes.ImageBatchManager.record_job_completed",
                new_callable=AsyncMock,
            ) as mock_completed,
            patch(
                "src.api.content_routes._deliver_concept_image_callback",
                new_callable=AsyncMock,
            ),
        ):
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db_mock)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await _process_concept_image_batch_job_bg(
                service,
                "img-job-1",
                "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
                "batch-1",
                budget_limit=5.0,
                quota_service=None,
            )

        service.unregister_task.assert_called_once_with("img-job-1")
        mock_completed.assert_called_once()

    async def test_batch_child_fails(self) -> None:
        service = MagicMock()
        service.process_job = AsyncMock(return_value=0.0)
        service._storage = MagicMock()
        service._storage.presign_url = MagicMock(return_value=None)
        service.unregister_task = MagicMock()

        fake_job = _FakeImageJob(status="failed", callback_url=None)
        fake_batch = _FakeImageBatch(total_cost=0.0)
        db_mock = AsyncMock()
        db_mock.commit = AsyncMock()

        with (
            patch("src.api.content_routes.get_db_session") as mock_ctx,
            patch(
                "src.api.content_routes.ImageBatchManager.get_batch",
                new_callable=AsyncMock,
                return_value=fake_batch,
            ),
            patch(
                "src.api.content_routes.ImageJobManager.get_job",
                new_callable=AsyncMock,
                return_value=fake_job,
            ),
            patch(
                "src.api.content_routes.ImageBatchManager.record_job_failed",
                new_callable=AsyncMock,
            ) as mock_failed,
            patch(
                "src.api.content_routes._deliver_concept_image_callback",
                new_callable=AsyncMock,
            ),
        ):
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db_mock)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await _process_concept_image_batch_job_bg(
                service,
                "img-job-1",
                "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
                "batch-1",
                budget_limit=5.0,
                quota_service=None,
            )

        mock_failed.assert_called_once()
        service.unregister_task.assert_called_once_with("img-job-1")


class TestCreateConceptImageWithQuota:
    async def test_quota_rejected(self) -> None:
        service = MagicMock()
        quota_result = MagicMock()
        quota_result.allowed = False
        quota_result.reason = "Daily limit reached"
        quota_service = AsyncMock()
        quota_service.check_quota = AsyncMock(return_value=quota_result)

        registry = _make_image_registry({"openai_images": MagicMock()})
        app = _build_app(
            imagegen_service=service,
            image_provider_registry=registry,
            quota_service=quota_service,
        )

        db = _mock_db()
        from src.database.session import get_db

        app.dependency_overrides[get_db] = lambda: db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/concept-images",
                json={"prompt": "test", "provider": "openai_images"},
            )

        assert resp.status_code == 429
        assert "Daily limit reached" in resp.json()["detail"]
