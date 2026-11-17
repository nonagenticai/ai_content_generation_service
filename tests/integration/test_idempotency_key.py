"""``Idempotency-Key`` on the job-creating POSTs, against the real Postgres.

The orchestrator re-submits a generation after its client-side wait times out
and sends ``Idempotency-Key`` expecting the ORIGINAL job back. ACG used to
ignore it, so every re-submit queued a second render on the single-lane image
backend (28 renders for 16 delivered images in one window). These tests drive
the real routes + real job managers + real unique index, and count the
background tasks the route starts -- the GPU work -- not just the rows.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.content_routes import router as concept_images_router
from src.api.routes import router as videos_router
from src.database.models import ImageGenJob, VideoGenerationJob
from src.database.session import get_db
from src.services.imagegen_service import ImageGenService
from src.services.video_generation_service import VideoGenerationService
from tests.conftest import TENANT_A, TENANT_B

_IMAGE_URL = "/api/concept-images"
_VIDEO_URL = "/api/content/videos/generate"
_IMAGE_BODY = {"prompt": "a lighthouse at dusk", "provider": "stub"}
_VIDEO_BODY = {"prompt": "waves on a pier", "provider": "fake", "duration_s": 5}


class _TaskCounter:
    """Wraps ``asyncio.create_task``: counts (and never runs) the routes' generation work.

    ``patch("src.api....asyncio.create_task")`` patches the asyncio MODULE, i.e. every
    task the event loop starts (httpx, psycopg). Only the route's ``run_with_lease``
    coroutine is intercepted; everything else goes to the real ``create_task``.
    """

    def __init__(self) -> None:
        self.started = 0
        self._real = asyncio.create_task

    def __call__(self, coro: Any, **kwargs: Any) -> asyncio.Future[None]:
        if getattr(coro, "__qualname__", "") != "run_with_lease":
            return self._real(coro, **kwargs)
        coro.close()
        self.started += 1
        # A real, never-finishing awaitable, so the cancel path can cancel + await it.
        return asyncio.get_running_loop().create_future()


def _image_registry() -> MagicMock:
    registry = MagicMock()
    registry.get.side_effect = lambda name: MagicMock() if name == "stub" else None
    registry.provider_names = ["stub"]
    return registry


def _build_app(factory: async_sessionmaker[AsyncSession], registry: Any) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def _tenant(request, call_next):  # pragma: no cover - harness
        # AuthMiddleware's job in production; the test picks the tenant per call.
        request.state.tenant_id = request.headers.get("x-test-tenant", TENANT_A)
        return await call_next(request)

    app.include_router(concept_images_router)
    app.include_router(videos_router)
    image_registry = _image_registry()
    app.state.image_provider_registry = image_registry
    app.state.imagegen_service = ImageGenService(
        storage=MagicMock(), registry=image_registry, default_provider="stub"
    )
    app.state.provider_registry = registry
    app.state.video_generation_service = VideoGenerationService(
        storage=MagicMock(), registry=registry, default_provider="fake"
    )

    async def _db() -> AsyncIterator[AsyncSession]:
        # One real session (one connection) per request, like production.
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _db
    return app


@pytest.fixture
async def harness(
    db_session_factory: async_sessionmaker[AsyncSession], registry: Any
) -> AsyncIterator[tuple[AsyncClient, _TaskCounter]]:
    counter = _TaskCounter()
    app = _build_app(db_session_factory, registry)
    with patch("asyncio.create_task", side_effect=counter):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac, counter


async def _count(factory: async_sessionmaker[AsyncSession], model: Any) -> int:
    async with factory() as session:
        return (await session.execute(select(func.count()).select_from(model))).scalar_one()


def _post(
    ac: AsyncClient, url: str, body: dict[str, Any], key: str | None, tenant: str = TENANT_A
) -> Any:
    headers = {"x-test-tenant": tenant}
    if key is not None:
        headers["Idempotency-Key"] = key
    return ac.post(url, json=body, headers=headers)


_ROUTES = [
    pytest.param(_IMAGE_URL, _IMAGE_BODY, ImageGenJob, id="concept-image"),
    pytest.param(_VIDEO_URL, _VIDEO_BODY, VideoGenerationJob, id="video"),
]


@pytest.mark.parametrize(("url", "body", "model"), _ROUTES)
async def test_repeat_key_returns_the_same_job_and_starts_one_task(
    harness, db_session_factory, url, body, model
) -> None:
    ac, counter = harness
    first = await _post(ac, url, body, "run1:ai_image_gen:m1")
    second = await _post(ac, url, body, "run1:ai_image_gen:m1")

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["poll_url"] == first.json()["poll_url"]
    assert second.headers.get("Idempotent-Replayed") == "true"
    assert "Idempotent-Replayed" not in first.headers
    assert counter.started == 1
    assert await _count(db_session_factory, model) == 1


@pytest.mark.parametrize(("url", "body", "model"), _ROUTES)
async def test_concurrent_duplicates_create_exactly_one_job(
    harness, db_session_factory, url, body, model
) -> None:
    ac, counter = harness
    responses = await asyncio.gather(*(_post(ac, url, body, "race-key") for _ in range(6)))

    assert {r.status_code for r in responses} == {200}, [r.text for r in responses]
    assert len({r.json()["job_id"] for r in responses}) == 1
    assert counter.started == 1
    assert await _count(db_session_factory, model) == 1


@pytest.mark.parametrize(("url", "body", "model"), _ROUTES)
async def test_same_key_different_tenant_gets_its_own_job(
    harness, db_session_factory, url, body, model
) -> None:
    ac, counter = harness
    a = await _post(ac, url, body, "shared-key", tenant=TENANT_A)
    b = await _post(ac, url, body, "shared-key", tenant=TENANT_B)

    assert a.status_code == b.status_code == 200
    assert a.json()["job_id"] != b.json()["job_id"]
    assert counter.started == 2
    assert await _count(db_session_factory, model) == 2


@pytest.mark.parametrize(("url", "body", "model"), _ROUTES)
async def test_same_key_with_a_different_body_is_422(
    harness, db_session_factory, url, body, model
) -> None:
    ac, counter = harness
    ok = await _post(ac, url, body, "k1")
    clash = await _post(ac, url, {**body, "prompt": "something else"}, "k1")

    assert ok.status_code == 200
    assert clash.status_code == 422
    assert "Idempotency-Key" in clash.json()["detail"]
    assert counter.started == 1
    assert await _count(db_session_factory, model) == 1


@pytest.mark.parametrize(("url", "body", "model"), _ROUTES)
async def test_no_key_still_creates_a_job_per_request(
    harness, db_session_factory, url, body, model
) -> None:
    ac, counter = harness
    a = await _post(ac, url, body, None)
    b = await _post(ac, url, body, None)

    assert a.json()["job_id"] != b.json()["job_id"]
    assert counter.started == 2
    assert await _count(db_session_factory, model) == 2


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize(("url", "body", "model"), _ROUTES)
async def test_a_terminal_failure_releases_the_key_for_a_fresh_attempt(
    harness, db_session_factory, url, body, model, status
) -> None:
    ac, counter = harness
    first = (await _post(ac, url, body, "retry-me")).json()["job_id"]
    async with db_session_factory() as session:
        await session.execute(update(model).where(model.job_id == first).values(status=status))
        await session.commit()

    retry = await _post(ac, url, body, "retry-me")

    assert retry.status_code == 200
    assert retry.json()["job_id"] != first
    assert "Idempotent-Replayed" not in retry.headers
    assert counter.started == 2
    async with db_session_factory() as session:
        holders = (
            (await session.execute(select(model.job_id).where(model.idempotency_key == "retry-me")))
            .scalars()
            .all()
        )
    assert holders == [retry.json()["job_id"]]


async def test_replay_reports_the_jobs_current_status(harness, db_session_factory) -> None:
    ac, counter = harness
    first = (await _post(ac, _IMAGE_URL, _IMAGE_BODY, "k-status")).json()["job_id"]
    async with db_session_factory() as session:
        await session.execute(
            update(ImageGenJob).where(ImageGenJob.job_id == first).values(status="completed")
        )
        await session.commit()

    replay = await _post(ac, _IMAGE_URL, _IMAGE_BODY, "k-status")

    assert replay.json() == {
        "job_id": first,
        "status": "completed",
        "poll_url": f"/api/concept-images/jobs/{first}",
        "result_url": f"/api/concept-images/jobs/{first}/result",
    }
    assert counter.started == 1


@pytest.mark.parametrize("bad", ["", "   ", "x" * 256])
async def test_an_unusable_key_is_422(harness, db_session_factory, bad: str) -> None:
    ac, counter = harness
    resp = await _post(ac, _IMAGE_URL, _IMAGE_BODY, bad)
    assert resp.status_code == 422
    assert counter.started == 0
    assert await _count(db_session_factory, ImageGenJob) == 0


async def test_cancel_route_cancels_a_pending_image_job_and_frees_its_key(
    harness, db_session_factory
) -> None:
    ac, counter = harness
    job_id = (await _post(ac, _IMAGE_URL, _IMAGE_BODY, "cancel-me")).json()["job_id"]

    cancelled = await ac.post(
        f"/api/concept-images/jobs/{job_id}/cancel", headers={"x-test-tenant": TENANT_A}
    )
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json() == {"job_id": job_id, "status": "cancelled"}
    async with db_session_factory() as session:
        status = (
            await session.execute(select(ImageGenJob.status).where(ImageGenJob.job_id == job_id))
        ).scalar_one()
    assert status == "cancelled"

    again = await ac.post(
        f"/api/concept-images/jobs/{job_id}/cancel", headers={"x-test-tenant": TENANT_A}
    )
    assert again.status_code == 409

    retry = await _post(ac, _IMAGE_URL, _IMAGE_BODY, "cancel-me")
    assert retry.json()["job_id"] != job_id
    assert counter.started == 2


async def test_cancel_route_never_touches_another_tenants_job(harness, db_session_factory) -> None:
    ac, _ = harness
    job_id = (await _post(ac, _IMAGE_URL, _IMAGE_BODY, None, tenant=TENANT_A)).json()["job_id"]

    foreign = await ac.post(
        f"/api/concept-images/jobs/{job_id}/cancel", headers={"x-test-tenant": TENANT_B}
    )

    assert foreign.status_code == 409
    async with db_session_factory() as session:
        status = (
            await session.execute(select(ImageGenJob.status).where(ImageGenJob.job_id == job_id))
        ).scalar_one()
    assert status == "pending"
