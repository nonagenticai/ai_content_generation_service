"""Tenant scoping of every by-id job / asset / batch lookup.

These are the application-layer guards that sit in FRONT of Phase D RLS: a
by-id read for a job owned by another tenant must be indistinguishable from a
read for an id that does not exist (404, byte-identical body) rather than
relying on the database policy alone. The batch tracker in particular is
Redis-backed and has NO RLS backstop, so its scoping is load-bearing.

Runs against the real postgres test database (``db_session``) with a
superuser role, so RLS is NOT in play here — a leak in these tests is a leak
in the application layer, which is exactly what is under test.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.content_routes import router as content_router
from src.api.routes import router as video_router
from src.database.session import get_db
from src.services.batch_tracker import BatchTracker
from src.services.image_batch_manager import ImageBatchManager
from src.services.image_job_manager import ImageJobManager
from src.services.job_manager import JobManager
from tests.conftest import TENANT_A, TENANT_B

# ---------------------------------------------------------------------------
# App / client harness
# ---------------------------------------------------------------------------


def _storage_mock() -> MagicMock:
    storage = MagicMock()
    storage.presign_url = AsyncMock(return_value=None)
    storage.retrieve = AsyncMock(return_value=b"fake-bytes")
    return storage


def _build_app(
    db_session: Any,
    *,
    redis_client: Any = None,
    tracker: Any = None,
    quota_service: Any = None,
) -> FastAPI:
    """Minimal app wiring both routers onto the real test session."""
    app = FastAPI()
    app.include_router(video_router)
    app.include_router(content_router)

    app.state.quota_service = quota_service
    app.state.storage_service = _storage_mock()
    app.state.batch_tracker = tracker if tracker is not None else BatchTracker()
    app.state.video_generation_service = MagicMock()
    app.state.imagegen_service = MagicMock()
    app.state.image_provider_registry = MagicMock()
    app.state.provider_registry = MagicMock()
    if redis_client is not None:
        app.state.redis_client = redis_client

    @app.middleware("http")
    async def _tenant_mw(request, call_next):
        request.state.tenant_id = request.headers.get("X-Test-Tenant", TENANT_A)
        return await call_next(request)

    async def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _get(app: FastAPI, path: str, tenant: str = TENANT_A):
    async with _client(app) as client:
        return await client.get(path, headers={"X-Test-Tenant": tenant})


def _assert_indistinguishable_404(foreign, missing, foreign_id: str, missing_id: str) -> None:
    """The cross-tenant 404 must be byte-identical to the genuine not-found.

    Both bodies embed the requested id, so equality is asserted after
    normalising that one substitution — anything else differing (a distinct
    detail string, an extra field) would make the endpoint an enumeration
    oracle over uuid7 ids.
    """
    assert foreign.status_code == 404, foreign.text
    assert missing.status_code == 404, missing.text
    normalised = missing.content.replace(missing_id.encode(), foreign_id.encode())
    assert foreign.content == normalised


# ---------------------------------------------------------------------------
# Seeded fixtures: one row per tenant, same shape
# ---------------------------------------------------------------------------


@pytest.fixture
async def video_jobs(db_session) -> dict[str, str]:
    """A completed video job for tenant-a and one for tenant-b."""
    ids: dict[str, str] = {}
    for key, tenant in (("a", TENANT_A), ("b", TENANT_B)):
        job = await JobManager.create_job(
            db_session,
            tenant_id=tenant,
            provider="fake",
            prompt="a cat",
        )
        job.status = "completed"
        job.video_path = "/tmp/v.mp4"
        job.cost = 0.05
        ids[key] = job.job_id
    await db_session.commit()
    return ids


@pytest.fixture
async def image_jobs(db_session) -> dict[str, str]:
    """A completed image job (+asset) for tenant-a and one for tenant-b."""
    ids: dict[str, str] = {}
    for key, tenant in (("a", TENANT_A), ("b", TENANT_B)):
        job = await ImageJobManager.create_job(
            db_session,
            tenant_id=tenant,
            provider="mock",
            prompt="a hero image",
        )
        job.status = "completed"
        job.image_path = "images/x.png"
        ids[key] = job.job_id
        ids[f"asset_{key}"] = job.asset_id
    await db_session.commit()
    return ids


@pytest.fixture
async def image_batches(db_session) -> dict[str, str]:
    """A completed image batch for tenant-a and one for tenant-b."""
    ids: dict[str, str] = {}
    for key, tenant in (("a", TENANT_A), ("b", TENANT_B)):
        batch = await ImageBatchManager.create_batch(
            db_session,
            tenant_id=tenant,
            budget_limit=5.0,
            job_count=0,
        )
        batch.status = "completed"
        ids[key] = batch.batch_id
    await db_session.commit()
    return ids


def _missing_id() -> str:
    return str(uuid.uuid7())


# ---------------------------------------------------------------------------
# Video job routes (src/api/routes/jobs.py)
# ---------------------------------------------------------------------------

VIDEO_JOB_ROUTES = [
    "/api/content/videos/jobs/{id}",
    "/api/content/videos/jobs/{id}/result",
    "/api/content/videos/jobs/{id}/costs",
    "/api/content/videos/jobs/{id}/video",
]


def _assert_identifies_job(resp, job_id: str) -> None:
    """A 200 must actually carry the requested job — an empty fixture cannot fake it."""
    if not resp.headers.get("content-type", "").startswith("application/json"):
        assert resp.content == b"fake-bytes"
        return
    body = resp.json()
    if "job_id" in body:
        assert body["job_id"] == job_id
    else:  # /costs returns a CostReport keyed per clip
        assert [c["job_id"] for c in body["per_clip_costs"]] == [job_id]


@pytest.mark.parametrize("template", VIDEO_JOB_ROUTES)
async def test_video_job_route_owner_gets_200(db_session, video_jobs, template) -> None:
    app = _build_app(db_session)
    resp = await _get(app, template.format(id=video_jobs["a"]))
    assert resp.status_code == 200, resp.text
    _assert_identifies_job(resp, video_jobs["a"])


@pytest.mark.parametrize("template", VIDEO_JOB_ROUTES)
async def test_video_job_route_foreign_tenant_404s(db_session, video_jobs, template) -> None:
    app = _build_app(db_session)
    missing = _missing_id()
    foreign = await _get(app, template.format(id=video_jobs["b"]), tenant=TENANT_A)
    absent = await _get(app, template.format(id=missing), tenant=TENANT_A)
    _assert_indistinguishable_404(foreign, absent, video_jobs["b"], missing)


# ---------------------------------------------------------------------------
# Concept-image job routes (src/api/content_routes.py)
# ---------------------------------------------------------------------------

IMAGE_JOB_ROUTES = [
    "/api/concept-images/jobs/{id}",
    "/api/concept-images/jobs/{id}/result",
]


@pytest.mark.parametrize("template", IMAGE_JOB_ROUTES)
async def test_image_job_route_owner_gets_200(db_session, image_jobs, template) -> None:
    app = _build_app(db_session)
    resp = await _get(app, template.format(id=image_jobs["a"]))
    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"] == image_jobs["a"]


@pytest.mark.parametrize("template", IMAGE_JOB_ROUTES)
async def test_image_job_route_foreign_tenant_404s(db_session, image_jobs, template) -> None:
    app = _build_app(db_session)
    missing = _missing_id()
    foreign = await _get(app, template.format(id=image_jobs["b"]), tenant=TENANT_A)
    absent = await _get(app, template.format(id=missing), tenant=TENANT_A)
    _assert_indistinguishable_404(foreign, absent, image_jobs["b"], missing)


# ---------------------------------------------------------------------------
# Asset-id routes — same hole on a different key
# ---------------------------------------------------------------------------

ASSET_ROUTES = [
    "/api/concept-images/assets/{id}",
    "/api/concept-images/assets/{id}/download",
]


@pytest.mark.parametrize("template", ASSET_ROUTES)
async def test_asset_route_owner_gets_200(db_session, image_jobs, template) -> None:
    app = _build_app(db_session)
    resp = await _get(app, template.format(id=image_jobs["asset_a"]))
    assert resp.status_code == 200, resp.text
    _assert_identifies_job(resp, image_jobs["a"])


@pytest.mark.parametrize("template", ASSET_ROUTES)
async def test_asset_route_foreign_tenant_404s(db_session, image_jobs, template) -> None:
    app = _build_app(db_session)
    missing = _missing_id()
    foreign = await _get(app, template.format(id=image_jobs["asset_b"]), tenant=TENANT_A)
    absent = await _get(app, template.format(id=missing), tenant=TENANT_A)
    _assert_indistinguishable_404(foreign, absent, image_jobs["asset_b"], missing)


# ---------------------------------------------------------------------------
# Concept-image batch routes
# ---------------------------------------------------------------------------

IMAGE_BATCH_ROUTES = [
    "/api/concept-images/batches/{id}",
    "/api/concept-images/batches/{id}/result",
]


@pytest.mark.parametrize("template", IMAGE_BATCH_ROUTES)
async def test_image_batch_route_owner_gets_200(db_session, image_batches, template) -> None:
    app = _build_app(db_session)
    resp = await _get(app, template.format(id=image_batches["a"]))
    assert resp.status_code == 200, resp.text
    assert resp.json()["batch_id"] == image_batches["a"]


@pytest.mark.parametrize("template", IMAGE_BATCH_ROUTES)
async def test_image_batch_route_foreign_tenant_404s(db_session, image_batches, template) -> None:
    app = _build_app(db_session)
    missing = _missing_id()
    foreign = await _get(app, template.format(id=image_batches["b"]), tenant=TENANT_A)
    absent = await _get(app, template.format(id=missing), tenant=TENANT_A)
    _assert_indistinguishable_404(foreign, absent, image_batches["b"], missing)


async def test_image_batch_jobs_are_tenant_scoped(db_session, image_batches) -> None:
    """A foreign job carrying the same batch_id must not leak into the results."""
    own = await ImageJobManager.create_job(
        db_session,
        tenant_id=TENANT_A,
        provider="mock",
        prompt="mine",
        batch_id=image_batches["a"],
    )
    foreign = await ImageJobManager.create_job(
        db_session,
        tenant_id=TENANT_B,
        provider="mock",
        prompt="theirs",
        batch_id=image_batches["a"],
    )
    await db_session.commit()

    app = _build_app(db_session)
    resp = await _get(app, f"/api/concept-images/batches/{image_batches['a']}")
    assert resp.status_code == 200, resp.text
    job_ids = resp.json()["job_ids"]
    assert own.job_id in job_ids
    assert foreign.job_id not in job_ids


# ---------------------------------------------------------------------------
# Video batch routes — BatchTracker (Redis-backed, NO RLS backstop)
# ---------------------------------------------------------------------------

VIDEO_BATCH_ROUTES = [
    "/api/content/videos/batches/{id}/status",
    "/api/content/videos/batches/{id}/result",
]


@pytest.fixture
async def video_batches(video_jobs) -> tuple[BatchTracker, dict[str, str]]:
    """One tracked batch per tenant, each already terminal."""
    tracker = BatchTracker()
    ids = {"a": str(uuid.uuid7()), "b": str(uuid.uuid7())}
    await tracker.create_batch(ids["a"], [video_jobs["a"]], budget_limit=5.0, tenant_id=TENANT_A)
    await tracker.create_batch(ids["b"], [video_jobs["b"]], budget_limit=5.0, tenant_id=TENANT_B)
    await tracker.record_clip_completed(ids["a"], cost=0.05)
    await tracker.record_clip_completed(ids["b"], cost=0.05)
    return tracker, ids


@pytest.mark.parametrize("template", VIDEO_BATCH_ROUTES)
async def test_video_batch_route_owner_gets_200(db_session, video_batches, template) -> None:
    tracker, ids = video_batches
    app = _build_app(db_session, tracker=tracker)
    resp = await _get(app, template.format(id=ids["a"]))
    assert resp.status_code == 200, resp.text
    assert resp.json()["batch_id"] == ids["a"]


@pytest.mark.parametrize("template", VIDEO_BATCH_ROUTES)
async def test_video_batch_route_foreign_tenant_404s(db_session, video_batches, template) -> None:
    tracker, ids = video_batches
    app = _build_app(db_session, tracker=tracker)
    missing = _missing_id()
    foreign = await _get(app, template.format(id=ids["b"]), tenant=TENANT_A)
    absent = await _get(app, template.format(id=missing), tenant=TENANT_A)
    _assert_indistinguishable_404(foreign, absent, ids["b"], missing)


async def test_batch_status_skips_foreign_clips(db_session, video_jobs) -> None:
    """A batch listing a foreign clip drops that clip — it does not 404 the batch."""
    tracker = BatchTracker()
    batch_id = str(uuid.uuid7())
    await tracker.create_batch(
        batch_id,
        [video_jobs["a"], video_jobs["b"]],
        budget_limit=5.0,
        tenant_id=TENANT_A,
    )
    app = _build_app(db_session, tracker=tracker)
    resp = await _get(app, f"/api/content/videos/batches/{batch_id}/status")
    assert resp.status_code == 200, resp.text
    clip_ids = [c["job_id"] for c in resp.json()["clips"]]
    assert clip_ids == [video_jobs["a"]]


# ---------------------------------------------------------------------------
# Cache: a warm entry for tenant B must not soften the DB filter for tenant A
# ---------------------------------------------------------------------------


async def test_warm_foreign_cache_entry_still_404s_video_job(
    db_session, video_jobs, real_redis_client
) -> None:
    """Warm tenant B's cached status, then read it as tenant A: still 404.

    The cache key is not tenant-scoped (only the cached VALUE is tenant-tagged),
    so tenant A never hits the cache and always falls through to the DB read.
    Warm-cache is therefore the case that proves both the tag check AND the new
    tenant predicate on the DB read.
    """
    app = _build_app(db_session, redis_client=real_redis_client)

    warm = await _get(app, f"/api/content/videos/jobs/{video_jobs['b']}", tenant=TENANT_B)
    assert warm.status_code == 200, warm.text
    assert await real_redis_client.get(f"content_generation:job:{video_jobs['b']}:status")

    missing = _missing_id()
    foreign = await _get(app, f"/api/content/videos/jobs/{video_jobs['b']}", tenant=TENANT_A)
    absent = await _get(app, f"/api/content/videos/jobs/{missing}", tenant=TENANT_A)
    _assert_indistinguishable_404(foreign, absent, video_jobs["b"], missing)


async def test_warm_foreign_cache_entry_still_404s_image_job(
    db_session, image_jobs, real_redis_client
) -> None:
    """Same shape on the concept-image status route."""
    app = _build_app(db_session, redis_client=real_redis_client)

    warm = await _get(app, f"/api/concept-images/jobs/{image_jobs['b']}", tenant=TENANT_B)
    assert warm.status_code == 200, warm.text
    assert await real_redis_client.get(f"content:image:{image_jobs['b']}:status")

    missing = _missing_id()
    foreign = await _get(app, f"/api/concept-images/jobs/{image_jobs['b']}", tenant=TENANT_A)
    absent = await _get(app, f"/api/concept-images/jobs/{missing}", tenant=TENANT_A)
    _assert_indistinguishable_404(foreign, absent, image_jobs["b"], missing)


# ---------------------------------------------------------------------------
# Worker regression: background callers pass NO tenant and must keep working
# ---------------------------------------------------------------------------


async def test_unscoped_get_job_still_returns_row(db_session, video_jobs) -> None:
    job = await JobManager.get_job(db_session, video_jobs["b"])
    assert job is not None
    assert job.job_id == video_jobs["b"]


async def test_unscoped_image_lookups_still_return_rows(db_session, image_jobs) -> None:
    assert await ImageJobManager.get_job(db_session, image_jobs["b"]) is not None
    assert await ImageJobManager.get_job_by_asset_id(db_session, image_jobs["asset_b"]) is not None


async def test_unscoped_image_batch_lookups_still_return_rows(db_session, image_batches) -> None:
    assert await ImageBatchManager.get_batch(db_session, image_batches["b"]) is not None
    assert await ImageBatchManager.get_batch_jobs(db_session, image_batches["b"]) == []


async def test_unscoped_tracker_get_batch_still_returns_state(video_batches) -> None:
    tracker, ids = video_batches
    assert await tracker.get_batch(ids["b"]) is not None


async def test_tracker_get_batch_scoped_filters_foreign(video_batches) -> None:
    tracker, ids = video_batches
    assert await tracker.get_batch(ids["b"], tenant_id=TENANT_A) is None
    assert await tracker.get_batch(ids["b"], tenant_id=TENANT_B) is not None


# ---------------------------------------------------------------------------
# GET /quotas/{tenant_id} — owner id came from the URL, not the identity
# ---------------------------------------------------------------------------

QUOTA_ROUTE = "/api/content/videos/quotas/{tenant}"


def _quota_service_mock() -> AsyncMock:
    """A quota service that answers for whichever tenant it is asked about.

    Deliberately faithful to the real one: ``get_quota_status`` never refuses,
    it synthesises a status for any id. All the tenant safety therefore has to
    live in the route.
    """
    numbers = {
        TENANT_A: {"daily_generations_used": 3, "daily_spend_used": 0.75},
        TENANT_B: {"daily_generations_used": 91, "daily_spend_used": 47.5},
    }

    async def _status(tenant_id: str) -> dict[str, Any]:
        return {
            "tenant_id": tenant_id,
            "daily_generations_limit": 100,
            "daily_spend_limit": 50.0,
            "concurrent_jobs": 1,
            "concurrent_jobs_limit": 10,
            **numbers.get(tenant_id, {"daily_generations_used": 0, "daily_spend_used": 0.0}),
        }

    svc = AsyncMock()
    svc.get_quota_status = AsyncMock(side_effect=_status)
    return svc


async def test_quota_owner_gets_200(db_session) -> None:
    """The caller reading their OWN quota is unaffected by the scoping."""
    svc = _quota_service_mock()
    app = _build_app(db_session, quota_service=svc)
    resp = await _get(app, QUOTA_ROUTE.format(tenant=TENANT_A), tenant=TENANT_A)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tenant_id"] == TENANT_A
    assert body["daily_generations_used"] == 3
    assert body["daily_spend_used"] == 0.75


async def test_quota_foreign_tenant_is_refused(db_session) -> None:
    """Tenant A asking for tenant B's quota must be refused, not answered.

    Quota is Redis-backed and has no RLS backstop, so this route check is the
    ONLY thing between the URL and another tenant's spend figures.
    """
    svc = _quota_service_mock()
    app = _build_app(db_session, quota_service=svc)
    resp = await _get(app, QUOTA_ROUTE.format(tenant=TENANT_B), tenant=TENANT_A)
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "Quota is readable only for your own tenant"
    # The victim's numbers were never even looked up.
    svc.get_quota_status.assert_not_awaited()


# ---------------------------------------------------------------------------
# Rollout hazard: batch snapshots written by the PREVIOUS release carry no owner
# ---------------------------------------------------------------------------


async def test_legacy_ownerless_batch_is_refused_even_for_its_owner(db_session, video_jobs) -> None:
    """A pre-upgrade batch (``tenant_id is None``) is refused for EVERYBODY.

    This pins a deliberate decision, not an accident: at deploy, in-flight
    batches written by the previous release have no stored owner, so their
    rightful owner 404s on /status and /result for up to the 2h Redis TTL
    while workers keep charging the batch. The accepted trade is that an
    ownerless batch stays unreadable rather than becoming readable by whoever
    guesses its id -- adopting it into the requesting tenant would reopen the
    very leak the scoping closes. If this test ever has to be relaxed, that is
    a decision to be argued, not a detail to be patched.
    """
    tracker = BatchTracker()
    batch_id = str(uuid.uuid7())
    # Exactly what an old release persisted: no tenant_id argument at all.
    await tracker.create_batch(batch_id, [video_jobs["a"]], budget_limit=5.0)
    stored = await tracker.get_batch(batch_id)
    assert stored is not None
    assert stored.tenant_id is None

    # Refused for its real owner ...
    assert await tracker.get_batch(batch_id, tenant_id=TENANT_A) is None
    # ... and, the point of the trade, for everyone else too.
    assert await tracker.get_batch(batch_id, tenant_id=TENANT_B) is None

    app = _build_app(db_session, tracker=tracker)
    for template in VIDEO_BATCH_ROUTES:
        resp = await _get(app, template.format(id=batch_id), tenant=TENANT_A)
        assert resp.status_code == 404, f"{template}: {resp.text}"
