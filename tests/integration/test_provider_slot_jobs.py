"""Tenant provider-slot jobs (P5-2) against the real Postgres + a fake provider HTTP API.

What is proven here, end to end through the real job rows:

* a slot job builds a PER-JOB provider that authenticates with the SLOT's key
  against the SLOT's base_url -- and never touches the platform registry;
* the secret is SEALED at rest: the raw row holds no plaintext key;
* the sealed secret is PURGED by the DB trigger when the job turns terminal,
  including when the terminal write is the reclaim function's SQL-only path;
* a reclaimed job (the reaper's resume) UNSEALS the secret from the row, also
  after a key rotation; an unopenable envelope fails the job permanently;
* a provider 401 fails the job permanently, names the slot, is not retried and
  never falls back to a platform provider -- and the error carries no secret;
* every slot request is re-validated at CONNECT time and pinned to the validated
  IP; a host that resolves to a private address then fails the job permanently;
* a completed slot job is metered as tenant-funded BYO (``usage_extra.byo``,
  top-level ``funding=tenant_byo``, ``cost_usd=0``), never at platform prices.

The provider HTTP API is faked at the transport layer (``httpx.MockTransport``),
so the real provider classes build and send the real requests.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import text, update

from src.config import clear_settings_cache
from src.database.models import VideoGenerationJob
from src.services.image_job_manager import ImageJobManager
from src.services.image_providers.base import ImageGenProvider, ImageProviderRegistry
from src.services.imagegen_service import ImageGenService
from src.services.job_manager import JobManager
from src.services.provider_slots import ImageProviderSlot, VideoProviderSlot, seal_slot
from src.services.providers.base import VideoGenerationProvider, VideoProviderRegistry
from src.services.slot_http import SlotSSRFGuardTransport
from src.services.video_generation_service import VideoGenerationService
from tests.conftest import TENANT_A

TENANT_KEY = "sk-tenant-slot-key-0123456789abcdef"  # pragma: allowlist secret
PLATFORM_KEY = "sk-platform-key-should-never-be-sent"  # pragma: allowlist secret
_KEY_A = base64.b64encode(b"A" * 32).decode()
_KEY_B = base64.b64encode(b"B" * 32).decode()
_CONNECTION = "0195f48b-5555-7000-8000-00000000c0de"
#: The IP the (stubbed) connect-time validator pins every slot request to.
_PINNED_IP = "93.184.216.34"


@pytest.fixture
def keyring(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], None]]:
    """Set ``PROVIDER_SECRET_KEYS`` (and the platform keys) for the test."""

    def _set(value: str) -> None:
        monkeypatch.setenv("PROVIDER_SECRET_KEYS", value)
        clear_settings_cache()

    monkeypatch.setenv("KLING_API_KEY", PLATFORM_KEY)
    monkeypatch.setenv("DALLE_API_KEY", PLATFORM_KEY)
    _set(f"ka:{_KEY_A}")
    yield _set
    clear_settings_cache()


class _FakeProviderAPI:
    """Records every request; answers from a per-path handler."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self.validated: list[str] = []  # URLs the connect-time guard checked
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)


@pytest.fixture
def fake_http(monkeypatch: pytest.MonkeyPatch) -> Callable[..., _FakeProviderAPI]:
    """Route every httpx.AsyncClient built during the test through a MockTransport.

    A slot client's connect-time SSRF guard is KEPT: the mock replaces only its
    inner (network) transport, and the DNS validator is stubbed to pin a public
    IP -- so every request still passes through the real guard + pin logic.
    """
    import src.services.slot_http as slot_http

    real_client = httpx.AsyncClient

    def _install(handler: Callable[[httpx.Request], httpx.Response]) -> _FakeProviderAPI:
        api = _FakeProviderAPI(handler)

        def _client(**kwargs: Any) -> httpx.AsyncClient:
            kwargs.pop("http2", None)
            guard = kwargs.get("transport")
            if isinstance(guard, SlotSSRFGuardTransport):
                guard.inner = httpx.MockTransport(api)
                return real_client(**kwargs)
            return real_client(transport=httpx.MockTransport(api), **kwargs)

        def _validate(url: str, **_: Any) -> str:
            api.validated.append(url)
            return _PINNED_IP

        monkeypatch.setattr(httpx, "AsyncClient", _client)
        monkeypatch.setattr(slot_http, "validate_public_url", _validate)
        return api

    return _install


class _PlatformVideoProvider(VideoGenerationProvider):
    """The platform ``kling`` in the registry. A slot job must never reach it."""

    name = "kling"

    def __init__(self) -> None:
        self.calls = 0

    async def submit(self, prompt: str, **kwargs: Any) -> str:
        self.calls += 1
        raise AssertionError("platform provider used for a tenant-slot job")

    async def check_status(self, task_id: str) -> dict:
        raise AssertionError("platform provider used for a tenant-slot job")

    async def download(self, task_id: str, video_url: str) -> bytes:
        raise AssertionError("platform provider used for a tenant-slot job")

    def estimate_cost(self, duration_s: float) -> float:
        return 0.0


class _PlatformImageProvider(ImageGenProvider):
    name = "openai_images"

    async def generate(self, **kwargs: Any) -> Any:
        raise AssertionError("platform provider used for a tenant-slot job")

    def estimate_cost(self, *, size: str | None = None, quality: str | None = None) -> float:
        return 0.0


def _video_service(platform: _PlatformVideoProvider) -> VideoGenerationService:
    registry = VideoProviderRegistry()
    registry.register(platform)
    return VideoGenerationService(
        storage=AsyncMock(),
        registry=registry,
        default_provider="kling",
        max_retries=2,
        fallback_enabled=True,
        metering_redis=AsyncMock(),
    )


@pytest.fixture
def cost_events(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every cost event the services schedule during the test."""
    from src.services import metering

    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        metering, "schedule_cost_event", lambda _redis, **kwargs: events.append(kwargs)
    )
    return events


def _assert_tenant_byo(event: dict[str, Any], *, slot: str, kind: str, version: int) -> None:
    """A tenant-key job is metered as BYO: zero platform cost, tenant funding."""
    assert event["cost_usd"] == 0.0
    extra = event["usage_extra"]
    assert extra["byo"] == "true"  # billing-service: usage_extra.byo == "true"
    assert "owner_provided" not in extra
    assert event["funding"] == "tenant_byo"  # top-level (w2v-common 0.94 kwarg)
    assert "funding" not in extra
    assert extra["slot"] == slot
    assert extra["slot_kind"] == kind
    assert extra["connection_id"] == _CONNECTION
    assert extra["version"] == version
    assert TENANT_KEY not in json.dumps(event, default=str)


def _video_slot(**overrides: Any) -> VideoProviderSlot:
    data: dict[str, Any] = {
        "slot": "video_gen",
        "kind": "kling",
        "base_url": "https://kling.tenant.example/v1",
        "model": "kling-v1-5",
        "options": {},
        "secret": TENANT_KEY,
        "ref": {"connection_id": _CONNECTION, "version": 3},
    }
    data.update(overrides)
    return VideoProviderSlot.model_validate(data)


async def _create_video_slot_job(db: Any, slot: VideoProviderSlot) -> str:
    job = await JobManager.create_job(
        db,
        tenant_id=TENANT_A,
        provider="kling",
        prompt="a tenant clip",
        duration_s=5.0,
        # A slot job never carries fallbacks; this one does on purpose, to prove
        # the slot path ignores them even if a row somehow has them.
        fallback_providers=["kling"],
    )
    persisted, sealed = seal_slot(slot, job_id=job.job_id)
    await JobManager.attach_provider_slot(
        db, job.job_id, provider_slot=persisted, provider_secret_sealed=sealed
    )
    await JobManager.register_dispatch(
        db,
        job.job_id,
        options={"kind": "video_single", "budget_limit": 5.0},
        worker_id="pod-a",
        lease_seconds=900,
    )
    await db.commit()
    return job.job_id


async def _raw_row(db: Any, table: str, job_id: str) -> dict[str, Any]:
    row = (
        (
            await db.execute(
                text(f"SELECT * FROM {table} WHERE job_id = :j"),  # noqa: S608 -- fixed table names
                {"j": job_id},
            )
        )
        .mappings()
        .one()
    )
    return dict(row)


def _row_bytes(row: dict[str, Any]) -> bytes:
    """Every column of a raw row, as one blob to grep for the plaintext key."""
    parts = []
    for value in row.values():
        if isinstance(value, bytes | memoryview):
            parts.append(bytes(value))
        else:
            parts.append(json.dumps(value, default=str).encode())
    return b"\x00".join(parts)


def _kling_ok(request: httpx.Request) -> httpx.Response:
    if request.method == "POST":
        return httpx.Response(200, json={"data": {"task_id": "tenant-task-1"}})
    return httpx.Response(200, json={"data": {"task_status": "succeed", "task_result": {}}})


async def test_video_slot_job_uses_the_slot_key_and_base_url_and_purges_on_completion(
    db_session, keyring, fake_http, cost_events
):
    api = fake_http(_kling_ok)
    platform = _PlatformVideoProvider()
    svc = _video_service(platform)
    job_id = await _create_video_slot_job(db_session, _video_slot())

    # Sealed at rest: the envelope is there, the plaintext is nowhere in the row.
    before = await _raw_row(db_session, "video_generation_jobs", job_id)
    assert before["provider_secret_sealed"] is not None
    assert bytes(before["provider_secret_sealed"]).startswith(b"v1|ka|")
    assert TENANT_KEY.encode() not in _row_bytes(before)
    assert before["provider_slot"]["has_secret"] is True
    assert "secret" not in before["provider_slot"]

    cost = await svc.process_job(db_session, job_id, TENANT_A, budget_limit=5.0)

    # The tenant's key paid: zero platform cost on the return (-> quota spend).
    assert cost == 0.0
    assert platform.calls == 0
    assert [r.method for r in api.requests] == ["POST", "GET"]
    for request in api.requests:
        # Connected to the IP validated at connect time; TLS + Host stay on the name.
        assert request.url.host == _PINNED_IP
        assert request.headers["host"] == "kling.tenant.example"
        assert request.extensions["sni_hostname"] == "kling.tenant.example"
        assert request.headers["authorization"] == f"Bearer {TENANT_KEY}"
    assert [httpx.URL(u).host for u in api.validated] == ["kling.tenant.example"] * 2
    assert json.loads(api.requests[0].content)["model_name"] == "kling-v1-5"

    # Metered as tenant-funded BYO -- never at the platform's provider price.
    assert len(cost_events) == 1
    assert cost_events[0]["provider"] == "kling"
    _assert_tenant_byo(cost_events[0], slot="video_gen", kind="kling", version=3)

    db_session.expunge_all()
    after = await _raw_row(db_session, "video_generation_jobs", job_id)
    assert after["status"] == "completed"
    assert after["provider_secret_sealed"] is None  # purged on terminal
    assert after["provider_slot"]["ref"]["connection_id"] == _CONNECTION  # audit kept
    assert after["cost"] == 0.0  # never the platform's provider price


async def test_video_slot_job_is_outside_the_platform_budget_and_cost_metric(
    db_session, keyring, fake_http, cost_events, monkeypatch
):
    """BYO is not platform spend: a budget below the platform price does not block it, the row
    records cost 0, and the platform cost metric is fed 0 (so it never counts)."""
    metric_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "src.services.video_generation_service.record_job_completed",
        lambda **kw: metric_calls.append(kw),
    )
    fake_http(_kling_ok)

    class _PricedPlatform(_PlatformVideoProvider):
        # The platform price must be non-zero, or "priced at 0" is indistinguishable.
        def estimate_cost(self, duration_s: float) -> float:
            return 0.5

    svc = _video_service(_PricedPlatform())
    job_id = await _create_video_slot_job(db_session, _video_slot())

    cost = await svc.process_job(db_session, job_id, TENANT_A, budget_limit=0.1)

    assert cost == 0.0
    db_session.expunge_all()
    row = await _raw_row(db_session, "video_generation_jobs", job_id)
    assert row["status"] == "completed", row["error_message"]
    assert row["cost"] == 0.0
    assert [c["cost"] for c in metric_calls] == [0.0]
    assert len(cost_events) == 1
    _assert_tenant_byo(cost_events[0], slot="video_gen", kind="kling", version=3)


async def test_video_slot_auth_failure_is_permanent_names_the_slot_and_never_falls_back(
    db_session, keyring, fake_http
):
    api = fake_http(lambda request: httpx.Response(401, json={"error": "bad key"}))
    platform = _PlatformVideoProvider()
    svc = _video_service(platform)
    job_id = await _create_video_slot_job(db_session, _video_slot())

    cost = await svc.process_job(db_session, job_id, TENANT_A, budget_limit=5.0)

    assert cost == 0.0
    assert len(api.requests) == 1, "a 401 must not be retried"
    assert platform.calls == 0, "a tenant slot must never fall back to a platform provider"
    db_session.expunge_all()
    row = await _raw_row(db_session, "video_generation_jobs", job_id)
    assert row["status"] == "failed"
    assert f"video_gen/kling (connection {_CONNECTION} v3)" in row["error_message"]
    assert "HTTP 401" in row["error_message"]
    assert TENANT_KEY not in row["error_message"]
    assert row["provider_secret_sealed"] is None


async def test_reclaimed_slot_job_unseals_after_key_rotation(db_session, keyring, fake_http):
    """The reaper's resume path: real reclaim SQL, then the same process_job."""
    api = fake_http(_kling_ok)
    svc = _video_service(_PlatformVideoProvider())
    job_id = await _create_video_slot_job(db_session, _video_slot())

    # The pod died mid-job: processing, lease lapsed. Then the keyring rotated:
    # a new primary was prepended, the sealing key still decrypts.
    await db_session.execute(
        update(VideoGenerationJob)
        .where(VideoGenerationJob.job_id == job_id)
        .values(status="processing", lease_expires_at=datetime.now(UTC) - timedelta(seconds=30))
    )
    await db_session.commit()
    keyring(f"kb:{_KEY_B},ka:{_KEY_A}")

    claimed = await JobManager.reclaim_orphaned(
        db_session, worker_id="pod-b", lease_seconds=900, max_attempts=5
    )
    assert claimed is not None and claimed.job_id == job_id
    db_session.expunge_all()

    cost = await svc.process_job(db_session, job_id, claimed.tenant_id, budget_limit=5.0)

    assert cost == 0.0
    assert api.requests[0].headers["authorization"] == f"Bearer {TENANT_KEY}"
    db_session.expunge_all()
    assert (await _raw_row(db_session, "video_generation_jobs", job_id))["status"] == "completed"


async def test_unopenable_envelope_fails_the_job_permanently(db_session, keyring, fake_http):
    api = fake_http(_kling_ok)
    platform = _PlatformVideoProvider()
    svc = _video_service(platform)
    job_id = await _create_video_slot_job(db_session, _video_slot())
    keyring(f"kb:{_KEY_B}")  # the sealing key is gone

    cost = await svc.process_job(db_session, job_id, TENANT_A, budget_limit=5.0)

    assert cost == 0.0
    assert api.requests == [] and platform.calls == 0
    db_session.expunge_all()
    row = await _raw_row(db_session, "video_generation_jobs", job_id)
    assert row["status"] == "failed"
    assert "could not be unsealed" in row["error_message"]
    assert "video_gen/kling" in row["error_message"]


async def test_envelope_is_bound_to_its_job(db_session, keyring, fake_http):
    """A sealed secret copied onto ANOTHER job row cannot be opened there."""
    fake_http(_kling_ok)
    svc = _video_service(_PlatformVideoProvider())
    donor = await _create_video_slot_job(db_session, _video_slot())
    victim = await _create_video_slot_job(db_session, _video_slot(secret=None))
    donor_row = await _raw_row(db_session, "video_generation_jobs", donor)
    await db_session.execute(
        update(VideoGenerationJob)
        .where(VideoGenerationJob.job_id == victim)
        .values(
            provider_secret_sealed=bytes(donor_row["provider_secret_sealed"]),
            provider_slot={**donor_row["provider_slot"]},
        )
    )
    await db_session.commit()

    assert await svc.process_job(db_session, victim, TENANT_A, budget_limit=5.0) == 0.0
    db_session.expunge_all()
    row = await _raw_row(db_session, "video_generation_jobs", victim)
    assert row["status"] == "failed"
    assert "could not be unsealed" in row["error_message"]


async def test_sql_only_terminal_write_also_purges_the_secret(db_session, keyring):
    """The reclaim function fails a poison job in SQL; the trigger still purges."""
    job_id = await _create_video_slot_job(db_session, _video_slot())
    await db_session.execute(
        update(VideoGenerationJob)
        .where(VideoGenerationJob.job_id == job_id)
        .values(attempts=5, lease_expires_at=datetime.now(UTC) - timedelta(seconds=30))
    )
    await db_session.commit()

    assert (
        await JobManager.reclaim_orphaned(
            db_session, worker_id="pod-b", lease_seconds=900, max_attempts=5
        )
        is None
    )
    db_session.expunge_all()
    row = await _raw_row(db_session, "video_generation_jobs", job_id)
    assert row["status"] == "failed"
    assert row["provider_secret_sealed"] is None


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


def _image_slot(**overrides: Any) -> ImageProviderSlot:
    data: dict[str, Any] = {
        "slot": "image",
        "kind": "openai",
        "base_url": None,
        "model": None,
        "options": {},
        "secret": TENANT_KEY,
        "ref": {"connection_id": _CONNECTION, "version": 1},
    }
    data.update(overrides)
    return ImageProviderSlot.model_validate(data)


def _image_service(*, metering: bool = False) -> ImageGenService:
    registry = ImageProviderRegistry()
    registry.register(_PlatformImageProvider())
    storage = AsyncMock()
    storage.store_image.return_value = "content_generation/img.png"
    return ImageGenService(
        storage=storage,
        registry=registry,
        default_provider="openai_images",
        metering_redis=AsyncMock() if metering else None,
    )


async def _create_image_slot_job(db: Any, slot: ImageProviderSlot) -> str:
    job = await ImageJobManager.create_job(
        db, tenant_id=TENANT_A, provider="openai_images", prompt="a tenant image", size="1024x1024"
    )
    persisted, sealed = seal_slot(slot, job_id=job.job_id)
    await ImageJobManager.attach_provider_slot(
        db, job.job_id, provider_slot=persisted, provider_secret_sealed=sealed
    )
    await db.commit()
    return job.job_id


_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()


async def test_image_slot_job_uses_contract_defaults_and_the_slot_key(
    db_session, keyring, fake_http, cost_events
):
    api = fake_http(lambda request: httpx.Response(200, json={"data": [{"b64_json": _PNG_B64}]}))
    job_id = await _create_image_slot_job(db_session, _image_slot())
    assert TENANT_KEY.encode() not in _row_bytes(
        await _raw_row(db_session, "imagegen_jobs", job_id)
    )

    cost = await _image_service(metering=True).process_job(db_session, job_id, TENANT_A)

    assert cost == 0.0  # tenant's key paid -- no platform cost to record
    assert len(api.requests) == 1
    request = api.requests[0]
    # Contract defaults, NOT the platform DALLE_* settings -- validated at
    # connect time and pinned to the validated IP.
    assert api.validated == ["https://api.openai.com/v1/images/generations"]
    assert request.url.host == _PINNED_IP
    assert request.url.path == "/v1/images/generations"
    assert request.headers["host"] == "api.openai.com"
    assert request.extensions["sni_hostname"] == "api.openai.com"
    assert request.headers["authorization"] == f"Bearer {TENANT_KEY}"
    assert json.loads(request.content)["model"] == "gpt-image-1"
    db_session.expunge_all()
    row = await _raw_row(db_session, "imagegen_jobs", job_id)
    assert row["status"] == "completed"
    assert row["provider_secret_sealed"] is None
    assert row["cost"] == 0.0
    # Still metered (as BYO) even though the platform cost is 0.
    assert len(cost_events) == 1
    _assert_tenant_byo(cost_events[0], slot="image", kind="openai", version=1)


async def test_image_slot_auth_failure_is_permanent_and_names_the_slot(
    db_session, keyring, fake_http
):
    fake_http(lambda request: httpx.Response(401, json={"error": {"message": "bad key"}}))
    job_id = await _create_image_slot_job(
        db_session, _image_slot(base_url="https://images.tenant.example/v1")
    )

    assert await _image_service().process_job(db_session, job_id, TENANT_A) == 0.0

    db_session.expunge_all()
    row = await _raw_row(db_session, "imagegen_jobs", job_id)
    assert row["status"] == "failed"
    assert f"image/openai (connection {_CONNECTION} v1)" in row["error_message"]
    assert "HTTP 401" in row["error_message"]
    assert TENANT_KEY not in row["error_message"]
    assert row["provider_secret_sealed"] is None


async def test_platform_job_without_a_slot_runs_on_the_registry_provider(
    db_session, keyring, cost_events
):
    """No slot -> both new columns stay NULL and the registry provider runs, as before."""
    from src.services.image_providers.base import ImageGenResult

    class _OkPlatform(ImageGenProvider):
        name = "openai_images"
        calls = 0

        async def generate(self, **kwargs: Any) -> ImageGenResult:
            type(self).calls += 1
            return ImageGenResult(success=True, image_data=b"png", width=1, height=1)

        def estimate_cost(self, *, size: str | None = None, quality: str | None = None) -> float:
            return 0.04

    registry = ImageProviderRegistry()
    registry.register(_OkPlatform())
    storage = AsyncMock()
    storage.store_image.return_value = "content_generation/img.png"
    svc = ImageGenService(
        storage=storage,
        registry=registry,
        default_provider="openai_images",
        metering_redis=AsyncMock(),
    )
    job = await ImageJobManager.create_job(
        db_session, tenant_id=TENANT_A, provider="openai_images", prompt="platform image"
    )
    await db_session.commit()

    assert await svc.process_job(db_session, job.job_id, TENANT_A) == pytest.approx(0.04)

    assert _OkPlatform.calls == 1
    db_session.expunge_all()
    row = await _raw_row(db_session, "imagegen_jobs", job.job_id)
    assert row["status"] == "completed"
    assert row["provider_slot"] is None and row["provider_secret_sealed"] is None
    assert row["cost"] == pytest.approx(0.04)  # platform job keeps its platform price
    # Platform spend is metered exactly as before: platform price, no BYO marks.
    assert len(cost_events) == 1
    assert cost_events[0]["cost_usd"] == pytest.approx(0.04)
    assert not {"byo", "funding", "owner_provided"} & set(cost_events[0]["usage_extra"])
    assert cost_events[0]["funding"] == "platform_external"  # stated, never derived (G5)
    assert cost_events[0]["app_id"] == "w2v"


async def test_video_slot_host_resolving_private_at_connect_time_fails_permanently(
    db_session, keyring, fake_http, cost_events, monkeypatch
):
    """DNS rebinding: the enqueue-time check passed, the host now answers 10.x.

    The REAL validator runs here (only DNS is faked): the request is refused
    before any connection, not retried, never falls back, and is not metered.
    """
    import socket

    from w2v_common.net import validate_public_url

    import src.services.slot_http as slot_http

    api = fake_http(_kling_ok)
    monkeypatch.setattr(slot_http, "validate_public_url", validate_public_url)
    real_getaddrinfo = socket.getaddrinfo
    lookups: list[str] = []

    def _rebound(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host == "kling.tenant.example":
            lookups.append(host)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 0))]
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _rebound)
    platform = _PlatformVideoProvider()
    svc = _video_service(platform)
    job_id = await _create_video_slot_job(db_session, _video_slot())

    assert await svc.process_job(db_session, job_id, TENANT_A, budget_limit=5.0) == 0.0

    assert api.requests == [], "no request may leave for a private address"
    assert platform.calls == 0
    assert cost_events == []
    db_session.expunge_all()
    row = await _raw_row(db_session, "video_generation_jobs", job_id)
    assert row["status"] == "failed"
    assert lookups == ["kling.tenant.example"], "an SSRF refusal is final, not retried"
    assert f"video_gen/kling (connection {_CONNECTION} v3)" in row["error_message"]
    assert "SSRF guard" in row["error_message"]
    assert TENANT_KEY not in row["error_message"]
    assert row["provider_secret_sealed"] is None


async def test_image_slot_host_resolving_private_at_connect_time_fails_permanently(
    db_session, keyring, fake_http, cost_events, monkeypatch
):
    """Image twin of the rebinding test: the real validator refuses at connect time."""
    import socket

    from w2v_common.net import validate_public_url

    import src.services.slot_http as slot_http

    api = fake_http(lambda request: httpx.Response(200, json={"data": [{"b64_json": _PNG_B64}]}))
    monkeypatch.setattr(slot_http, "validate_public_url", validate_public_url)
    real_getaddrinfo = socket.getaddrinfo

    def _rebound(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host == "images.tenant.example":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))]
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _rebound)
    job_id = await _create_image_slot_job(
        db_session, _image_slot(base_url="https://images.tenant.example/v1")
    )

    assert await _image_service(metering=True).process_job(db_session, job_id, TENANT_A) == 0.0

    assert api.requests == []
    assert cost_events == []
    db_session.expunge_all()
    row = await _raw_row(db_session, "imagegen_jobs", job_id)
    assert row["status"] == "failed"
    assert f"image/openai (connection {_CONNECTION} v1)" in row["error_message"]
    assert "SSRF guard" in row["error_message"]
    assert TENANT_KEY not in row["error_message"]
    assert row["provider_secret_sealed"] is None
