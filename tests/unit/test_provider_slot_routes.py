"""``provider_slot`` at the API boundary (P5-2): validation, fail-closed, sealing.

The two single-job endpoints accept an optional tenant slot. These tests pin:

* unknown ``kind`` -> 422, and a 422 never echoes the slot's secret;
* an empty ``PROVIDER_SECRET_KEYS`` keyring -> 503 before any job exists;
* with a keyring, the job row gets the slot's provider, no fallbacks, the
  non-secret metadata and a SEALED secret -- and the response never carries it;
* an internal ``base_url`` -> 400 (same SSRF rule as every caller URL);
* a slot inside a batch item -> 422 (single-job contract; ignoring it would
  silently run on a platform provider);
* no slot -> nothing new happens (the slot persistence is never called).
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from w2v_common.asgi.error_envelope import register_error_envelope_handlers

from src.api.content_routes import router as image_router
from src.api.routes import router as video_router
from src.config import clear_settings_cache
from src.database.session import get_db
from src.services.provider_secrets import Keyring
from src.services.provider_slots import ProviderSlot

TENANT = "0195f48b-1234-7000-8000-0000000000aa"
SECRET = "sk-tenant-route-secret-abcdef0123456789"  # pragma: allowlist secret
_KEY = base64.b64encode(b"K" * 32).decode()


def _slot(slot: str, kind: str, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "slot": slot,
        "kind": kind,
        "base_url": None,
        "model": None,
        "options": {},
        "secret": SECRET,
        "ref": {"connection_id": "0195f48b-5555-7000-8000-00000000c0de", "version": 2},
    }
    data.update(overrides)
    return data


@pytest.fixture
def keyring_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("PROVIDER_SECRET_KEYS", f"k1:{_KEY}")
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def no_keyring(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("PROVIDER_SECRET_KEYS", raising=False)
    clear_settings_cache()
    yield
    clear_settings_cache()


def _app(*, video_service: Any = None, image_service: Any = None) -> FastAPI:
    app = FastAPI()
    register_error_envelope_handlers(app)

    @app.middleware("http")
    async def _tenant(request: Any, call_next: Any) -> Any:  # pragma: no cover - harness
        request.state.tenant_id = TENANT
        return await call_next(request)

    app.include_router(video_router)
    app.include_router(image_router)
    app.state.video_generation_service = video_service or MagicMock()
    app.state.imagegen_service = image_service or MagicMock()
    # An EMPTY platform registry: a slot request must not need any platform provider.
    for attr in ("provider_registry", "image_provider_registry"):
        registry = MagicMock()
        registry.get.return_value = None
        registry.provider_names = []
        setattr(app.state, attr, registry)
    app.state.storage_service = MagicMock()
    app.state.batch_tracker = MagicMock()
    db = AsyncMock()
    app.dependency_overrides[get_db] = lambda: db
    return app


def _service(job_id: str) -> MagicMock:
    service = MagicMock()
    service.create_job = AsyncMock(return_value=job_id)
    service.register_task = MagicMock()
    return service


def _close(coro: Any) -> MagicMock:
    coro.close()
    return MagicMock()


async def _post(app: FastAPI, path: str, body: dict[str, Any]) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        return await ac.post(path, json=body)


# ---------------------------------------------------------------------------
# Video: POST /api/content/videos/generate
# ---------------------------------------------------------------------------


async def test_video_unknown_kind_is_422_and_never_echoes_the_secret(keyring_env) -> None:
    service = _service("job-v")
    resp = await _post(
        _app(video_service=service),
        "/api/content/videos/generate",
        {"prompt": "p", "provider_slot": _slot("video_gen", "pika")},
    )
    assert resp.status_code == 422
    assert SECRET not in resp.text
    service.create_job.assert_not_called()


async def test_video_slot_missing_ref_422_does_not_echo_the_secret(keyring_env) -> None:
    body = _slot("video_gen", "kling")
    del body["ref"]  # a whole-object error: pydantic's ``input`` would be the slot
    resp = await _post(
        _app(), "/api/content/videos/generate", {"prompt": "p", "provider_slot": body}
    )
    assert resp.status_code == 422
    assert SECRET not in resp.text


async def test_video_slot_on_the_image_endpoint_is_422(keyring_env) -> None:
    resp = await _post(
        _app(), "/api/concept-images", {"prompt": "p", "provider_slot": _slot("video_gen", "kling")}
    )
    assert resp.status_code == 422


async def test_video_slot_without_keyring_is_503_and_creates_nothing(no_keyring) -> None:
    service = _service("job-v")
    resp = await _post(
        _app(video_service=service),
        "/api/content/videos/generate",
        {"prompt": "p", "provider_slot": _slot("video_gen", "kling")},
    )
    assert resp.status_code == 503
    service.create_job.assert_not_called()


async def test_video_slot_internal_base_url_is_400(keyring_env) -> None:
    resp = await _post(
        _app(),
        "/api/content/videos/generate",
        {
            "prompt": "p",
            "provider_slot": _slot("video_gen", "kling", base_url="http://10.0.0.7/v1"),
        },
    )
    assert resp.status_code == 400


async def test_video_slot_job_is_created_on_the_slot_provider_with_a_sealed_secret(
    keyring_env,
) -> None:
    service = _service("job-v")
    with (
        patch(
            "src.api.routes.generation.JobManager.attach_provider_slot", new=AsyncMock()
        ) as attach,
        patch("src.api.routes.generation.JobManager.register_dispatch", new=AsyncMock()),
        patch("src.api.routes.generation.asyncio.create_task", side_effect=_close),
    ):
        resp = await _post(
            _app(video_service=service),
            "/api/content/videos/generate",
            {
                "prompt": "p",
                "provider": "sora",  # ignored: the slot replaces provider selection
                "fallback_providers": ["sora"],  # ignored too: slots never fall back
                "provider_slot": _slot("video_gen", "kling"),
            },
        )

    assert resp.status_code == 200, resp.text
    assert SECRET not in resp.text
    kwargs = service.create_job.await_args.kwargs
    assert kwargs["provider"] == "kling"
    assert kwargs["fallback_providers"] is None
    attach.assert_awaited_once()
    assert attach.await_args is not None
    persisted = attach.await_args.kwargs["provider_slot"]
    sealed = attach.await_args.kwargs["provider_secret_sealed"]
    assert persisted["kind"] == "kling" and persisted["has_secret"] is True
    assert SECRET not in repr(persisted)
    assert SECRET.encode() not in sealed
    # The envelope opens for THIS job id only.
    assert Keyring.from_env_value(f"k1:{_KEY}").decrypt(sealed, job_id="job-v") == SECRET


async def test_video_slot_in_a_batch_item_is_422(keyring_env) -> None:
    resp = await _post(
        _app(),
        "/api/content/videos/generate/batch",
        {"items": [{"prompt": "p", "provider_slot": _slot("video_gen", "kling")}]},
    )
    assert resp.status_code == 422


async def test_video_without_slot_never_touches_slot_persistence(keyring_env) -> None:
    service = _service("job-v")
    registry = MagicMock()
    provider = MagicMock()
    provider.get_max_duration_s.return_value = None
    registry.get.return_value = provider
    app = _app(video_service=service)
    app.state.provider_registry = registry
    with (
        patch(
            "src.api.routes.generation.JobManager.attach_provider_slot", new=AsyncMock()
        ) as attach,
        patch("src.api.routes.generation.JobManager.register_dispatch", new=AsyncMock()),
        patch("src.api.routes.generation.asyncio.create_task", side_effect=_close),
    ):
        resp = await _post(
            app,
            "/api/content/videos/generate",
            {"prompt": "p", "provider": "sora", "fallback_providers": ["runway"]},
        )
    assert resp.status_code == 200, resp.text
    attach.assert_not_called()
    kwargs = service.create_job.await_args.kwargs
    assert kwargs["provider"] == "sora"
    assert kwargs["fallback_providers"] == ["runway"]


# ---------------------------------------------------------------------------
# Image: POST /api/concept-images
# ---------------------------------------------------------------------------


async def test_image_unknown_kind_is_422(keyring_env) -> None:
    service = _service("job-i")
    resp = await _post(
        _app(image_service=service),
        "/api/concept-images",
        {"prompt": "p", "provider_slot": _slot("image", "midjourney")},
    )
    assert resp.status_code == 422
    assert SECRET not in resp.text
    service.create_job.assert_not_called()


async def test_image_slot_without_keyring_is_503(no_keyring) -> None:
    service = _service("job-i")
    resp = await _post(
        _app(image_service=service),
        "/api/concept-images",
        {"prompt": "p", "provider_slot": _slot("image", "openai")},
    )
    assert resp.status_code == 503
    service.create_job.assert_not_called()


async def test_image_slot_job_is_created_with_a_sealed_secret(keyring_env) -> None:
    service = _service("job-i")
    with (
        patch(
            "src.api.content_routes.ImageJobManager.attach_provider_slot", new=AsyncMock()
        ) as attach,
        patch("src.api.content_routes.ImageJobManager.register_dispatch", new=AsyncMock()),
        patch("src.api.content_routes.asyncio.create_task", side_effect=_close),
    ):
        resp = await _post(
            _app(image_service=service),
            "/api/concept-images",
            {"prompt": "p", "provider": "local_image", "provider_slot": _slot("image", "openai")},
        )
    assert resp.status_code == 200, resp.text
    assert SECRET not in resp.text
    assert service.create_job.await_args.kwargs["provider"] == "openai_images"
    assert attach.await_args is not None
    sealed = attach.await_args.kwargs["provider_secret_sealed"]
    assert SECRET.encode() not in sealed
    assert Keyring.from_env_value(f"k1:{_KEY}").decrypt(sealed, job_id="job-i") == SECRET


async def test_image_slot_in_a_batch_item_is_422(keyring_env) -> None:
    resp = await _post(
        _app(),
        "/api/concept-images/batches",
        {"items": [{"prompt": "p", "provider_slot": _slot("image", "openai")}]},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Model-level redaction
# ---------------------------------------------------------------------------


def test_slot_repr_and_persisted_form_never_carry_the_secret() -> None:
    slot = ProviderSlot.model_validate(_slot("image", "openai"))
    assert SECRET not in repr(slot)
    assert SECRET not in str(slot)
    assert SECRET not in repr(slot.to_persisted())
    assert slot.secret is not None and slot.secret.get_secret_value() == SECRET


# ---------------------------------------------------------------------------
# BYO spend: a slot job is not platform spend at admission
# ---------------------------------------------------------------------------


def _exhausted_quota() -> Any:
    """A REAL QuotaService whose tenant is already OVER the platform daily cap."""
    from src.services.quota_service import QuotaService

    redis = AsyncMock()

    async def fake_get(key: str) -> Any:
        if "daily" in key:
            return {"generations": 0, "spend": 55.0}
        return 0

    redis.get.side_effect = fake_get
    return QuotaService(redis_client=redis, max_daily_spend=50.0)


async def test_video_slot_is_admitted_past_an_exhausted_platform_spend_cap(keyring_env) -> None:
    service = _service("job-v")
    service.estimate_request_cost = MagicMock(return_value=99.0)
    app = _app(video_service=service)
    app.state.quota_service = _exhausted_quota()
    app.state.provider_registry.get.return_value = MagicMock()  # control: a known platform
    app.state.provider_registry.provider_names = ["sora"]
    with (
        patch("src.api.routes.generation.JobManager.attach_provider_slot", new=AsyncMock()),
        patch("src.api.routes.generation.JobManager.register_dispatch", new=AsyncMock()),
        patch("src.api.routes.generation.asyncio.create_task", side_effect=_close),
        patch.object(app.state.quota_service, "record_generation_start", new=AsyncMock()),
    ):
        slot_resp = await _post(
            app,
            "/api/content/videos/generate",
            {"prompt": "p", "provider_slot": _slot("video_gen", "kling")},
        )
        platform_resp = await _post(
            app, "/api/content/videos/generate", {"prompt": "p", "provider": "sora"}
        )

    assert slot_resp.status_code == 200, slot_resp.text
    # Control: the same exhausted cap still refuses a platform job.
    assert platform_resp.status_code == 429, platform_resp.text
    # The tenant's key is never priced at platform rates: only the control was.
    service.estimate_request_cost.assert_called_once()


async def test_image_slot_is_admitted_past_an_exhausted_platform_spend_cap(keyring_env) -> None:
    service = _service("job-i")
    service.estimate_request_cost = MagicMock(return_value=99.0)
    app = _app(image_service=service)
    app.state.quota_service = _exhausted_quota()
    app.state.image_provider_registry.get.return_value = MagicMock()  # control
    app.state.image_provider_registry.provider_names = ["local_image"]
    with (
        patch("src.api.content_routes.ImageJobManager.attach_provider_slot", new=AsyncMock()),
        patch("src.api.content_routes.ImageJobManager.register_dispatch", new=AsyncMock()),
        patch("src.api.content_routes.asyncio.create_task", side_effect=_close),
        patch.object(app.state.quota_service, "record_generation_start", new=AsyncMock()),
    ):
        slot_resp = await _post(
            app, "/api/concept-images", {"prompt": "p", "provider_slot": _slot("image", "openai")}
        )
        platform_resp = await _post(
            app, "/api/concept-images", {"prompt": "p", "provider": "local_image"}
        )

    assert slot_resp.status_code == 200, slot_resp.text
    assert platform_resp.status_code == 429, platform_resp.text
    service.estimate_request_cost.assert_called_once()


# ---------------------------------------------------------------------------
# W2V_PROVIDER_SLOTS=enforce: ACG resolves the run's slot; body slot ignored
# (real-DB coverage: tests/integration/test_provider_slots_enforce.py)
# ---------------------------------------------------------------------------

_RESOLVE = "src.api.routes.dependencies.resolve_provider_slot"


def _resolve_raising(exc: BaseException) -> AsyncMock:
    return AsyncMock(side_effect=exc)


def _enforce_errors() -> list[Any]:
    from w2v_common.providers import (
        ProviderHopRefusedError,
        ProviderSlotConfigError,
        ProviderSlotRequiredError,
        ProviderSlotUnavailableError,
    )
    from w2v_common.tenant import UnvouchedTenantError

    return [
        pytest.param(ProviderSlotRequiredError("image"), 403, id="byo-required"),
        pytest.param(
            ProviderSlotUnavailableError("image", "rolling", transient=True), 503, id="transient"
        ),
        pytest.param(ProviderSlotUnavailableError("image", "revoked"), 409, id="permanent"),
        pytest.param(
            ProviderSlotConfigError("image", "cell_lane_required", status=403), 503, id="config"
        ),
        pytest.param(ProviderHopRefusedError("image"), 403, id="hop"),
        pytest.param(UnvouchedTenantError("unvouched"), 403, id="unvouched"),
    ]


@pytest.mark.parametrize(("exc", "status"), _enforce_errors())
@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/concept-images", {"prompt": "p", "provider_slot": _slot("image", "openai")}),
        (
            "/api/content/videos/generate",
            {"prompt": "p", "provider_slot": _slot("video_gen", "kling")},
        ),
    ],
)
async def test_enforce_resolve_failure_never_falls_back_to_the_body_or_platform(
    keyring_env, monkeypatch, exc, status, path, body
) -> None:
    monkeypatch.setenv("W2V_PROVIDER_SLOTS", "enforce")
    video, image = _service("job-v"), _service("job-i")
    with patch(_RESOLVE, new=_resolve_raising(exc)):
        resp = await _post(_app(video_service=video, image_service=image), path, body)
    assert resp.status_code == status, resp.text
    assert SECRET not in resp.text
    video.create_job.assert_not_called()
    image.create_job.assert_not_called()


async def test_enforce_byo_required_carries_the_orchestrator_failure_detail(
    keyring_env, monkeypatch
) -> None:
    from w2v_common.providers import ProviderSlotRequiredError

    monkeypatch.setenv("W2V_PROVIDER_SLOTS", "enforce")
    with patch(_RESOLVE, new=_resolve_raising(ProviderSlotRequiredError("image"))):
        resp = await _post(_app(), "/api/concept-images", {"prompt": "p"})
    assert resp.status_code == 403
    assert "byo_provider_required" in resp.text


async def test_enforce_unusable_resolved_kind_is_409_without_the_secret(
    keyring_env, monkeypatch
) -> None:
    from pydantic import SecretStr
    from w2v_common.providers import ProviderSlot as Resolved
    from w2v_common.providers import ProviderSlotRef

    monkeypatch.setenv("W2V_PROVIDER_SLOTS", "enforce")
    resolved = Resolved(
        slot="image",
        source="tenant",
        kind="midjourney",  # not an ACG image kind
        secret=SecretStr(SECRET),
        ref=ProviderSlotRef(
            connection_id=uuid.UUID(_slot("image", "x")["ref"]["connection_id"]),
            version=2,
        ),
    )
    service = _service("job-i")
    with patch(_RESOLVE, new=AsyncMock(return_value=resolved)):
        resp = await _post(_app(image_service=service), "/api/concept-images", {"prompt": "p"})
    assert resp.status_code == 409, resp.text
    assert SECRET not in resp.text
    service.create_job.assert_not_called()


async def test_a_typo_in_the_switch_fails_closed_503(keyring_env, monkeypatch) -> None:
    monkeypatch.setenv("W2V_PROVIDER_SLOTS", "enforced")  # typo: not off, not enforce
    service = _service("job-i")
    with patch(_RESOLVE, new=AsyncMock()) as resolve:
        resp = await _post(
            _app(image_service=service),
            "/api/concept-images",
            {"prompt": "p", "provider_slot": _slot("image", "openai")},
        )
    assert resp.status_code == 503
    resolve.assert_not_called()
    service.create_job.assert_not_called()


@pytest.mark.parametrize(
    ("path", "slot_name", "kind"),
    [
        ("/api/concept-images/batches", "image", "openai"),
        ("/api/content/videos/generate/batch", "video_gen", "kling"),
    ],
)
async def test_enforce_batch_for_a_run_with_a_tenant_slot_is_422(
    keyring_env, monkeypatch, path, slot_name, kind
) -> None:
    from pydantic import SecretStr
    from w2v_common.providers import ProviderSlot as Resolved
    from w2v_common.providers import ProviderSlotRef

    monkeypatch.setenv("W2V_PROVIDER_SLOTS", "enforce")
    resolved = Resolved(
        slot=slot_name,
        source="tenant",
        kind=kind,
        secret=SecretStr(SECRET),
        ref=ProviderSlotRef(
            connection_id=uuid.UUID("0195f48b-5555-7000-8000-00000000c0de"),
            version=2,
        ),
    )
    with patch(_RESOLVE, new=AsyncMock(return_value=resolved)):
        resp = await _post(_app(), path, {"items": [{"prompt": "p"}]})
    assert resp.status_code == 422, resp.text
    assert "single-job" in resp.text
    assert SECRET not in resp.text


async def test_off_never_resolves(keyring_env, monkeypatch) -> None:
    monkeypatch.delenv("W2V_PROVIDER_SLOTS", raising=False)  # default = off
    with patch(_RESOLVE, new=AsyncMock()) as resolve:
        resp = await _post(
            _app(),
            "/api/concept-images/batches",
            {"items": [{"prompt": "p", "provider_slot": _slot("image", "openai")}]},
        )
    assert resp.status_code == 422  # today's batch rule, unchanged
    resolve.assert_not_called()
