"""``W2V_PROVIDER_SLOTS`` on the single-job POSTs, against the real Postgres.

Phase 2 of run provider slots: under ``enforce`` ACG resolves the run's
``image`` / ``video_gen`` slot itself (``w2v_common.providers.resolve_provider_slot``,
faked here at the import site) and IGNORES any body ``provider_slot``; under
``off`` (the default) the body slot is honoured exactly as before.

Pinned through the real routes, job managers, unique index and purge trigger:

* off   -> body slot honoured, resolve never called;
* enforce -> the RESOLVED slot runs the job, the body slot (different kind,
  different secret, even an internal base_url) is ignored;
* enforce + platform answer -> a platform job, body slot ignored;
* neither secret is ever readable in the raw row (only a job-bound AES-GCM
  envelope, which opens to the RESOLVED secret);
* ACG #144 semantics unchanged for a tenant job: the same Idempotency-Key
  replays the original job with NO second resolve and NO second task; cancel
  purges the sealed secret and frees the key for a fresh attempt.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from w2v_common.providers import ProviderSlot as ResolvedSlot
from w2v_common.providers import ProviderSlotRef

from src.config import clear_settings_cache
from src.database.models import ImageGenJob, VideoGenerationJob
from src.services.provider_secrets import Keyring
from tests.conftest import TENANT_A
from tests.integration.test_idempotency_key import _build_app, _TaskCounter

RESOLVED_SECRET = "sk-resolved-run-slot-0123456789abcdef"  # pragma: allowlist secret
BODY_SECRET = "sk-body-slot-must-be-ignored-fedcba98"  # pragma: allowlist secret
_KEY = base64.b64encode(b"E" * 32).decode()
_RESOLVED_CONN = uuid.UUID("0195f48b-7777-7000-8000-00000000beef")
_BODY_CONN = "0195f48b-5555-7000-8000-00000000c0de"
_RESOLVE = "src.api.routes.dependencies.resolve_provider_slot"

_IMAGE_URL = "/api/concept-images"
_VIDEO_URL = "/api/content/videos/generate"


def _body_slot(slot: str, kind: str) -> dict[str, Any]:
    return {
        "slot": slot,
        "kind": kind,
        "base_url": "http://10.0.0.7/v1",  # internal: a 400 if it were honoured
        "model": None,
        "options": {},
        "secret": BODY_SECRET,
        "ref": {"connection_id": _BODY_CONN, "version": 1},
    }


# (url, model, resolved slot name, resolved kind, row provider, body with a DIFFERENT slot)
_CASES = [
    pytest.param(
        _IMAGE_URL,
        ImageGenJob,
        "image",
        "openai",
        "openai_images",
        {"prompt": "a lighthouse", "provider": "stub"},
        "openai_compatible",
        id="image",
    ),
    pytest.param(
        _VIDEO_URL,
        VideoGenerationJob,
        "video_gen",
        "kling",
        "kling",
        {"prompt": "waves", "provider": "fake", "duration_s": 5},
        "runway",
        id="video",
    ),
]


def _resolved(slot: str, kind: str) -> ResolvedSlot:
    return ResolvedSlot(
        slot=slot,
        source="tenant",
        kind=kind,
        secret=SecretStr(RESOLVED_SECRET),
        ref=ProviderSlotRef(connection_id=_RESOLVED_CONN, version=4),
    )


@pytest.fixture
def keyring(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("PROVIDER_SECRET_KEYS", f"ke:{_KEY}")
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
async def harness(
    db_session_factory: async_sessionmaker[AsyncSession], registry: Any, keyring: None
) -> AsyncIterator[tuple[AsyncClient, _TaskCounter]]:
    counter = _TaskCounter()
    app = _build_app(db_session_factory, registry)
    with patch("asyncio.create_task", side_effect=counter):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac, counter


def _post(ac: AsyncClient, url: str, body: dict[str, Any], key: str | None = None) -> Any:
    headers = {"x-test-tenant": TENANT_A}
    if key is not None:
        headers["Idempotency-Key"] = key
    return ac.post(url, json=body, headers=headers)


async def _raw_row(factory: async_sessionmaker[AsyncSession], model: Any, job_id: str) -> str:
    """The WHOLE row as Postgres renders it (every column, bytea as hex + escape)."""
    table = model.__tablename__
    async with factory() as session:
        row = (
            await session.execute(
                text(
                    f"SELECT t::text, coalesce(encode(t.provider_secret_sealed, 'escape'), '') "  # noqa: S608
                    f"FROM {table} t WHERE job_id = :j"
                ),
                {"j": job_id},
            )
        ).one()
    return f"{row[0]}\n{row[1]}"


async def _job(factory: async_sessionmaker[AsyncSession], model: Any, job_id: str) -> Any:
    async with factory() as session:
        return (await session.execute(select(model).where(model.job_id == job_id))).scalar_one()


@pytest.mark.parametrize(("url", "model", "slot", "kind", "provider", "body", "body_kind"), _CASES)
async def test_off_honours_the_body_slot_and_never_resolves(
    harness, db_session_factory, monkeypatch, url, model, slot, kind, provider, body, body_kind
) -> None:
    monkeypatch.setenv("W2V_PROVIDER_SLOTS", "off")
    ac, counter = harness
    sent = {**body, "provider_slot": {**_body_slot(slot, kind), "base_url": None}}
    with patch(_RESOLVE, new=AsyncMock()) as resolve:
        resp = await _post(ac, url, sent)

    assert resp.status_code == 200, resp.text
    resolve.assert_not_called()
    job = await _job(db_session_factory, model, resp.json()["job_id"])
    assert job.provider == provider
    assert job.provider_slot["ref"]["connection_id"] == _BODY_CONN
    sealed = job.provider_secret_sealed
    assert Keyring.from_env_value(f"ke:{_KEY}").decrypt(sealed, job_id=job.job_id) == BODY_SECRET
    assert BODY_SECRET not in await _raw_row(db_session_factory, model, job.job_id)
    assert counter.started == 1


@pytest.mark.parametrize(("url", "model", "slot", "kind", "provider", "body", "body_kind"), _CASES)
async def test_enforce_runs_on_the_resolved_slot_and_ignores_the_body_slot(
    harness, db_session_factory, monkeypatch, url, model, slot, kind, provider, body, body_kind
) -> None:
    monkeypatch.setenv("W2V_PROVIDER_SLOTS", "enforce")
    ac, counter = harness
    sent = {**body, "provider_slot": _body_slot(slot, body_kind)}
    with patch(_RESOLVE, new=AsyncMock(return_value=_resolved(slot, kind))) as resolve:
        resp = await _post(ac, url, sent)

    # 200, not the 400 the body slot's internal base_url would earn if honoured.
    assert resp.status_code == 200, resp.text
    resolve.assert_awaited_once_with(slot)
    assert RESOLVED_SECRET not in resp.text and BODY_SECRET not in resp.text
    job = await _job(db_session_factory, model, resp.json()["job_id"])
    assert job.provider == provider
    assert job.provider_slot["kind"] == kind
    assert job.provider_slot["ref"] == {"connection_id": str(_RESOLVED_CONN), "version": 4}
    assert job.provider_slot["has_secret"] is True
    # At rest: only a job-bound envelope, which opens to the RESOLVED secret.
    raw = await _raw_row(db_session_factory, model, job.job_id)
    assert RESOLVED_SECRET not in raw
    assert BODY_SECRET not in raw
    keyring = Keyring.from_env_value(f"ke:{_KEY}")
    assert keyring.decrypt(job.provider_secret_sealed, job_id=job.job_id) == RESOLVED_SECRET
    assert counter.started == 1


@pytest.mark.parametrize(("url", "model", "slot", "kind", "provider", "body", "body_kind"), _CASES)
async def test_enforce_platform_answer_is_a_platform_job_even_with_a_body_slot(
    harness, db_session_factory, monkeypatch, url, model, slot, kind, provider, body, body_kind
) -> None:
    monkeypatch.setenv("W2V_PROVIDER_SLOTS", "enforce")
    ac, _ = harness
    platform = ResolvedSlot(slot=slot, source="deployment", kind="platform")
    sent = {**body, "provider_slot": _body_slot(slot, body_kind)}
    with patch(_RESOLVE, new=AsyncMock(return_value=platform)):
        resp = await _post(ac, url, sent)

    assert resp.status_code == 200, resp.text
    job = await _job(db_session_factory, model, resp.json()["job_id"])
    assert job.provider == body["provider"]
    assert job.provider_slot is None
    assert job.provider_secret_sealed is None


@pytest.mark.parametrize(("url", "model", "slot", "kind", "provider", "body", "body_kind"), _CASES)
async def test_enforce_same_key_retry_replays_without_a_second_resolve(
    harness, db_session_factory, monkeypatch, url, model, slot, kind, provider, body, body_kind
) -> None:
    monkeypatch.setenv("W2V_PROVIDER_SLOTS", "enforce")
    ac, counter = harness
    sent = {**body, "provider_slot": _body_slot(slot, body_kind)}
    with patch(_RESOLVE, new=AsyncMock(return_value=_resolved(slot, kind))) as resolve:
        first = await _post(ac, url, sent, key="run1:stage:m1")
        second = await _post(ac, url, sent, key="run1:stage:m1")

    assert first.status_code == second.status_code == 200, second.text
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.headers.get("Idempotent-Replayed") == "true"
    assert resolve.await_count == 1
    assert counter.started == 1


async def test_enforce_cancel_purges_the_sealed_secret_and_frees_the_key(
    harness, db_session_factory, monkeypatch
) -> None:
    monkeypatch.setenv("W2V_PROVIDER_SLOTS", "enforce")
    ac, counter = harness
    body = {"prompt": "a lighthouse", "provider": "stub"}
    with patch(_RESOLVE, new=AsyncMock(return_value=_resolved("image", "openai"))):
        job_id = (await _post(ac, _IMAGE_URL, body, key="cancel-me")).json()["job_id"]
        assert (await _job(db_session_factory, ImageGenJob, job_id)).provider_secret_sealed

        cancelled = await ac.post(
            f"/api/concept-images/jobs/{job_id}/cancel", headers={"x-test-tenant": TENANT_A}
        )
        assert cancelled.status_code == 200, cancelled.text
        job = await _job(db_session_factory, ImageGenJob, job_id)
        assert job.status == "cancelled"
        assert job.provider_secret_sealed is None  # purge trigger on terminal

        retry = await _post(ac, _IMAGE_URL, body, key="cancel-me")

    assert retry.status_code == 200, retry.text
    assert retry.json()["job_id"] != job_id
    assert "Idempotent-Replayed" not in retry.headers
    assert counter.started == 2
