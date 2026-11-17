"""The reaper's re-dispatch must carry the ENQUEUE-TIME tenant provenance.

``ReaperService`` reclaims jobs across ALL tenants and re-dispatches them through
the same ``_process_*_bg`` entrypoints the request path uses. Those entrypoints
re-seed the tenant for RLS, and provenance is bound to a tenant: re-seeding a
DIFFERENT tenant drops the inherited ``TenantSource`` to ``None``, which
``build_outbound_identity`` reports as ``'unrecorded'`` and refuses for a metered
destination (modality_gateway). So on the reaper path the ambient ContextVar is
either empty — the re-dispatch fails closed at its first paid call — or it holds
the PREVIOUS row's tenant, which would be worse.

The persisted job row is the only honest source, and nothing was persisting it:
``build_*_options`` carried ``budget_limit``/``callback_url``/``batch_id`` and no
provenance at all.

These tests assert the provenance VALUE actually established inside the real
``_process_*_bg`` function — read back out of the ContextVar the real code set —
not that some mock was called. A call-count assertion here would go green against
a reaper that passed the wrong tenant's provenance.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from w2v_common.auth.identity_context import (
    clear_identity_context,
    get_tenant_source_from_context,
    set_identity_context,
)
from w2v_common.tenant import TenantSource

from src.services.durable_dispatch import (
    build_image_batch_options,
    build_image_single_options,
    build_video_batch_options,
    build_video_single_options,
    restore_tenant_source,
)
from src.services.job_manager import ClaimedJob
from src.services.reaper_service import ReaperService

# The tenant whose job was enqueued, and an unrelated one the reaper loop happens
# to be carrying when it reaches that row. They must be DIFFERENT: with the same
# tenant, ambient inheritance would mask a missing restore and the test would pass
# against the bug.
ENQUEUE_TENANT = "e148e8af-3751-49bf-a0fa-56ab46d31ddc"
OTHER_TENANT = "b3d4c5e6-0000-4a1b-9c2d-7f8e9a0b1c2d"


@pytest.fixture(autouse=True)
def _clean_identity():
    clear_identity_context()
    yield
    clear_identity_context()


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: build_video_single_options(budget_limit=3.0, callback_url=None),
        lambda: build_video_batch_options(batch_id="b", budget_limit=3.0),
        lambda: build_image_single_options(budget_limit=3.0),
        lambda: build_image_batch_options(batch_id="b", budget_limit=3.0),
    ],
    ids=["video_single", "video_batch", "image_single", "image_batch"],
)
def test_every_options_bundle_persists_the_enqueue_provenance(build) -> None:
    """All four dispatch kinds, because the reaper re-dispatches all four."""
    set_identity_context(ENQUEUE_TENANT, None, None, tenant_source=TenantSource.LICENSE_TOKEN)

    options = build()

    assert restore_tenant_source(options.get("tenant_source")) is TenantSource.LICENSE_TOKEN, (
        "provenance was not persisted on the job row — the reaper has nothing to "
        "restore and every metered hop of the re-dispatch fails closed"
    )


def test_an_unvouched_enqueue_round_trips_as_unvouched() -> None:
    """The bundle records what happened; it does not upgrade it."""
    set_identity_context(ENQUEUE_TENANT, None, None, tenant_source=TenantSource.UNVOUCHED_HEADER)

    options = build_image_single_options(budget_limit=1.0)

    assert restore_tenant_source(options["tenant_source"]) is TenantSource.UNVOUCHED_HEADER


def test_an_enqueue_with_no_provenance_stores_an_empty_marker() -> None:
    clear_identity_context()

    options = build_image_single_options(budget_limit=1.0)

    assert options["tenant_source"] == ""
    assert restore_tenant_source(options["tenant_source"]) is None


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", 12, "not_a_real_source", {"a": 1}])
def test_restore_never_guesses_a_provenance(value) -> None:
    """A legacy row, or a garbled one, restores as unvouched — never as vouched.

    Fail-closed is the point: substituting ``LICENSE_TOKEN`` for "I don't know"
    would forge the vouch and spend a tenant's credentials on provenance this
    process cannot stand behind.
    """
    assert restore_tenant_source(value) is None


def test_restore_accepts_the_value_the_builder_wrote() -> None:
    for source in TenantSource:
        assert restore_tenant_source(source.value) is source


# ---------------------------------------------------------------------------
# The crossing itself — what the bg task actually ends up bound to
# ---------------------------------------------------------------------------


class _StopAtTheSessionError(Exception):
    """Raised from the patched session factory to end the bg task early."""


def _reaper(app: FastAPI) -> ReaperService:
    return ReaperService(
        app, interval_s=1.0, lease_seconds=900, max_attempts=5, heartbeat_enabled=False
    )


def _record_provenance_then_stop(seen: list):
    """A ``get_db_session`` stand-in that captures the bg task's bound provenance.

    This reads the ContextVar the REAL ``_process_*_bg`` just set, one statement
    after the seeding line — which is the value every metered outbound call in
    that task will be made under.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _factory():
        seen.append(get_tenant_source_from_context())
        raise _StopAtTheSessionError
        yield  # pragma: no cover — unreachable, keeps this a generator

    return _factory


async def _drive(monkeypatch, module, dispatch, claimed) -> TenantSource | None:
    seen: list = []
    monkeypatch.setattr(module, "get_db_session", _record_provenance_then_stop(seen))
    # The reaper loop is standing on ANOTHER tenant when it reaches this row.
    set_identity_context(OTHER_TENANT, None, None, tenant_source=TenantSource.LICENSE_TOKEN)
    with pytest.raises(_StopAtTheSessionError):
        await dispatch(claimed)
    assert seen, "the bg task never reached its session; nothing was observed"
    return seen[0]


async def test_reclaimed_image_job_runs_under_the_enqueue_tenants_provenance(monkeypatch):
    from src.api import content_routes

    app = FastAPI()
    app.state.imagegen_service = MagicMock()
    claimed = ClaimedJob(
        job_id=str(uuid.uuid7()),
        tenant_id=ENQUEUE_TENANT,
        options={
            "kind": "image_single",
            "budget_limit": 2.0,
            "tenant_source": TenantSource.LICENSE_TOKEN.value,
        },
    )

    bound = await _drive(monkeypatch, content_routes, _reaper(app)._dispatch_image, claimed)

    assert bound is TenantSource.LICENSE_TOKEN, (
        "the re-dispatched image job is bound to "
        f"{bound!r} instead of the provenance persisted at enqueue — modality_gateway "
        "refuses this as 'unrecorded'"
    )


async def test_reclaimed_video_job_runs_under_the_enqueue_tenants_provenance(monkeypatch):
    from src.api.routes import generation

    app = FastAPI()
    app.state.video_generation_service = MagicMock()
    claimed = ClaimedJob(
        job_id=str(uuid.uuid7()),
        tenant_id=ENQUEUE_TENANT,
        options={
            "kind": "video_single",
            "budget_limit": 2.0,
            "callback_url": None,
            "tenant_source": TenantSource.UNVOUCHED_HEADER.value,
        },
    )

    bound = await _drive(monkeypatch, generation, _reaper(app)._dispatch_video, claimed)

    assert bound is TenantSource.UNVOUCHED_HEADER, (
        f"the re-dispatched video job is bound to {bound!r}, not the enqueue-time "
        "provenance — a restored value must never be upgraded or inherited"
    )


async def test_a_legacy_row_without_provenance_fails_closed(monkeypatch):
    """A job enqueued before this key existed must NOT inherit the loop's tenant.

    This is the assertion that makes the fix a fix rather than a shuffle: the
    reaper is standing on OTHER_TENANT with LICENSE_TOKEN bound, and the
    re-dispatch of ENQUEUE_TENANT's legacy row must still come out unvouched.
    """
    from src.api import content_routes

    app = FastAPI()
    app.state.imagegen_service = MagicMock()
    claimed = ClaimedJob(
        job_id=str(uuid.uuid7()),
        tenant_id=ENQUEUE_TENANT,
        options={"kind": "image_single", "budget_limit": 2.0},
    )

    bound = await _drive(monkeypatch, content_routes, _reaper(app)._dispatch_image, claimed)

    assert bound is None, (
        f"a legacy row was vouched as {bound!r} — that is another tenant's proof "
        "being spent on this one"
    )
