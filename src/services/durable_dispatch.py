"""Durable multi-replica dispatch helpers (claim-lease + heartbeat).

A request-time ``asyncio.create_task`` dies with its pod, orphaning the job.
This module is the durability layer shared by the enqueue paths and the reaper:

* :func:`pod_id` — stable per-pod worker identity (for ``claimed_by``).
* ``build_*_options`` — the JSONB bundle persisted on the job row so a DIFFERENT
  pod (the reaper) can reconstruct the dispatch after a crash.
* :func:`run_with_lease` — wraps a job coroutine with a lease heartbeat so the
  reaper never reclaims a job that is still actively running on a live pod.

The lease itself is stamped by ``JobManager.register_dispatch`` at enqueue and by
the SECURITY DEFINER reclaim function on reclaim; this module only renews it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from w2v_common.auth.identity_context import get_tenant_source_from_context
from w2v_common.tenant import TenantSource

from src.database.session import get_db_session

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Dispatch kinds — the reaper routes a reclaimed job by this tag.
KIND_VIDEO_SINGLE = "video_single"
KIND_VIDEO_BATCH = "video_batch"
KIND_IMAGE_SINGLE = "image_single"
KIND_IMAGE_BATCH = "image_batch"


def pod_id() -> str:
    """Stable per-pod worker id (hostname, clamped to the ``claimed_by`` width)."""
    return socket.gethostname()[:64]


# ---------------------------------------------------------------------------
# Tenant provenance across the crash boundary
# ---------------------------------------------------------------------------
#
# Provenance (``TenantSource``) records HOW a tenant was established. It is not
# bookkeeping alongside the identity — it IS part of it: ``build_outbound_identity``
# REFUSES to delegate a tenant whose provenance is ``None``, reporting it as
# ``'unrecorded'``, and every re-dispatch here makes metered outbound calls to
# modality_gateway. So a re-dispatch that cannot state provenance cannot generate.
#
# On the request path provenance lives in a ContextVar that ``asyncio.create_task``
# copies into the background task. The reaper has no such inheritance to use: it
# runs on a DIFFERENT pod, in a loop that walks rows for MANY tenants, and its
# ambient value is either unset (→ the re-dispatch fails closed) or, worse, the
# PREVIOUS row's tenant — which would vouch tenant A's proof for tenant B's spend.
# ``set_identity_context`` deliberately drops an inherited source when the tenant
# changes for exactly this reason, so there is nothing ambient to fall back on and
# there must not be.
#
# The only honest source is therefore the JOB ROW, written at enqueue while the
# request context was still live. Persist it here; read it back with
# :func:`restore_tenant_source` in the reaper. Stored as the enum's ``.value`` so
# it round-trips through JSONB, ``""`` when the enqueue had no provenance.
#
# Mirrors ``video_compiler_service.durable_dispatch``, which is this estate's
# reference implementation of the same crossing.


def _persisted_tenant_source() -> str:
    """The live request's provenance, JSONB-safe. ``""`` when there is none."""
    source = get_tenant_source_from_context()
    return source.value if source is not None else ""


def restore_tenant_source(value: object) -> TenantSource | None:
    """Re-hydrate the enqueue-time ``TenantSource`` from persisted job options.

    Returns ``None`` — never a guessed default — for a job enqueued before this
    key existed, or one carrying an unrecognised value. ``None`` is both the
    honest answer and the fail-CLOSED one: the delegation gate refuses an
    unvouched tenant, so such a job raises at its first metered call instead of
    spending some arbitrary tenant's credentials under provenance this process
    cannot vouch for. Substituting ``LICENSE_TOKEN`` here would be forging the
    vouch, which is why no default is offered.

    That is also why this needs no migration or backfill: in-flight rows have no
    ``tenant_source`` key, read back as ``None``, and behave exactly as they do
    today.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return TenantSource(value)
    except ValueError:
        # TenantSource is a plain StrEnum, so this is genuinely "unknown member",
        # not a swallowed tenant refusal. Degrade to unvouched rather than raise —
        # the delegation gate downstream is the thing that must refuse, and will.
        logger.warning("Unrecognised persisted tenant_source %r; treating as unvouched", value)
        return None


def build_video_single_options(*, budget_limit: float, callback_url: str | None) -> dict:
    """Persisted dispatch bundle for a single video job."""
    return {
        "kind": KIND_VIDEO_SINGLE,
        "budget_limit": budget_limit,
        "callback_url": callback_url,
        "tenant_source": _persisted_tenant_source(),
    }


def build_video_batch_options(*, batch_id: str, budget_limit: float) -> dict:
    """Persisted dispatch bundle for a video job that belongs to a batch."""
    return {
        "kind": KIND_VIDEO_BATCH,
        "batch_id": batch_id,
        "budget_limit": budget_limit,
        "tenant_source": _persisted_tenant_source(),
    }


def build_image_single_options(*, budget_limit: float) -> dict:
    """Persisted dispatch bundle for a single concept-image job."""
    return {
        "kind": KIND_IMAGE_SINGLE,
        "budget_limit": budget_limit,
        "tenant_source": _persisted_tenant_source(),
    }


def build_image_batch_options(*, batch_id: str, budget_limit: float) -> dict:
    """Persisted dispatch bundle for a concept-image job that belongs to a batch."""
    return {
        "kind": KIND_IMAGE_BATCH,
        "batch_id": batch_id,
        "budget_limit": budget_limit,
        "tenant_source": _persisted_tenant_source(),
    }


# ---------------------------------------------------------------------------
# Lease heartbeat
# ---------------------------------------------------------------------------

# renew_lease(db, job_id, lease_seconds) — the manager static method for the
# job's table (JobManager.renew_lease or ImageJobManager.renew_lease).
RenewLease = Callable[["AsyncSession", str, int], Awaitable[None]]


async def _heartbeat(
    job_id: str, lease_seconds: int, stop: asyncio.Event, renew: RenewLease
) -> None:
    """Renew ``job_id``'s lease every ``lease_seconds/3`` until ``stop`` is set."""
    interval = max(5, lease_seconds // 3)
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return  # stop set → job finished, terminal write owns the lease clear
        except TimeoutError:
            pass
        try:
            async with get_db_session() as db:
                await renew(db, job_id, lease_seconds)
                await db.commit()
        except Exception:  # best-effort; a missed beat just shortens the lease
            logger.warning("Lease renew failed for job %s", job_id, exc_info=True)


async def run_with_lease(
    *,
    job_id: str,
    lease_seconds: int,
    renew: RenewLease,
    work: Awaitable[None],
    heartbeat_enabled: bool = True,
) -> None:
    """Run ``work`` while heartbeating the job's lease.

    Keeps the lease fresh for the whole (possibly multi-minute) generation so a
    reaper on another pod does not reclaim a job that is still running here. If
    THIS pod dies the heartbeat stops, the lease lapses, and the reaper takes over.
    """
    if not heartbeat_enabled:
        await work
        return
    stop = asyncio.Event()
    hb = asyncio.create_task(_heartbeat(job_id, lease_seconds, stop, renew))
    try:
        await work
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            await hb
