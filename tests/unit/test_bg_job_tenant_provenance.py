"""Background jobs must PRESERVE tenant provenance, not destroy it.

Regression for the 2026-08-28 panel-image failure in the w2v_frontend E2E
post-promote matrix: modality-gateway refused to delegate tenant
``e148e8af-…`` because provenance was ``'unrecorded'``.

The trap is that this looks like a propagation bug and is not.
``asyncio.create_task`` COPIES the caller's context, so ``AuthMiddleware``'s
provenance is already present inside the task. The original defect was that
``set_identity_context`` did ``_tenant_source_var.set(tenant_source)``
UNCONDITIONALLY, so ``set_identity_context(tenant, None, None)`` did not merely
fail to set provenance — it DESTROYED what the task had inherited, leaving
``None``, which ``build_outbound_identity`` reports as ``'unrecorded'`` and
refuses for any metered destination.

⚠️ THE LIBRARY DEFAULT CHANGED — w2v-common 0.67.7, ``fix(auth)!`` bd1e30d.
That per-caller workaround is no longer what protects us; the class is fixed at
the source via a ``KEEP_TENANT_SOURCE`` sentinel, and the contract is now
three-way:

    omitted + SAME tenant       -> provenance PRESERVED
    omitted + DIFFERENT tenant  -> provenance DROPPED to None
    explicit value (incl. None) -> always wins

The tenant-equality half is load-bearing, not defensive: provenance is bound to
a TENANT — it is what unlocks that tenant's BYO credentials and bills them — so
a bare "preserve on omit" would let a reaper walking rows for many tenants carry
tenant A's proof onto tenant B. The tests below assert BOTH halves on purpose,
so that neither the original clobber bug nor the tempting naive fix can pass.

The explicit ``tenant_source=`` keyword at the ``_bg`` call sites is now
redundant, and is KEPT deliberately: it is self-documenting, it costs nothing,
and ``test_api_bg_entrypoints_all_preserve_provenance`` below still enforces it
so that a future caller cannot quietly go back to relying on an implicit
default it does not state.
"""

from __future__ import annotations

import asyncio

from w2v_common.auth.identity_context import (
    clear_identity_context,
    get_tenant_source_from_context,
    set_identity_context,
)
from w2v_common.tenant import TenantSource

TENANT = "e148e8af-3751-49bf-a0fa-56ab46d31ddc"
OTHER_TENANT = "b3d4c5e6-0000-4a1b-9c2d-7f8e9a0b1c2d"


def test_omitting_tenant_source_preserves_provenance_for_the_same_tenant() -> None:
    """The mechanism itself, as of w2v-common 0.67.7.

    Tripwire in the ORIGINAL direction: if omission ever starts DESTROYING
    provenance again, the 'unrecorded' panel-image failure is back and this test
    says so before a metered hop fails closed in production.
    """
    clear_identity_context()
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.LICENSE_TOKEN)
    assert get_tenant_source_from_context() is TenantSource.LICENSE_TOKEN

    set_identity_context(TENANT, None, None)  # the bg-task call shape, same tenant
    assert get_tenant_source_from_context() is TenantSource.LICENSE_TOKEN, (
        "omitting tenant_source destroyed inherited provenance for the SAME "
        "tenant — the w2v-common 0.67.7 KEEP_TENANT_SOURCE default has "
        "regressed; every metered hop from a background job now fails closed "
        "as 'unrecorded'"
    )
    clear_identity_context()


def test_omitting_tenant_source_drops_provenance_across_tenants() -> None:
    """The security half — a naive "preserve always" fix must NOT pass.

    Provenance vouches for a specific tenant's right to spend its own BYO
    credentials. Carrying it onto a different tenant is exactly the
    "unauthenticated caller spends an arbitrary tenant's credentials" hole the
    outbound guard exists to close, so rebinding to another tenant must drop it.
    """
    clear_identity_context()
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.LICENSE_TOKEN)
    assert get_tenant_source_from_context() is TenantSource.LICENSE_TOKEN

    set_identity_context(OTHER_TENANT, None, None)  # e.g. a reaper walking rows
    assert get_tenant_source_from_context() is None, (
        "provenance survived a rebind to a DIFFERENT tenant — tenant A's proof "
        "would vouch for tenant B's spend"
    )
    clear_identity_context()


def test_explicit_none_still_clears_provenance() -> None:
    """An explicit value always wins over the preserve default, including None."""
    clear_identity_context()
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.LICENSE_TOKEN)
    set_identity_context(TENANT, None, None, tenant_source=None)
    assert get_tenant_source_from_context() is None, (
        "an EXPLICIT tenant_source=None was swallowed by the preserve default — "
        "callers can no longer state 'there is genuinely no provenance'"
    )
    clear_identity_context()


def test_created_task_inherits_provenance_and_the_reseed_keeps_it() -> None:
    """A detached task inherits provenance; re-seeding its own tenant keeps it.

    Models exactly what the `_bg` entrypoints do: request scope binds provenance,
    `asyncio.create_task` copies the context, and the task re-seeds the tenant for
    RLS. As of w2v-common 0.67.7 that re-seed preserves provenance whether or not
    the caller spells `tenant_source=` out — but only because the tenant is the
    SAME. A task that rebinds a DIFFERENT tenant still loses it, by design.
    """
    seen: dict[str, object] = {}

    async def bg_with_explicit_preserve() -> None:
        # what src/api/*.py does — redundant since 0.67.7, kept as documentation
        set_identity_context(TENANT, None, None, tenant_source=get_tenant_source_from_context())
        seen["explicit"] = get_tenant_source_from_context()

    async def bg_relying_on_the_default() -> None:
        set_identity_context(TENANT, None, None)
        seen["implicit"] = get_tenant_source_from_context()

    async def bg_rebinding_another_tenant() -> None:
        set_identity_context(OTHER_TENANT, None, None)
        seen["cross_tenant"] = get_tenant_source_from_context()

    async def drive() -> None:
        clear_identity_context()
        set_identity_context(TENANT, None, None, tenant_source=TenantSource.LICENSE_TOKEN)
        # each task gets its own copy of the current context
        await asyncio.gather(asyncio.create_task(bg_with_explicit_preserve()))
        await asyncio.gather(asyncio.create_task(bg_relying_on_the_default()))
        await asyncio.gather(asyncio.create_task(bg_rebinding_another_tenant()))
        clear_identity_context()

    asyncio.run(drive())

    assert seen["explicit"] is TenantSource.LICENSE_TOKEN, (
        "the preserve pattern did not keep provenance; every metered hop from a "
        "background job will fail closed as 'unrecorded'"
    )
    assert seen["implicit"] is TenantSource.LICENSE_TOKEN, (
        "re-seeding the SAME tenant without tenant_source destroyed inherited "
        "provenance — the 0.67.7 preserve default has regressed"
    )
    assert seen["cross_tenant"] is None, (
        "a task that rebound a DIFFERENT tenant kept the original tenant's "
        "provenance — that vouches tenant A's proof for tenant B's spend"
    )


def test_api_bg_entrypoints_all_preserve_provenance() -> None:
    """Every request-dispatched `_bg` seeding call under src/api/ must preserve.

    A source-level assertion on purpose: the defect is a MISSING KEYWORD, which no
    behavioural test of one entrypoint can generalise across the others. This fails
    the moment someone adds a fifth `_bg` task with the old call shape.

    src/services/cleanup_service.py is deliberately excluded — that reaper has no
    inbound request and rebinds a different tenant per row, so preserving an
    inherited value there would vouch one tenant's provenance for another's.
    """
    import pathlib
    import re

    api = pathlib.Path(__file__).resolve().parents[2] / "src" / "api"
    offenders: list[str] = []
    call = re.compile(r"set_identity_context\(\s*[^)]*?\)", re.S)
    for path in api.rglob("*.py"):
        text = path.read_text()
        for m in call.finditer(text):
            if "tenant_source" not in m.group(0):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{path.name}:{line}")

    assert not offenders, (
        "set_identity_context() without tenant_source under src/api/ — these "
        f"destroy inherited provenance and fail every metered hop: {offenders}"
    )
