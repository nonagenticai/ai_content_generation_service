"""Phase D RLS — re-bind the tenant GUC at the start of every transaction.

The imagegen/video generation pipelines process a job across several ``commit()``
calls on one session; the GUC is transaction-scoped (``set_config is_local=true``)
and dropped on each commit. This installs an ``after_begin`` listener that re-applies
``app.current_tenant`` from the identity ContextVar at the start of EVERY transaction.
Inert unless ``TENANT_RLS_MODE=enforce`` and a tenant is in context. Scope is this
process only; the cross-tenant cleanup read goes through a SECURITY DEFINER resolver.
"""

from __future__ import annotations

from sqlalchemy import Connection, event
from sqlalchemy.orm import Session, SessionTransaction
from w2v_common.auth.identity_context import get_tenant_id_from_context
from w2v_common.database.rls import tenant_rls_mode

_installed = False


def install_rls_tenant_rebind() -> None:
    """Register the after_begin GUC re-bind listener once per process."""
    global _installed
    if _installed:
        return

    @event.listens_for(Session, "after_begin")
    def _rebind_tenant(
        session: Session, transaction: SessionTransaction, connection: Connection
    ) -> None:
        if tenant_rls_mode() != "enforce":
            return
        tenant = get_tenant_id_from_context()
        if not tenant:
            return
        # str() at the boundary: set_config() has no uuid overload, and a
        # tenant read back from a resolver is now a uuid.UUID, not a str.
        connection.exec_driver_sql(
            "SELECT set_config('app.current_tenant', %s, true)", (str(tenant),)
        )

    _installed = True
