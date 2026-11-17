"""Proves the Phase D RLS policies ISOLATE tenants for ai_content_generation_service.

* ``video_generation_jobs`` — strict own-only (representative of all 4 strict tables).
* ``expired_video_generation_jobs`` — the SECURITY DEFINER cross-tenant cleanup READ
  resolver still finds jobs across tenants (owner bypasses ENABLE).

Connects AS a throwaway non-owner ``NOBYPASSRLS`` role (the runtime posture under
enforce). Needs a superuser test DB; skips loudly in CI where it can't CREATE ROLE.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from tests.conftest import TENANT_A, TENANT_B

_ROLE = "w2v_ai_content_app_rls_test"
_INS = (
    "INSERT INTO video_generation_jobs (job_id, tenant_id, provider, prompt, status, created_at, updated_at) "
    "VALUES (:id, :t, 'runway', 'p', :st, now(), now())"
)


@pytest.mark.asyncio
async def test_rls_isolates_tenants_under_restricted_role(db_engine) -> None:
    async with db_engine.begin() as conn:
        try:
            await conn.execute(text(f'CREATE ROLE "{_ROLE}" NOSUPERUSER NOBYPASSRLS NOINHERIT'))
        except ProgrammingError as exc:
            pytest.skip(f"cannot CREATE ROLE (needs superuser/CREATEROLE): {exc.orig}")
        await conn.execute(
            text(f'GRANT SELECT, INSERT, UPDATE, DELETE ON video_generation_jobs TO "{_ROLE}"')
        )

        await conn.execute(text(_INS), {"id": "job-a", "t": TENANT_A, "st": "completed"})
        await conn.execute(text(_INS), {"id": "job-b", "t": TENANT_B, "st": "completed"})

        try:
            await conn.execute(text(f'SET LOCAL ROLE "{_ROLE}"'))

            n = (await conn.execute(text("SELECT count(*) FROM video_generation_jobs"))).scalar()
            assert n == 0, f"leaked {n} rows with no GUC"

            await conn.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": TENANT_A}
            )
            rows = (
                (await conn.execute(text("SELECT tenant_id FROM video_generation_jobs")))
                .scalars()
                .all()
            )
            assert [str(r) for r in rows] == [TENANT_A], f"not isolated under GUC=A: {rows}"

            await conn.execute(text("SAVEPOINT s"))
            with pytest.raises(ProgrammingError):
                await conn.execute(text(_INS), {"id": "job-x", "t": TENANT_B, "st": "completed"})
            await conn.execute(text("ROLLBACK TO SAVEPOINT s"))
        finally:
            await conn.execute(text("RESET ROLE"))
            await conn.execute(text(f'DROP OWNED BY "{_ROLE}"'))
            await conn.execute(text(f'DROP ROLE IF EXISTS "{_ROLE}"'))


@pytest.mark.asyncio
async def test_cleanup_read_resolver_is_cross_tenant(db_engine) -> None:
    """expired_video_generation_jobs returns expired jobs across tenants (SECURITY DEFINER)."""
    async with db_engine.begin() as conn:
        for jid, tid in (("exp-a", TENANT_A), ("exp-b", TENANT_B)):
            await conn.execute(
                text(
                    "INSERT INTO video_generation_jobs "
                    "(job_id, tenant_id, provider, prompt, status, created_at, updated_at, completed_at) "
                    "VALUES (:id, :t, 'runway', 'p', 'completed', now(), now(), now() - interval '10 days')"
                ),
                {"id": jid, "t": tid},
            )
        rows = (
            await conn.execute(
                text(
                    "SELECT job_id, tenant_id FROM "
                    "expired_video_generation_jobs('completed', now(), 100)"
                )
            )
        ).all()
        found = {(r.job_id, str(r.tenant_id)) for r in rows}
        assert ("exp-a", TENANT_A) in found and ("exp-b", TENANT_B) in found, found
