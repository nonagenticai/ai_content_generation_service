"""Migration 016 against a FRESH Postgres database, through real Alembic.

The rest of the suite builds its schema with ``create_all``, so nothing else
ever executes the migration chain. This test bootstraps a throwaway database
from zero to head, proves 016 created the columns + the PARTIAL unique index
(and that the index really is per-tenant and NULL-tolerant), then downgrades to
015 and back up -- the reversibility + drift-tolerance the revision promises.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from tests.conftest import _TEST_DATABASE_URL, TENANT_A, TENANT_B

_TABLES = ("video_generation_jobs", "imagegen_jobs")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def fresh_db_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    base = make_url(_TEST_DATABASE_URL)
    name = f"acg_mig016_{uuid.uuid4().hex[:10]}"
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = base.set(database=name).render_as_string(hide_password=False)
    # env.py reads DATABASE_URL at import time.
    monkeypatch.setenv("DATABASE_URL", url)
    try:
        yield url
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _alembic() -> Config:
    # No ini file on purpose: env.py runs ``logging.config.fileConfig`` when given
    # one, which DISABLES every existing logger and breaks later caplog tests.
    cfg = Config()
    cfg.set_main_option("script_location", os.path.join(_ROOT, "src/database/migrations"))
    return cfg


def _columns(url: str, table: str) -> set[str]:
    engine = create_engine(url)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_name = :t"),
            {"t": table},
        )
        cols = {r[0] for r in rows}
    engine.dispose()
    return cols


def _index_def(url: str, table: str) -> str | None:
    engine = create_engine(url)
    with engine.connect() as conn:
        found = conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :i"),
            {"i": f"uq_{table}_tenant_idempotency_key"},
        ).scalar_one_or_none()
    engine.dispose()
    return found


def test_016_upgrades_downgrades_and_reupgrades(fresh_db_url: str) -> None:
    cfg = _alembic()
    command.upgrade(cfg, "head")

    for table in _TABLES:
        assert {"idempotency_key", "idempotency_fingerprint"} <= _columns(fresh_db_url, table)
        indexdef = _index_def(fresh_db_url, table)
        assert indexdef is not None, f"016 did not create the key index on {table}"
        assert "UNIQUE" in indexdef
        assert "(tenant_id, idempotency_key)" in indexdef
        assert "WHERE (idempotency_key IS NOT NULL)" in indexdef

    command.downgrade(cfg, "015_provider_slot")
    for table in _TABLES:
        assert not {"idempotency_key", "idempotency_fingerprint"} & _columns(fresh_db_url, table)
        assert _index_def(fresh_db_url, table) is None

    command.upgrade(cfg, "head")
    for table in _TABLES:
        assert _index_def(fresh_db_url, table) is not None


def test_016_index_is_unique_per_tenant_and_ignores_null_keys(fresh_db_url: str) -> None:
    command.upgrade(_alembic(), "head")
    engine = create_engine(fresh_db_url)
    insert = text(
        "INSERT INTO imagegen_jobs (job_id, asset_id, tenant_id, status, provider, prompt, "
        "attempts, idempotency_key) VALUES (:j, :a, :t, 'pending', 'stub', 'p', 0, :k)"
    )

    def _row(tenant: str, key: str | None) -> dict[str, str | None]:
        return {"j": uuid.uuid4().hex, "a": uuid.uuid4().hex, "t": tenant, "k": key}

    with engine.begin() as conn:
        conn.execute(insert, _row(TENANT_A, "k"))
        conn.execute(insert, _row(TENANT_B, "k"))  # same key, other tenant: allowed
        conn.execute(insert, _row(TENANT_A, None))  # NULL keys never collide
        conn.execute(insert, _row(TENANT_A, None))
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(insert, _row(TENANT_A, "k"))
    engine.dispose()
