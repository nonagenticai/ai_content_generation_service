"""Tests for database/connection.py to cover missing lines.

Covers: _ensure_database_exists, create_engine_with_retry, get_engine,
health_check, close_engine.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import w2v_common.database.connection as conn_mod


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset global engine between tests."""
    original = conn_mod._engine
    conn_mod._engine = None
    yield
    conn_mod._engine = original


# ============================================================================
# _ensure_database_exists
# ============================================================================


class TestEnsureDatabaseExists:
    async def test_invalid_db_name_raises(self) -> None:
        """Invalid database name with special chars raises ValueError."""
        with patch.object(conn_mod, "DB_NAME", "bad-name!"):
            with pytest.raises(ValueError, match="invalid characters"):
                await conn_mod._ensure_database_exists()

    async def test_database_already_exists(self) -> None:
        """When database already exists, returns without creating."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1  # database exists
        mock_conn.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_engine.dispose = MagicMock()

        with (
            patch.object(conn_mod, "DB_NAME", "test_db"),
            patch("w2v_common.database.connection.create_engine", return_value=mock_engine),
        ):
            await conn_mod._ensure_database_exists(max_retries=1)

        # Only one execute call (the SELECT check), no CREATE DATABASE
        assert mock_conn.execute.call_count == 1
        executed_sql = str(mock_conn.execute.call_args_list[0][0][0])
        assert "pg_database" in executed_sql
        mock_engine.dispose.assert_called_once()

    async def test_database_created_successfully(self) -> None:
        """When database does not exist, it is created."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = None  # database doesn't exist

        executed: list[str] = []

        def _execute(statement, *_args, **_kwargs):
            # A side_effect FUNCTION, not a fixed-length list or a positional
            # call_args_list index: the ONLY thing that matters is that call 1 is the
            # existence probe and must return a result whose .scalar() is None; every
            # later call is provisioning DDL whose return value is unused. Pinning the
            # exact statement COUNT broke the moment w2v_common added a legitimate
            # third statement (REVOKE CONNECT ... FROM PUBLIC, the born-LOCKED policy).
            executed.append(str(statement))
            return mock_result if len(executed) == 1 else None

        mock_conn.execute.side_effect = _execute

        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_engine.dispose = MagicMock()

        with (
            patch.object(conn_mod, "DB_NAME", "new_db"),
            patch("w2v_common.database.connection.create_engine", return_value=mock_engine),
        ):
            await conn_mod._ensure_database_exists(max_retries=1)

        # The point of this test: an absent database gets CREATED. Deliberately does
        # NOT assert the REVOKE text -- that would re-pin the exact statement list.
        assert executed, "no statement was executed at all"
        assert "pg_database" in executed[0], executed
        assert any("CREATE DATABASE new_db" in stmt for stmt in executed), executed
        mock_engine.dispose.assert_called_once()

    async def test_database_create_already_exists_race(self) -> None:
        """ProgrammingError with 'already exists' is handled."""
        from sqlalchemy.exc import ProgrammingError

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = None

        call_count = 0

        def exec_side_effect(stmt, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_result
            raise ProgrammingError(
                statement="CREATE DATABASE",
                params={},
                orig=Exception("database already exists"),
            )

        mock_conn.execute.side_effect = exec_side_effect

        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_engine.dispose = MagicMock()

        with (
            patch.object(conn_mod, "DB_NAME", "test_db"),
            patch("w2v_common.database.connection.create_engine", return_value=mock_engine),
        ):
            # Should NOT raise — the ProgrammingError "already exists" is swallowed
            await conn_mod._ensure_database_exists(max_retries=1)

        # Both the SELECT and the CREATE DATABASE were attempted
        assert call_count == 2
        mock_engine.dispose.assert_called_once()

    async def test_connection_failure_retries_and_raises(self) -> None:
        """OperationalError retries and then raises ConnectionError."""
        from sqlalchemy.exc import OperationalError

        mock_engine = MagicMock()
        mock_engine.connect.side_effect = OperationalError(
            statement="SELECT 1", params={}, orig=Exception("Connection refused")
        )
        mock_engine.dispose = MagicMock()

        with (
            patch.object(conn_mod, "DB_NAME", "test_db"),
            patch("w2v_common.database.connection.create_engine", return_value=mock_engine),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                await conn_mod._ensure_database_exists(max_retries=2)

        # Should have retried: 2 connect attempts, 1 sleep between them
        assert mock_engine.connect.call_count == 2
        mock_sleep.assert_awaited_once_with(2)  # attempt=1 → sleep(1*2)
        # dispose called in finally for each attempt
        assert mock_engine.dispose.call_count == 2

    async def test_unexpected_error_raises(self) -> None:
        """Unexpected error during ensure_database_exists is re-raised."""
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = TypeError("unexpected")
        mock_engine.dispose = MagicMock()

        with (
            patch.object(conn_mod, "DB_NAME", "test_db"),
            patch("w2v_common.database.connection.create_engine", return_value=mock_engine),
        ):
            with pytest.raises(TypeError, match="unexpected"):
                await conn_mod._ensure_database_exists(max_retries=1)

        # No retries on unexpected errors — only 1 connect attempt
        assert mock_engine.connect.call_count == 1
        mock_engine.dispose.assert_called_once()


# ============================================================================
# create_engine_with_retry
# ============================================================================


class TestCreateEngineWithRetry:
    async def test_success_on_first_attempt(self) -> None:
        """Engine created successfully on first attempt."""
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        with patch("w2v_common.database.connection.create_async_engine", return_value=mock_engine):
            engine = await conn_mod.create_engine_with_retry(max_retries=1)

        assert engine is mock_engine

    async def test_database_not_exist_creates_it(self) -> None:
        """When database doesn't exist, _ensure_database_exists is called."""
        from sqlalchemy.exc import ProgrammingError

        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        call_count = 0

        def create_engine_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProgrammingError(
                    statement="SELECT 1",
                    params={},
                    orig=Exception('database "test" does not exist'),
                )
            return mock_engine

        with (
            patch(
                "w2v_common.database.connection.create_async_engine",
                side_effect=create_engine_side_effect,
            ),
            patch(
                "w2v_common.database.connection._ensure_database_exists",
                new_callable=AsyncMock,
            ) as mock_ensure,
        ):
            engine = await conn_mod.create_engine_with_retry(max_retries=2)

        # _ensure_database_exists was called with no arguments (uses defaults)
        mock_ensure.assert_awaited_once_with()
        # Engine returned successfully after the database was created
        assert engine is mock_engine
        # create_async_engine was called twice: first raised, second succeeded
        assert call_count == 2

    async def test_timeout_error_retries(self) -> None:
        """TimeoutError triggers retry."""
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        call_count = 0

        def create_engine_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Connection timed out")
            return mock_engine

        with (
            patch(
                "w2v_common.database.connection.create_async_engine",
                side_effect=create_engine_side_effect,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            engine = await conn_mod.create_engine_with_retry(max_retries=2, retry_delay=0.01)

        assert engine is mock_engine
        # Retried after first failure with initial delay
        mock_sleep.assert_awaited_once_with(0.01)
        # create_async_engine called twice (fail + succeed)
        assert call_count == 2

    async def test_all_retries_exhausted_raises(self) -> None:
        """All retries exhausted raises ConnectionError."""
        with (
            patch(
                "w2v_common.database.connection.create_async_engine",
                side_effect=OSError("Connection refused"),
            ) as mock_create,
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                await conn_mod.create_engine_with_retry(max_retries=2, retry_delay=0.01)

        # Attempted exactly max_retries times
        assert mock_create.call_count == 2
        # Slept between attempts with exponential backoff: 0.01, then 0.02
        mock_sleep.assert_awaited_once_with(0.01)  # only 1 sleep for 2 attempts

    async def test_general_exception_retries(self) -> None:
        """General exception triggers retry."""
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        call_count = 0

        def create_engine_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Unexpected error")
            return mock_engine

        with (
            patch(
                "w2v_common.database.connection.create_async_engine",
                side_effect=create_engine_side_effect,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            engine = await conn_mod.create_engine_with_retry(max_retries=2, retry_delay=0.01)

        assert engine is mock_engine
        # Slept once between the failed and successful attempt
        mock_sleep.assert_awaited_once_with(0.01)
        assert call_count == 2

    async def test_operational_error_non_db_doesnt_exist(self) -> None:
        """OperationalError that isn't 'database does not exist' retries normally."""
        from sqlalchemy.exc import OperationalError

        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        call_count = 0

        def create_engine_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OperationalError(
                    statement="SELECT 1",
                    params={},
                    orig=Exception("connection refused"),
                )
            return mock_engine

        with (
            patch(
                "w2v_common.database.connection.create_async_engine",
                side_effect=create_engine_side_effect,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch(
                "w2v_common.database.connection._ensure_database_exists",
                new_callable=AsyncMock,
            ) as mock_ensure,
        ):
            engine = await conn_mod.create_engine_with_retry(max_retries=2, retry_delay=0.01)

        assert engine is mock_engine
        # Non-"does not exist" OperationalError should NOT trigger _ensure_database_exists
        mock_ensure.assert_not_awaited()
        # Slept once between the failed and successful attempt
        mock_sleep.assert_awaited_once_with(0.01)
        assert call_count == 2


# ============================================================================
# get_engine
# ============================================================================


class TestGetEngine:
    async def test_get_engine_creates_engine(self) -> None:
        """get_engine creates engine when None."""
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        conn_mod._engine = None

        with patch("w2v_common.database.connection.create_async_engine", return_value=mock_engine):
            engine = await conn_mod.get_engine()

        assert engine is mock_engine
        # Module-level _engine is now set
        assert conn_mod._engine is mock_engine

    async def test_get_engine_returns_existing(self) -> None:
        """get_engine returns existing engine."""
        mock_engine = MagicMock()
        conn_mod._engine = mock_engine

        engine = await conn_mod.get_engine()
        assert engine is mock_engine


# ============================================================================
# close_engine
# ============================================================================


class TestCloseEngine:
    async def test_close_engine_when_none(self) -> None:
        """close_engine with no engine is a no-op."""
        conn_mod._engine = None
        await conn_mod.close_engine()
        assert conn_mod._engine is None

    async def test_close_engine_disposes(self) -> None:
        """close_engine disposes the engine and sets it to None."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()
        conn_mod._engine = mock_engine

        await conn_mod.close_engine()

        mock_engine.dispose.assert_awaited_once()
        assert conn_mod._engine is None


# ============================================================================
# health_check
# ============================================================================


class TestHealthCheck:
    async def test_health_check_success(self) -> None:
        """health_check returns True when DB is accessible."""
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        mock_engine = AsyncMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        conn_mod._engine = mock_engine

        result = await conn_mod.health_check()
        assert result is True

    async def test_health_check_failure(self) -> None:
        """health_check returns False when DB is not accessible."""
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(side_effect=OSError("DB down"))

        mock_engine = AsyncMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        conn_mod._engine = mock_engine

        result = await conn_mod.health_check()
        assert result is False
