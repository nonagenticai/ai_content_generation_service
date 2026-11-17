"""Unit tests for database session management module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import w2v_common.database.session as sess_mod


@pytest.fixture(autouse=True)
def _reset_session_maker():
    """Ensure global session maker is reset between tests."""
    sess_mod._async_session_maker = None
    yield
    sess_mod._async_session_maker = None


# ---------------------------------------------------------------------------
# get_session_maker
# ---------------------------------------------------------------------------


@patch("w2v_common.database.session.get_engine")
async def test_get_session_maker_creates_maker(mock_get_engine):
    """get_session_maker creates an async_sessionmaker on first call."""
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    maker = await sess_mod.get_session_maker()

    assert maker is not None
    mock_get_engine.assert_awaited_once()


@patch("w2v_common.database.session.get_engine")
async def test_get_session_maker_reuses_cached(mock_get_engine):
    """get_session_maker returns the same maker on subsequent calls."""
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    maker1 = await sess_mod.get_session_maker()
    maker2 = await sess_mod.get_session_maker()

    assert maker1 is maker2
    # get_engine should only be called once since maker is cached
    mock_get_engine.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_db
# ---------------------------------------------------------------------------


@patch("w2v_common.database.session.get_engine")
async def test_get_db_yields_session_and_commits(mock_get_engine):
    """get_db yields a session and commits on success."""
    mock_session = AsyncMock()
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    # We need to make the sessionmaker return our mock session
    # The code does: async with session_maker() as session
    # session_maker() returns an AsyncSession which is used as an async context manager
    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    # Create a first session maker, then replace it
    await sess_mod.get_session_maker()

    # Now replace with our controlled mock
    mock_maker = MagicMock()
    mock_maker.return_value = mock_session_ctx
    sess_mod._async_session_maker = mock_maker

    gen = sess_mod.get_db()
    session = await gen.__anext__()

    assert session is mock_session

    # Simulate normal completion (StopAsyncIteration expected)
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    mock_session.commit.assert_awaited_once()
    mock_session.close.assert_awaited_once()


@patch("w2v_common.database.session.get_engine")
async def test_get_db_rollback_on_exception(mock_get_engine):
    """get_db rolls back the session when an exception occurs."""
    mock_session = AsyncMock()
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    await sess_mod.get_session_maker()

    mock_maker = MagicMock()
    mock_maker.return_value = mock_session_ctx
    sess_mod._async_session_maker = mock_maker

    gen = sess_mod.get_db()
    session = await gen.__anext__()
    assert session is mock_session

    # Throw an exception into the generator
    with pytest.raises(ValueError, match="db error"):
        await gen.athrow(ValueError("db error"))

    mock_session.rollback.assert_awaited_once()
    mock_session.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_db_session
# ---------------------------------------------------------------------------


@patch("w2v_common.database.session.get_engine")
async def test_get_db_session_commits_on_success(mock_get_engine):
    """get_db_session context manager commits on success."""
    mock_session = AsyncMock()
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    await sess_mod.get_session_maker()

    mock_maker = MagicMock()
    mock_maker.return_value = mock_session_ctx
    sess_mod._async_session_maker = mock_maker

    async with sess_mod.get_db_session() as session:
        assert session is mock_session

    mock_session.commit.assert_awaited_once()
    mock_session.close.assert_awaited_once()


@patch("w2v_common.database.session.get_engine")
async def test_get_db_session_rollback_on_exception(mock_get_engine):
    """get_db_session context manager rolls back on exception."""
    mock_session = AsyncMock()
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    await sess_mod.get_session_maker()

    mock_maker = MagicMock()
    mock_maker.return_value = mock_session_ctx
    sess_mod._async_session_maker = mock_maker

    with pytest.raises(RuntimeError, match="query failed"):
        async with sess_mod.get_db_session() as session:
            assert session is mock_session
            raise RuntimeError("query failed")

    mock_session.rollback.assert_awaited_once()
    mock_session.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# close_session_maker
# ---------------------------------------------------------------------------


@patch("w2v_common.database.session.get_engine")
async def test_close_session_maker_resets_global(mock_get_engine):
    """close_session_maker sets the global maker to None."""
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    await sess_mod.get_session_maker()
    assert sess_mod._async_session_maker is not None

    await sess_mod.close_session_maker()
    assert sess_mod._async_session_maker is None
