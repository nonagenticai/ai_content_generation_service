"""Unit tests for Redis client caching layer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import RedisError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The actual implementation now lives in w2v_common; the src.cache.redis_client
# module is a thin re-export.
REDIS_MODULE = "w2v_common.cache.redis_client"


def _make_client():
    """Create a RedisClient with default env-var settings."""
    from src.cache.redis_client import RedisClient

    return RedisClient()


def _connected_client():
    """Create a RedisClient pre-wired with a mocked internal _client."""
    client = _make_client()
    client._client = AsyncMock()
    return client


# ===========================================================================
# __init__
# ===========================================================================


class TestInit:
    async def test_init_reads_defaults(self):
        client = _make_client()

        assert client._redis_url == "redis://localhost:6379/0"
        assert client._pool is None
        assert client._client is None

    async def test_init_custom_url_overrides_default(self):
        from src.cache.redis_client import RedisClient

        client = RedisClient(redis_url="redis://custom:6380/1")
        assert client._redis_url == "redis://custom:6380/1"


# ===========================================================================
# connect / disconnect
# ===========================================================================


class TestConnectDisconnect:
    async def test_connect_creates_pool_and_client(self):
        client = _make_client()

        mock_pool = MagicMock()
        mock_redis_instance = AsyncMock()
        # ping returns an awaitable
        mock_redis_instance.ping = AsyncMock(return_value=True)

        with (
            patch(
                f"{REDIS_MODULE}.ConnectionPool.from_url", return_value=mock_pool
            ) as mock_from_url,
            patch(f"{REDIS_MODULE}.redis.Redis", return_value=mock_redis_instance),
        ):
            await client.connect()

        mock_from_url.assert_called_once()
        assert client._pool is mock_pool
        assert client._client is mock_redis_instance

    async def test_connect_skips_when_pool_exists(self):
        client = _make_client()
        client._pool = MagicMock()  # already has a pool

        with patch(f"{REDIS_MODULE}.ConnectionPool.from_url") as mock_from_url:
            await client.connect()

        mock_from_url.assert_not_called()

    async def test_connect_raises_on_ping_failure(self):
        client = _make_client()

        mock_pool = MagicMock()
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping = MagicMock(side_effect=ConnectionError("refused"))

        with (
            patch(f"{REDIS_MODULE}.ConnectionPool.from_url", return_value=mock_pool),
            patch(f"{REDIS_MODULE}.redis.Redis", return_value=mock_redis_instance),
        ):
            with pytest.raises(ConnectionError, match="refused"):
                await client.connect()

    async def test_connect_handles_non_awaitable_ping(self):
        """When ping() returns a plain bool instead of a coroutine."""
        client = _make_client()

        mock_pool = MagicMock()
        mock_redis_instance = MagicMock()
        # Return a plain True -- not an awaitable
        mock_redis_instance.ping = MagicMock(return_value=True)

        with (
            patch(f"{REDIS_MODULE}.ConnectionPool.from_url", return_value=mock_pool),
            patch(f"{REDIS_MODULE}.redis.Redis", return_value=mock_redis_instance),
        ):
            await client.connect()

        assert client._client is mock_redis_instance

    async def test_disconnect_closes_client_and_pool(self):
        client = _make_client()
        mock_client_inner = AsyncMock()
        mock_pool = AsyncMock()
        client._client = mock_client_inner
        client._pool = mock_pool

        await client.disconnect()

        mock_client_inner.aclose.assert_awaited_once()
        mock_pool.aclose.assert_awaited_once()
        assert client._client is None
        assert client._pool is None

    async def test_disconnect_noop_when_already_disconnected(self):
        client = _make_client()
        # No _client or _pool set
        await client.disconnect()  # should not raise
        assert client._client is None
        assert client._pool is None


# ===========================================================================
# async context manager
# ===========================================================================


class TestAsyncContextManager:
    async def test_aenter_connects_and_returns_self(self):
        client = _make_client()

        mock_pool = MagicMock()
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping = AsyncMock(return_value=True)

        with (
            patch(f"{REDIS_MODULE}.ConnectionPool.from_url", return_value=mock_pool),
            patch(f"{REDIS_MODULE}.redis.Redis", return_value=mock_redis_instance),
        ):
            result = await client.__aenter__()

        assert result is client

    async def test_aexit_disconnects(self):
        client = _make_client()
        mock_client_inner = AsyncMock()
        mock_pool = AsyncMock()
        client._client = mock_client_inner
        client._pool = mock_pool

        await client.__aexit__(None, None, None)

        mock_client_inner.aclose.assert_awaited_once()
        assert client._client is None


# ===========================================================================
# _ensure_connected
# ===========================================================================


class TestEnsureConnected:
    async def test_raises_when_not_connected(self):
        client = _make_client()

        with pytest.raises(RuntimeError, match="not connected"):
            client._ensure_connected()

    async def test_returns_client_when_connected(self):
        client = _make_client()
        mock_inner = AsyncMock()
        client._client = mock_inner

        result = client._ensure_connected()
        assert result is mock_inner


# ===========================================================================
# _serialize / _deserialize
# ===========================================================================


class TestSerialization:
    async def test_serialize_dict(self):
        client = _make_client()
        result = client._serialize({"key": "value", "count": 42})
        assert json.loads(result) == {"key": "value", "count": 42}

    async def test_serialize_list(self):
        client = _make_client()
        result = client._serialize([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]

    async def test_serialize_string(self):
        client = _make_client()
        result = client._serialize("hello")
        assert json.loads(result) == "hello"

    async def test_serialize_pydantic_model(self):
        """Objects with model_dump() (e.g., Pydantic) are dumped before JSON encoding."""
        client = _make_client()
        mock_model = MagicMock()
        mock_model.model_dump.return_value = {"id": "abc", "status": "done"}

        result = client._serialize(mock_model)
        mock_model.model_dump.assert_called_once_with(mode="json")
        assert json.loads(result) == {"id": "abc", "status": "done"}

    async def test_deserialize_valid_json(self):
        client = _make_client()
        result = client._deserialize('{"x": 1}')
        assert result == {"x": 1}

    async def test_deserialize_none_returns_none(self):
        client = _make_client()
        assert client._deserialize(None) is None

    async def test_deserialize_invalid_json_returns_none(self):
        client = _make_client()
        assert client._deserialize("not json {{{") is None

    async def test_serialize_deserialize_roundtrip(self):
        client = _make_client()
        original = {"nested": {"list": [1, 2, 3], "flag": True}, "value": None}
        serialized = client._serialize(original)
        deserialized = client._deserialize(serialized)
        assert deserialized == original


# ===========================================================================
# get
# ===========================================================================


class TestGet:
    async def test_get_returns_deserialized_value(self):
        client = _connected_client()
        client._client.get = AsyncMock(return_value='{"status": "ok"}')

        result = await client.get("my-key")

        assert result == {"status": "ok"}
        client._client.get.assert_awaited_once_with("my-key")

    async def test_get_returns_none_for_missing_key(self):
        client = _connected_client()
        client._client.get = AsyncMock(return_value=None)

        result = await client.get("missing")
        assert result is None

    async def test_get_returns_none_for_empty_string(self):
        """An empty string is falsy, so get() returns None."""
        client = _connected_client()
        client._client.get = AsyncMock(return_value="")

        result = await client.get("empty")
        assert result is None

    async def test_get_returns_none_on_redis_error(self):
        client = _connected_client()
        client._client.get = AsyncMock(side_effect=RedisError("connection lost"))

        result = await client.get("some-key")
        assert result is None

    async def test_get_returns_none_when_not_connected(self):
        client = _make_client()

        result = await client.get("key")
        assert result is None


# ===========================================================================
# set
# ===========================================================================


class TestSet:
    async def test_set_without_ttl(self):
        client = _connected_client()
        client._client.set = AsyncMock(return_value=True)

        result = await client.set("k", {"data": 1})

        assert result is True
        client._client.set.assert_awaited_once_with("k", json.dumps({"data": 1}))

    async def test_set_with_ttl_uses_setex(self):
        client = _connected_client()
        client._client.setex = AsyncMock(return_value=True)

        result = await client.set("k", "val", ttl=300)

        assert result is True
        client._client.setex.assert_awaited_once_with("k", 300, json.dumps("val"))

    async def test_set_returns_false_on_error(self):
        client = _connected_client()
        client._client.set = AsyncMock(side_effect=RedisError("write error"))

        result = await client.set("k", "v")
        assert result is False

    async def test_set_returns_false_when_not_connected(self):
        client = _make_client()

        result = await client.set("k", "v")
        assert result is False


# ===========================================================================
# delete
# ===========================================================================


class TestDelete:
    async def test_delete_returns_true_on_success(self):
        client = _connected_client()
        client._client.delete = AsyncMock(return_value=1)

        assert await client.delete("k") is True

    async def test_delete_returns_false_when_key_absent(self):
        client = _connected_client()
        client._client.delete = AsyncMock(return_value=0)

        assert await client.delete("missing") is False

    async def test_delete_returns_false_on_error(self):
        client = _connected_client()
        client._client.delete = AsyncMock(side_effect=RedisError("timeout"))

        assert await client.delete("k") is False


# ===========================================================================
# exists
# ===========================================================================


class TestExists:
    async def test_exists_returns_true(self):
        client = _connected_client()
        client._client.exists = AsyncMock(return_value=1)

        assert await client.exists("k") is True

    async def test_exists_returns_false(self):
        client = _connected_client()
        client._client.exists = AsyncMock(return_value=0)

        assert await client.exists("k") is False

    async def test_exists_returns_false_on_error(self):
        client = _connected_client()
        client._client.exists = AsyncMock(side_effect=RedisError("oops"))

        assert await client.exists("k") is False


# ===========================================================================
# incr
# ===========================================================================


class TestIncr:
    async def test_incr_returns_new_value(self):
        client = _connected_client()
        client._client.incr = AsyncMock(return_value=5)

        assert await client.incr("counter") == 5

    async def test_incr_returns_zero_on_error(self):
        client = _connected_client()
        client._client.incr = AsyncMock(side_effect=OSError("err"))

        assert await client.incr("counter") == 0


# ===========================================================================
# ping
# ===========================================================================


class TestPing:
    async def test_ping_returns_true_awaitable(self):
        client = _connected_client()
        client._client.ping = AsyncMock(return_value=True)

        assert await client.ping() is True

    async def test_ping_returns_true_non_awaitable(self):
        client = _connected_client()
        client._client.ping = MagicMock(return_value=True)

        assert await client.ping() is True

    async def test_ping_returns_false_on_error(self):
        client = _connected_client()
        client._client.ping = MagicMock(side_effect=RedisError("unreachable"))

        assert await client.ping() is False


# ===========================================================================
# Module-level functions: get_redis_client, init_redis, close_redis
# ===========================================================================


class TestModuleFunctions:
    async def test_get_redis_client_singleton(self):
        import w2v_common.cache.redis_client as mod

        # Reset module state
        mod._redis_client = None
        mod.get_redis_client.cache_clear()

        client1 = mod.get_redis_client()
        assert client1 is not None
        assert mod._redis_client is client1

        # Second call returns the same instance
        client2 = mod.get_redis_client()
        assert client2 is client1

        # Cleanup
        mod._redis_client = None
        mod.get_redis_client.cache_clear()

    async def test_init_redis_connects_and_returns_client(self):
        import w2v_common.cache.redis_client as mod

        mod._redis_client = None
        mod.get_redis_client.cache_clear()

        mock_pool = MagicMock()
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping = AsyncMock(return_value=True)

        with (
            patch(f"{REDIS_MODULE}.ConnectionPool.from_url", return_value=mock_pool),
            patch(f"{REDIS_MODULE}.redis.Redis", return_value=mock_redis_instance),
        ):
            client = await mod.init_redis()

        assert client is not None
        assert client._client is mock_redis_instance

        # Cleanup
        mod._redis_client = None
        mod.get_redis_client.cache_clear()

    async def test_close_redis_disconnects_and_resets(self):
        import w2v_common.cache.redis_client as mod

        mod._redis_client = None
        mod.get_redis_client.cache_clear()

        # Set up a fake connected client
        fake_client = MagicMock()
        fake_client.disconnect = AsyncMock()
        mod._redis_client = fake_client

        await mod.close_redis()

        fake_client.disconnect.assert_awaited_once()
        assert mod._redis_client is None

        # Cleanup
        mod.get_redis_client.cache_clear()

    async def test_close_redis_noop_when_no_client(self):
        import w2v_common.cache.redis_client as mod

        mod._redis_client = None
        mod.get_redis_client.cache_clear()

        await mod.close_redis()  # should not raise

        assert mod._redis_client is None
        mod.get_redis_client.cache_clear()
