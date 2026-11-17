"""Unit tests for provider health monitoring (Phase 4)."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

from src.services.provider_health import (
    _REDIS_KEY_PREFIX,
    _REDIS_TTL_SECONDS,
    ProviderHealthMonitor,
    ProviderHealthState,
)


def test_initial_state_healthy():
    monitor = ProviderHealthMonitor()
    state = monitor.get_health("test")
    assert state.is_healthy is True
    assert state.consecutive_failures == 0
    assert state.average_latency_ms is None


def test_record_success():
    monitor = ProviderHealthMonitor()
    monitor.record_success("test", 100.0)
    state = monitor.get_health("test")
    assert state.is_healthy is True
    assert state.consecutive_failures == 0
    assert state.average_latency_ms == 100.0


def test_record_multiple_successes():
    monitor = ProviderHealthMonitor()
    monitor.record_success("test", 100.0)
    monitor.record_success("test", 200.0)
    state = monitor.get_health("test")
    assert state.average_latency_ms == 150.0
    assert state.total_successes == 2


def test_record_failure_below_threshold():
    monitor = ProviderHealthMonitor(max_consecutive_failures=5)
    for _ in range(4):
        monitor.record_failure("test")
    state = monitor.get_health("test")
    assert state.is_healthy is True
    assert state.consecutive_failures == 4


def test_auto_disable_after_max_failures():
    monitor = ProviderHealthMonitor(max_consecutive_failures=3)
    for _ in range(3):
        monitor.record_failure("test")
    state = monitor.get_health("test")
    assert state.is_healthy is False
    assert state.disabled_at is not None
    assert state.cooldown_until is not None


def test_is_available_healthy():
    monitor = ProviderHealthMonitor()
    assert monitor.is_available("test") is True


def test_is_available_disabled():
    monitor = ProviderHealthMonitor(max_consecutive_failures=1, cooldown_seconds=300)
    monitor.record_failure("test")
    assert monitor.is_available("test") is False


def test_cooldown_re_enables():
    monitor = ProviderHealthMonitor(max_consecutive_failures=1, cooldown_seconds=1)
    monitor.record_failure("test")
    assert monitor.is_available("test") is False

    # Simulate time passing
    state = monitor._get_state("test")
    state.cooldown_until = time.time() - 1

    assert monitor.is_available("test") is True
    state = monitor.get_health("test")
    assert state.is_healthy is True
    assert state.consecutive_failures == 0


def test_success_resets_failures():
    monitor = ProviderHealthMonitor(max_consecutive_failures=5)
    for _ in range(4):
        monitor.record_failure("test")
    monitor.record_success("test", 50.0)
    state = monitor.get_health("test")
    assert state.consecutive_failures == 0
    assert state.is_healthy is True


def test_check_cooldowns():
    monitor = ProviderHealthMonitor(max_consecutive_failures=1, cooldown_seconds=1)
    monitor.record_failure("p1")
    monitor.record_failure("p2")

    # Expire both
    for name in ["p1", "p2"]:
        monitor._get_state(name).cooldown_until = time.time() - 1

    monitor.check_cooldowns()
    assert monitor.is_available("p1") is True
    assert monitor.is_available("p2") is True


def test_health_state_dataclass():
    state = ProviderHealthState()
    assert state.is_healthy is True
    assert state.consecutive_failures == 0
    assert state.total_successes == 0
    assert state.total_latency_ms == 0.0
    assert state.disabled_at is None
    assert state.cooldown_until is None
    assert state.average_latency_ms is None


# ===========================================================================
# ProviderHealthState serialization
# ===========================================================================


def test_health_state_to_dict():
    state = ProviderHealthState(
        is_healthy=False,
        consecutive_failures=3,
        total_successes=10,
        total_latency_ms=5000.0,
        disabled_at=1000.0,
        cooldown_until=1300.0,
    )
    d = state.to_dict()
    assert d["is_healthy"] is False
    assert d["consecutive_failures"] == 3
    assert d["total_successes"] == 10
    assert d["total_latency_ms"] == 5000.0
    assert d["disabled_at"] == 1000.0
    assert d["cooldown_until"] == 1300.0


def test_health_state_from_dict():
    d = {
        "is_healthy": False,
        "consecutive_failures": 5,
        "total_successes": 20,
        "total_latency_ms": 8000.0,
        "disabled_at": 2000.0,
        "cooldown_until": 2300.0,
    }
    state = ProviderHealthState.from_dict(d)
    assert state.is_healthy is False
    assert state.consecutive_failures == 5
    assert state.total_successes == 20
    assert state.total_latency_ms == 8000.0
    assert state.disabled_at == 2000.0
    assert state.cooldown_until == 2300.0


def test_health_state_roundtrip():
    original = ProviderHealthState(
        is_healthy=True,
        consecutive_failures=0,
        total_successes=5,
        total_latency_ms=1234.5,
    )
    roundtripped = ProviderHealthState.from_dict(original.to_dict())
    assert roundtripped.is_healthy == original.is_healthy
    assert roundtripped.consecutive_failures == original.consecutive_failures
    assert roundtripped.total_successes == original.total_successes
    assert roundtripped.total_latency_ms == original.total_latency_ms
    assert roundtripped.disabled_at == original.disabled_at
    assert roundtripped.cooldown_until == original.cooldown_until


def test_health_state_from_dict_with_missing_keys():
    """from_dict should use defaults for missing keys."""
    state = ProviderHealthState.from_dict({})
    assert state.is_healthy is True
    assert state.consecutive_failures == 0
    assert state.total_successes == 0
    assert state.total_latency_ms == 0.0
    assert state.disabled_at is None
    assert state.cooldown_until is None


# ===========================================================================
# Redis persistence
# ===========================================================================


class TestRedisPersistence:
    def test_redis_key_format(self):
        monitor = ProviderHealthMonitor()
        assert monitor._redis_key("runway") == f"{_REDIS_KEY_PREFIX}runway"

    async def test_persist_state_writes_to_redis(self):
        redis_mock = AsyncMock()
        redis_mock.set = AsyncMock(return_value=True)

        monitor = ProviderHealthMonitor(redis_client=redis_mock)
        state = ProviderHealthState(is_healthy=True, consecutive_failures=0, total_successes=5)
        await monitor._persist_state("runway", state)

        redis_mock.set.assert_awaited_once()
        call_args = redis_mock.set.call_args
        assert call_args[0][0] == f"{_REDIS_KEY_PREFIX}runway"
        payload = json.loads(call_args[0][1])
        assert payload["total_successes"] == 5
        assert call_args[1]["ttl"] == _REDIS_TTL_SECONDS

    async def test_persist_state_no_redis_is_noop(self):
        monitor = ProviderHealthMonitor(redis_client=None)
        # Should not raise
        await monitor._persist_state("test", ProviderHealthState())

    async def test_persist_state_redis_error_is_silent(self):
        redis_mock = AsyncMock()
        redis_mock.set = AsyncMock(side_effect=OSError("connection lost"))

        monitor = ProviderHealthMonitor(redis_client=redis_mock)
        # Should not raise
        await monitor._persist_state("test", ProviderHealthState())

    async def test_load_state_from_redis_dict(self):
        """When RedisClient returns deserialized dict."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(
            return_value={
                "is_healthy": False,
                "consecutive_failures": 3,
                "total_successes": 10,
                "total_latency_ms": 5000.0,
                "disabled_at": 1000.0,
                "cooldown_until": 1300.0,
            }
        )

        monitor = ProviderHealthMonitor(redis_client=redis_mock)
        state = await monitor._load_state("kling")
        assert state is not None
        assert state.is_healthy is False
        assert state.consecutive_failures == 3

    async def test_load_state_from_redis_json_string(self):
        """When Redis returns a raw JSON string."""
        data = {
            "is_healthy": True,
            "consecutive_failures": 0,
            "total_successes": 7,
            "total_latency_ms": 700.0,
            "disabled_at": None,
            "cooldown_until": None,
        }
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=json.dumps(data))

        monitor = ProviderHealthMonitor(redis_client=redis_mock)
        state = await monitor._load_state("veo")
        assert state is not None
        assert state.total_successes == 7

    async def test_load_state_cache_miss_returns_none(self):
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)

        monitor = ProviderHealthMonitor(redis_client=redis_mock)
        state = await monitor._load_state("unknown")
        assert state is None

    async def test_load_state_no_redis_returns_none(self):
        monitor = ProviderHealthMonitor(redis_client=None)
        state = await monitor._load_state("test")
        assert state is None

    async def test_load_state_redis_error_returns_none(self):
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(side_effect=OSError("timeout"))

        monitor = ProviderHealthMonitor(redis_client=redis_mock)
        state = await monitor._load_state("test")
        assert state is None

    async def test_load_all_from_redis(self):
        data_runway = {
            "is_healthy": True,
            "consecutive_failures": 0,
            "total_successes": 15,
            "total_latency_ms": 3000.0,
            "disabled_at": None,
            "cooldown_until": None,
        }
        data_kling = {
            "is_healthy": False,
            "consecutive_failures": 5,
            "total_successes": 3,
            "total_latency_ms": 600.0,
            "disabled_at": 100.0,
            "cooldown_until": 400.0,
        }

        async def mock_get(key):
            if "runway" in key:
                return data_runway
            if "kling" in key:
                return data_kling
            return None

        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(side_effect=mock_get)

        monitor = ProviderHealthMonitor(redis_client=redis_mock)
        await monitor.load_all_from_redis(["runway", "kling", "veo"])

        assert monitor._states["runway"].total_successes == 15
        assert monitor._states["kling"].is_healthy is False
        assert "veo" not in monitor._states

    async def test_load_all_from_redis_handles_errors(self):
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(side_effect=OSError("boom"))

        monitor = ProviderHealthMonitor(redis_client=redis_mock)
        # Should not raise
        await monitor.load_all_from_redis(["test"])
        # Falls back to no cached state
        assert "test" not in monitor._states

    def test_schedule_persist_no_redis_is_noop(self):
        monitor = ProviderHealthMonitor(redis_client=None)
        # Should not raise
        monitor._schedule_persist("test", ProviderHealthState())

    def test_schedule_persist_no_event_loop_is_silent(self):
        redis_mock = MagicMock()
        monitor = ProviderHealthMonitor(redis_client=redis_mock)
        # No running event loop -- should silently skip
        monitor._schedule_persist("test", ProviderHealthState())

    async def test_schedule_persist_task_ref_kept_then_discarded(self, monkeypatch):
        """Round 5: prove the persist task is held strongly while in-flight
        and removed once done -- same fix shape as image_gen_backend#3."""
        import asyncio

        redis_mock = AsyncMock()
        redis_mock.set = AsyncMock(return_value=True)
        monitor = ProviderHealthMonitor(redis_client=redis_mock)

        # Gate the inner _persist_state so we observe the task mid-flight.
        gate = asyncio.Event()
        real_persist = monitor._persist_state

        async def _gated(provider, state):
            await gate.wait()
            await real_persist(provider, state)

        monkeypatch.setattr(monitor, "_persist_state", _gated)

        assert monitor._background_tasks == set()
        monitor._schedule_persist("runway", ProviderHealthState(total_successes=1))
        assert len(monitor._background_tasks) == 1
        (task,) = tuple(monitor._background_tasks)
        gate.set()
        await task
        assert monitor._background_tasks == set()
        redis_mock.set.assert_awaited_once()
