"""The cleanup and reaper loops must run a pass BEFORE their first sleep.

Regression guard. Both ``_loop`` implementations used to ``await asyncio.sleep``
as the first statement of the loop body. For the cleanup loop that interval is
3600s while the pod rolls far more often than hourly, so it never completed a
single cycle in production. The reaper's 30s interval made its own case a mere
startup delay rather than a defect, but the shape is identical and worth pinning
so neither can regress.

Each test drives the real loop and asserts on an ordered log of what executed.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.cleanup_service import CleanupService
from src.services.reaper_service import ReaperService


async def test_cleanup_runs_a_cycle_before_sleeping() -> None:
    service = CleanupService(interval_seconds=3600)
    events: list[str] = []

    async def record_cleanup() -> None:
        events.append("run_cleanup")

    async def fake_sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")
        raise asyncio.CancelledError

    with (
        patch.object(service, "run_cleanup", AsyncMock(side_effect=record_cleanup)),
        patch("src.services.cleanup_service.asyncio.sleep", fake_sleep),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await service._loop()

    assert events == ["run_cleanup", "sleep:3600"]


async def test_cleanup_survives_a_failing_cycle_and_still_waits() -> None:
    """A raising cycle is swallowed AND still backs off for a full interval.

    Skipping the sleep on the error path would spin the loop hot against a
    database that is already unwell.
    """
    service = CleanupService(interval_seconds=3600)
    events: list[str] = []

    async def boom() -> None:
        events.append("run_cleanup")
        raise RuntimeError("db down")

    async def fake_sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")
        raise asyncio.CancelledError

    with (
        patch.object(service, "run_cleanup", AsyncMock(side_effect=boom)),
        patch("src.services.cleanup_service.asyncio.sleep", fake_sleep),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await service._loop()

    assert events == ["run_cleanup", "sleep:3600"]


async def test_reaper_sweeps_before_sleeping() -> None:
    service = ReaperService(
        MagicMock(),
        interval_s=30.0,
        lease_seconds=120,
        max_attempts=3,
    )
    events: list[str] = []

    async def record_sweep() -> int:
        events.append("reap_once")
        return 0

    async def fake_sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")
        raise asyncio.CancelledError

    with (
        patch.object(service, "reap_once", AsyncMock(side_effect=record_sweep)),
        patch("src.services.reaper_service.asyncio.sleep", fake_sleep),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await service._loop()

    assert events == ["reap_once", "sleep:30.0"]
