"""An un-normalized clip must not be recorded as a normalized one.

``VideoNormalizer.normalize`` returned the RAW bytes on every failure path --
non-zero ffmpeg rc, timeout, missing output file, and ``ffmpeg`` not on PATH at
all -- with the same type and shape as success. The caller therefore could not
tell "scaled to 1920x1080@30fps, audio stripped" from "the provider's original
mp4, whatever resolution and audio it carries". ``_maybe_normalize`` added a
second such arm of its own.

Downstream, ``_persist_success`` marked the job ``completed`` with no marker at
all, the batch tracker counted it via ``record_clip_completed``, and
video_compiler concatenated the clip assuming uniform resolution/fps and no
audio track. The delivered video jumps resolution mid-way or carries provider
audio, and every job in the run is green. The ``FileNotFoundError`` arm makes
that fleet-wide and silent: one image without ffmpeg degrades EVERY clip it
touches while reporting nothing.

So the contract asserted here is: normalization reports an OUTCOME, the outcome
distinguishes "normalized" from "returned as-is", and a returned-as-is clip
reaches the terminal DB write as ``degradation``, not as a clean completion.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.normalizer import VideoNormalizer
from src.services.providers.base import VideoProviderRegistry
from src.services.video_generation_service import VideoGenerationService

RAW = b"video-data"


def _fake_process(returncode: int) -> AsyncMock:
    process = AsyncMock()
    process.communicate = AsyncMock(return_value=(b"", b"stderr text"))
    process.returncode = returncode
    return process


class TestNormalizerReportsItsOutcome:
    """The normalizer's return value must carry whether it normalized."""

    async def test_ffmpeg_missing_is_not_reported_as_normalized(self) -> None:
        """The fleet-wide arm: no ffmpeg binary at all."""
        norm = VideoNormalizer()
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("ffmpeg")):
            outcome = await norm.normalize(RAW, "job-1")

        assert outcome.data == RAW, "the clip itself must still be returned, never dropped"
        assert outcome.normalized is False
        assert outcome.reason == "ffmpeg_missing"

    async def test_nonzero_returncode_is_not_reported_as_normalized(self) -> None:
        norm = VideoNormalizer()
        with patch("asyncio.create_subprocess_exec", return_value=_fake_process(1)):
            outcome = await norm.normalize(RAW, "job-1")

        assert outcome.data == RAW
        assert outcome.normalized is False
        assert outcome.reason is not None and outcome.reason.startswith("ffmpeg_failed")

    async def test_timeout_is_not_reported_as_normalized(self) -> None:
        norm = VideoNormalizer(timeout_seconds=1)
        process = MagicMock()
        process.communicate = MagicMock()
        process.returncode = 1
        with (
            patch("asyncio.create_subprocess_exec", return_value=process),
            patch("asyncio.wait_for", side_effect=asyncio.TimeoutError),
        ):
            outcome = await norm.normalize(RAW, "job-1")

        assert outcome.data == RAW
        assert outcome.normalized is False
        assert outcome.reason == "ffmpeg_timeout"

    async def test_missing_output_file_is_not_reported_as_normalized(self) -> None:
        """rc==0 but ffmpeg wrote nothing -- the quietest arm of the four."""
        norm = VideoNormalizer()
        with patch("asyncio.create_subprocess_exec", return_value=_fake_process(0)):
            outcome = await norm.normalize(RAW, "job-1")

        assert outcome.data == RAW
        assert outcome.normalized is False
        assert outcome.reason == "ffmpeg_no_output"

    async def test_success_is_reported_as_normalized(self) -> None:
        """The positive control: without it every assertion above passes vacuously."""
        norm = VideoNormalizer()

        real_write_bytes = __import__("pathlib").Path.write_bytes

        async def _exec(*args: Any, **kwargs: Any) -> Any:
            del kwargs
            # ffmpeg's job: produce the output file the args name.
            from pathlib import Path

            real_write_bytes(Path(args[-1]), b"normalized-bytes")
            return _fake_process(0)

        with patch("asyncio.create_subprocess_exec", _exec):
            outcome = await norm.normalize(RAW, "job-1")

        assert outcome.data == b"normalized-bytes"
        assert outcome.normalized is True
        assert outcome.reason is None

    async def test_empty_input_is_not_a_degradation(self) -> None:
        """Nothing to normalize is not the same as failing to normalize."""
        outcome = await VideoNormalizer().normalize(b"", "job-1")
        assert outcome.data == b""
        assert outcome.reason is None


def _make_svc(**kwargs: Any) -> VideoGenerationService:
    return VideoGenerationService(
        storage=kwargs.pop("storage", AsyncMock()),
        registry=kwargs.pop("registry", VideoProviderRegistry()),
        default_provider=kwargs.pop("default_provider", "fake"),
        **kwargs,
    )


class TestServiceCarriesTheOutcome:
    async def test_normalizer_crash_is_a_degradation_not_a_clean_pass(self) -> None:
        """``_maybe_normalize``'s own ``except`` arm was the second silent one."""
        normalizer = MagicMock()
        normalizer.normalize = AsyncMock(side_effect=OSError("ffmpeg crashed"))
        svc = _make_svc(normalizer=normalizer)

        outcome = await svc._maybe_normalize(RAW, "job1")

        assert outcome.data == RAW
        assert outcome.normalized is False
        assert outcome.reason is not None

    async def test_no_normalizer_configured_is_not_a_degradation(self) -> None:
        """Normalization switched off is a deliberate posture, not a failure."""
        outcome = await _make_svc(normalizer=None)._maybe_normalize(RAW, "job1")
        assert outcome.data == RAW
        assert outcome.reason is None


class TestDegradationReachesTheTerminalWrite:
    """The whole point: the marker must survive to ``mark_completed``."""

    def _result(self) -> MagicMock:
        result = MagicMock()
        result.success = True
        result.video_data = RAW
        result.video_url = None
        result.duration_s = 4.0
        result.provider_job_id = "prov-1"
        return result

    async def _run_attempt(self, normalizer: Any) -> dict[str, Any]:
        storage = AsyncMock()
        storage.store_video = AsyncMock(return_value="/tmp/clip.mp4")
        storage.upload_to_minio = AsyncMock(return_value="minio/clip.mp4")
        svc = _make_svc(storage=storage, normalizer=normalizer)
        svc._is_cancelled = AsyncMock(return_value=False)  # type: ignore[method-assign]
        svc._try_generate_once = AsyncMock(return_value=(self._result(), 3.0))  # type: ignore[method-assign]

        db = AsyncMock()
        with patch("src.services.video_generation_service.JobManager") as jm:
            jm.set_provider_job_id = AsyncMock()
            jm.mark_completed = AsyncMock()
            jm.record_usage = AsyncMock()
            outcome, _cost = await svc._attempt_provider(
                db=db,
                job=MagicMock(),
                job_id="job1",
                tenant_id="t1",
                provider=MagicMock(),
                provider_name="fake",
                duration_s=4.0,
                estimated_cost=0.05,
                provider_errors={},
                settings=MagicMock(),
            )
        assert outcome == "completed", "precondition: the attempt reached the success path"
        assert jm.mark_completed.await_count == 1
        call = jm.mark_completed.await_args
        assert call is not None
        return dict(call.kwargs)

    async def test_a_failed_normalization_is_persisted_as_degraded(self) -> None:
        normalizer = MagicMock()
        normalizer.normalize = AsyncMock(side_effect=OSError("ffmpeg crashed"))

        kwargs = await self._run_attempt(normalizer)

        assert kwargs.get("degradation"), (
            "the job was marked completed with no degradation marker -- an "
            "un-normalized clip is indistinguishable from a normalized one, and "
            "video_compiler will concatenate it assuming uniform resolution/fps"
        )

    async def test_a_successful_normalization_is_persisted_clean(self) -> None:
        """Control: a normalized clip must NOT be marked degraded."""
        from src.services.normalizer import NormalizationOutcome

        normalizer = MagicMock()
        normalizer.normalize = AsyncMock(
            return_value=NormalizationOutcome(data=b"norm", normalized=True, reason=None)
        )

        kwargs = await self._run_attempt(normalizer)

        assert kwargs.get("degradation") is None


@pytest.mark.parametrize("reason", ["ffmpeg_missing", "ffmpeg_timeout"])
def test_reason_survives_into_the_persisted_string(reason: str) -> None:
    """The persisted marker must name WHICH failure, not just 'degraded'."""
    from src.services.video_generation_service import format_normalization_degradation

    marker = format_normalization_degradation(reason)
    assert marker is not None
    assert reason in marker
