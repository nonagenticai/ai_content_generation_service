"""FFmpeg-based video normalization for consistent output format."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NormalizationOutcome:
    """What ``VideoNormalizer.normalize`` actually did to a clip.

    Every failure arm of ``normalize`` returns the ORIGINAL bytes, which are
    the same type and shape as a success. Returning bare ``bytes`` therefore
    made "scaled to the target resolution/fps with audio stripped"
    indistinguishable from "the provider's mp4, untouched" -- including the
    case where ``ffmpeg`` is not installed at all, which degrades every clip
    the image touches while reporting nothing. The caller needs the outcome,
    not just the payload.

    Attributes:
        data: The bytes to persist. On a failure arm these are the ORIGINAL
            bytes -- a clip we already paid for is never dropped.
        normalized: True only when ffmpeg produced the output file and it was
            read back. False on every failure arm AND when there was nothing
            to normalize.
        reason: Machine-readable identifier of the failure, or None when
            nothing went wrong. ``degraded`` is the flag to branch on --
            ``normalized is False`` alone also covers the empty-input case,
            which is not a fault.
    """

    data: bytes
    normalized: bool
    reason: str | None = None

    @property
    def degraded(self) -> bool:
        """True iff normalization was attempted and did not happen."""
        return self.reason is not None


class VideoNormalizer:
    """Normalizes video clips to a consistent format using FFmpeg."""

    def __init__(
        self,
        target_resolution: str = "1920:1080",
        target_fps: int = 30,
        target_codec: str = "libx264",
        strip_audio: bool = True,
        timeout_seconds: int = 120,
    ) -> None:
        self._resolution = target_resolution
        self._fps = target_fps
        self._codec = target_codec
        self._strip_audio = strip_audio
        self._timeout = timeout_seconds

    def _build_ffmpeg_args(self, input_path: str, output_path: str) -> list[str]:
        """Build FFmpeg command arguments."""
        args = [
            "ffmpeg",
            "-i",
            input_path,
            "-vf",
            f"scale={self._resolution}",
            "-r",
            str(self._fps),
            "-c:v",
            self._codec,
        ]
        if self._strip_audio:
            args.append("-an")
        args.extend(["-y", output_path])
        return args

    @staticmethod
    async def _reap(process: asyncio.subprocess.Process | None) -> None:
        """Kill and wait for an ffmpeg child that is still running.

        ``asyncio.wait_for`` only cancels ``communicate()``; the ffmpeg child
        keeps running with a handle on an input file that the enclosing
        ``TemporaryDirectory`` is about to delete. A finished process
        (``returncode`` set) is left alone.
        """
        if process is None or process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(Exception):
            await process.wait()

    async def normalize(self, video_data: bytes, job_id: str) -> NormalizationOutcome:
        """Normalize video data.

        Returns:
            A :class:`NormalizationOutcome`. On every failure arm the ORIGINAL
            bytes come back with ``normalized=False`` and a ``reason`` -- the
            clip is never dropped, but the caller can tell it apart from a
            normalized one and record the job as degraded.
        """
        if not video_data:
            # Nothing to normalize is not a failure: no reason, so not degraded.
            return NormalizationOutcome(data=video_data, normalized=False, reason=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = str(Path(tmpdir) / f"{job_id}_input.mp4")
            output_path = str(Path(tmpdir) / f"{job_id}_output.mp4")

            Path(input_path).write_bytes(video_data)

            args = self._build_ffmpeg_args(input_path, output_path)
            logger.info("Normalizing video for job %s: %s", job_id, " ".join(args))

            process: asyncio.subprocess.Process | None = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(process.communicate(), timeout=self._timeout)

                if process.returncode != 0:
                    logger.error(
                        "FFmpeg failed for job %s (rc=%d): %s",
                        job_id,
                        process.returncode,
                        stderr.decode(errors="replace")[:500],
                    )
                    return NormalizationOutcome(
                        data=video_data,
                        normalized=False,
                        reason=f"ffmpeg_failed_rc_{process.returncode}",
                    )

                output = Path(output_path)
                if output.exists():
                    result = output.read_bytes()
                    logger.info(
                        "Normalized video for job %s: %d -> %d bytes",
                        job_id,
                        len(video_data),
                        len(result),
                    )
                    return NormalizationOutcome(data=result, normalized=True, reason=None)

                # rc==0 but no output file: the quietest arm of the four.
                logger.error("FFmpeg reported success for job %s but wrote no output file", job_id)
                return NormalizationOutcome(
                    data=video_data, normalized=False, reason="ffmpeg_no_output"
                )

            except TimeoutError:
                logger.error("FFmpeg timed out for job %s after %ds", job_id, self._timeout)
                return NormalizationOutcome(
                    data=video_data, normalized=False, reason="ffmpeg_timeout"
                )
            except FileNotFoundError:
                # Louder than the per-clip arms on purpose: a missing binary is
                # not a transient, it is a property of the IMAGE. Every clip
                # this pod ever produces is un-normalized, and nothing else in
                # the pipeline will say so.
                logger.critical(
                    "FFmpeg not found in PATH -- normalization is enabled but IMPOSSIBLE on "
                    "this pod; EVERY clip it produces will be delivered un-normalized "
                    "(job %s)",
                    job_id,
                )
                return NormalizationOutcome(
                    data=video_data, normalized=False, reason="ffmpeg_missing"
                )
            finally:
                # Reaping lives here, not in the timeout arm, because the timeout
                # is not the only way out with a live child:
                # asyncio.CancelledError is a BaseException, so a cancel (local
                # task.cancel() or a cross-pod cancel) sails past both except
                # arms and would leave ffmpeg orphaned — still writing, while
                # the TemporaryDirectory below yanks its input file away.
                # No-op on the success path, where returncode is already set.
                await self._reap(process)
