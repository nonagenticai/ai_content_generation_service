"""Regression: a provider-side locator must never be published as ``video_url``.

``StubVideoProvider.check_status`` returned ``stub://video/<task_id>.mp4`` and
``WanVideoProvider.check_status`` returns a bare on-pod filesystem path. Both are
correct inputs to that provider's own ``download()`` -- and both are useless to
every OTHER consumer.

Production runs ``VIDEO_PROVIDER=stub``. The full chain that made this
product-breaking, and the reason an emptiness check could never catch it:

    stub.check_status         -> "stub://video/<id>.mp4"
    base.generate             -> VideoGenerationResult.video_url = "stub://..."
    _persist_success          -> jobs.video_url = "stub://..."
    GET /jobs/{id}/result     -> GenerationResultResponse.video_url = "stub://..."
    orchestrator _to_generated_video       -> GeneratedVideo.video_url = "stub://..."
    orchestrator _republish_relative       -> rewrites ONLY "/"-prefixed URLs, so
                                              "stub://" passes through UNTOUCHED
    orchestrator non-empty guard           -> PASSES ("stub://..." is a non-empty
                                              string and is not "none"/"null")
    orchestrator is_fetchable_url          -> REJECTS "stub://" -> shot DROPPED

so the stage reported COMPLETED with N videos while every ai_video shot was
silently absent from the finished render. A paid, apparently-successful run
delivered missing content.

These tests drive a REAL ``stub://`` value along that chain to the point of
consumption -- asserting on the provider's return alone would not prove the
pipeline stops accepting it.
"""

from __future__ import annotations

import shutil
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.api.routes.jobs import _preferred_video_url
from src.services.providers.base import VideoGenerationProvider, publishable_video_url
from src.services.providers.stub import StubVideoProvider

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE, reason="ffmpeg not installed in this environment"
)


def _orchestrator_is_fetchable_url(url: object) -> bool:
    """Byte-for-byte copy of the CONSUMER's predicate.

    Mirrors ``w2v_orchestrator/src/services/visual_media.py::is_fetchable_url``
    at origin/main. Copied rather than imported because the orchestrator is a
    separate service -- this is the exact rule that silently dropped the shot.
    """
    return isinstance(url, str) and (
        url.startswith(("http://", "https://")) or url.startswith("/api/runs/")
    )


def _orchestrator_republishes(url: object) -> bool:
    """The rewrite rule in ``ai_video_gen._republish_relative``: "/"-prefixed only.

    A URL this returns True for gets downloaded by the orchestrator and
    re-published into the run's artifact store as an ``/api/runs/...`` URL, which
    ``is_fetchable_url`` then accepts.
    """
    return isinstance(url, str) and url.startswith("/")


def _orchestrator_nonempty_guard_passes(url: object) -> bool:
    """The guard that was supposed to catch this and could not.

    ``ai_video_gen``: ``if not url or str(url).strip().lower() in {"none","null"}``
    -> raise. A non-empty ``"stub://..."`` sails straight through it.
    """
    return bool(url) and str(url).strip().lower() not in {"none", "null"}


def _job(*, job_id: str, video_url: str | None, minio_key: str | None, video_path: str | None):
    job = MagicMock()
    job.job_id = job_id
    job.video_url = video_url
    job.minio_key = minio_key
    job.video_path = video_path
    return job


class TestTheOldValueWouldStillFailToday:
    """Pin the defect itself, so the chain above cannot be dismissed as theory."""

    def test_stub_locator_passes_the_emptiness_guard_but_is_unfetchable(self) -> None:
        leaked = "stub://video/0199-abcd.mp4"
        # This is the whole bug in three assertions: loud guard says fine,
        # rewrite declines to touch it, consumer silently drops it.
        assert _orchestrator_nonempty_guard_passes(leaked) is True
        assert _orchestrator_republishes(leaked) is False
        assert _orchestrator_is_fetchable_url(leaked) is False

    def test_wan_pod_path_is_the_same_class(self) -> None:
        leaked = "/workspace/wan_outputs/0199-abcd.mp4"
        assert _orchestrator_nonempty_guard_passes(leaked) is True
        assert _orchestrator_is_fetchable_url(leaked) is False


class TestProducerNeverPublishesAnUnfetchableLocator:
    def test_stub_scheme_is_not_publishable(self) -> None:
        assert publishable_video_url("stub://video/x.mp4", provider="stub") is None

    def test_bare_pod_path_is_not_publishable(self) -> None:
        assert publishable_video_url("/workspace/out/x.mp4", provider="wan") is None

    def test_a_future_unknown_scheme_is_nulled_by_default(self) -> None:
        """Allow-list, not deny-list: this is why the defect cannot silently recur."""
        assert publishable_video_url("s3://bucket/key.mp4", provider="brandnew") is None
        assert publishable_video_url("file:///tmp/x.mp4", provider="brandnew") is None

    def test_a_real_provider_cdn_url_still_passes_through(self) -> None:
        cdn = "https://cdn.runwayml.com/abc.mp4"
        assert publishable_video_url(cdn, provider="runway") == cdn

    def test_none_stays_none(self) -> None:
        assert publishable_video_url(None, provider="stub") is None


class TestEndToEndFromRealStubProviderToConsumer:
    @needs_ffmpeg
    @pytest.mark.asyncio
    async def test_real_stub_generate_yields_bytes_but_no_leaked_locator(self) -> None:
        """The REAL provider, through the REAL ``generate()`` -- not a hand-built value.

        ``download()`` must still receive the provider locator (we get real MP4
        bytes back), while the PUBLISHED ``video_url`` must not carry it.
        """
        provider = StubVideoProvider()
        result = await provider.generate(prompt="a red square", duration_s=1.0, width=64, height=64)

        assert result.success is True
        # download() still worked -- the locator reached the one consumer it is for
        assert result.video_data is not None
        assert b"ftyp" in result.video_data[:32]
        # ...and did NOT reach any other consumer
        assert result.video_url is None, (
            f"provider leaked a non-fetchable locator: {result.video_url!r}"
        )

    @pytest.mark.asyncio
    async def test_any_provider_returning_a_scheme_locator_is_sanitised(self) -> None:
        """Covers the generic base path without needing ffmpeg."""

        class _LeakyProvider(VideoGenerationProvider):
            name = "leaky"
            seen_locator: str | None = None

            async def submit(self, prompt: str, **kwargs: Any) -> str:
                return "task-1"

            async def check_status(self, task_id: str) -> dict:
                return {
                    "status": "completed",
                    "video_url": f"leaky://video/{task_id}.mp4",
                    "error": None,
                }

            async def download(self, task_id: str, video_url: str) -> bytes:
                # The locator MUST still arrive here -- that is its legitimate use.
                type(self).seen_locator = video_url
                return b"\x00\x00\x00\x20ftypisom"

            def estimate_cost(self, duration_s: float) -> float:
                return 0.0

            async def health_check(self) -> bool:
                return True

            def get_models(self) -> list[str]:
                return ["leaky-v1"]

            def get_styles(self) -> list[str]:
                return ["default"]

        result = await _LeakyProvider().generate(prompt="prompt", duration_s=1.0)

        assert result.success is True
        assert _LeakyProvider.seen_locator == "leaky://video/task-1.mp4"
        assert result.video_url is None
        assert result.video_data == b"\x00\x00\x00\x20ftypisom"


class TestServicePublishesSomethingTheConsumerCanActuallyFetch:
    def test_stored_artifact_yields_a_republishable_path(self) -> None:
        """The positive half: the clip must actually LAND in the render."""
        url = _preferred_video_url(
            _job(job_id="job-1", video_url=None, minio_key="videos/job-1.mp4", video_path=None)
        )
        assert url == "/api/content/videos/jobs/job-1/video"
        # The orchestrator rewrites this one, which is what makes it reachable.
        assert _orchestrator_republishes(url) is True

    def test_local_artifact_also_yields_the_stream_through_path(self) -> None:
        url = _preferred_video_url(
            _job(job_id="job-2", video_url=None, minio_key=None, video_path="/data/job-2.mp4")
        )
        assert url == "/api/content/videos/jobs/job-2/video"

    def test_stored_artifact_wins_over_a_stale_provider_locator(self) -> None:
        """Belt and braces: even if a locator survived to the DB, it is not served."""
        url = _preferred_video_url(
            _job(
                job_id="job-3",
                video_url="stub://video/job-3.mp4",
                minio_key="videos/job-3.mp4",
                video_path=None,
            )
        )
        assert url == "/api/content/videos/jobs/job-3/video"
        assert "stub://" not in str(url)

    def test_provider_cdn_url_is_the_fallback_when_nothing_was_stored(self) -> None:
        cdn = "https://cdn.runwayml.com/job-4.mp4"
        url = _preferred_video_url(
            _job(job_id="job-4", video_url=cdn, minio_key=None, video_path=None)
        )
        assert url == cdn
        assert _orchestrator_is_fetchable_url(url) is True

    def test_nothing_fetchable_anywhere_is_none_so_the_stage_fails_loud(self) -> None:
        """The whole point: the un-fetchable case is expressible and loud.

        ``None`` trips the orchestrator's non-empty guard into
        ``StageDownstreamResponseError`` instead of being disguised as a URL and
        dropped in silence.
        """
        url = _preferred_video_url(
            _job(job_id="job-5", video_url=None, minio_key=None, video_path=None)
        )
        assert url is None
        assert _orchestrator_nonempty_guard_passes(url) is False
