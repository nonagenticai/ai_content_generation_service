"""Unit tests for the deterministic StubVideoProvider.

Tests assume ffmpeg is on PATH. Skip the rendering tests if it isn't.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.services.providers.stub import StubVideoProvider, _render_mp4

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not installed in this environment"
)


def _ffprobe(path: Path) -> dict:
    """Return ffprobe JSON for a file."""
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None, "ffprobe must be on PATH for this test"
    result = subprocess.run(  # noqa: S603 -- ffprobe with controlled args
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return json.loads(result.stdout)


@needs_ffmpeg
class TestStubVideoProviderRendering:
    async def test_default_lifecycle_returns_valid_mp4(self, tmp_path: Path) -> None:
        provider = StubVideoProvider()
        # Pass explicit small dimensions -- the base provider's generate()
        # defaults to 1920x1080 which is a slow encode for a unit test.
        result = await provider.generate(
            prompt="hello world", duration_s=1.0, width=256, height=256
        )
        assert result.success is True
        assert result.video_data is not None
        # MP4 ftyp box is in the first 32 bytes.
        assert b"ftyp" in result.video_data[:32]
        # Round-trip through ffprobe.
        path = tmp_path / "out.mp4"
        path.write_bytes(result.video_data)
        info = _ffprobe(path)
        streams = info["streams"]
        assert len(streams) == 1
        v = streams[0]
        assert v["codec_name"] == "h264"
        assert v["pix_fmt"] == "yuv420p"
        assert v["width"] == 256
        assert v["height"] == 256
        # Duration is approximate -- ffmpeg encodes at a frame boundary.
        duration = float(info["format"]["duration"])
        assert 0.8 <= duration <= 1.3

    async def test_custom_dimensions_and_duration(self, tmp_path: Path) -> None:
        provider = StubVideoProvider()
        result = await provider.generate(prompt="x", duration_s=2.0, width=320, height=240)
        assert result.video_data is not None
        path = tmp_path / "out.mp4"
        path.write_bytes(result.video_data)
        info = _ffprobe(path)
        v = info["streams"][0]
        assert v["width"] == 320
        assert v["height"] == 240
        assert 1.7 <= float(info["format"]["duration"]) <= 2.3

    async def test_odd_dimensions_are_rounded_up(self, tmp_path: Path) -> None:
        # yuv420p requires even dimensions -- the stub must coerce.
        provider = StubVideoProvider()
        result = await provider.generate(prompt="x", duration_s=1.0, width=255, height=255)
        assert result.video_data is not None
        path = tmp_path / "out.mp4"
        path.write_bytes(result.video_data)
        info = _ffprobe(path)
        v = info["streams"][0]
        # Coerced to 256x256.
        assert v["width"] == 256
        assert v["height"] == 256

    async def test_caching_per_resolution_and_prompt(self) -> None:
        provider = StubVideoProvider()
        r1 = await provider.generate(prompt="cache", duration_s=1.0)
        r2 = await provider.generate(prompt="cache", duration_s=1.0)
        # Same bytes object reused (cache hit).
        assert r1.video_data is r2.video_data

    async def test_different_resolutions_render_separately(self) -> None:
        provider = StubVideoProvider()
        r1 = await provider.generate(prompt="x", duration_s=1.0, width=256, height=256)
        r2 = await provider.generate(prompt="x", duration_s=1.0, width=128, height=128)
        assert r1.video_data is not None
        assert r2.video_data is not None
        assert r1.video_data != r2.video_data

    async def test_different_prompts_render_different_bytes(self) -> None:
        provider = StubVideoProvider()
        r1 = await provider.generate(prompt="A", duration_s=1.0, width=64, height=64)
        r2 = await provider.generate(prompt="B", duration_s=1.0, width=64, height=64)
        assert r1.video_data != r2.video_data

    async def test_estimate_cost_is_zero(self) -> None:
        provider = StubVideoProvider()
        assert provider.estimate_cost(1.0) == 0.0
        assert provider.estimate_cost(60.0) == 0.0

    async def test_health_check(self) -> None:
        provider = StubVideoProvider()
        assert await provider.health_check() is True

    def test_provider_metadata(self) -> None:
        provider = StubVideoProvider()
        assert provider.name == "stub"
        assert provider.get_models() == ["stub-v1"]
        assert provider.get_max_duration_s() == 60.0
        assert provider.supports_image_to_video() is True

    async def test_can_handle_within_max_duration(self) -> None:
        provider = StubVideoProvider()
        assert provider.can_handle(duration_s=5.0) is True
        assert provider.can_handle(duration_s=120.0) is False
        assert provider.can_handle(duration_s=5.0, image_url="https://x/y.png") is True

    def test_render_mp4_starts_with_ftyp(self, tmp_path: Path) -> None:
        data = _render_mp4(width=64, height=64, duration_s=0.5, prompt="x")
        assert b"ftyp" in data[:32]
        path = tmp_path / "raw.mp4"
        path.write_bytes(data)
        info = _ffprobe(path)
        assert info["streams"][0]["codec_name"] == "h264"


class TestStubVideoProviderDownloadParams:
    async def test_submit_remembers_params_for_download(self) -> None:
        provider = StubVideoProvider()
        if not FFMPEG_AVAILABLE:
            pytest.skip("ffmpeg not installed")
        task_id = await provider.submit("remembered prompt", width=128, height=128, duration_s=1.0)
        # check_status is synchronous-completion in the stub.
        status = await provider.check_status(task_id)
        assert status["status"] == "completed"
        data = await provider.download(task_id, status["video_url"])
        assert b"ftyp" in data[:32]

    async def test_download_uses_default_params_when_kwargs_missing(self) -> None:
        provider = StubVideoProvider()
        if not FFMPEG_AVAILABLE:
            pytest.skip("ffmpeg not installed")
        task_id = await provider.submit("only-prompt")
        data = await provider.download(task_id, "stub://x")
        assert b"ftyp" in data[:32]
