"""Unit tests for the deterministic StubImageProvider."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from src.services.image_providers.stub import (
    _PNG_SIGNATURE,
    StubImageProvider,
    _color_for_prompt,
    _parse_size,
)


class TestStubImageProviderRendering:
    async def test_default_size_returns_valid_png(self) -> None:
        provider = StubImageProvider()
        result = await provider.generate(prompt="hello world")
        assert result.success is True
        assert result.image_data is not None
        assert result.image_data[:8] == _PNG_SIGNATURE
        assert result.mime_type == "image/png"
        assert result.width == 256
        assert result.height == 256
        # Pillow can round-trip the bytes.
        img = Image.open(io.BytesIO(result.image_data))
        assert img.size == (256, 256)
        assert img.format == "PNG"

    async def test_explicit_size_is_honored(self) -> None:
        provider = StubImageProvider()
        result = await provider.generate(prompt="x", size="640x480")
        assert result.width == 640
        assert result.height == 480
        assert result.image_data is not None
        img = Image.open(io.BytesIO(result.image_data))
        assert img.size == (640, 480)

    async def test_rectangular_size(self) -> None:
        provider = StubImageProvider()
        result = await provider.generate(prompt="x", size="1920x1080")
        assert result.image_data is not None
        img = Image.open(io.BytesIO(result.image_data))
        assert img.size == (1920, 1080)

    async def test_deterministic_for_same_prompt(self) -> None:
        provider = StubImageProvider()
        r1 = await provider.generate(prompt="prompt-A", size="64x64")
        r2 = await provider.generate(prompt="prompt-A", size="64x64")
        assert r1.image_data == r2.image_data

    async def test_different_prompts_produce_different_bytes(self) -> None:
        provider = StubImageProvider()
        r1 = await provider.generate(prompt="prompt-A", size="64x64")
        r2 = await provider.generate(prompt="prompt-B", size="64x64")
        assert r1.image_data != r2.image_data

    async def test_caching_returns_same_object(self) -> None:
        provider = StubImageProvider()
        r1 = await provider.generate(prompt="cache-me", size="32x32")
        r2 = await provider.generate(prompt="cache-me", size="32x32")
        # Same bytes object is reused (cache hit).
        assert r1.image_data is r2.image_data

    async def test_invalid_size_falls_back_to_default(self) -> None:
        provider = StubImageProvider()
        result = await provider.generate(prompt="x", size="not-a-size")
        assert result.width == 256
        assert result.height == 256

    async def test_negative_dimensions_fall_back(self) -> None:
        provider = StubImageProvider()
        result = await provider.generate(prompt="x", size="-1x-1")
        assert result.width == 256
        assert result.height == 256

    async def test_provider_job_id_is_prompt_derived(self) -> None:
        provider = StubImageProvider()
        r1 = await provider.generate(prompt="A")
        r2 = await provider.generate(prompt="A")
        r3 = await provider.generate(prompt="B")
        assert r1.provider_job_id == r2.provider_job_id
        assert r1.provider_job_id != r3.provider_job_id

    async def test_extra_params_are_accepted_and_ignored(self) -> None:
        provider = StubImageProvider()
        r = await provider.generate(
            prompt="x",
            negative_prompt="bad",
            style="editorial",
            size="64x64",
            quality="hd",
            reference_image_urls=["https://example.com/ref.png"],
            extra_params={"background": "transparent"},
            usage_context={"brand": "Acme"},
            target_usage="hero",
        )
        assert r.success is True

    async def test_pillow_can_decode_pixels(self) -> None:
        """Round-trip through Pillow yields the expected solid background."""
        provider = StubImageProvider()
        prompt = "deterministic-pixel-check"
        result = await provider.generate(prompt=prompt, size="32x32")
        assert result.image_data is not None
        img = Image.open(io.BytesIO(result.image_data))
        rgb = img.convert("RGB")
        # The first pixel (1,1) is in the text area; sample the bottom-right
        # corner instead, which is guaranteed to be the solid background.
        expected = _color_for_prompt(prompt)
        assert rgb.getpixel((30, 30)) == expected

    async def test_estimate_cost_is_zero(self) -> None:
        provider = StubImageProvider()
        assert provider.estimate_cost() == 0.0
        assert provider.estimate_cost(size="1024x1024") == 0.0
        assert provider.estimate_cost(quality="hd") == 0.0

    async def test_health_check(self) -> None:
        assert await StubImageProvider().health_check() is True

    def test_provider_metadata(self) -> None:
        provider = StubImageProvider()
        assert provider.name == "stub"
        assert provider.get_models() == ["stub-v1"]
        assert provider.get_styles() == ["default"]


class TestStubImageProviderEdgeCases:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            (None, (256, 256)),
            ("", (256, 256)),
            ("100x", (256, 256)),
            ("100", (256, 256)),
            ("0x0", (256, 256)),
            ("8x8", (8, 8)),
        ],
    )
    def test_parse_size(self, size: str | None, expected: tuple[int, int]) -> None:
        assert _parse_size(size) == expected

    async def test_tiny_image_skips_label(self) -> None:
        """Images below the label-threshold still produce valid PNGs."""
        provider = StubImageProvider()
        result = await provider.generate(prompt="tiny", size="8x8")
        assert result.image_data is not None
        img = Image.open(io.BytesIO(result.image_data))
        assert img.size == (8, 8)
        # All pixels should be the deterministic background color.
        rgb = img.convert("RGB")
        expected = _color_for_prompt("tiny")
        assert rgb.getpixel((0, 0)) == expected
        assert rgb.getpixel((7, 7)) == expected

    async def test_delay_is_awaited(self) -> None:
        import time

        provider = StubImageProvider(delay_s=0.05)
        t0 = time.monotonic()
        result = await provider.generate(prompt="x", size="16x16")
        elapsed = time.monotonic() - t0
        assert result.success is True
        assert elapsed >= 0.04
