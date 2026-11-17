"""Unit tests for MockImageProvider."""

from __future__ import annotations

from src.services.image_providers.mock import _MINIMAL_PNG, MockImageProvider


class TestMockImageProvider:
    async def test_generate_default(self) -> None:
        provider = MockImageProvider()
        result = await provider.generate(prompt="a test image")
        assert result.success is True
        assert result.image_data is not None
        assert result.image_data == _MINIMAL_PNG
        assert len(result.image_data) > 0
        assert result.mime_type == "image/png"
        assert result.width == 1
        assert result.height == 1
        assert result.provider_job_id == "mock-image-job"
        assert result.error is None

    async def test_generate_with_size(self) -> None:
        provider = MockImageProvider()
        result = await provider.generate(prompt="test", size="512x512")
        assert result.width == 512
        assert result.height == 512
        # Size param doesn't change the image data -- still returns the canned PNG
        assert result.image_data == _MINIMAL_PNG

    async def test_generate_with_rectangular_size(self) -> None:
        provider = MockImageProvider()
        result = await provider.generate(prompt="test", size="1024x768")
        assert result.width == 1024
        assert result.height == 768

    async def test_generate_with_delay(self) -> None:
        import time

        provider = MockImageProvider(delay_s=0.05)
        t0 = time.monotonic()
        result = await provider.generate(prompt="test")
        elapsed = time.monotonic() - t0
        assert result.success is True
        assert elapsed >= 0.04  # Allow slight timing slack

    async def test_generate_with_all_params(self) -> None:
        provider = MockImageProvider()
        result = await provider.generate(
            prompt="test",
            negative_prompt="bad",
            style="editorial",
            size="256x256",
            quality="hd",
            reference_image_urls=["https://example.com/img.png"],
            extra_params={"background": "transparent"},
            usage_context={"brand": "Acme"},
            target_usage="hero",
        )
        assert result.success is True
        assert result.provider_job_id == "mock-image-job"
        assert result.width == 256
        assert result.height == 256
        # All extra params accepted without error -- mock ignores them

    async def test_generate_returns_valid_png(self) -> None:
        """The returned image_data is a valid PNG (starts with PNG signature)."""
        provider = MockImageProvider()
        result = await provider.generate(prompt="test")
        # PNG files start with the 8-byte signature: 137 80 78 71 13 10 26 10
        png_signature = b"\x89PNG\r\n\x1a\n"
        assert result.image_data is not None
        assert result.image_data[:8] == png_signature

    def test_estimate_cost(self) -> None:
        provider = MockImageProvider()
        assert provider.estimate_cost() == 0.01
        assert provider.estimate_cost(size="1024x1024") == 0.01
        assert provider.estimate_cost(quality="hd") == 0.01
        # Cost is constant regardless of parameters
        assert provider.estimate_cost(size="4096x4096", quality="ultra") == 0.01

    async def test_health_check(self) -> None:
        provider = MockImageProvider()
        assert await provider.health_check() is True

    def test_get_models(self) -> None:
        provider = MockImageProvider()
        assert provider.get_models() == ["mock-image-v1"]

    def test_get_styles(self) -> None:
        provider = MockImageProvider()
        assert provider.get_styles() == ["default"]

    def test_name(self) -> None:
        provider = MockImageProvider()
        assert provider.name == "mock_images"

    async def test_generate_no_size_defaults_to_1x1(self) -> None:
        """Without size parameter, default resolution is 1x1."""
        provider = MockImageProvider()
        result = await provider.generate(prompt="test", size=None)
        assert result.width == 1
        assert result.height == 1

    async def test_generate_deterministic(self) -> None:
        """Multiple calls return identical results."""
        provider = MockImageProvider()
        r1 = await provider.generate(prompt="a")
        r2 = await provider.generate(prompt="b")
        assert r1.image_data == r2.image_data
        assert r1.mime_type == r2.mime_type
