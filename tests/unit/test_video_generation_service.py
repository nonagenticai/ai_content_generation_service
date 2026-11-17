"""Unit tests for ai_content_generation_service."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.services.providers.base import (
    _POLL_INITIAL_DELAY,
    _POLL_MAX_DELAY,
    _POLL_MULTIPLIER,
    ProviderInfo,
    VideoGenerationProvider,
    VideoGenerationResult,
    VideoProviderRegistry,
)

# ==========================================================================
# Concrete test provider for testing ABC
# ==========================================================================


class FakeProvider(VideoGenerationProvider):
    """Minimal concrete provider for testing."""

    name = "fake"

    async def submit(self, prompt: str, **kwargs) -> str:
        return "task-123"

    async def check_status(self, task_id: str) -> dict:
        return {"status": "completed", "video_url": "https://example.com/v.mp4", "error": None}

    async def download(self, task_id: str, video_url: str) -> bytes:
        return b"fake-video-data"

    def estimate_cost(self, duration_s: float) -> float:
        return 0.01 * duration_s


class FailingProvider(VideoGenerationProvider):
    """Provider that always fails."""

    name = "failing"

    async def submit(self, prompt: str, **kwargs) -> str:
        return "fail-task"

    async def check_status(self, task_id: str) -> dict:
        return {"status": "failed", "video_url": None, "error": "Generation error"}

    async def download(self, task_id: str, video_url: str) -> bytes:
        return b""

    def estimate_cost(self, duration_s: float) -> float:
        return 0.0


# ==========================================================================
# VideoGenerationResult tests
# ==========================================================================


class TestVideoGenerationResult(unittest.TestCase):
    def test_success_result(self):
        result = VideoGenerationResult(
            success=True,
            video_data=b"data",
            video_url="https://example.com/v.mp4",
            provider_job_id="task-1",
            duration_s=5.0,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.video_data, b"data")
        self.assertEqual(result.video_url, "https://example.com/v.mp4")
        self.assertEqual(result.provider_job_id, "task-1")
        self.assertEqual(result.duration_s, 5.0)
        self.assertIsNone(result.error)

    def test_failure_result(self):
        result = VideoGenerationResult(success=False, error="Something went wrong")
        self.assertFalse(result.success)
        self.assertIsNone(result.video_data)
        self.assertIsNone(result.video_url)
        self.assertEqual(result.error, "Something went wrong")

    def test_defaults(self):
        result = VideoGenerationResult(success=True)
        self.assertIsNone(result.video_data)
        self.assertIsNone(result.video_url)
        self.assertIsNone(result.provider_job_id)
        self.assertIsNone(result.duration_s)
        self.assertIsNone(result.error)


# ==========================================================================
# ProviderInfo tests
# ==========================================================================


class TestProviderInfo(unittest.TestCase):
    def test_basic_info(self):
        info = ProviderInfo(
            name="test", available=True, models=["m1"], supports_image_to_video=True
        )
        self.assertEqual(info.name, "test")
        self.assertTrue(info.available)
        self.assertEqual(info.models, ["m1"])
        self.assertTrue(info.supports_image_to_video)

    def test_defaults(self):
        info = ProviderInfo(name="test", available=False)
        self.assertEqual(info.models, [])
        self.assertEqual(info.styles, [])
        self.assertIsNone(info.max_duration_s)
        self.assertFalse(info.supports_image_to_video)


# ==========================================================================
# VideoProviderRegistry tests
# ==========================================================================


class TestVideoProviderRegistry(unittest.TestCase):
    def test_register_and_get(self):
        registry = VideoProviderRegistry()
        provider = FakeProvider()
        registry.register(provider)
        self.assertIs(registry.get("fake"), provider)

    def test_get_unknown_returns_none(self):
        registry = VideoProviderRegistry()
        self.assertIsNone(registry.get("nonexistent"))

    def test_provider_names(self):
        registry = VideoProviderRegistry()
        registry.register(FakeProvider())
        registry.register(FailingProvider())
        self.assertEqual(sorted(registry.provider_names), ["failing", "fake"])

    def test_list_providers(self):
        registry = VideoProviderRegistry()
        provider = FakeProvider()
        registry.register(provider)
        infos = registry.list_providers()
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].name, "fake")
        self.assertTrue(infos[0].available)

    def test_empty_registry(self):
        registry = VideoProviderRegistry()
        self.assertEqual(registry.list_providers(), [])
        self.assertEqual(registry.provider_names, [])

    def test_register_overwrites(self):
        registry = VideoProviderRegistry()
        p1 = FakeProvider()
        p2 = FakeProvider()
        registry.register(p1)
        registry.register(p2)
        self.assertIs(registry.get("fake"), p2)
        self.assertEqual(len(registry.list_providers()), 1)


# ==========================================================================
# VideoGenerationProvider ABC tests
# ==========================================================================


class TestVideoGenerationProviderABC(unittest.TestCase):
    def test_default_methods(self):
        provider = FakeProvider()
        self.assertEqual(provider.get_models(), [])
        self.assertEqual(provider.get_styles(), [])
        self.assertIsNone(provider.get_max_duration_s())
        self.assertFalse(provider.supports_image_to_video())

    def test_estimate_cost(self):
        provider = FakeProvider()
        self.assertAlmostEqual(provider.estimate_cost(5.0), 0.05)
        self.assertAlmostEqual(provider.estimate_cost(10.0), 0.1)

    def test_health_check_default(self):
        provider = FakeProvider()
        result = asyncio.run(provider.health_check())
        self.assertTrue(result)

    def test_can_handle_no_constraints(self):
        """Provider with no max_duration accepts any duration."""
        provider = FakeProvider()
        self.assertTrue(provider.can_handle(duration_s=100.0))
        self.assertTrue(provider.can_handle())

    def test_can_handle_duration_within_limit(self):
        """Provider with max_duration_s=10 accepts 5s."""

        class LimitedProvider(FakeProvider):
            def get_max_duration_s(self):
                return 10.0

        p = LimitedProvider()
        self.assertTrue(p.can_handle(duration_s=5.0))
        self.assertTrue(p.can_handle(duration_s=10.0))

    def test_can_handle_duration_exceeds_limit(self):
        """Provider with max_duration_s=10 rejects 15s."""

        class LimitedProvider(FakeProvider):
            def get_max_duration_s(self):
                return 10.0

        p = LimitedProvider()
        self.assertFalse(p.can_handle(duration_s=15.0))

    def test_can_handle_image_to_video_not_supported(self):
        """Provider without i2v rejects image_url requests."""
        provider = FakeProvider()
        # FakeProvider.supports_image_to_video() returns False
        self.assertFalse(provider.can_handle(image_url="https://example.com/img.png"))

    def test_can_handle_image_to_video_supported(self):
        """Provider with i2v accepts image_url requests."""

        class I2VProvider(FakeProvider):
            def supports_image_to_video(self):
                return True

        p = I2VProvider()
        self.assertTrue(p.can_handle(image_url="https://example.com/img.png"))

    def test_can_handle_no_image_url_always_ok(self):
        """Without image_url, even non-i2v providers are fine."""
        provider = FakeProvider()
        self.assertTrue(provider.can_handle(image_url=None))
        self.assertTrue(provider.can_handle())

    def test_generate_success(self):
        provider = FakeProvider()
        result = asyncio.run(provider.generate(prompt="test prompt", duration_s=5.0))
        self.assertTrue(result.success)
        self.assertEqual(result.video_data, b"fake-video-data")
        self.assertEqual(result.provider_job_id, "task-123")
        self.assertEqual(result.duration_s, 5.0)

    def test_generate_failure(self):
        provider = FailingProvider()
        result = asyncio.run(provider.generate(prompt="test prompt"))
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn("Generation error", result.error)


# ==========================================================================
# Polling constants tests
# ==========================================================================


class TestPollingConstants(unittest.TestCase):
    def test_initial_delay(self):
        self.assertEqual(_POLL_INITIAL_DELAY, 2.0)

    def test_multiplier(self):
        self.assertEqual(_POLL_MULTIPLIER, 1.5)

    def test_max_delay(self):
        self.assertEqual(_POLL_MAX_DELAY, 30.0)

    def test_exponential_growth(self):
        delay = _POLL_INITIAL_DELAY
        for _ in range(10):
            delay = min(delay * _POLL_MULTIPLIER, _POLL_MAX_DELAY)
        self.assertEqual(delay, _POLL_MAX_DELAY)


# ==========================================================================
# Provider cost estimation tests
# ==========================================================================


class TestProviderCosts(unittest.TestCase):
    @patch("src.services.providers.runway.get_client")
    @patch("src.services.providers.runway.get_settings")
    def test_runway_cost(self, mock_settings, _mock_get_client):
        mock_settings.return_value = MagicMock(
            runway=MagicMock(api_key="test", model="gen-3", timeout=300.0)
        )
        from src.services.providers.runway import RunwayProvider

        p = RunwayProvider()
        self.assertAlmostEqual(p.estimate_cost(5.0), 0.05)
        self.assertAlmostEqual(p.estimate_cost(10.0), 0.10)
        # Min 5s charge
        self.assertAlmostEqual(p.estimate_cost(1.0), 0.05)

    @patch("src.services.providers.kling.get_client")
    @patch("src.services.providers.kling.get_settings")
    def test_kling_cost(self, mock_settings, _mock_get_client):
        mock_settings.return_value = MagicMock(
            kling=MagicMock(api_key="test", model="kling-v1", timeout=300.0)
        )
        from src.services.providers.kling import KlingProvider

        p = KlingProvider()
        self.assertAlmostEqual(p.estimate_cost(5.0), 0.03)
        self.assertAlmostEqual(p.estimate_cost(10.0), 0.06)

    @patch("src.services.providers.veo.get_client")
    @patch("src.services.providers.veo.get_settings")
    def test_veo_cost(self, mock_settings, _mock_get_client):
        mock_settings.return_value = MagicMock(
            veo=MagicMock(api_key="test", model="veo-2", timeout=300.0)
        )
        from src.services.providers.veo import VeoProvider

        p = VeoProvider()
        self.assertAlmostEqual(p.estimate_cost(5.0), 0.04)
        self.assertAlmostEqual(p.estimate_cost(10.0), 0.08)


# ==========================================================================
# Provider model info tests
# ==========================================================================


class TestProviderModels(unittest.TestCase):
    @patch("src.services.providers.runway.get_client")
    @patch("src.services.providers.runway.get_settings")
    def test_runway_models(self, mock_settings, _mock_get_client):
        mock_settings.return_value = MagicMock(
            runway=MagicMock(api_key="key", model="gen-3", timeout=300.0)
        )
        from src.services.providers.runway import RunwayProvider

        p = RunwayProvider()
        self.assertEqual(p.get_models(), ["gen-3", "gen-3-turbo"])
        self.assertTrue(p.supports_image_to_video())
        self.assertEqual(p.get_max_duration_s(), 10.0)

    @patch("src.services.providers.kling.get_client")
    @patch("src.services.providers.kling.get_settings")
    def test_kling_models(self, mock_settings, _mock_get_client):
        mock_settings.return_value = MagicMock(
            kling=MagicMock(api_key="key", model="kling-v1", timeout=300.0)
        )
        from src.services.providers.kling import KlingProvider

        p = KlingProvider()
        self.assertEqual(p.get_models(), ["kling-v1", "kling-v1-5"])
        self.assertTrue(p.supports_image_to_video())

    @patch("src.services.providers.veo.get_client")
    @patch("src.services.providers.veo.get_settings")
    def test_veo_models(self, mock_settings, _mock_get_client):
        mock_settings.return_value = MagicMock(
            veo=MagicMock(api_key="key", model="veo-2", timeout=300.0)
        )
        from src.services.providers.veo import VeoProvider

        p = VeoProvider()
        self.assertEqual(p.get_models(), ["veo-2"])
        self.assertFalse(p.supports_image_to_video())


# ==========================================================================
# StorageService tests
# ==========================================================================


class TestStorageService(unittest.TestCase):
    def _storage_mock(self, tmpdir):
        return MagicMock(
            storage=MagicMock(
                local_path=tmpdir,
                backend="local",
                minio_endpoint="",
                minio_access_key="",
                minio_secret_key="",
                minio_bucket="test",
                minio_secure=False,
            )
        )

    @patch("src.services.storage_service.get_settings")
    def test_store_and_retrieve(self, mock_settings):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.return_value = self._storage_mock(tmpdir)
            from src.services.storage_service import StorageService

            svc = StorageService()

            async def _run():
                path = await svc.store_video("job1", b"video-data", "clip.mp4")
                self.assertIn("job1", path)
                self.assertTrue(Path(path).exists())

                data = await svc.retrieve(path)
                self.assertEqual(data, b"video-data")

            asyncio.run(_run())

    @patch("src.services.storage_service.get_settings")
    def test_retrieve_missing(self, mock_settings):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.return_value = self._storage_mock(tmpdir)
            from src.services.storage_service import StorageService

            svc = StorageService()
            result = asyncio.run(svc.retrieve("/nonexistent/path.mp4"))
            self.assertIsNone(result)

    @patch("src.services.storage_service.get_settings")
    def test_get_job_dir(self, mock_settings):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.return_value = self._storage_mock(tmpdir)
            from src.services.storage_service import StorageService

            svc = StorageService()
            job_dir = svc.get_job_dir("test-job-id")
            self.assertTrue(job_dir.exists())
            self.assertEqual(job_dir.name, "test-job-id")

    @patch("src.services.storage_service.get_settings")
    def test_initialize_creates_directory(self, mock_settings):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "subdir" / "storage"
            mock_settings.return_value = MagicMock(
                storage=MagicMock(
                    local_path=str(target),
                    backend="local",
                    minio_endpoint="",
                    minio_access_key="",
                    minio_secret_key="",
                    minio_bucket="test",
                    minio_secure=False,
                )
            )
            from src.services.storage_service import StorageService

            svc = StorageService()
            asyncio.run(svc.initialize())
            self.assertTrue(target.exists())


# ==========================================================================
# VideoGenerationService tests
# ==========================================================================


class TestVideoGenerationService(unittest.TestCase):
    def test_default_provider_fallback(self):
        from src.services.video_generation_service import VideoGenerationService

        registry = VideoProviderRegistry()
        registry.register(FakeProvider())
        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=registry,
            default_provider="fake",
        )
        self.assertEqual(svc._default_provider, "fake")

    def test_registry_wired(self):
        from src.services.video_generation_service import VideoGenerationService

        registry = VideoProviderRegistry()
        registry.register(FakeProvider())
        svc = VideoGenerationService(
            storage=MagicMock(),
            registry=registry,
            default_provider="fake",
        )
        self.assertIs(svc._registry, registry)
        self.assertIsNotNone(svc._registry.get("fake"))


if __name__ == "__main__":
    unittest.main()
