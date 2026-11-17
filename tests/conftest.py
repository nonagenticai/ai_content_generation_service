"""Shared test fixtures for ai_content_generation_service."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from w2v_common.cache.redis_client import RedisClient
from w2v_common.testing import inject_fakeredis

from src.services.providers.base import VideoGenerationProvider, VideoProviderRegistry

# Capture the test DB URL BEFORE the autouse _clear_settings fixture
# monkeypatches DATABASE_URL away. CI sets DATABASE_URL to the per-job
# ephemeral postgres sidecar (see .github/workflows/ci-cd.yaml); locally
# falls back to the default below — start the test postgres with
# `docker compose up -d ai_content_generation_service_test_postgres`.
# Port 5472, not 5432, deliberately: a 5432 default resolves to whatever
# postgres the dev happens to be running, which answers the TCP connect and
# THEN fails auth — disguising a config error as a connection error.
_TEST_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://ci_test:ci_test_password_secure@127.0.0.1:5472/ai_content_generation_service_ci",  # pragma: allowlist secret
)

# w2v_common >= 0.15.0 dropped the localhost defaults on DatabaseSettings.url,
# RedisSettings.url, and StorageSettings.minio_endpoint. Settings() is
# instantiated at module-import time (e.g. when test files import from src.*
# during pytest collection), which happens BEFORE the autouse _clear_settings
# fixture runs. Seed module-level so collection succeeds; the autouse fixture
# restores the same placeholders after its delenv sweep.
os.environ.setdefault("DATABASE_URL", _TEST_DATABASE_URL)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("STORAGE_MINIO_ENDPOINT", "localhost:9000")

# ============================================================================
# Tenant ids
# ============================================================================
# Migration 011 flipped all four tenant_id columns (video_generation_jobs,
# video_generation_usage, imagegen_jobs, imagegen_batches) to native `uuid`,
# and the SECURITY DEFINER resolver expired_video_generation_jobs now RETURNS
# tenant_id uuid. A label like "tenant-a" therefore dies at INSERT with
# psycopg InvalidTextRepresentation. Every fixture, factory, parametrisation,
# request header and raw-SQL literal must supply a real uuid.
#
# These are deterministic and mutually DISTINCT on purpose: a test that used
# two different labels to prove tenant isolation still gets two genuinely
# different tenants, so the assertion keeps proving what it proved before.
TENANT_A = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
TENANT_B = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
TENANT_C = "cccccccc-3333-4333-8333-cccccccccccc"
TENANT_D = "dddddddd-4444-4444-8444-dddddddddddd"
TENANT_E = "eeeeeeee-5555-4555-8555-eeeeeeeeeeee"


# ============================================================================
# Settings cache management
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch):
    """Remove CI env vars that interfere with config-default tests."""
    for key in list(os.environ):
        if any(
            key.startswith(p)
            for p in (
                "DATABASE_",
                "AUTH_",
                "REDIS_",
                "API_",
                "STORAGE_",
                "LOG_",
                "CACHE_",
                "TRACING_",
                "LLM_",
                "FFMPEG_",
            )
        ):
            monkeypatch.delenv(key, raising=False)
    # w2v_common >= 0.15.0 makes DatabaseSettings.url / RedisSettings.url /
    # StorageSettings.minio_endpoint mandatory (no localhost defaults). Restore
    # placeholders so unit tests that instantiate Settings() — directly or via
    # get_settings() — don't ValidationError on missing fields.
    monkeypatch.setenv("DATABASE_URL", _TEST_DATABASE_URL)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "localhost:9000")


# ============================================================================
# Concrete test providers
# ============================================================================


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

    def get_max_duration_s(self) -> float | None:
        return 30.0

    def supports_image_to_video(self) -> bool:
        return True


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


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def fake_provider():
    return FakeProvider()


@pytest.fixture
def failing_provider():
    return FailingProvider()


@pytest.fixture
def registry():
    reg = VideoProviderRegistry()
    reg.register(FakeProvider())
    return reg


@pytest.fixture
def registry_with_failing():
    reg = VideoProviderRegistry()
    reg.register(FakeProvider())
    reg.register(FailingProvider())
    return reg


@pytest.fixture
def mock_settings():
    """Return a MagicMock mimicking Settings with all sub-sections."""
    settings = MagicMock()
    settings.storage.local_path = tempfile.mkdtemp()
    settings.storage.backend = "local"
    settings.storage.minio_endpoint = "localhost:9000"
    settings.storage.minio_access_key = ""
    settings.storage.minio_secret_key = ""
    settings.storage.minio_bucket = "test-bucket"
    settings.storage.minio_secure = False
    settings.generation.default_provider = "fake"
    settings.generation.default_duration_s = 5.0
    settings.generation.default_width = 1920
    settings.generation.default_height = 1080
    settings.generation.poll_interval_s = 0.1
    settings.generation.max_poll_time_s = 5.0
    settings.generation.parallel_requests = 3
    settings.generation.max_retries = 2
    settings.generation.fallback_enabled = True
    settings.generation.provider_strategy = "manual"
    settings.generation.fallback_chain = []
    settings.runway.api_key = "test-key"  # pragma: allowlist secret
    settings.runway.model = "gen-3"
    settings.runway.timeout = 300.0
    settings.kling.api_key = "test-key"  # pragma: allowlist secret
    settings.kling.model = "kling-v1"
    settings.kling.timeout = 300.0
    settings.veo.api_key = "test-key"  # pragma: allowlist secret
    settings.veo.model = "veo-2"
    settings.veo.timeout = 300.0
    settings.sora.api_key = "test-key"  # pragma: allowlist secret
    settings.sora.model = "sora-1"
    settings.sora.timeout = 300.0
    settings.seedance.api_key = "test-key"  # pragma: allowlist secret
    settings.seedance.model = "seedance-1"
    settings.seedance.timeout = 300.0
    settings.mock_provider.enabled = True
    settings.mock_provider.delay_s = 0.0
    settings.redis.url = "redis://localhost:6400/0"
    settings.redis.max_connections = 10
    settings.redis.socket_timeout = 5
    settings.redis.socket_connect_timeout = 5
    settings.redis.decode_responses = True
    settings.health_monitor.enabled = True
    settings.health_monitor.max_consecutive_failures = 5
    settings.health_monitor.cooldown_seconds = 300
    settings.quota.enabled = False
    settings.quota.default_max_daily_generations = 100
    settings.quota.default_max_daily_spend = 50.0
    settings.quota.default_max_concurrent_jobs = 10
    settings.rate_limit.enabled = False
    settings.cleanup.enabled = False
    settings.normalization.enabled = False
    return settings


# ---------------------------------------------------------------------------
# Real database fixtures (postgres+psycopg, TRUNCATE-based isolation)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_engine():
    """Per-test async postgres engine with TRUNCATE-based isolation.

    Connects to _TEST_DATABASE_URL captured at module load (CI postgres in
    CI, localhost:5434 locally). Runs Base.metadata.create_all (idempotent)
    then TRUNCATEs every model table so each test starts from a clean slate.
    """
    from src.database.models import Base

    engine = create_async_engine(_TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if Base.metadata.sorted_tables:
            tables = ", ".join(f'"{t.name}"' for t in reversed(Base.metadata.sorted_tables))
            await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session_factory(db_engine):
    """Real async_sessionmaker for ai_content_generation_service tests."""
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db_session(db_session_factory):
    """Real AsyncSession for a single test."""
    async with db_session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Real Redis fixtures (fakeredis)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_redis_client():
    """RedisClient backed by fakeredis (real commands, in-memory)."""
    client = RedisClient(redis_url="redis://localhost:6379/0")
    inject_fakeredis(client)
    return client
