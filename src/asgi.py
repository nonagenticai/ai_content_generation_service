"""
ASGI application for ai_content_generation_service.

FastAPI REST API following skill_builder/tts_service pattern.
"""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
from sqlalchemy.exc import SQLAlchemyError
from w2v_common.asgi import ASGIMiddleware, create_app
from w2v_common.observability import database_endpoint, redis_endpoint
from w2v_common.observability.request_context import RequestIdLogFilter

from src.cache.redis_client import close_redis, get_redis_client, init_redis
from src.config import Settings, get_settings
from src.database.connection import close_engine, get_engine

if TYPE_CHECKING:
    from src.cache.redis_client import RedisClient
    from src.services.cleanup_service import CleanupService
    from src.services.image_providers.base import ImageProviderRegistry
    from src.services.normalizer import VideoNormalizer
    from src.services.provider_health import ProviderHealthMonitor
    from src.services.providers.base import VideoProviderRegistry
    from src.services.storage_service import StorageService

load_dotenv()


def _setup_logging() -> None:
    """Configure logging (JSON format in production)."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    json_logs = os.getenv("LOG_JSON_LOGS", "").lower() in ("1", "true", "yes")

    if json_logs:
        import json as json_module

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_entry = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info and record.exc_info[1]:
                    log_entry["exception"] = self.formatException(record.exc_info)
                # Add trace context if available
                trace_id = getattr(record, "otelTraceID", None)
                if trace_id:
                    log_entry["trace_id"] = trace_id
                span_id = getattr(record, "otelSpanID", None)
                if span_id:
                    log_entry["span_id"] = span_id
                return json_module.dumps(log_entry)

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.root.handlers = [handler]
        logging.root.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


_setup_logging()
logger = logging.getLogger(__name__)

settings: Settings = get_settings()


# ============================================================================
# Health Check Logging Filter
# ============================================================================


class HealthCheckLoggingFilter(logging.Filter):
    """Suppress health-check request logs from uvicorn access logger."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for health-check paths to suppress their logs."""
        message = record.getMessage()
        return not any(
            path in message
            for path in ["/health", "/health/ready", "/health/live", "/readyz", "/healthz"]
        )


uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(HealthCheckLoggingFilter())


# ============================================================================
# Lifespan
# ============================================================================


def _setup_observability() -> None:
    """Initialize tracing + metrics if enabled in settings."""
    if settings.observability.tracing_enabled:
        from src.observability.tracing import setup_tracing

        setup_tracing(
            service_name=settings.observability.service_name,
            otlp_endpoint=settings.observability.otlp_endpoint or None,
        )

    if settings.observability.metrics_enabled:
        from src.observability.metrics import setup_metrics

        setup_metrics()


async def _init_database(app: FastAPI) -> None:
    """Connect to the database and store the engine on app.state.

    Raises on failure -- the service cannot start without a database.
    """
    # Phase D RLS: install the after_begin GUC re-bind listener before any session is
    # opened, so the multi-commit generation pipelines keep app.current_tenant bound
    # across each commit under enforce.
    from src.database.rls_events import install_rls_tenant_rebind

    install_rls_tenant_rebind()
    try:
        db_engine = await get_engine()
        app.state.db_engine = db_engine
        logger.info("Database connected: %s", database_endpoint(settings.database.url))
    except (OSError, SQLAlchemyError) as e:
        logger.error("Failed to connect to database: %s", e, exc_info=True)
        raise


async def _init_redis_optional(app: FastAPI) -> RedisClient | None:
    """Connect to Redis. Returns None on failure -- caching is optional."""
    try:
        redis_client = await init_redis()
        await redis_client.ping()
        app.state.redis_client = get_redis_client()
        logger.info("Redis connected: %s", redis_endpoint(settings.redis.url))
        return redis_client
    except OSError as e:
        logger.error("Failed to connect to Redis: %s", e, exc_info=True)
        logger.warning("Continuing without Redis -- caching will be disabled")
        return None


async def _init_storage(app: FastAPI) -> StorageService:
    """Initialize the storage service and stash it on app.state."""
    from src.services.storage_service import StorageService

    storage = StorageService()
    await storage.initialize()
    app.state.storage_service = storage
    return storage


async def _try_register_provider(
    registry: VideoProviderRegistry,
    name: str,
    factory: Any,
) -> None:
    """Instantiate ``factory()``, health-check it, register on success.

    Centralizes the boilerplate around the per-provider try/except block
    so each cloud provider does not need its own copy. ``factory`` is a
    zero-arg callable returning a provider instance (typically the class
    itself).
    """
    try:
        provider = factory()
        if await provider.health_check():
            registry.register(provider)
    except (OSError, ImportError, ValueError, RuntimeError) as e:
        logger.warning("Failed to initialize %s provider: %s", name, e)


async def _init_video_provider_registry() -> VideoProviderRegistry:
    """Build the video provider registry from configured cloud + local providers.

    Each provider is best-effort: failures are logged and skipped so a
    misconfigured key for one cloud provider does not break the whole
    service.
    """
    from src.services.providers.base import VideoProviderRegistry

    registry = VideoProviderRegistry()

    # Cloud providers -- each module is imported lazily so a missing
    # SDK only breaks the one provider, not service start-up.
    from src.services.providers.kling import KlingProvider
    from src.services.providers.runway import RunwayProvider
    from src.services.providers.seedance import SeedanceProvider
    from src.services.providers.sora import SoraProvider
    from src.services.providers.veo import VeoProvider

    await _try_register_provider(registry, "Runway", RunwayProvider)
    await _try_register_provider(registry, "Kling", KlingProvider)
    await _try_register_provider(registry, "Veo", VeoProvider)
    await _try_register_provider(registry, "Sora", SoraProvider)
    await _try_register_provider(registry, "Seedance", SeedanceProvider)

    if settings.wan.enabled:
        try:
            from src.services.providers.wan import WanProvider

            wan_provider = WanProvider()
            if await wan_provider.health_check():
                registry.register(wan_provider)
                logger.info("Wan2.2 provider registered")
        except (OSError, ImportError, ValueError, RuntimeError) as e:
            logger.warning("Failed to initialize Wan2.2 provider: %s", e)

    if settings.mock_provider.enabled:
        try:
            from src.services.providers.mock import MockProvider

            mock = MockProvider(delay_s=settings.mock_provider.delay_s)
            registry.register(mock)
            logger.info("Mock provider registered (delay=%.1fs)", settings.mock_provider.delay_s)
        except (OSError, ImportError, ValueError, RuntimeError) as e:
            logger.warning("Failed to initialize Mock provider: %s", e)

    if settings.stub.video_provider.lower() == "stub":
        try:
            from src.services.providers.stub import StubVideoProvider

            stub = StubVideoProvider()
            if await stub.health_check():
                registry.register(stub)
                logger.info("Stub video provider registered (VIDEO_PROVIDER=stub)")
            else:
                logger.error(
                    "Stub video provider health check FAILED (ffmpeg missing on PATH) -- "
                    "VIDEO_PROVIDER=stub is set, so this was the intended video provider; "
                    "no video generation can be served by this pod",
                )
        except (OSError, ImportError, ValueError, RuntimeError) as e:
            logger.warning("Failed to initialize stub video provider: %s", e)

    return registry


def _resolve_default_provider(registry: VideoProviderRegistry) -> None:
    """Adjust ``settings.generation.default_provider`` to one that registered.

    If the configured default did not come up (e.g. missing API key),
    fall back to the first successfully registered provider rather than
    serving traffic with a default that will always 404.

    When ``VIDEO_PROVIDER=stub`` is set, the stub provider becomes the
    default regardless of any other configured default -- the operator
    is explicitly opting into offline mode.
    """
    if settings.stub.video_provider.lower() == "stub" and "stub" in registry.provider_names:
        settings.generation.default_provider = "stub"
        return
    configured_default = settings.generation.default_provider
    if configured_default and configured_default not in registry.provider_names:
        if registry.provider_names:
            fallback = registry.provider_names[0]
            logger.warning(
                "Default provider %r not available, falling back to %r",
                configured_default,
                fallback,
            )
            settings.generation.default_provider = fallback
        else:
            logger.error(
                "NO VIDEO PROVIDERS REGISTERED -- default provider %r is unavailable and "
                "nothing else came up. Every video-generation request to this pod will "
                "fail; /health reports unhealthy (503) for as long as this holds.",
                configured_default,
            )


async def _init_health_monitor(
    app: FastAPI,
    registry: VideoProviderRegistry,
    redis_client: RedisClient | None,
) -> ProviderHealthMonitor | None:
    """Create + restore the provider health monitor (if enabled)."""
    from src.services.provider_health import ProviderHealthMonitor

    if not settings.health_monitor.enabled:
        return None
    health_monitor = ProviderHealthMonitor(
        max_consecutive_failures=settings.health_monitor.max_consecutive_failures,
        cooldown_seconds=settings.health_monitor.cooldown_seconds,
        redis_client=redis_client,
    )
    if redis_client is not None:
        try:
            await health_monitor.load_all_from_redis(registry.provider_names)
        except OSError as e:
            logger.warning("Failed to load health state from Redis: %s", e)
    app.state.health_monitor = health_monitor
    logger.info("Provider health monitor initialized")
    return health_monitor


def _build_normalizer() -> VideoNormalizer | None:
    """Build a VideoNormalizer iff normalization is enabled in settings."""
    if not settings.normalization.enabled:
        return None
    from src.services.normalizer import VideoNormalizer

    normalizer = VideoNormalizer(
        target_resolution=settings.normalization.target_resolution,
        target_fps=settings.normalization.target_fps,
        target_codec=settings.normalization.target_codec,
        strip_audio=settings.normalization.strip_audio,
        timeout_seconds=settings.normalization.timeout_seconds,
    )
    logger.info("Video normalizer enabled")
    return normalizer


async def _init_image_provider_registry() -> ImageProviderRegistry:
    """Build the image provider registry (OpenAI + optional mock)."""
    from src.services.image_providers.base import ImageProviderRegistry

    image_registry = ImageProviderRegistry()

    try:
        from src.services.image_providers.openai import OpenAIImageProvider

        image_provider = OpenAIImageProvider()
        if await image_provider.health_check():
            image_registry.register(image_provider)
    except (OSError, ImportError, ValueError, RuntimeError) as e:
        logger.warning("Failed to initialize OpenAI image provider: %s", e)

    if settings.local_image.enabled:
        try:
            from src.services.image_providers.local_image import LocalImageProvider

            local_provider = LocalImageProvider(settings.local_image)
            image_registry.register(local_provider)
            logger.info(
                "Local image provider registered (backend=%s)",
                settings.local_image.backend_url,
            )
        except (OSError, ImportError, ValueError, RuntimeError) as e:
            logger.warning("Failed to initialize local image provider: %s", e)

    if settings.mock_provider.enabled:
        try:
            from src.services.image_providers.mock import MockImageProvider

            mock_images = MockImageProvider(delay_s=settings.mock_provider.delay_s)
            image_registry.register(mock_images)
            logger.info(
                "Mock image provider registered (delay=%.1fs)", settings.mock_provider.delay_s
            )
        except (OSError, ImportError, ValueError, RuntimeError) as e:
            logger.warning("Failed to initialize mock image provider: %s", e)

    if settings.stub.image_provider.lower() == "stub":
        try:
            from src.services.image_providers.stub import StubImageProvider

            stub_images = StubImageProvider()
            image_registry.register(stub_images)
            logger.info("Stub image provider registered (IMAGE_PROVIDER=stub)")
        except (OSError, ImportError, ValueError, RuntimeError) as e:
            logger.warning("Failed to initialize stub image provider: %s", e)

    return image_registry


def _resolve_default_image_provider(image_registry: ImageProviderRegistry) -> None:
    """Adjust ``settings.image_generation.default_provider`` to one that registered.

    When ``IMAGE_PROVIDER=stub`` is set, the stub provider wins -- the
    operator has explicitly opted into offline mode.
    """
    if settings.stub.image_provider.lower() == "stub" and "stub" in image_registry.provider_names:
        settings.image_generation.default_provider = "stub"
        return
    configured_image_default = settings.image_generation.default_provider
    if configured_image_default and configured_image_default not in image_registry.provider_names:
        if image_registry.provider_names:
            fallback_image_provider = image_registry.provider_names[0]
            logger.warning(
                "Default image provider %r not available, falling back to %r",
                configured_image_default,
                fallback_image_provider,
            )
            settings.image_generation.default_provider = fallback_image_provider
        else:
            logger.warning(
                "Default image provider %r not available and no image providers registered",
                configured_image_default,
            )


def _maybe_init_quota(app: FastAPI, redis_client: RedisClient | None) -> None:
    """Build a QuotaService and attach to app.state if enabled."""
    if not settings.quota.enabled:
        return
    from src.services.quota_service import QuotaService

    quota_service = QuotaService(
        redis_client=redis_client,
        max_daily_generations=settings.quota.default_max_daily_generations,
        max_daily_spend=settings.quota.default_max_daily_spend,
        max_concurrent_jobs=settings.quota.default_max_concurrent_jobs,
        degraded_spend_budget=settings.quota.degraded_spend_budget,
    )
    app.state.quota_service = quota_service
    logger.info("Quota service initialized")


async def _maybe_start_cleanup(storage: StorageService) -> CleanupService | None:
    """Start the periodic cleanup service if enabled, return it for shutdown."""
    if not settings.cleanup.enabled:
        return None
    from src.database.session import get_session_maker
    from src.services.cleanup_service import CleanupService

    cleanup_service = CleanupService(
        interval_seconds=settings.cleanup.interval_seconds,
        completed_ttl_days=settings.cleanup.completed_ttl_days,
        failed_ttl_days=settings.cleanup.failed_ttl_days,
        cancelled_ttl_days=settings.cleanup.cancelled_ttl_days,
        storage=storage,
        session_factory=await get_session_maker(),
    )
    await cleanup_service.start()
    logger.info("Cleanup service started")
    return cleanup_service


async def _maybe_start_reaper(app: FastAPI) -> Any:
    """Start the durable-dispatch reaper if enabled, return it for shutdown."""
    if not settings.dispatch.reaper_enabled:
        return None
    from src.services.reaper_service import ReaperService

    reaper = ReaperService(
        app,
        interval_s=settings.dispatch.reaper_interval_s,
        lease_seconds=settings.dispatch.lease_seconds,
        max_attempts=settings.dispatch.max_attempts,
        max_lifetime_seconds=settings.dispatch.max_job_lifetime_seconds,
        batch_stale_seconds=settings.dispatch.batch_reap_stale_seconds,
        heartbeat_enabled=settings.dispatch.heartbeat_enabled,
    )
    await reaper.start()
    return reaper


async def _shutdown(app: FastAPI) -> None:
    """Run the shutdown sequence: reaper, cleanup, http clients, redis, db.

    Each close() is wrapped independently so a failure in one does not
    leak the others. We still log the failure for visibility.
    """
    logger.info("Shutting down AI Content Generation Service")
    reaper_service = getattr(app.state, "reaper_service", None)
    if reaper_service is not None:
        # Stop first so its shutdown lease-release lands before the DB closes,
        # letting a sibling reclaim this pod's in-flight jobs immediately.
        await reaper_service.stop()
    cleanup_service = getattr(app.state, "cleanup_service", None)
    if cleanup_service is not None:
        await cleanup_service.stop()
    try:
        from src.services.metering import close_metering_redis

        await close_metering_redis(getattr(app.state, "metering_redis", None))
    except ImportError:
        pass
    try:
        from src.services.job_manager import set_default_job_manager

        set_default_job_manager(None)
    except ImportError:
        pass
    try:
        from src.services.http_client import close_all

        await close_all()
    except OSError as e:
        logger.error("Error closing httpx clients: %s", e, exc_info=True)
    try:
        await close_redis()
    except OSError as e:
        logger.error("Error closing Redis: %s", e, exc_info=True)
    try:
        await close_engine()
    except (OSError, SQLAlchemyError) as e:
        logger.error("Error closing database: %s", e, exc_info=True)
    logger.info("AI Content Generation Service shutdown complete")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Back-compat context manager composing the create_app on_startup/on_shutdown
    hooks. The app itself is built via ``create_app`` (which wires its own lifespan
    from those hooks); this wrapper exists so tests can drive the full
    startup→shutdown cycle as a single ``async with``.
    """
    await _startup(app)
    try:
        yield
    finally:
        await _shutdown(app)


async def _startup(app: FastAPI) -> None:
    """Init DB, Redis, providers (create_app on_startup).

    This is intentionally a thin orchestrator: each phase is a small
    helper above. Adding a new dependency means adding a helper, not
    growing this function.
    """
    logger.info("=" * 70)
    logger.info("Starting AI Content Generation Service")
    logger.info("=" * 70)

    _setup_observability()
    await _init_database(app)
    redis_client = await _init_redis_optional(app)
    storage = await _init_storage(app)

    # Cross-pod cancellation flag store (shared by both generation services).
    from src.services.cancel_registry import CancelRegistry

    cancel_registry = CancelRegistry(redis_client)
    app.state.cancel_registry = cancel_registry

    registry = await _init_video_provider_registry()
    app.state.provider_registry = registry
    logger.info("Registered providers: %s", registry.provider_names)
    _resolve_default_provider(registry)

    health_monitor = await _init_health_monitor(app, registry, redis_client)

    from src.services.provider_selector import create_selector

    selector = create_selector(
        strategy=settings.generation.provider_strategy,
        default_provider=settings.generation.default_provider,
        fallback_chain=settings.generation.fallback_chain,
        redis_client=redis_client,
    )

    normalizer = _build_normalizer()

    from src.services.batch_tracker import BatchTracker

    app.state.batch_tracker = BatchTracker(redis_client=redis_client)

    # JobManager singleton wired with injected collaborators.
    # The legacy static-method interface continues to work; this
    # instance is the preferred entry point for new code via
    # ``Depends(get_job_manager)``.
    #
    # ``session_factory`` is left ``None`` because every existing
    # call site already passes an explicit ``db`` session via
    # FastAPI's ``Depends(get_db)`` chain. The factory is reserved
    # for future call sites that need to own their session lifetime
    # (background tasks, callbacks, etc.) and can be injected lazily
    # via ``app.state.job_manager._session_factory = ...`` if needed.
    from src.services.job_manager import JobManager, set_default_job_manager

    job_manager = JobManager(redis_client=redis_client)
    app.state.job_manager = job_manager
    set_default_job_manager(job_manager)

    from src.services.metering import init_metering_redis

    app.state.metering_redis = await init_metering_redis(settings.metering.redis_url)

    from src.services.video_generation_service import VideoGenerationService

    app.state.video_generation_service = VideoGenerationService(
        storage=storage,
        registry=registry,
        default_provider=settings.generation.default_provider,
        parallel_requests=settings.generation.parallel_requests,
        max_retries=settings.generation.max_retries,
        fallback_enabled=settings.generation.fallback_enabled,
        health_monitor=health_monitor,
        selector=selector,
        normalizer=normalizer,
        metering_redis=app.state.metering_redis,
        cancel_registry=cancel_registry,
    )

    image_registry = await _init_image_provider_registry()
    app.state.image_provider_registry = image_registry
    logger.info("Registered image providers: %s", image_registry.provider_names)
    _resolve_default_image_provider(image_registry)

    from src.services.imagegen_service import ImageGenService

    app.state.imagegen_service = ImageGenService(
        storage=storage,
        registry=image_registry,
        default_provider=settings.image_generation.default_provider,
        parallel_requests=settings.image_generation.parallel_requests,
        metering_redis=app.state.metering_redis,
        cancel_registry=cancel_registry,
    )

    _maybe_init_quota(app, redis_client)
    app.state.cleanup_service = await _maybe_start_cleanup(storage)
    app.state.reaper_service = await _maybe_start_reaper(app)

    logger.info("=" * 70)
    logger.info("AI Content Generation Service started successfully")
    logger.info("Environment: %s", settings.environment)
    logger.info("API: %s:%d", settings.api.host, settings.api.port)
    logger.info("=" * 70)


# ============================================================================
# Application
# ============================================================================

# Routers are imported here so create_app receives them via routers=.
from src.api.content_routes import router as concept_images_router  # noqa: E402
from src.api.health import router as health_router  # noqa: E402
from src.api.routes import router as api_router  # noqa: E402

# Service-specific middleware appended OUTSIDE the canonical stack. The rate
# limiter is opt-in (Phase 5) and takes per-endpoint RPM config; it shares the
# module-load redis singleton that lifespan's init_redis later connect()s. Its
# dispatch already tolerates a not-yet-connected redis (degrades to allow).
_extra_middleware: list[tuple[type[ASGIMiddleware], dict[str, Any]]] = []
if settings.rate_limit.enabled:
    from src.middleware.rate_limit import RateLimitMiddleware

    _extra_middleware.append(
        (
            RateLimitMiddleware,
            {
                "redis_client": get_redis_client(),
                "generate_rpm": settings.rate_limit.generate_requests_per_minute,
                "status_rpm": settings.rate_limit.status_requests_per_minute,
                "batch_rpm": settings.rate_limit.batch_requests_per_minute,
            },
        )
    )
    logger.info("Rate limiting middleware enabled")

app = create_app(
    settings,
    routers=[health_router, api_router, concept_images_router],
    title=settings.app_name,
    description="Multi-provider AI content generation microservice for video and image assets",
    version=settings.app_version,
    on_startup=[_startup],
    on_shutdown=[_shutdown],
    extra_middleware=_extra_middleware,
    redirect_slashes=False,
    swagger_ui_parameters={"persistAuthorization": True},
    # Fallback unique-id generator. Every route still sets `operation_id=`
    # explicitly per OPERATION_ID_PLAN.md; this exists so a future route
    # that forgets the kwarg degrades to `<tag>_<func>` instead of FastAPI's
    # default `<func>_<path>_<method>` (garbage in generated clients).
    generate_unique_id_function=lambda route: (
        f"{route.tags[0] if route.tags else 'default'}_{route.name}"
    ),
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness, dependency probes."},
        {
            "name": "content-videos",
            "description": (
                "Multi-provider AI video generation: submission, jobs, batches, "
                "providers, preview, usage/quota."
            ),
        },
        {
            "name": "concept-images",
            "description": (
                "AI-generated concept images: jobs, batches, asset retrieval and download."
            ),
        },
    ],
)

# Bind request_id onto every log record (factory sets the ContextVar via
# RequestIdMiddleware; this filter surfaces it in the log formatter).
logging.getLogger().addFilter(RequestIdLogFilter())


# ============================================================================
# Root & Health Endpoints
# ============================================================================


@app.get(
    "/", operation_id="service_root", tags=["health"], summary="Service banner + endpoint index"
)
async def root() -> dict[str, Any]:
    """Return service info and available endpoint map."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Multi-provider AI content generation microservice for video and image assets",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "generate_video": "/api/content/videos/generate",
            "video_job": "/api/content/videos/jobs/{id}",
            "video_result": "/api/content/videos/jobs/{id}/video",
            "concept_images": "/api/concept-images",
            "concept_image_job": "/api/concept-images/jobs/{id}",
            "concept_image_batch": "/api/concept-images/batches/{id}",
            "providers": "/api/content/videos/providers",
            "concept_image_asset": "/api/concept-images/assets/{id}",
        },
        "status": "running",
        "environment": settings.environment,
    }


@app.get(
    "/metrics",
    operation_id="prometheus_metrics",
    tags=["health"],
    summary="Prometheus metrics scrape endpoint",
)
async def metrics_endpoint() -> Response:
    """Prometheus metrics endpoint."""
    from src.observability.metrics import get_metrics_response

    data, content_type = get_metrics_response()
    return Response(content=data, media_type=content_type)


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Main entry point for running the ASGI application."""
    import uvicorn

    uvicorn.run(
        "src.asgi:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
        reload=settings.api.reload,
    )


if __name__ == "__main__":
    main()
