"""
Configuration management for ai_content_generation_service using Pydantic Settings.

Hierarchical settings with sub-configs.
Follows skill_builder/tts_service config pattern.
"""

import base64
import binascii
import re
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from w2v_common.config import APISettings as _BaseAPISettings
from w2v_common.config import AuthSettings, DatabaseSettings, LoggingSettings, RedisSettings
from w2v_common.config import StorageSettings as _BaseStorageSettings

__all__ = [
    "AuthSettings",
    "DatabaseSettings",
    "LoggingSettings",
    "RedisSettings",
]


class APISettings(_BaseAPISettings):
    """AI content generation service API settings (port 6310)."""

    port: int = Field(default=6310, description="API server port")


class StorageSettings(_BaseStorageSettings):
    """File storage configuration."""

    minio_bucket: str = Field(
        default="ai-content-generation-service", description="MinIO bucket name"
    )
    minio_external_endpoint: str | None = Field(
        default=None,
        description="External MinIO endpoint for presigned URLs (replaces internal host when set)",
    )


class RunwaySettings(BaseSettings):
    """Runway ML configuration."""

    api_key: str = Field(default="", description="Runway API key")
    model: str = Field(default="gen-3", description="Runway model")
    timeout: float = Field(default=300.0, description="Request timeout")

    model_config = SettingsConfigDict(
        env_prefix="RUNWAY_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class KlingSettings(BaseSettings):
    """Kling AI configuration."""

    api_key: str = Field(default="", description="Kling API key")
    model: str = Field(default="kling-v1", description="Kling model")
    timeout: float = Field(default=300.0, description="Request timeout")

    model_config = SettingsConfigDict(
        env_prefix="KLING_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class VeoSettings(BaseSettings):
    """Google Veo configuration."""

    api_key: str = Field(default="", description="Google API key")
    model: str = Field(default="veo-2", description="Veo model")
    timeout: float = Field(default=300.0, description="Request timeout")

    model_config = SettingsConfigDict(
        env_prefix="VEO_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class SoraSettings(BaseSettings):
    """OpenAI Sora configuration."""

    api_key: str = Field(default="", description="OpenAI API key")
    model: str = Field(default="sora-1", description="Sora model")
    timeout: float = Field(default=300.0, description="Request timeout")

    model_config = SettingsConfigDict(
        env_prefix="SORA_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class SeedanceSettings(BaseSettings):
    """Seedance configuration."""

    api_key: str = Field(default="", description="Seedance API key")
    model: str = Field(default="seedance-1", description="Seedance model")
    timeout: float = Field(default=300.0, description="Request timeout")

    model_config = SettingsConfigDict(
        env_prefix="SEEDANCE_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class WanSettings(BaseSettings):
    """Wan2.2 local video generation configuration."""

    enabled: bool = Field(default=False, description="Enable Wan2.2 local provider")
    model_name: str = Field(default="Wan-AI/Wan2.2-TI2V-5B", description="HuggingFace model name")
    device: str = Field(default="auto", description="Device: auto/cuda/mps/cpu")
    offload_model: bool = Field(default=True, description="Offload model layers to save VRAM")
    convert_model_dtype: bool = Field(default=True, description="Convert to float16")
    t5_cpu: bool = Field(default=True, description="Run T5 text encoder on CPU")
    timeout: float = Field(default=600.0, description="Generation timeout in seconds")

    model_config = SettingsConfigDict(
        env_prefix="WAN_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class LocalImageSettings(BaseSettings):
    """Local on-device image generation (mflux/FLUX on mini1)."""

    enabled: bool = Field(default=False, description="Enable local mflux image provider")
    backend_url: str = Field(
        default="http://image-gen.test:9204",
        description="Base URL of the mflux/FLUX backend",
    )
    # mflux on mini1 generates a 896x512 image and LANCZOS-upscales it to 1536px
    # in ~54-70s; the old 60s ceiling clipped the slower gens and forced the
    # caller (ai_image_gen) into retry churn. 180s clears the gen + upscale +
    # transfer with headroom; the backend bounds the actual work.
    timeout: float = Field(default=180.0, description="Request timeout in seconds")
    # Service-account license token for authenticating to an in-cluster
    # modality-gateway (which is fail-closed: ENABLE_AUTH=true, no trusted-network
    # bypass). When set, it is sent verbatim as ``X-Tenant-Id: Bearer <token>`` on
    # every outbound call; ``identity_event_hook`` preserves that Bearer form and
    # ADDS ``X-Effective-Tenant-Id: <customer tenant>`` from the request context,
    # so the gateway resolves per-customer config + attributes cost to the real
    # tenant (the token's subject must be in the gateway's AUTH_DELEGATING_SUBJECTS).
    # Empty (default) => no Bearer header => talk directly to the mflux backend
    # exactly as before (bare-UUID X-Tenant-Id). See identity_event_hook docstring.
    service_token: str = Field(
        default="",
        description="SA license token for X-Tenant-Id Bearer auth to the modality-gateway",
    )
    # The mflux backend on mini1 is SINGLE-FLIGHT: it generates one image at a
    # time and returns HTTP 503 while busy. A batch dispatched as per-child
    # concurrent tasks (dispatch layer) would otherwise fire N concurrent
    # requests and 503 them all. `max_concurrency` caps the number of in-pod
    # concurrent backend calls (default 1 = match the backend's single-flight
    # nature); a batch's children serialize on the semaphore WITHIN a pod while
    # still holding their renewing lease, so durability is preserved.
    max_concurrency: int = Field(
        default=1,
        ge=1,
        description="Max concurrent in-pod calls to the single-flight mflux backend",
    )
    # Cross-pod 503/429 handling: with >1 ai-content replica, each pod's single
    # slot can still collide at the backend (two pods, one image-gen box). On a
    # 503/429 the provider sleeps and retries (honouring Retry-After) up to
    # `busy_retry_attempts` with exponential backoff from `busy_retry_base_delay`.
    busy_retry_attempts: int = Field(
        default=5,
        ge=0,
        description="Retries on a busy (503/429) backend before failing the generation",
    )
    busy_retry_base_delay: float = Field(
        default=2.0,
        ge=0.0,
        description="Base seconds for exponential backoff between busy-backend retries",
    )

    model_config = SettingsConfigDict(
        env_prefix="LOCAL_IMAGE_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class MockProviderSettings(BaseSettings):
    """Mock provider configuration for testing/development."""

    enabled: bool = Field(default=False, description="Enable mock provider")
    delay_s: float = Field(default=0.0, description="Simulated generation delay in seconds")

    model_config = SettingsConfigDict(
        env_prefix="MOCK_PROVIDER_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class StubProviderSettings(BaseSettings):
    """Top-level stub provider switches.

    When ``image_provider == "stub"`` the service registers a deterministic
    image stub provider (real PNG bytes) and uses it as the default image
    provider. Same for ``video_provider``: a deterministic, ffprobe-valid
    H.264/MP4 stub is registered and used as the default video provider.

    These switches are intended for offline pipelines (CI, demos, local
    dev) where paid APIs are unavailable. They are opt-in; default
    behavior is unchanged.
    """

    image_provider: str = Field(
        default="",
        description="Set to 'stub' to use the deterministic image stub provider",
    )
    video_provider: str = Field(
        default="",
        description="Set to 'stub' to use the deterministic video stub provider",
    )

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class DalleImageSettings(BaseSettings):
    """OpenAI image generation configuration."""

    enabled: bool = Field(default=False, description="Enable OpenAI image generation")
    api_key: str = Field(default="", description="OpenAI API key for image generation")
    model: str = Field(default="gpt-image-1", description="OpenAI image model")
    size: str = Field(default="1792x1024", description="Default image size")
    quality: str = Field(default="high", description="Image quality")
    background: str = Field(default="auto", description="Background handling mode")
    output_format: str = Field(default="png", description="Image output format")
    timeout: float = Field(default=120.0, description="Request timeout")
    base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")

    model_config = SettingsConfigDict(
        env_prefix="DALLE_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class ObservabilitySettings(BaseSettings):
    """Observability configuration for tracing and metrics."""

    otlp_endpoint: str = Field(default="", description="OTLP gRPC endpoint for traces")
    service_name: str = Field(
        default="ai-content-generation-service", description="Service name for traces"
    )
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics endpoint")
    tracing_enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing")

    model_config = SettingsConfigDict(
        env_prefix="OTEL_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class GenerationSettings(BaseSettings):
    """Default generation configuration."""

    default_provider: str = Field(default="runway", description="Default provider name")
    default_duration_s: float = Field(default=5.0, description="Default clip duration")
    default_width: int = Field(default=1920, description="Default video width")
    default_height: int = Field(default=1080, description="Default video height")
    max_retries: int = Field(default=2, description="Max retries per generation")
    poll_interval_s: float = Field(default=5.0, description="Polling interval for async providers")
    max_poll_time_s: float = Field(default=600.0, description="Max polling time before timeout")
    parallel_requests: int = Field(default=3, ge=1, le=10, description="Max concurrent generations")
    fallback_enabled: bool = Field(default=True, description="Enable provider fallback on failure")
    provider_strategy: str = Field(default="manual", description="Provider selection strategy")
    fallback_chain: list[str] = Field(
        default_factory=list, description="Ordered fallback providers"
    )

    model_config = SettingsConfigDict(
        env_prefix="GENERATION_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class ImageGenerationSettings(BaseSettings):
    """Default image generation configuration."""

    default_provider: str = Field(default="openai_images", description="Default image provider")
    parallel_requests: int = Field(default=4, ge=1, le=20, description="Max concurrent requests")
    poll_interval_s: float = Field(default=2.0, description="Polling interval for async jobs")
    max_poll_time_s: float = Field(default=180.0, description="Maximum polling time")

    model_config = SettingsConfigDict(
        env_prefix="IMAGE_GENERATION_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class HealthMonitorSettings(BaseSettings):
    """Provider health monitoring configuration."""

    enabled: bool = Field(default=True, description="Enable provider health monitoring")
    max_consecutive_failures: int = Field(
        default=5, description="Failures before disabling provider"
    )
    cooldown_seconds: int = Field(default=300, description="Cooldown period after disabling")

    model_config = SettingsConfigDict(
        env_prefix="HEALTH_MONITOR_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class QuotaSettings(BaseSettings):
    """Per-tenant usage quota configuration."""

    enabled: bool = Field(default=True, description="Enable quota enforcement")
    default_max_daily_generations: int = Field(default=100, description="Max daily generations")
    default_max_daily_spend: float = Field(default=50.0, description="Max daily spend in USD")
    default_max_concurrent_jobs: int = Field(
        default=10, description="Max concurrent jobs per tenant"
    )
    degraded_spend_budget: float = Field(
        default=25.0,
        ge=0.0,
        description=(
            "USD of estimated generation this pod may admit while the quota store "
            "is unreachable, before admission is refused. Bounds an outage's "
            "unmetered spend; 0 makes the quota check fail closed."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="QUOTA_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class RateLimitSettings(BaseSettings):
    """API rate limiting configuration."""

    enabled: bool = Field(default=False, description="Enable rate limiting")
    generate_requests_per_minute: int = Field(default=10, description="Generate endpoint RPM")
    status_requests_per_minute: int = Field(default=60, description="Status endpoint RPM")
    batch_requests_per_minute: int = Field(default=5, description="Batch endpoint RPM")

    model_config = SettingsConfigDict(
        env_prefix="RATE_LIMIT_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class CleanupSettings(BaseSettings):
    """Job cleanup/expiry configuration."""

    enabled: bool = Field(default=False, description="Enable automatic job cleanup")
    interval_seconds: int = Field(default=3600, description="Cleanup interval in seconds")
    completed_ttl_days: int = Field(default=30, description="TTL for completed jobs in days")
    failed_ttl_days: int = Field(default=7, description="TTL for failed jobs in days")
    cancelled_ttl_days: int = Field(
        default=7, description="TTL for cancelled jobs in days (row AND its MinIO object)"
    )

    model_config = SettingsConfigDict(
        env_prefix="CLEANUP_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class NormalizationSettings(BaseSettings):
    """Video normalization configuration."""

    enabled: bool = Field(default=False, description="Enable video normalization")
    target_resolution: str = Field(default="1920:1080", description="Target resolution (W:H)")
    target_fps: int = Field(default=30, description="Target frame rate")
    target_codec: str = Field(default="libx264", description="Target video codec")
    strip_audio: bool = Field(default=True, description="Strip audio from output")
    timeout_seconds: int = Field(default=120, description="FFmpeg timeout")

    model_config = SettingsConfigDict(
        env_prefix="NORMALIZATION_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class MeteringSettings(BaseSettings):
    """Cost-event metering configuration.

    Emits one event per successful video/image generation to a Redis stream
    consumed by metering_service. ``redis_url`` is SEPARATE from the
    service's regular cache/job Redis. Leave empty to disable silently.

    The cost-bus stream name is resolved centrally by ``w2v_common.metering``
    (``METERING_STREAM`` env → canonical ``improvement-loop:costs``), so it is
    not a field here.
    """

    redis_url: str = Field(default="", description="Metering Redis URL")

    model_config = SettingsConfigDict(
        env_prefix="METERING_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class DispatchSettings(BaseSettings):
    """Durable multi-replica dispatch (claim-lease + heartbeat + reaper).

    A fire-and-forget ``asyncio.create_task`` dies with its pod. These knobs
    drive the durability layer: every dispatched job carries a lease renewed by
    a heartbeat, and a per-pod reaper reclaims (or fails) jobs whose lease
    lapsed because the owning pod crashed — so any replica can resume them.
    """

    reaper_enabled: bool = Field(
        default=True, description="Run the per-pod orphaned-job reaper loop"
    )
    reaper_interval_s: float = Field(default=30.0, description="Seconds between reaper sweeps")
    lease_seconds: int = Field(
        default=900, description="Job lease TTL; a processing job past this is reclaimable"
    )
    heartbeat_enabled: bool = Field(
        default=True, description="Renew the lease while a job runs (prevents premature reclaim)"
    )
    max_attempts: int = Field(
        default=5, description="Reclaim attempts before a job is failed as poison"
    )
    # Kept as-is alongside the attempt REFUND: the refund only fires on a GRACEFUL
    # release (rollout/eviction), so a job that keeps genuinely killing its worker
    # never reaches release_claims and still trips this counter fast. The counter
    # is the fast guard; max_job_lifetime_seconds is the slow, wall-clock backstop.
    max_job_lifetime_seconds: int = Field(
        default=3600,
        description=(
            "Wall-clock ceiling on a job's total RUNNING life, measured from its "
            "first claim (started_at) — not from created_at, so a backlogged job "
            "never ages out before it ran. 0 disables the bound."
        ),
    )

    # The batch AGGREGATE row (imagegen_batches) is reaped separately from its
    # children. This is the quiet-period a batch must sit in — no pending or
    # processing child, and no child terminal write — before the reaper derives
    # its counters from the child rows and closes it out. It exists to keep the
    # reconcile from racing the LAST child, which marks itself terminal and bumps
    # the parent in two transactions milliseconds apart.
    batch_reap_stale_seconds: int = Field(
        default=300,
        description=(
            "Quiet period before a non-terminal imagegen batch whose children are "
            "all terminal is closed out from the child rows"
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="DISPATCH_", env_file=".env", case_sensitive=False, extra="ignore"
    )


#: A provider-secret key id: short, printable, and free of the ``|`` separator the
#: ciphertext envelope uses (``v1|kid|nonce|ct``). Same shape as w2v_orchestrator.
PROVIDER_KEY_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,32}")
#: AES-256 => exactly 32 key bytes.
PROVIDER_KEY_BYTES = 32


def parse_provider_secret_keys(raw: str) -> list[tuple[str, bytes]]:
    """Parse ``"kid:b64,kid2:b64"`` into ``[(kid, 32-byte key), ...]`` (order kept).

    Accepts standard or urlsafe base64. Raises ``ValueError`` on a malformed entry,
    a duplicate kid, or a key that is not 32 bytes -- the message names the kid or
    the entry POSITION, never key material.
    """
    entries: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for position, item in enumerate(raw.split(","), start=1):
        item = item.strip()
        if not item:
            continue
        kid, sep, b64 = item.partition(":")
        kid, b64 = kid.strip(), b64.strip()
        if not sep or not PROVIDER_KEY_ID_PATTERN.fullmatch(kid):
            raise ValueError(f"PROVIDER_SECRET_KEYS entry #{position} is not 'kid:base64'")
        if kid in seen:
            raise ValueError(f"PROVIDER_SECRET_KEYS repeats key id {kid!r}")
        try:
            key = base64.b64decode(b64.replace("-", "+").replace("_", "/"), validate=True)
        except binascii.Error, ValueError:
            raise ValueError(f"PROVIDER_SECRET_KEYS key {kid!r} is not valid base64") from None
        if len(key) != PROVIDER_KEY_BYTES:
            raise ValueError(
                f"PROVIDER_SECRET_KEYS key {kid!r} must decode to {PROVIDER_KEY_BYTES} bytes"
            )
        seen.add(kid)
        entries.append((kid, key))
    return entries


class ProviderSecretSettings(BaseSettings):
    """ACG's OWN keyring sealing per-job tenant provider-slot secrets at rest (P5-2).

    ``PROVIDER_SECRET_KEYS="kid:base64,kid2:base64"`` -- AES-256 keys (32 bytes  # pragma: allowlist secret
    each), from ACG's own Secret (never shared with another service). The FIRST
    entry encrypts; every entry decrypts. Empty => a request carrying
    ``provider_slot`` is refused with 503 (fail closed) and a resumed slot job
    cannot be unsealed. A malformed value fails config load: it must never
    degrade to "no keys". See ``src/services/provider_secrets.py``.
    """

    provider_secret_keys: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="PROVIDER_SECRET_KEYS",
        description="Comma-separated kid:base64 AES-256 keys for slot secrets (first encrypts).",
    )

    @field_validator("provider_secret_keys")
    @classmethod
    def _keys_parse(cls, value: SecretStr) -> SecretStr:
        parse_provider_secret_keys(value.get_secret_value())
        return value

    def keyring_entries(self) -> list[tuple[str, bytes]]:
        """The parsed keyring, primary (encrypting) key first."""
        return parse_provider_secret_keys(self.provider_secret_keys.get_secret_value())

    # ``hide_input_in_errors`` is load-bearing: by default a ValidationError embeds
    # ``input_value`` -- the RAW env string, i.e. key material in the pod's crash log.
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore", hide_input_in_errors=True
    )


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""

    app_name: str = Field(default="AI Content Generation Service", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(
        default="development", description="Environment (development, staging, production)"
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    runway: RunwaySettings = Field(default_factory=RunwaySettings)
    kling: KlingSettings = Field(default_factory=KlingSettings)
    veo: VeoSettings = Field(default_factory=VeoSettings)
    sora: SoraSettings = Field(default_factory=SoraSettings)
    seedance: SeedanceSettings = Field(default_factory=SeedanceSettings)
    wan: WanSettings = Field(default_factory=WanSettings)
    local_image: LocalImageSettings = Field(default_factory=LocalImageSettings)
    mock_provider: MockProviderSettings = Field(default_factory=MockProviderSettings)
    stub: StubProviderSettings = Field(default_factory=StubProviderSettings)
    dalle: DalleImageSettings = Field(default_factory=DalleImageSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    image_generation: ImageGenerationSettings = Field(default_factory=ImageGenerationSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    health_monitor: HealthMonitorSettings = Field(default_factory=HealthMonitorSettings)
    quota: QuotaSettings = Field(default_factory=QuotaSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    cleanup: CleanupSettings = Field(default_factory=CleanupSettings)
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)
    metering: MeteringSettings = Field(default_factory=MeteringSettings)
    dispatch: DispatchSettings = Field(default_factory=DispatchSettings)
    provider_secrets: ProviderSecretSettings = Field(default_factory=ProviderSecretSettings)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    def get_database_url(self) -> str:
        return self.database.url

    def get_redis_url(self) -> str:
        return self.redis.url

    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the cached settings instance (useful for testing)."""
    get_settings.cache_clear()
