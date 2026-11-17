# Ai Content Generation Service

Generates AI image and video assets on demand across multiple cloud and on-device providers, stores them in MinIO, and returns fetchable URLs. Port 6310.

## What Ai Content Generation Service Does

When an upstream caller — chiefly `w2v_orchestrator`'s visual stages, or `html_render_service` requesting AI concept images — needs a generated asset, it POSTs a prompt plus an optional provider hint. The service picks a provider from the appropriate registry, kicks off the generation asynchronously, and returns a job the caller can poll. Video and image generation are two independent surfaces: `/api/content/videos/*` for clips, `/api/concept-images/*` for stills, each backed by its own provider registry assembled at startup from whatever credentials are configured.

Each media type carries a fleet of providers built best-effort at boot. On the video side the cloud APIs are Runway, Kling, Veo, Sora, and Seedance; an optional local Wan2.2 (TI2V) provider registers only on a CUDA host and is off by default (`WAN_ENABLED=false`, not deployed in the cluster); and `mock` plus a deterministic ffmpeg `stub` cover offline CI and demos. On the image side the providers are OpenAI (`gpt-image-1`), `local_image` (which POSTs to the mflux/FLUX backend at `image-gen.test:9204`), `mock`, and a deterministic Pillow `stub`. A missing key for any single provider is logged and skipped — never fatal — and if the configured default fails to register the service falls back to the first provider that did.

Once a generation completes, the resulting asset is uploaded to MinIO and a fetchable (optionally presigned) URL is handed back through the job/result endpoints. Every successful generation emits exactly one cost event to a Redis metering stream consumed by `metering_service`, and per-tenant quotas (daily generation count and USD spend) gate new work before a provider is ever called. Batch endpoints let callers submit and track many generations under one request, and a preview endpoint estimates a video request without committing it.

The service is stateful and tenant-isolated. It runs on Postgres 18 through psycopg 3 (never asyncpg or SQLite), with Alembic migrations applied by a Kubernetes initContainer. Row-Level Security is enforce-capable: the runtime connects as the restricted `w2v_ai_content_app` role with `TENANT_RLS_MODE=enforce`, re-binding the tenant GUC on every commit so the multi-commit generation pipelines stay tenant-scoped end to end. Redis is used for caching, job tracking, quotas, and the metering stream, but degrades gracefully — a failed Redis connection at startup disables caching rather than crashing the service.

Because the live production image provider is `local_image` (on-device FLUX at `image-gen.test:9204`), the heavy GPU work happens off-cluster on mini1; the service itself stays a lightweight amd64 pod that orchestrates providers, storage, and metering rather than running models in-process.

## Features

- **Dual generation surfaces** — separate video (`/api/content/videos`) and concept-image (`/api/concept-images`) APIs, each with its own provider registry, so a caller can request either media type without a shared bottleneck
- **Multi-provider video** — Runway, Kling, Veo, Sora, and Seedance cloud APIs plus an optional local Wan2.2 provider, so tenants are not locked to a single video vendor and generation survives a provider outage via fallback
- **Multi-provider image** — OpenAI `gpt-image-1` and on-device `local_image` (mflux/FLUX at `image-gen.test:9204`), so production image generation runs on local hardware while a cloud fallback stays available
- **Best-effort registry assembly** — each provider is instantiated, health-checked, and registered independently at startup, so a missing or invalid API key for one provider is logged and skipped rather than breaking service start-up
- **Deterministic offline stubs** — `IMAGE_PROVIDER=stub` renders real PNG bytes via Pillow and `VIDEO_PROVIDER=stub` renders a valid H.264/MP4 via ffmpeg, so CI and demos produce parseable assets with no paid APIs
- **Async job model** — generations run in the background and are tracked by job ID with status, result, cost, and download sub-resources, so callers poll instead of holding long-lived HTTP connections
- **Batch generation** — submit and track many generations under a single batch, so high-volume pipelines avoid per-asset request overhead
- **Per-tenant quotas** — daily generation-count and USD-spend limits gate work before any provider call, so a single tenant cannot exhaust shared provider budgets
- **Cost metering** — one cost event per successful generation is emitted to a Redis stream consumed by `metering_service`, so spend is attributable per tenant
- **Provider health monitoring** — consecutive-failure tracking with cooldown disables a flapping provider and restores state from Redis, so bad providers are routed around automatically
- **RLS tenant isolation** — enforce-capable Row-Level Security under the restricted `w2v_ai_content_app` role with per-commit GUC re-bind, so cross-tenant data leakage is prevented at the database layer
- **License-token authentication** — inbound `X-Tenant-Id: Bearer <token>` introspected against the licenses service with a 5-minute cache, so every request is tenant-scoped without per-call latency

## Authentication

Auth enabled by default (`AUTH_ENABLED=true`). The `AuthMiddleware` (from `w2v_common`) reads the license token from the `X-Tenant-Id: Bearer <license_token>` header — **not** `Authorization`, which is reserved for Keycloak JWTs from the SPA — introspects it against the licenses service, and caches the result for 5 minutes (`AUTH_CACHE_TTL=300`). Service-to-service calls forward `X-Tenant-Id`, `X-License-Id`, and `X-Subject-Id` as plain headers (no `Bearer` prefix).

| Priority | Method | Details |
|----------|--------|---------|
| 1 | License token | `X-Tenant-Id: Bearer <token>` introspected against the licenses service, 5-minute cache |
| 2 | Trusted network | Requests from private CIDRs (127.0.0.1/32, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) get the default tenant without a token |

Public paths are always exempt: `/`, `/health`, `/health/live`, `/health/ready`, `/healthz`, `/readyz`, `/ready`, `/docs`, `/redoc`, `/openapi.json`, `/favicon.ico`, `/metrics` (plus the `/docs`, `/redoc`, `/health/` prefixes).

Row-Level Security is enforced at the database layer: the runtime connects as `w2v_ai_content_app` with `TENANT_RLS_MODE=enforce` and re-binds the tenant GUC on each commit.

## Quick Start

```bash
uv sync
uv run uvicorn src.asgi:app --host 0.0.0.0 --port 6310
uv run pytest tests/ -v --cov=src
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

### Video generation `/api/content/videos`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/generate` | Start a video generation job |
| POST | `/generate/batch` | Start a batch of video generations |
| POST | `/preview` | Preview/estimate a video generation without committing |
| GET | `/jobs` | List video jobs |
| GET | `/jobs/{job_id}` | Get video job status |
| GET | `/jobs/{job_id}/result` | Get video job result |
| GET | `/jobs/{job_id}/costs` | Get cost breakdown for a job |
| GET | `/jobs/{job_id}/video` | Download the generated video |
| POST | `/jobs/{job_id}/cancel` | Cancel a running video job |
| GET | `/batches/{batch_id}/status` | Get batch status |
| GET | `/batches/{batch_id}/result` | Get batch result |
| GET | `/providers` | List registered video providers |
| GET | `/providers/{name}/health` | Get health for a video provider |
| GET | `/usage` | Get video usage for the tenant |
| GET | `/quotas/{tenant_id}` | Get quota status. `{tenant_id}` must be the caller's own tenant — any other value is `403` |

### Concept images `/api/concept-images`

| Method | Path | Description |
|--------|------|-------------|
| POST | `` | Create a concept image generation job |
| POST | `/batches` | Create a concept image batch |
| GET | `/jobs/{job_id}` | Get concept image job status |
| GET | `/jobs/{job_id}/result` | Get concept image job result |
| GET | `/batches/{batch_id}` | Get concept image batch status |
| GET | `/batches/{batch_id}/result` | Get concept image batch result |
| GET | `/assets/{asset_id}` | Get a generated image asset |
| GET | `/assets/{asset_id}/download` | Download a generated image asset |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe (checks DB) |
| GET | `/healthz` | Liveness probe (alias) |
| GET | `/readyz` | Readiness probe (alias) |
| GET | `/metrics` | Prometheus metrics scrape endpoint |

## Configuration

All configuration via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Listen host |
| `API_PORT` | `6310` | Listen port |
| `API_DEBUG` | `false` | Enable debug mode |
| `API_RELOAD` | `false` | Auto-reload on code changes |
| `AUTH_ENABLED` | `true` | Toggle authentication middleware |
| ~~`AUTH_DEFAULT_TENANT_ID`~~ | -- | **REMOVED.** A `tenant_id` may never hold a sentinel; an absent tenant is a 400, never a substituted default. |
| `AUTH_CACHE_TTL` | `300` | License introspection cache TTL (seconds) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_JSON_LOGS` | `false` | Structured JSON logging |
| `DATABASE_URL` | -- | PostgreSQL connection string (psycopg 3) |
| `DATABASE_POOL_SIZE` | `10` | Connection pool size |
| `DATABASE_MAX_OVERFLOW` | `20` | Max connections beyond pool size |
| `TENANT_RLS_MODE` | `enforce` (in cluster) | RLS mode for the restricted app role |
| `REDIS_URL` | -- | Redis URL (cache, jobs, quotas) |
| `STORAGE_MINIO_BUCKET` | `ai-content-generation-service` | MinIO bucket for assets |
| `STORAGE_MINIO_EXTERNAL_ENDPOINT` | -- | External MinIO endpoint for presigned URLs |
| `GENERATION_DEFAULT_PROVIDER` | `runway` | Default video provider (falls back to first registered) |
| `GENERATION_PARALLEL_REQUESTS` | `3` | Max concurrent video generations |
| `GENERATION_FALLBACK_ENABLED` | `true` | Enable provider fallback on failure |
| `VIDEO_PROVIDER` | -- | Set to `stub` for the deterministic ffmpeg video provider |
| `IMAGE_PROVIDER` | -- | Set to `stub` for the deterministic Pillow image provider |
| `IMAGE_GENERATION_DEFAULT_PROVIDER` | `openai_images` | Default image provider (wired to `local_image` in cluster) |
| `IMAGE_GENERATION_PARALLEL_REQUESTS` | `4` | Max concurrent image generations |
| `RUNWAY_API_KEY` | -- | Runway API key (also `KLING_`, `VEO_`, `SORA_`, `SEEDANCE_`) |
| `DALLE_ENABLED` | `false` | Enable OpenAI image generation |
| `DALLE_API_KEY` | -- | OpenAI API key for image generation |
| `DALLE_MODEL` | `gpt-image-1` | OpenAI image model |
| `LOCAL_IMAGE_ENABLED` | `false` | Enable the on-device mflux/FLUX image provider |
| `LOCAL_IMAGE_BACKEND_URL` | `http://image-gen.test:9204` | mflux/FLUX backend URL |
| `LOCAL_IMAGE_TIMEOUT` | `180.0` | Local image request timeout (seconds) |
| `WAN_ENABLED` | `false` | Enable local Wan2.2 video provider (CUDA host only) |
| `MOCK_PROVIDER_ENABLED` | `false` | Register mock video + image providers |
| `QUOTA_ENABLED` | `true` | Enable per-tenant quota enforcement |
| `QUOTA_DEFAULT_MAX_DAILY_GENERATIONS` | `100` | Max daily generations per tenant |
| `QUOTA_DEFAULT_MAX_DAILY_SPEND` | `50.0` | Max daily spend per tenant (USD) |
| `RATE_LIMIT_ENABLED` | `false` | Enable API rate limiting |
| `HEALTH_MONITOR_ENABLED` | `true` | Enable provider health monitoring |
| `CLEANUP_ENABLED` | `false` | Enable periodic job/asset cleanup |
| `NORMALIZATION_ENABLED` | `false` | Enable ffmpeg video normalization |
| `METERING_REDIS_URL` | -- | Redis URL for the cost-event stream (separate from cache Redis) |
| `OTEL_TRACING_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `OTEL_METRICS_ENABLED` | `true` | Enable Prometheus metrics endpoint |

## Architecture

```
src/
├── asgi.py                          # FastAPI app + lifespan; builds video + image registries, middleware, health
├── config.py                        # Pydantic Settings (per-provider sub-configs, quotas, metering, image-gen)
├── constants.py                     # Shared constants
├── api/
│   ├── content_routes.py            # /api/concept-images image generation + batch endpoints
│   ├── content_schemas.py           # Concept-image request/response schemas
│   ├── schemas.py                   # Shared API schemas
│   ├── health.py                    # /health, /healthz, /readyz probes
│   └── routes/
│       ├── __init__.py              # /api/content/videos router aggregation
│       ├── generation.py            # Video generate + generate/batch
│       ├── jobs.py                  # Video job status/result/costs/video/cancel
│       ├── batches.py               # Video batch status/result
│       ├── preview.py               # Video generation preview
│       ├── providers.py             # Provider listing + health
│       ├── usage.py                 # Usage + quota status
│       └── dependencies.py          # FastAPI dependency injection helpers
├── auth/
│   ├── middleware.py                # License-token AuthMiddleware (re-export from w2v_common)
│   └── models.py                    # Auth data models
├── cache/
│   └── redis_client.py              # Redis client init/close
├── database/
│   ├── connection.py                # Async engine (psycopg 3)
│   ├── session.py                   # Session maker
│   ├── models.py                    # SQLAlchemy ORM models
│   ├── rls_events.py                # Per-commit tenant GUC re-bind listener
│   ├── rls_resolvers.py             # SECURITY DEFINER cross-tenant cleanup resolver
│   └── migrations/
│       ├── env.py                   # Alembic environment
│       └── versions/                # 001..008 (008_rls.py enforce-capable RLS)
├── middleware/
│   ├── rate_limit.py                # Redis-backed rate limiting middleware
│   └── request_id.py                # Request-ID ContextVar binding
├── observability/
│   ├── metrics.py                   # Prometheus metrics
│   └── tracing.py                   # OpenTelemetry tracing setup
└── services/
    ├── video_generation_service.py  # Video generation orchestration
    ├── imagegen_service.py          # Image generation orchestration
    ├── job_manager.py               # Video job lifecycle
    ├── image_job_manager.py         # Image job lifecycle
    ├── batch_tracker.py             # Batch state tracking
    ├── image_batch_manager.py       # Image batch lifecycle
    ├── provider_selector.py         # Provider selection strategy
    ├── provider_health.py           # Consecutive-failure health monitor
    ├── quota_service.py             # Per-tenant quota enforcement
    ├── metering.py                  # Per-generation cost events → Redis stream
    ├── storage_service.py           # MinIO asset upload + presigned URLs
    ├── normalizer.py                # ffmpeg video normalization
    ├── cleanup_service.py           # Periodic job/asset expiry
    ├── callback_service.py          # Completion callbacks
    ├── http_client.py               # Shared httpx client pool
    ├── errors.py / exceptions.py    # Service error types
    ├── providers/                   # Video providers: runway, kling, veo, sora, seedance, wan, mock, stub
    └── image_providers/             # Image providers: openai, local_image, mock, stub
```

## Testing

```bash
uv run pytest tests/ -v --cov=src          # All tests with coverage
uv run pytest tests/unit/ -v               # Unit tests
uv run pytest tests/integration/ -v        # Integration tests
uv run pytest tests/test_rls_isolation.py -v   # RLS tenant-isolation tests
uv run ruff check src/ tests/              # Lint
uv run ruff format --check src/ tests/     # Format check
```

Coverage threshold: 80% (enforced in `pyproject.toml`). Run `bash scripts/ci-local.sh` to mirror the CI lint/unit job exactly before pushing.

## CI/CD

`.github/workflows/ci-public.yml` is self-contained and runs on GitHub-hosted runners: a betterleaks secret scan, the `no-private-refs` guard, then `ruff check`, `ruff format --check`, `pytest` against a Postgres 18 service container, and `scripts/pip-audit.sh`.

No image is published from this repository; build it locally with the Dockerfiles.

`pip-audit` ignores `PYSEC-2025-183` (pyjwt, fleet-standard) and `CVE-2025-3000` (torch, GPU-only `local-video` extra with no upstream fix — owner decision 2026-06-17).

## Docker

```bash
docker build -t ai-content-generation-service .
docker run -p 6310:6310 ai-content-generation-service
docker run -p 6310:6310 -e LOG_LEVEL=DEBUG -e DATABASE_URL=postgresql+psycopg://host/db ai-content-generation-service
```

`Dockerfile` is the standard amd64 cluster image (ffmpeg present for the video stub). `Dockerfile.gpu` is the GPU/local-video build (CUDA + torch, `uv sync --group local-video`) for running Wan2.2 on a CUDA host — not used in the cluster. `Dockerfile.production` is the production variant.

## Dependencies

- **w2v_orchestrator** — upstream caller for video/image visual stages
- **html_render_service** — upstream caller for AI concept images
- **image_gen_backend** (`image-gen.test:9204`) — on-device mflux/FLUX backend for the `local_image` provider (macOS launchd, no CI)
- **licenses service** — validates inbound license tokens via introspection
- **metering_service** — consumes the per-generation cost-event Redis stream
- **cloud provider APIs** — Runway, Kling, Veo, Sora, Seedance (video); OpenAI `gpt-image-1` (image)
- **PostgreSQL 18** — persistent storage via psycopg 3, through PgBouncer in-cluster (`pgbouncer.pgbouncer:6432`)
- **Redis** — caching, job tracking, quotas, and the metering stream
- **MinIO** — generated-asset object storage
- **FastAPI** / **Uvicorn** — web framework + ASGI server
- **Pydantic v2** / **pydantic-settings** — schema and settings validation
- **SQLAlchemy** (async) + **Alembic** — ORM and migrations
- **httpx** — async HTTP client for provider and backend calls
- **Pillow** — image stub rendering and asset handling
- **prometheus-client** / **OpenTelemetry** — metrics and tracing
- **w2v-common** (>=0.25.0) — shared config, auth middleware, error envelope

## Known Issues

- **torch CVE-2025-3000 suppressed in CI** — no upstream fix; torch is only pulled by the GPU-only `local-video` dependency group (Wan2.2), which the cluster image never installs. `pip-audit` ignores it by owner decision (2026-06-17).
- **Wan2.2 local video provider not deployed** — `WAN_ENABLED=false` by default; it registers only on a CUDA host and is absent from the cluster. Production image generation runs through `local_image` → `image-gen.test:9204`, not a bundled GPU deployment.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. Tenant and licence identity is forwarded from the caller. It runs with
no licence switch set; unset means off.
