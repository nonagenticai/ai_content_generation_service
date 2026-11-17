# Multi-stage Dockerfile for ai_content_generation_service
#
# Single-context build: w2v-common resolves from the public git source declared
# via the [[tool.uv.index]] entry in pyproject.toml.
# Build with: docker buildx build -f Dockerfile .

ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Resolve and install deps; w2v-common comes from its public git tag
RUN uv sync --frozen --no-dev --no-install-project

# Now copy source and install the project itself
COPY src/ ./src/
COPY alembic.ini ./alembic.ini

RUN uv sync --frozen --no-dev

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# `ffmpeg` is REQUIRED, not optional. Two production paths shell out to it:
# `services/providers/stub.py` (the stub video provider — the ONLY video
# provider activated in prod, see clusters/strato/services/w2v/
# ai-content-generation-service-config.env `VIDEO_PROVIDER=stub`; every cloud
# provider is keyless there) and `services/normalizer.py`. Without it the stub's
# `health_check()` returns False, the registry registers ZERO video providers,
# and the pod still goes Ready — serving no video at all. Measured live on both
# replicas 2026-08-20. DO NOT STRIP.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 contentgenservice && \
    mkdir -p /app /app/logs /app/data && \
    chown -R contentgenservice:contentgenservice /app

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=contentgenservice:contentgenservice /app/.venv /app/.venv

# Copy application code
COPY --chown=contentgenservice:contentgenservice src/ ./src/
COPY --chown=contentgenservice:contentgenservice alembic.ini ./alembic.ini

ENV PATH="/app/.venv/bin:$PATH"

USER 1000:1000

# Port 6310 (AI Content Generation Service)
EXPOSE 6310

# Kubernetes livenessProbe / readinessProbe in clusters/strato/services/w2v/
# are the source of truth for health. A Dockerfile HEALTHCHECK is dead
# weight under k8s and only causes drift if the probe paths change.

CMD ["sh", "-c", "alembic -c /app/alembic.ini upgrade head && python -m src.asgi"]
