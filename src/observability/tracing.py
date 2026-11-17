"""OpenTelemetry tracing setup for ai_content_generation_service."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def setup_tracing(
    service_name: str = "ai-content-generation-service", otlp_endpoint: str | None = None
) -> None:
    """Initialize OpenTelemetry tracing with auto-instrumentation."""
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if otlp_endpoint:
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)

        # Auto-instrument
        FastAPIInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()

        logger.info(
            "OpenTelemetry tracing initialized (service=%s, endpoint=%s)",
            service_name,
            otlp_endpoint,
        )
    except ImportError:
        logger.warning("OpenTelemetry packages not installed -- tracing disabled")
    except (OSError, ValueError, RuntimeError) as e:
        logger.error("Failed to initialize OpenTelemetry: %s", e)


def get_tracer(name: str = "ai-content-generation-service") -> Any:
    """Get a tracer instance for creating custom spans."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return None
