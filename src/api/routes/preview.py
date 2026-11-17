"""Preview endpoint: POST /preview."""

from __future__ import annotations

import asyncio
import base64
import logging
import time

from fastapi import APIRouter, HTTPException, Request

from src.api.routes.dependencies import _get_registry
from src.api.schemas import PreviewRequest, PreviewResponse
from src.constants import PREVIEW_MAX_VIDEO_SIZE_BYTES, PREVIEW_TIMEOUT_S
from src.services.exceptions import format_error

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/preview",
    operation_id="preview_video_generation",
    responses={
        422: {"description": "Unknown provider or validation error"},
    },
)
async def preview_video(
    body: PreviewRequest,
    request: Request,
) -> PreviewResponse:
    """Quick preview generation -- no DB record, returns base64 video inline.

    Args:
        body: Preview request with prompt and optional provider/duration.
        request: FastAPI request.

    Returns:
        PreviewResponse with base64-encoded video or error details.
    """
    from src.config import get_settings

    settings = get_settings()
    registry = _get_registry(request)
    provider_name = body.provider or settings.generation.default_provider

    logger.info(
        "Preview request received: provider=%s prompt_len=%d duration=%.1fs",
        provider_name,
        len(body.prompt),
        body.duration_s,
    )

    provider = registry.get(provider_name)
    if provider is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown provider: {provider_name}",
        )

    estimated_cost = provider.estimate_cost(body.duration_s)

    try:
        start = time.monotonic()
        result = await asyncio.wait_for(
            provider.generate(
                prompt=body.prompt,
                duration_s=body.duration_s,
                width=settings.generation.default_width,
                height=settings.generation.default_height,
            ),
            timeout=PREVIEW_TIMEOUT_S,
        )
        elapsed = time.monotonic() - start

        if not result.success:
            logger.warning(
                "Preview generation failed: provider=%s error=%s elapsed=%.2fs",
                provider_name,
                result.error,
                elapsed,
            )
            return PreviewResponse(
                status="failed",
                provider=provider_name,
                error=result.error,
                estimated_cost=estimated_cost,
            )

        video_base64 = None
        if result.video_data and len(result.video_data) < PREVIEW_MAX_VIDEO_SIZE_BYTES:
            video_base64 = base64.b64encode(result.video_data).decode()

        logger.info(
            "Preview completed: provider=%s elapsed=%.2fs",
            provider_name,
            elapsed,
        )
        return PreviewResponse(
            status="completed",
            provider=provider_name,
            video_base64=video_base64,
            estimated_cost=estimated_cost,
        )

    except TimeoutError:
        logger.warning("Preview timed out: provider=%s", provider_name)
        return PreviewResponse(
            status="timeout",
            provider=provider_name,
            error="Preview generation timed out (60s)",
            estimated_cost=estimated_cost,
        )
    except Exception as exc:
        logger.error("Preview failed: provider=%s error=%s", provider_name, exc, exc_info=True)
        return PreviewResponse(
            status="failed",
            provider=provider_name,
            error=format_error(exc),
            estimated_cost=estimated_cost,
        )
