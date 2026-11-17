"""Wan2.2 TI2V-5B local video generation provider (text+image to video).

Requires CUDA — the wan library uses torch.cuda internally for autocast,
synchronization, and memory management. MPS/CPU are not supported.
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx
from w2v_common.net import ResponseTooLargeError, SSRFError, safe_get

from src.config import get_settings
from src.services.providers.base import VideoGenerationProvider

logger = logging.getLogger(__name__)


def _resolve_checkpoint_dir(model_name: str) -> str:
    """Resolve HuggingFace model name to a local checkpoint directory.

    If the model has been downloaded via ``huggingface_hub.snapshot_download``,
    returns the cached snapshot path.  Otherwise returns the model name as-is
    (the caller is expected to have placed weights at that path).
    """
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                # Return the most recent revision's snapshot path
                revisions = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                if revisions:
                    return str(revisions[0].snapshot_path)
    except OSError, ValueError, ImportError:
        logger.debug(
            "Could not resolve local HF cache path for %s, using model name directly", model_name
        )
    return model_name


class WanProvider(VideoGenerationProvider):
    """Local Wan2.2 TI2V-5B provider — text+image to video generation.

    Lazy-loads the model on first ``submit()`` call.  Wraps synchronous
    inference with ``asyncio.to_thread`` so the event loop stays free.

    **Requires CUDA** — the wan library uses ``torch.amp.autocast('cuda')``,
    ``torch.cuda.synchronize()``, and ``torch.cuda.empty_cache()`` internally.
    """

    name = "wan"

    def __init__(self) -> None:
        settings = get_settings()
        self._cfg = settings.wan
        self._pipeline = None
        self._output_dir = Path(tempfile.mkdtemp(prefix="wan_"))
        logger.info("WanProvider configured: model=%s", self._cfg.model_name)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_pipeline(self) -> None:
        """Load the Wan2.2 TI2V pipeline if not already loaded (called inside thread)."""
        if self._pipeline is not None:
            return

        import torch

        # wan's T5EncoderModel uses torch.cuda.current_device() as a default
        # param at class-definition time, which fails on non-CUDA machines.
        # Patch it so the module can at least import for testing.
        if not torch.cuda.is_available():
            _orig = torch.cuda.current_device
            torch.cuda.current_device = lambda: 0  # type: ignore[assignment]
            import wan
            from wan.configs import WAN_CONFIGS

            torch.cuda.current_device = _orig  # type: ignore[assignment]
        else:
            import wan
            from wan.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS["ti2v-5B"]
        ckpt_dir = _resolve_checkpoint_dir(self._cfg.model_name)

        self._pipeline = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_cpu=self._cfg.t5_cpu,
            convert_model_dtype=self._cfg.convert_model_dtype,
        )
        logger.info("Wan2.2 TI2V pipeline loaded from %s", ckpt_dir)

    # ------------------------------------------------------------------
    # Synchronous inference (runs in thread)
    # ------------------------------------------------------------------

    def _generate_sync(
        self,
        prompt: str,
        image_path: str | None,
        seed: int | None,
    ) -> str:
        """Run Wan2.2 inference synchronously. Returns task_id."""
        self._ensure_pipeline()
        if self._pipeline is None:
            raise RuntimeError("Wan pipeline failed to initialize")

        task_id = str(uuid.uuid7())
        out_path = str(self._output_dir / f"{task_id}.mp4")

        img = None
        if image_path:
            from PIL import Image

            img = Image.open(image_path).convert("RGB")

        video = self._pipeline.generate(
            input_prompt=prompt,
            img=img,
            size=(1280, 704),
            frame_num=81,
            seed=seed if seed is not None else -1,
            offload_model=self._cfg.offload_model,
        )

        # Save output video using wan's utility
        from wan.utils.utils import save_video

        save_video(
            tensor=video[None],
            save_file=out_path,
            fps=24,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        logger.info("Wan2.2 generation complete: %s", out_path)
        return task_id

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    async def _download_image(self, image_url: str) -> str:
        """Fetch a caller-supplied image URL SSRF-safely; return the temp file path.

        The API boundary (``src.api.routes.dependencies._reject_internal_url``)
        already resolved this URL, but it **discards** the pinned IP — its check is
        advisory. A bare ``client.get(image_url)`` here would re-resolve the
        hostname and reopen the DNS-rebinding TOCTOU window (public answer on the
        boundary lookup, ``169.254.169.254`` / ``10.x`` on the fetch lookup).
        :func:`w2v_common.net.safe_get` is the actual enforcement point: it
        re-validates, pins the connection to the validated IP, and re-checks every
        redirect hop with ``follow_redirects`` forced off.
        """
        from src.services.exceptions import ProviderError, format_error

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await safe_get(client, image_url)
        except (SSRFError, ResponseTooLargeError) as exc:
            # Both subclass ValueError, and ``VideoGenerationProvider.generate``
            # catches ``(OSError, RuntimeError, ValueError)`` and DEGRADES it into
            # ``VideoGenerationResult(success=False)``. A blocked-target refusal
            # would then be indistinguishable from an ordinary provider fault and
            # fall through the chain to the next (paid) provider, re-attempting the
            # same blocked fetch. ProviderError is not a ValueError, so re-raising
            # as one propagates the refusal as a real, non-retriable failure.
            raise ProviderError(
                f"Refused to fetch image_url: {format_error(exc)}",
                provider=self.name,
                retriable=False,
                cause=exc,
            ) from exc
        resp.raise_for_status()
        suffix = ".png" if "png" in resp.headers.get("content-type", "") else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(resp.content)
            return tmp.name

    async def submit(self, prompt: str, **kwargs: Any) -> str:
        image_url: str | None = kwargs.get("image_url")
        seed: int | None = kwargs.get("seed")

        # Download remote image to local temp file if provided
        image_path: str | None = None
        if image_url:
            image_path = await self._download_image(image_url)

        return await asyncio.to_thread(self._generate_sync, prompt, image_path, seed)

    async def check_status(self, task_id: str) -> dict:
        out_path = self._output_dir / f"{task_id}.mp4"
        if out_path.exists():
            return {
                "status": "completed",
                "video_url": str(out_path),
                "error": None,
            }
        return {
            "status": "failed",
            "video_url": None,
            "error": f"Output file not found: {out_path}",
        }

    async def download(self, task_id: str, video_url: str) -> bytes:
        return Path(video_url).read_bytes()

    def estimate_cost(self, duration_s: float) -> float:
        return 0.0  # Local inference, no API cost

    async def health_check(self) -> bool:
        if not self._cfg.enabled:
            return False
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA not available — WanProvider requires CUDA")
                return False
            return True
        except ImportError:
            logger.warning("PyTorch not installed — WanProvider unavailable")
            return False

    def get_models(self) -> list[str]:
        return ["Wan2.2-TI2V-5B"]

    def get_max_duration_s(self) -> float | None:
        # 81 frames @ 24fps ≈ 3.4s
        return 3.4

    def supports_image_to_video(self) -> bool:
        return True
