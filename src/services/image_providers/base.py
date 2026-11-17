"""Abstract interfaces and registry for AI image generation providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ImageGenResult:
    """Result from an image generation request."""

    success: bool
    image_data: bytes | None = None
    image_url: str | None = None
    provider_job_id: str | None = None
    mime_type: str = "image/png"
    width: int | None = None
    height: int | None = None
    error: str | None = None


@dataclass
class ImageProviderInfo:
    """Information about a registered image provider."""

    name: str
    available: bool
    models: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)


class ImageGenProvider(ABC):
    """Abstract interface for AI image generation providers."""

    name: str

    @abstractmethod
    async def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        style: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        reference_image_urls: list[str] | None = None,
        extra_params: dict | None = None,
        usage_context: dict | None = None,
        target_usage: str | None = None,
    ) -> ImageGenResult:
        """Generate a single image."""

    @abstractmethod
    def estimate_cost(self, *, size: str | None = None, quality: str | None = None) -> float:
        """Estimate cost in USD."""

    async def health_check(self) -> bool:
        """Check whether the provider is available."""
        return True

    async def aclose(self) -> None:  # noqa: B027 -- deliberate no-op default
        """Release per-instance resources (per-job slot instances override this)."""

    def get_models(self) -> list[str]:
        return []

    def get_styles(self) -> list[str]:
        return []


class ImageProviderRegistry:
    """Registry managing image generation providers."""

    def __init__(self) -> None:
        self._providers: dict[str, ImageGenProvider] = {}

    def register(self, provider: ImageGenProvider) -> None:
        self._providers[provider.name] = provider
        logger.info("Registered image provider: %s", provider.name)

    def get(self, name: str) -> ImageGenProvider | None:
        return self._providers.get(name)

    def list_providers(self) -> list[ImageProviderInfo]:
        result: list[ImageProviderInfo] = []
        for name, provider in self._providers.items():
            try:
                result.append(
                    ImageProviderInfo(
                        name=name,
                        available=True,
                        models=provider.get_models(),
                        styles=provider.get_styles(),
                    )
                )
            except OSError, RuntimeError, ValueError:
                result.append(ImageProviderInfo(name=name, available=False))
        return result

    @property
    def provider_names(self) -> list[str]:
        return list(self._providers.keys())
