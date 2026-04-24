"""Abstract base class for image generators (Strategy Pattern)."""

from abc import ABC, abstractmethod

from PIL import Image

from src.config.models import ProductConfig, RegionConfig


class ImageGenerator(ABC):
    """Strategy interface for all GenAI image providers."""

    @abstractmethod
    def generate(
        self,
        product: ProductConfig,
        region: RegionConfig,
        brand_style: str,
        target_audience: str = "",
        aspect_ratio: str = "1:1",
    ) -> Image.Image:
        """
        Generate a product image. Raises GenerationError on unrecoverable failure.

        aspect_ratio: "1:1" | "9:16" | "16:9"
          Passed through to the provider so it can natively compose for the target
          format when --native-ratios is active. Defaults to "1:1" (base generation).
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...


class GenerationError(Exception):
    """Raised when all retry attempts for image generation are exhausted."""
