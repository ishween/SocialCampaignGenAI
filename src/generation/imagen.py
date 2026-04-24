"""
Google Imagen 4 image generator via google-genai SDK.
"""

import io
import logging
import os

from PIL import Image

from src.config.models import ProductConfig, RegionConfig
from src.generation.base import GenerationError, ImageGenerator
from src.generation.prompt_builder import build_image_prompt

logger = logging.getLogger(__name__)

IMAGEN_MODEL = "imagen-4.0-generate-001"


class ImagenGenerator(ImageGenerator):
    """Google Imagen 3 via google-genai SDK."""

    provider_name = "google-imagen4"

    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError("GOOGLE_API_KEY environment variable not set")

        import google.genai as genai
        self._client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        product: ProductConfig,
        region: RegionConfig,
        brand_style: str,
        target_audience: str = "",
        aspect_ratio: str = "1:1",
    ) -> Image.Image:
        from google.genai import types as genai_types

        prompt = build_image_prompt(product, region, brand_style, target_audience)
        logger.debug(f"  Imagen 4 prompt [{aspect_ratio}]: {prompt[:120]}...")

        response = self._client.models.generate_images(
            model=IMAGEN_MODEL,
            prompt=prompt,
            config=genai_types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/png",
                aspect_ratio=aspect_ratio,  # Native ratio when --native-ratios active
                safety_filter_level="block_low_and_above",
                person_generation="allow_adult",  # Sports backgrounds include athletes
            ),
        )

        if not response.generated_images:
            raise GenerationError("Imagen 4 returned no images -> possible safety filter block")

        image_bytes = response.generated_images[0].image.image_bytes
        return Image.open(io.BytesIO(image_bytes)).convert("RGBA")
