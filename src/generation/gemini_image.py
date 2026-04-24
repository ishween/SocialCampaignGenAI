"""
Gemini Flash image generation: default generator for standard Gemini API keys.
"""

import io
import logging
import os

from PIL import Image

from src.config.models import ProductConfig, RegionConfig
from src.generation.base import GenerationError, ImageGenerator
from src.generation.prompt_builder import build_image_prompt

logger = logging.getLogger(__name__)

GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image-preview"  # Nano Banana 2: 1K/day paid tier


class GeminiImageGenerator(ImageGenerator):
    """Gemini 2.0 Flash with IMAGE response modality."""

    provider_name = "gemini-flash-image"

    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError("GOOGLE_API_KEY environment variable not set")

        import google.genai as genai
        self._client = genai.Client(api_key=self.api_key)

    # Gemini Flash Image doesn't expose a native aspect-ratio API parameter, so the
    # desired format is appended to the prompt. This gives the model directional
    # guidance, though results are less reliable than Imagen 4's native support.
    _ASPECT_HINTS: dict[str, str] = {
        "9:16": "portrait vertical format (9:16 aspect ratio, taller than wide)",
        "16:9": "landscape horizontal format (16:9 aspect ratio, wider than tall)",
    }

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
        hint = self._ASPECT_HINTS.get(aspect_ratio)
        if hint:
            prompt = f"{prompt}\n\nCompose this image in {hint}."
        logger.debug(f"  Gemini image prompt [{aspect_ratio}]: {prompt[:120]}...")

        response = self._client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_bytes = part.inline_data.data
                return Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        raise GenerationError(
            "Gemini returned no image -> the model may have declined the prompt "
            "or returned text only. Try rephrasing the prompt or check safety settings."
        )
