"""
Generator factory: instantiates the configured image generator.

Provider options:
  'auto'   -> GeminiImageGenerator  (Gemini 2.0 Flash, IMAGE modality)
               Works with any Gemini API key, including free tier. Default.
  'imagen' -> ImagenGenerator       (Google Imagen 4)
               Requires billing enabled + Imagen API access in Google Cloud.
               404s on a standard free-tier Gemini API key. Use this for
               production or when you have Vertex AI / paid API setup.

Extensibility:
  The factory is the single point of change for adding new providers or
  enabling user-configurable model selection. To add a provider, implement
  the ImageGenerator interface in a new module and add a branch here.
  No other pipeline code needs to change.

  This design also supports fallback logic: if the preferred provider fails
  at initialisation (missing key, quota exhausted), a secondary provider can
  be returned here without the rest of the pipeline knowing. For example,
  falling back from Imagen 4 to Gemini Flash Image when the Imagen API is
  unavailable is a one-line change in this factory.

On provider misconfiguration:
  The factory fails fast if the requested provider cannot be initialised.
  Silent auto-switching to a different model would change output quality
  without the caller knowing, making failures hard to diagnose.
"""

import logging

from src.generation.base import ImageGenerator

logger = logging.getLogger(__name__)


def get_generator(preferred: str = "auto") -> ImageGenerator:
    """
    Return the configured image generator.
    preferred: 'auto' | 'imagen'

    To add a new provider: implement ImageGenerator, import it here,
    and add a branch for its preferred key string.
    """
    if preferred == "imagen":
        from src.generation.imagen import ImagenGenerator
        gen = ImagenGenerator()
        logger.info(f"Using image generator: {gen.provider_name}")
        return gen

    # Default ('auto'): Gemini Flash image generation, works with standard API key
    from src.generation.gemini_image import GeminiImageGenerator
    gen = GeminiImageGenerator()
    logger.info(f"Using image generator: {gen.provider_name}")
    return gen
