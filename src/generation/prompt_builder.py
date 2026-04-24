"""
Structured prompt builder for GenAI image generation.

Two modes driven by YAML fields:
  - label_text set    → instructs the model to render text ON the product label
                        (like VOLT bottle: brand name, specs are part of the image)
  - label_text null   → clean product photography, no text in image
                        (text is added post-process via overlay.py)

  - background_scene set → lifestyle/action context behind the product
                           (sports stadium, kitchen counter, outdoor trail, etc.)
  - background_scene null → studio white/neutral background

Why this matters:
  Post-process overlay (overlay.py) adds campaign message + logo on TOP of the image.
  Label text generated INTO the image is for product identity (brand name on bottle label).
  These are two different layers with different purposes:
    Layer 1: product image with label (GenAI), brand identity baked in
    Layer 2: campaign overlay (PIL), message, logo, localization
"""

from src.config.models import ProductConfig, RegionConfig


def build_image_prompt(
    product: ProductConfig,
    region: RegionConfig,
    brand_style: str,
    target_audience: str,
) -> str:
    """
    Build a detailed, structured image generation prompt.
    Uses product.label_text and product.background_scene when provided.
    """
    style_keywords = ", ".join(product.style_keywords) if product.style_keywords else brand_style
    locale_hint = f" for {region.locale_name}" if region.locale_name else ""

    parts = []

    # ── Subject: product + label ─────────────────────────────────────────────
    if product.label_text:
        label_lines = product.label_text.replace("\\n", "\n").strip()
        parts.append(
            f"Professional advertising photograph of {product.name}. "
            f"{product.description}. "
            f"The product label clearly displays the following text in bold commercial typography: "
            f'"{label_lines}". '
            f"The label text is legible, well-designed, and integrated into the product packaging. "
            f"Product is the hero of the image, centered and in sharp focus."
        )
    else:
        parts.append(
            f"Professional product photography of {product.name}. "
            f"{product.description}. "
            f"Product centered, sharp focus, no text on product."
        )

    # ── Background / scene ───────────────────────────────────────────────────
    if product.background_scene:
        parts.append(
            f"Background scene: {product.background_scene}. "
            f"Background is slightly out of focus to keep product as hero. "
            f"Dramatic cinematic lighting, high energy, commercial photography quality."
        )
    else:
        parts.append(
            "Clean neutral studio background, soft gradient, professional studio lighting."
        )

    # ── Style, audience, quality ─────────────────────────────────────────────
    parts.append(
        f"Visual style: {style_keywords}. "
        f"Target audience: {target_audience}{locale_hint}. "
        f"Ultra high resolution, 8K commercial quality, award-winning product photography, "
        f"photorealistic, no watermarks."
    )

    return " ".join(parts)
