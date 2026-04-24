"""
Region message adaptation via Gemini 3.1 Flash Lite.

Message resolution priority (highest to lowest):
  1. region.message_override: explicit translation set in YAML, used as-is
  2. LLM-adapted message: Gemini 3.1 Flash Lite adapts the brief message for
                           the region's language and cultural context
  3. brief.message: original English message, used if LLM unavailable

Why LLM adaptation over static overrides only:
  Static overrides require a human to write every locale string. At scale
  (10 regions × 50 campaigns) that is 500 manual copy entries per quarter.
  LLM adaptation generates culturally-appropriate copy automatically, with
  the human override available whenever a market team wants control.

Validation after generation:
  The LLM-generated message is validated against:
    - prohibited_words from the campaign brief (brand compliance)
    - injection patterns from input_validator (security)
  A violation causes fallback to the original brief message and logs a warning.
  This ensures the LLM cannot accidentally produce brand-unsafe or adversarial copy.

Model choice: Gemini 3.1 Flash Lite
  Text-only task, binary decision-making, speed matters. Flash Lite costs
  ~5× less than Flash and completes in <1s. Same model used for prompt
  refinement and input validation classification.
"""

import logging
import os
import re

from src.config.models import RegionConfig

logger = logging.getLogger(__name__)

MODEL = "gemini-3.1-flash-lite-preview"

# Patterns reused from input_validator, kept minimal here to avoid circular import
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|all|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"(generate|create|produce)\s+(adult|explicit|nsfw|illegal)", re.IGNORECASE),
]


def _is_safe(message: str, prohibited_words: list[str]) -> bool:
    """Return True if generated message passes security and brand checks."""
    for pat in _INJECTION_PATTERNS:
        if pat.search(message):
            logger.warning(f"  [MsgGen] Generated message failed injection check: '{pat.pattern}'")
            return False
    for word in prohibited_words:
        if re.search(rf"\b{re.escape(word)}\b", message, re.IGNORECASE):
            logger.warning(f"  [MsgGen] Generated message contains prohibited word: '{word}'")
            return False
    return True


def generate_region_message(
    brief_message: str,
    region: RegionConfig,
    target_audience: str,
    prohibited_words: list[str] | None = None,
    api_key: str = "",
) -> str | None:
    """
    Adapt the campaign message for a specific region using Gemini 3.1 Flash Lite.

    Called when region.message_override is not set. Returns an adapted message
    string on success, or None on failure. Caller falls back to brief.message.

    Validation:
      Generated text is checked against prohibited_words and injection patterns
      before being returned. Fails to None if either check fails.
    """
    api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        logger.debug("  [MsgGen] No API key, skipping LLM message adaptation")
        return None

    prohibited_words = prohibited_words or []
    locale_label = region.locale_name or region.code
    lang_label = region.language.upper()

    prompt = (
        f"You are a professional copywriter specialising in global marketing campaigns.\n\n"
        f"Adapt the following campaign message for {locale_label} ({lang_label}).\n\n"
        f"Rules:\n"
        f"- Translate to {lang_label} if it is not already in that language.\n"
        f"- Keep the same energy, brevity, and meaning as the original.\n"
        f"- Make it culturally resonant for {locale_label}.\n"
        f"- Target audience: {target_audience}.\n"
        f"- Same length or shorter than the original.\n"
        f"- Return ONLY the adapted message. No quotes. No explanation.\n\n"
        f"Original message: {brief_message}"
    )

    try:
        import google.genai as genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=MODEL, contents=[prompt])
        adapted = response.text.strip().strip('"').strip("'")

        if not adapted:
            logger.warning(f"  [MsgGen] Empty response for {region.code}, using brief message")
            return None

        if not _is_safe(adapted, prohibited_words):
            logger.warning(f"  [MsgGen] {region.code} message failed validation, using brief message")
            return None

        logger.info(f"  [MsgGen] {region.code} ({lang_label}): '{adapted}'")
        return adapted

    except Exception as e:
        logger.warning(f"  [MsgGen] {region.code} adaptation failed ({e}), using brief message")
        return None
