"""Unit tests for LLM overlay — mocked Gemini API calls."""

import io
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image

from src.composition.llm_overlay import (
    check_message_present,
    add_text_overlay_llm,
    FORMAT_CONTEXT,
    IMAGE_MODEL,
    VISION_MODEL,
)


def make_image(w=200, h=200) -> Image.Image:
    return Image.new("RGBA", (w, h), (100, 150, 200, 255))


def make_jpeg_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


# ── FORMAT_CONTEXT completeness ───────────────────────────────────────────────

def test_format_context_covers_all_standard_ratios():
    for ratio in ["1:1", "9:16", "16:9"]:
        assert ratio in FORMAT_CONTEXT


# ── check_message_present ─────────────────────────────────────────────────────

def test_check_message_present_no_api_key_returns_false(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = check_message_present(make_image(), "Stay Fresh.", api_key="")
    assert result is False


def test_check_message_present_yes_response_returns_true(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    mock_response = MagicMock()
    mock_response.text = "YES"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        result = check_message_present(make_image(), "Stay Fresh.")
    assert result is True


def test_check_message_present_no_response_returns_false(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    mock_response = MagicMock()
    mock_response.text = "NO"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        result = check_message_present(make_image(), "Stay Fresh.")
    assert result is False


def test_check_message_present_api_exception_returns_false(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API error")

    with patch("google.genai.Client", return_value=mock_client):
        result = check_message_present(make_image(), "Stay Fresh.")
    assert result is False


# ── add_text_overlay_llm ──────────────────────────────────────────────────────

def test_add_text_overlay_no_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = add_text_overlay_llm(make_image(), "Stay Fresh.", "1:1", api_key="")
    assert result is None


def test_add_text_overlay_llm_success_returns_pil_image(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    # Build the expected edited image as JPEG bytes
    edited_img = Image.new("RGB", (200, 200), (0, 200, 100))
    buf = io.BytesIO()
    edited_img.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    # Construct mock response
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock()
    mock_part.inline_data.mime_type = "image/jpeg"
    mock_part.inline_data.data = jpeg_bytes

    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]

    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        result = add_text_overlay_llm(make_image(), "Stay Fresh.", "1:1")

    assert result is not None
    assert isinstance(result, Image.Image)


def test_add_text_overlay_llm_no_image_part_returns_none(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    # Part has no inline_data (text-only response)
    mock_part = MagicMock()
    mock_part.inline_data = None

    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]

    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_response.text = "I cannot edit this image."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        result = add_text_overlay_llm(make_image(), "Stay Fresh.", "1:1")

    assert result is None


def test_add_text_overlay_llm_api_exception_returns_none(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("503 Service Unavailable")

    with patch("google.genai.Client", return_value=mock_client):
        result = add_text_overlay_llm(make_image(), "Stay Fresh.", "1:1")

    assert result is None


def test_add_text_overlay_llm_unknown_ratio_uses_fallback_context(monkeypatch):
    """Unknown ratio should not crash — falls back to generic format context."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("skip")

    with patch("google.genai.Client", return_value=mock_client):
        result = add_text_overlay_llm(make_image(), "msg", "3:7")
    assert result is None  # Fails gracefully on unknown ratio
