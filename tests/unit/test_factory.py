"""Unit tests for generator factory — routing by 'preferred' argument."""

import os
from unittest.mock import patch

import pytest

from src.generation.factory import get_generator
from src.generation.imagen import ImagenGenerator
from src.generation.gemini_image import GeminiImageGenerator


# Generators validate API keys in __init__; set a fake key so construction succeeds.
FAKE_ENV = {"GOOGLE_API_KEY": "fake-google-key"}


def test_preferred_imagen_returns_imagen_generator():
    with patch.dict(os.environ, FAKE_ENV):
        gen = get_generator("imagen")
    assert isinstance(gen, ImagenGenerator)


def test_preferred_auto_returns_gemini_generator():
    with patch.dict(os.environ, FAKE_ENV):
        gen = get_generator("auto")
    assert isinstance(gen, GeminiImageGenerator)


def test_default_argument_returns_gemini_generator():
    with patch.dict(os.environ, FAKE_ENV):
        gen = get_generator()
    assert isinstance(gen, GeminiImageGenerator)


def test_generators_have_provider_name():
    """All active generators expose a non-empty provider_name string."""
    cases = [
        ("imagen", ImagenGenerator),
        ("auto", GeminiImageGenerator),
    ]
    with patch.dict(os.environ, FAKE_ENV):
        for pref, expected_type in cases:
            gen = get_generator(pref)
            assert hasattr(gen, "provider_name"), f"{expected_type} missing provider_name"
            assert isinstance(gen.provider_name, str)
            assert len(gen.provider_name) > 0
