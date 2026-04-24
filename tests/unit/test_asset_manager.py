"""Unit tests for AssetManager — asset discovery, matching, persistence."""

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from src.assets.manager import AssetManager


def make_png(path: Path, size=(100, 100), color=(255, 0, 0)) -> Path:
    img = Image.new("RGB", size, color)
    img.save(path, "PNG")
    return path


# ── Explicit path resolution ──────────────────────────────────────────────────

def test_explicit_path_exists_returns_it(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    img_path = tmp_path / "product.png"
    make_png(img_path)

    manager = AssetManager(assets_dir)
    result = manager.get_asset("anything", explicit_path=str(img_path))
    assert result == img_path


def test_explicit_path_missing_falls_back_to_index(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    # Place a matching image in the assets dir
    make_png(assets_dir / "myproduct.png")

    manager = AssetManager(assets_dir)
    # Explicit path does not exist — should fall back to fuzzy match
    result = manager.get_asset("myproduct", explicit_path="/nonexistent/file.png")
    assert result is not None
    assert result.name == "myproduct.png"


# ── Directory index / fuzzy match ─────────────────────────────────────────────

def test_normalized_name_exact_match(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    make_png(assets_dir / "hydrate_plus.png")

    manager = AssetManager(assets_dir)
    result = manager.get_asset("hydrate_plus")
    assert result is not None


def test_name_with_spaces_normalized(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    make_png(assets_dir / "hydrate_plus.png")

    manager = AssetManager(assets_dir)
    result = manager.get_asset("Hydrate Plus")
    assert result is not None


def test_name_with_dashes_normalized(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    make_png(assets_dir / "sport_bottle.png")

    manager = AssetManager(assets_dir)
    result = manager.get_asset("sport-bottle")
    assert result is not None


def test_no_match_returns_none(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    manager = AssetManager(assets_dir)
    result = manager.get_asset("nonexistent_product_xyz")
    assert result is None


# ── save_generated ────────────────────────────────────────────────────────────

def test_save_generated_persists_file(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    manager = AssetManager(assets_dir)

    img = Image.new("RGBA", (64, 64), (0, 255, 0, 255))
    saved_path = manager.save_generated(img, "New Product")
    assert saved_path.exists()
    assert saved_path.suffix == ".png"


def test_save_generated_updates_index(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    manager = AssetManager(assets_dir)

    img = Image.new("RGBA", (64, 64), (0, 0, 255, 255))
    manager.save_generated(img, "Fresh Drink")

    # Index should now contain the generated asset
    result = manager.get_asset("Fresh Drink")
    assert result is not None


# ── load_image ────────────────────────────────────────────────────────────────

def test_load_image_returns_rgba(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    img_path = assets_dir / "test.png"
    make_png(img_path)

    manager = AssetManager(assets_dir)
    loaded = manager.load_image(img_path)
    assert loaded.mode == "RGBA"


# ── content_hash ──────────────────────────────────────────────────────────────

def test_content_hash_is_stable(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    img_path = assets_dir / "stable.png"
    make_png(img_path)

    h1 = AssetManager.content_hash(img_path)
    h2 = AssetManager.content_hash(img_path)
    assert h1 == h2


def test_content_hash_different_files_differ(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    p1 = make_png(assets_dir / "a.png", color=(255, 0, 0))
    p2 = make_png(assets_dir / "b.png", color=(0, 255, 0))

    assert AssetManager.content_hash(p1) != AssetManager.content_hash(p2)
