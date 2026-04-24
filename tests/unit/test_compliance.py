"""Unit tests for brand compliance checker."""

from PIL import Image

from src.compliance.checker import (
    check_brand_colors,
    check_prohibited_words,
    check_logo_present,
    ComplianceResult,
)


def make_image(color=(0, 168, 232)) -> Image.Image:
    img = Image.new("RGBA", (1080, 1080), (*color, 255))
    return img


def test_prohibited_words_detected():
    found = check_prohibited_words("This is guaranteed to work", ["guaranteed", "cure"])
    assert "guaranteed" in found
    assert "cure" not in found


def test_prohibited_words_case_insensitive():
    found = check_prohibited_words("GUARANTEED results", ["guaranteed"])
    assert "guaranteed" in found


def test_no_prohibited_words():
    found = check_prohibited_words("Stay Fresh. Stay Hydrated.", ["guaranteed", "miracle"])
    assert found == []


def test_brand_colors_matched():
    # Create image with exactly the brand color
    brand_color = "#00A8E8"  # RGB: (0, 168, 232)
    img = make_image(color=(0, 168, 232))
    matched, analysis = check_brand_colors(img, [brand_color], tolerance=50)
    assert matched is True


def test_brand_colors_no_match():
    # Use a multi-color gradient image and a brand color very far from any of its colors
    img = Image.new("RGBA", (1080, 1080), (255, 0, 0, 255))
    # Draw green patch to add variety, then check against pure yellow which won't appear
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 540, 540], fill=(0, 255, 0, 255))
    # Yellow (255, 255, 0) is far from both red and green when tolerance=5
    matched, _ = check_brand_colors(img, ["#8B00FF"], tolerance=5)  # Violet — very far
    assert matched is False


def test_no_brand_colors_configured():
    img = make_image()
    matched, _ = check_brand_colors(img, [])
    assert matched is True  # No constraint = always pass


def test_compliance_result_fails_on_prohibited_word():
    result = ComplianceResult(
        prohibited_words_found=["guaranteed"],
        logo_present=True,
        brand_colors_matched=True,
    )
    assert result.passed is False


def test_compliance_result_passes_clean():
    result = ComplianceResult(
        prohibited_words_found=[],
        logo_present=True,
        brand_colors_matched=True,
    )
    assert result.passed is True


# ── ComplianceResult.summary() ────────────────────────────────────────────────

def test_compliance_result_summary_pass():
    result = ComplianceResult(
        prohibited_words_found=[],
        logo_present=True,
        brand_colors_matched=True,
    )
    assert "PASS" in result.summary()


def test_compliance_result_summary_fail():
    result = ComplianceResult(
        prohibited_words_found=["guaranteed"],
        logo_present=True,
        brand_colors_matched=True,
    )
    assert "FAIL" in result.summary()


def test_compliance_result_summary_contains_logo_and_color_status():
    result = ComplianceResult(logo_present=True, brand_colors_matched=False, prohibited_words_found=[])
    s = result.summary()
    assert "logo=" in s
    assert "brand_colors=" in s


# ── check_logo_present ────────────────────────────────────────────────────────

def test_check_logo_present_no_logo_path_returns_true():
    img = make_image()
    assert check_logo_present(img, None) is True


def test_check_logo_present_nonexistent_file_returns_true():
    img = make_image()
    assert check_logo_present(img, "/nonexistent/logo.png") is True


def test_check_logo_present_uniform_region_returns_false():
    """A completely uniform top-right corner has low unique pixel count → no logo detected."""
    from src.compliance.checker import check_logo_present as clp
    import tempfile, pathlib
    from PIL import Image as PILImage

    # Solid color image — no logo composited → unique pixels ≤ 10
    img = PILImage.new("RGBA", (1080, 1080), (50, 50, 50, 255))

    # Save a dummy logo file so the path-exists check passes
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        logo_path = pathlib.Path(f.name)
        PILImage.new("RGBA", (10, 10), (255, 0, 0, 255)).save(logo_path)

    result = clp(img, str(logo_path))
    logo_path.unlink(missing_ok=True)
    # Solid color → unique pixels = 1 ≤ 10 → returns False
    assert result is False


# ── run_compliance_check ──────────────────────────────────────────────────────

def test_run_compliance_check_passes_clean_message():
    from src.compliance.checker import run_compliance_check
    from src.config.models import BrandConfig

    brand = BrandConfig(name="TestBrand")
    img = make_image(color=(0, 168, 232))
    result = run_compliance_check(img, brand, "Stay Fresh.", prohibited_words=["guaranteed"])
    assert result.passed is True
    assert result.prohibited_words_found == []


def test_run_compliance_check_fails_on_prohibited_word():
    from src.compliance.checker import run_compliance_check
    from src.config.models import BrandConfig

    brand = BrandConfig(name="TestBrand")
    img = make_image()
    result = run_compliance_check(img, brand, "guaranteed results", prohibited_words=["guaranteed"])
    assert result.passed is False
    assert "guaranteed" in result.prohibited_words_found


def test_run_compliance_check_warns_on_prohibited_word():
    from src.compliance.checker import run_compliance_check
    from src.config.models import BrandConfig

    brand = BrandConfig(name="TestBrand")
    img = make_image()
    result = run_compliance_check(img, brand, "guaranteed", prohibited_words=["guaranteed"])
    assert any("Prohibited" in w for w in result.warnings)
