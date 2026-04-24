"""Asset Manager: resolves existing assets or signals generation is needed."""

import hashlib
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class AssetManager:
    """Manages input assets: checks local folder, returns path if found."""

    def __init__(self, assets_dir: Path):
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._build_index()

    def _build_index(self) -> dict[str, Path]:
        """Index all available assets by normalized product name."""
        index: dict[str, Path] = {}
        for ext in SUPPORTED_EXTENSIONS:
            for f in self.assets_dir.rglob(f"*{ext}"):
                key = self._normalize(f.stem)
                index[key] = f
        logger.debug(f"Asset index built: {len(index)} assets in {self.assets_dir}")
        return index

    def _normalize(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")

    def get_asset(self, product_name: str, explicit_path: str | None = None) -> Path | None:
        """Return existing asset path if available, else None."""
        # 1. Explicit path from brief takes priority
        if explicit_path:
            p = Path(explicit_path)
            if p.exists():
                logger.info(f"  Using explicit asset: {p}")
                return p
            logger.warning(f"  Explicit asset not found: {p}, will generate instead")

        # 2. Fuzzy match in assets directory
        key = self._normalize(product_name)
        if key in self._index:
            path = self._index[key]
            logger.info(f"  Found cached asset for '{product_name}': {path}")
            return path

        logger.info(f"  No existing asset for '{product_name}': generation required")
        return None

    def save_generated(self, image: Image.Image, product_name: str) -> Path:
        """Persist a newly generated base image for reuse."""
        dest = self.assets_dir / f"{self._normalize(product_name)}_generated.png"
        image.save(dest, "PNG")
        self._index[self._normalize(product_name)] = dest
        logger.info(f"  Saved generated asset: {dest}")
        return dest

    def load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGBA")

    @staticmethod
    def content_hash(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()
