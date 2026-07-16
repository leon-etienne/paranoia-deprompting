"""Small I/O helpers shared by command-line examples."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image


def load_rgb_image(value: str | Path, *, timeout: float = 20.0) -> Image.Image:
    """Load an RGB PIL image from a local path or HTTP(S) URL."""
    text = str(value)
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"}:
        response = requests.get(text, timeout=timeout, headers={"User-Agent": "clip-token-lab/0.1"})
        response.raise_for_status()
        from io import BytesIO

        return Image.open(BytesIO(response.content)).convert("RGB")
    path = Path(text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def read_nonempty_lines(path: str | Path) -> list[str]:
    """Read stripped, non-empty lines from a UTF-8 text file."""
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
