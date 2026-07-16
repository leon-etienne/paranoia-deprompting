"""Random-image sources used by the loop examples."""

from __future__ import annotations

import time
from io import BytesIO

import requests
from PIL import Image

SOURCES = {
    "picsum": "https://picsum.photos/512",
    "thispersondoesnotexist": "https://thispersondoesnotexist.com/",
}


def fetch_random_image(source: str = "picsum", *, timeout: float = 15.0) -> Image.Image:
    """Fetch one cache-busted image from a named source or URL."""
    base_url = SOURCES.get(source, source)
    cache_buster = time.time_ns()
    separator = "&" if "?" in base_url else "?"
    url = f"{base_url}{separator}random={cache_buster}"
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "clip-token-lab/0.1"})
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")
