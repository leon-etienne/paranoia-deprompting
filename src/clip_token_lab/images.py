"""PIL image layout helpers."""

from __future__ import annotations

import math
from typing import Sequence

from PIL import Image


def make_grid(images: Sequence[Image.Image], columns: int | None = None) -> Image.Image:
    """Arrange equally sized images in a simple RGB grid."""
    if not images:
        raise ValueError("At least one image is required.")
    rgb = [image.convert("RGB") for image in images]
    width, height = rgb[0].size
    if any(image.size != (width, height) for image in rgb):
        raise ValueError("All images must have the same size.")
    columns = columns or math.ceil(math.sqrt(len(rgb)))
    rows = math.ceil(len(rgb) / columns)
    canvas = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(rgb):
        canvas.paste(image, ((index % columns) * width, (index // columns) * height))
    return canvas
