"""Finite and streaming random-image search loops."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator

from PIL import Image

from .clip import ClipEmbedder
from .image_sources import fetch_random_image


@dataclass
class MatchState:
    iteration: int
    current_image: Image.Image
    current_score: float
    best_image: Image.Image
    best_score: float
    worst_image: Image.Image
    worst_score: float


def _iterate_matches(
    target_embedding,
    embedder: ClipEmbedder,
    *,
    source: str,
    delay: float,
    iterations: int | None,
    fetcher: Callable[[str], Image.Image],
) -> Iterator[MatchState]:
    best_score = float("-inf")
    worst_score = float("inf")
    best_image = None
    worst_image = None
    step = 0

    while iterations is None or step < iterations:
        step += 1
        image = fetcher(source)
        image_embedding = embedder.encode_images([image])
        score = float((image_embedding @ target_embedding.T).item())
        if score > best_score:
            best_score, best_image = score, image.copy()
        if score < worst_score:
            worst_score, worst_image = score, image.copy()
        assert best_image is not None and worst_image is not None
        yield MatchState(step, image, score, best_image, best_score, worst_image, worst_score)
        if delay > 0:
            time.sleep(delay)


def iter_text_to_random_images(
    prompt: str,
    embedder: ClipEmbedder,
    *,
    source: str = "picsum",
    delay: float = 0.25,
    iterations: int | None = None,
    fetcher: Callable[[str], Image.Image] = fetch_random_image,
) -> Iterator[MatchState]:
    """Stream random images ranked against a text embedding."""
    target = embedder.encode_texts([prompt.strip()])
    yield from _iterate_matches(target, embedder, source=source, delay=delay, iterations=iterations, fetcher=fetcher)


def iter_image_to_random_images(
    target_image: Image.Image,
    embedder: ClipEmbedder,
    *,
    source: str = "picsum",
    delay: float = 0.25,
    iterations: int | None = None,
    fetcher: Callable[[str], Image.Image] = fetch_random_image,
) -> Iterator[MatchState]:
    """Stream random images ranked against a target image embedding."""
    target = embedder.encode_images([target_image.convert("RGB")])
    yield from _iterate_matches(target, embedder, source=source, delay=delay, iterations=iterations, fetcher=fetcher)
