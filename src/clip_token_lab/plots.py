"""Matplotlib helpers used by the Gradio apps."""

from __future__ import annotations

import textwrap
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .clip import RankedScore
from .loops import MatchState


def ranked_bar(results: Sequence[RankedScore], title: str, *, probability: bool = False):
    labels = ["\n".join(textwrap.wrap(item.label, width=36)) for item in results]
    scores = [item.score for item in results]
    height = max(4.5, 1.4 + 0.65 * len(results))
    fig, axis = plt.subplots(figsize=(11, height))
    positions = np.arange(len(results))
    axis.barh(positions, scores)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel("Probability" if probability else "Cosine similarity")
    if probability:
        axis.set_xlim(0.0, 1.0)
    for position, score in zip(positions, scores):
        label = f"{score:.1%}" if probability else f"{score:.3f}"
        axis.text(score, position, f"  {label}", va="center")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def image_ranking(images: Sequence[Image.Image], results: Sequence[RankedScore], title: str, *, probability: bool = False):
    index_by_label = {f"image_{index + 1}": image for index, image in enumerate(images)}
    rows = len(results)
    fig, axes = plt.subplots(rows, 2, figsize=(10, max(3.5, rows * 2.4)), squeeze=False, gridspec_kw={"width_ratios": [1, 3]})
    for row, result in enumerate(results):
        axes[row, 0].imshow(index_by_label[result.label])
        axes[row, 0].axis("off")
        axes[row, 1].barh([0], [result.score])
        axes[row, 1].set_yticks([0], [result.label])
        if probability:
            axes[row, 1].set_xlim(0.0, 1.0)
        label = f"{result.score:.1%}" if probability else f"{result.score:.3f}"
        axes[row, 1].text(result.score, 0, f"  {label}", va="center")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def loop_figure(state: MatchState, *, target_image: Image.Image | None = None, title: str = "CLIP loop"):
    items = []
    if target_image is not None:
        items.append((target_image, "Target"))
    items.extend(
        [
            (state.current_image, f"Current\n{state.current_score:.3f}"),
            (state.best_image, f"Best\n{state.best_score:.3f}"),
            (state.worst_image, f"Worst\n{state.worst_score:.3f}"),
        ]
    )
    fig, axes = plt.subplots(1, len(items), figsize=(4 * len(items), 4))
    if len(items) == 1:
        axes = [axes]
    for axis, (image, item_title) in zip(axes, items):
        axis.imshow(image)
        axis.set_title(item_title)
        axis.axis("off")
    fig.suptitle(f"{title} — iteration {state.iteration}")
    fig.tight_layout()
    return fig
