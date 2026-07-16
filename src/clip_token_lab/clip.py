"""CLIP embeddings and the four similarity pathways used in the intro UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image

from .config import CLIP_BASE_MODEL
from .device import resolve_device


@dataclass(frozen=True)
class RankedScore:
    label: str
    score: float


class ClipEmbedder:
    """Thin wrapper around Hugging Face ``CLIPModel`` and ``CLIPProcessor``."""

    def __init__(self, model_id: str = CLIP_BASE_MODEL, device: str = "auto") -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = resolve_device(device)
        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("At least one text is required.")
        batch = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        batch = self._to_device(batch)
        outputs = self.model.text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        features = self.model.text_projection(outputs.pooler_output)
        return self._normalize(features)

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required.")
        rgb_images = [image.convert("RGB") for image in images]
        batch = self.processor(images=rgb_images, return_tensors="pt")
        batch = self._to_device(batch)
        outputs = self.model.vision_model(pixel_values=batch["pixel_values"])
        features = self.model.visual_projection(outputs.pooler_output)
        return self._normalize(features)

    @staticmethod
    def _rank(labels: Sequence[str], scores: torch.Tensor) -> list[RankedScore]:
        values = scores.detach().float().cpu().tolist()
        return sorted(
            [RankedScore(label=str(label), score=float(score)) for label, score in zip(labels, values)],
            key=lambda item: item.score,
            reverse=True,
        )

    @torch.inference_mode()
    def image_to_text(self, image: Image.Image, texts: Sequence[str], *, probabilities: bool = True) -> list[RankedScore]:
        """Rank candidate texts for one image."""
        if not texts:
            raise ValueError("At least one candidate text is required.")
        batch = self.processor(text=list(texts), images=image.convert("RGB"), return_tensors="pt", padding=True, truncation=True)
        batch = self._to_device(batch)
        logits = self.model(**batch).logits_per_image[0]
        scores = logits.softmax(dim=0) if probabilities else logits
        return self._rank(texts, scores)

    @torch.inference_mode()
    def text_to_images(self, text: str, images: Sequence[Image.Image], *, probabilities: bool = True) -> list[RankedScore]:
        """Rank candidate images for one text prompt."""
        if not text.strip():
            raise ValueError("A text prompt is required.")
        if not images:
            raise ValueError("At least one candidate image is required.")
        labels = [f"image_{index + 1}" for index in range(len(images))]
        batch = self.processor(text=[text.strip()], images=[im.convert("RGB") for im in images], return_tensors="pt", padding=True, truncation=True)
        batch = self._to_device(batch)
        logits = self.model(**batch).logits_per_text[0]
        scores = logits.softmax(dim=0) if probabilities else logits
        return self._rank(labels, scores)

    def text_to_text(self, reference: str, candidates: Sequence[str]) -> list[RankedScore]:
        """Rank texts by cosine similarity to a reference text."""
        if not reference.strip() or not candidates:
            raise ValueError("A reference and at least one candidate text are required.")
        features = self.encode_texts([reference, *candidates])
        scores = features[0] @ features[1:].T
        return self._rank(candidates, scores)

    def image_to_images(self, reference: Image.Image, candidates: Sequence[Image.Image]) -> list[RankedScore]:
        """Rank images by cosine similarity to a reference image."""
        if not candidates:
            raise ValueError("At least one candidate image is required.")
        features = self.encode_images([reference, *candidates])
        scores = features[0] @ features[1:].T
        labels = [f"image_{index + 1}" for index in range(len(candidates))]
        return self._rank(labels, scores)
