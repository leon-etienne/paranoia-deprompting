"""Rank text labels for one image with CLIP."""

import torch
from diffusers.utils import load_image
from transformers import CLIPModel, CLIPProcessor

# Change these values, then run the file.
image = "https://picsum.photos/seed/person/512"
texts = [
    "a portrait of a person",
    "a landscape photograph",
    "a close-up of food",
    "an abstract painting",
]

# CLIP compares one image against all candidate labels at once.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
inputs = processor(text=texts, images=load_image(image).convert("RGB"), return_tensors="pt", padding=True).to(device)

# Scores are probabilities relative to the labels listed above.
with torch.inference_mode():
    scores = model(**inputs).logits_per_image[0].softmax(dim=0).cpu().tolist()

for label, score in sorted(zip(texts, scores, strict=True), key=lambda item: item[1], reverse=True):
    print(f"{score:8.4%}  {label}")
