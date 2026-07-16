"""Rank candidate texts against a reference text with CLIP cosine similarity."""

import torch
from transformers import CLIPModel, CLIPProcessor

# Change these values, then run the file.
reference = "a happy dog"
candidates = [
    "a joyful puppy",
    "a quiet office desk",
    "a sleeping cat",
    "a running animal",
]

# Text-to-text uses cosine similarity between CLIP text embeddings.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
inputs = processor(text=[reference, *candidates], return_tensors="pt", padding=True, truncation=True).to(device)

with torch.inference_mode():
    features = model.get_text_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    scores = (features[0] @ features[1:].T).cpu().tolist()

for label, score in sorted(zip(candidates, scores, strict=True), key=lambda item: item[1], reverse=True):
    print(f"{score:8.4f}  {label}")
