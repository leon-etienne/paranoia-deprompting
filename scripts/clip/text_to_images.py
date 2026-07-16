"""Rank images for one text prompt with CLIP."""

import torch
from diffusers.utils import load_image
from transformers import CLIPModel, CLIPProcessor

# Change these values, then run the file.
text = "a red sports car"
images = [
    "https://picsum.photos/seed/car/512",
    "https://picsum.photos/seed/forest/512",
    "https://picsum.photos/seed/city/512",
]

# Load the candidate images before sending them through CLIP.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
loaded_images = [load_image(path).convert("RGB") for path in images]
inputs = processor(text=[text], images=loaded_images, return_tensors="pt", padding=True).to(device)

# CLIP returns relative probabilities within this small candidate set.
with torch.inference_mode():
    scores = model(**inputs).logits_per_text[0].softmax(dim=0).cpu().tolist()

for path, score in sorted(zip(images, scores, strict=True), key=lambda item: item[1], reverse=True):
    print(f"{score:8.4%}  {path}")
