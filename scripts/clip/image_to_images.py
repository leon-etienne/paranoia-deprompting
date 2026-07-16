"""Rank candidate images against a reference image with CLIP cosine similarity."""

import torch
from diffusers.utils import load_image
from transformers import CLIPModel, CLIPProcessor

# Change these values, then run the file.
reference = "https://picsum.photos/seed/reference/512"
images = [
    "https://picsum.photos/seed/reference-like/512",
    "https://picsum.photos/seed/other-one/512",
    "https://picsum.photos/seed/other-two/512",
]

# Encode the reference and each candidate as normalized CLIP image vectors.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
loaded = [load_image(path).convert("RGB") for path in [reference, *images]]
inputs = processor(images=loaded, return_tensors="pt").to(device)

# Higher cosine similarity means closer in CLIP's image embedding space.
with torch.inference_mode():
    features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    scores = (features[0] @ features[1:].T).cpu().tolist()

for path, score in sorted(zip(images, scores, strict=True), key=lambda item: item[1], reverse=True):
    print(f"{score:8.4f}  {path}")
