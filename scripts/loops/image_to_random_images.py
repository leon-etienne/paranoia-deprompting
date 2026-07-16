"""Search random images for the closest and furthest CLIP match to an image."""

from pathlib import Path

import torch
from diffusers.utils import load_image
from transformers import CLIPModel, CLIPProcessor

# Change these values, then run the file.
target_image = "https://picsum.photos/seed/target/512"
iterations = 1000
image_url = "https://picsum.photos/seed/{i}/512"
output_dir = Path("outputs/image_loop")

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "openai/clip-vit-large-patch14"

# Load CLIP and encode the target image once.
model = CLIPModel.from_pretrained(name).to(device).eval()
processor = CLIPProcessor.from_pretrained(name)
target = load_image(target_image).convert("RGB")
target_inputs = processor(images=target, return_tensors="pt").to(device)

with torch.inference_mode():
    target_features = model.get_image_features(**target_inputs)
    target_features = target_features / target_features.norm(dim=-1, keepdim=True)

lo = float("inf")
hi = float("-inf")
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(iterations):
    # Load a candidate image and compare normalized CLIP image embeddings.
    image = load_image(image_url.format(i=i)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        score = (image_features @ target_features.T).item()

    image.save(output_dir / "current.jpg")

    # Save the furthest image seen so far.
    if score < lo:
        lo = score
        image.save(output_dir / "min.jpg")

    # Save the closest image seen so far.
    if score > hi:
        hi = score
        image.save(output_dir / "max.jpg")

    print(score, lo, hi)
