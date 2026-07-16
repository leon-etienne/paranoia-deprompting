"""Search random images for the closest and furthest CLIP match to a text."""

from pathlib import Path

import torch
from diffusers.utils import load_image
from transformers import CLIPModel, CLIPProcessor

# Change these values, then run the file.
text = "Programmer"
iterations = 1000
image_url = "https://thispersondoesnotexist.com/random-person.jpeg?{i}"
output_dir = Path("outputs/text_loop")

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "openai/clip-vit-large-patch14"

# Load CLIP once; the loop only changes the image.
model = CLIPModel.from_pretrained(name).to(device).eval()
processor = CLIPProcessor.from_pretrained(name)

lo = float("inf")
hi = float("-inf")
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(iterations):
    # Load a fresh random image and score it against the text.
    image = load_image(image_url.format(i=i)).convert("RGB")
    inputs = processor(text=[text], images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        score = model(**inputs).logits_per_image.item()

    image.save(output_dir / "current.jpg")

    # Keep the lowest-scoring image as the negative edge case.
    if score < lo:
        lo = score
        image.save(output_dir / "min.jpg")

    # Keep the highest-scoring image as the strongest accidental match.
    if score > hi:
        hi = score
        image.save(output_dir / "max.jpg")

    print(score, lo, hi)
