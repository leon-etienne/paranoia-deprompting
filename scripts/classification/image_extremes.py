"""Sample generated portraits and retain the CLIP score extremes."""

from pathlib import Path

import torch
from diffusers.utils import load_image
from transformers import CLIPModel, CLIPProcessor

label = "a happy person"
image_url = "https://thispersondoesnotexist.com/"
iterations = 20
output_dir = Path("outputs/classification/extremes")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
output_dir.mkdir(parents=True, exist_ok=True)
minimum = (float("inf"), None)
maximum = (float("-inf"), None)
for iteration in range(1, iterations + 1):
    image = load_image(image_url).convert("RGB")
    inputs = processor(text=[label], images=image, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        score = model(**inputs).logits_per_image.item()
    if score < minimum[0]:
        minimum = (score, image.copy())
        minimum[1].save(output_dir / "minimum.jpg")
    if score > maximum[0]:
        maximum = (score, image.copy())
        maximum[1].save(output_dir / "maximum.jpg")
    image.save(output_dir / "current.jpg")
    print(f"{iteration:4}/{iterations}  score={score:8.3f}  min={minimum[0]:8.3f}  max={maximum[0]:8.3f}")
