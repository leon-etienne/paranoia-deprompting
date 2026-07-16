"""Sample images and rank editable content labels with CLIP."""

from pathlib import Path

import torch
from diffusers.utils import load_image
from transformers import CLIPModel, CLIPProcessor

image_url = "https://thispersondoesnotexist.com/"
labels = ["safe portrait", "suggestive content", "nudity", "explicit sexual content"]
iterations = 20
output_dir = Path("outputs/classification/content_scan")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
output_dir.mkdir(parents=True, exist_ok=True)
for iteration in range(1, iterations + 1):
    image = load_image(image_url).convert("RGB")
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        probabilities = model(**inputs).logits_per_image[0].softmax(dim=0).cpu()
    best = int(probabilities.argmax())
    image.save(output_dir / f"sample_{iteration:04}.jpg")
    print(f"{iteration:4}/{iterations}  {probabilities[best]:7.2%}  {labels[best]}")

print("These are relative zero-shot label scores, not a calibrated safety classifier.")
