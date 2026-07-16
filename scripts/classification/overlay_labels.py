"""Rank labels for an image and save the results as an image overlay."""

from pathlib import Path

import torch
from diffusers.utils import load_image
from PIL import ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor

image_source = "https://picsum.photos/seed/food/768"
labels = ["smoked salmon", "schnitzel", "jollof rice", "tajine", "pizza", "sushi", "ramen", "burger", "tacos"]
output = Path("outputs/classification/label_overlay.jpg")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
image = load_image(image_source).convert("RGB")
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
with torch.inference_mode():
    probabilities = model(**inputs).logits_per_image[0].softmax(dim=0).cpu()

ranked = sorted(zip(labels, probabilities.tolist(), strict=True), key=lambda row: row[1], reverse=True)
draw = ImageDraw.Draw(image, "RGBA")
font = ImageFont.load_default(size=18)
for row, (label, probability) in enumerate(ranked):
    text = f"{probability:6.2%}  {label}"
    box = draw.textbbox((12, 12 + row * 25), text, font=font)
    draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0, 180))
    draw.text((12, 12 + row * 25), text, fill="yellow" if row < 3 else "white", font=font)
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output)
print(f"Saved {output}")
