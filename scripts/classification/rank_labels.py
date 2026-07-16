"""Rank labels for an image and display both CLIP logits and probabilities."""

import torch
from diffusers.utils import load_image
from transformers import CLIPModel, CLIPProcessor

image = "http://images.cocodataset.org/val2017/000000039769.jpg"
labels = ["a cat", "a dog", "a person", "an animal", "a pet"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
inputs = processor(text=labels, images=load_image(image).convert("RGB"), return_tensors="pt", padding=True).to(device)
with torch.inference_mode():
    logits = model(**inputs).logits_per_image[0]
    probabilities = logits.softmax(dim=0)

results = zip(labels, logits.cpu().tolist(), probabilities.cpu().tolist(), strict=True)
for rank, (label, logit, probability) in enumerate(sorted(results, key=lambda row: row[1], reverse=True), 1):
    print(f"{rank:2}. {probability:7.2%}  logit={logit:7.2f}  {label}")
