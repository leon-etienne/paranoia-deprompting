"""Project a small collection of CLIP text embeddings into two dimensions."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import umap
from transformers import CLIPModel, CLIPProcessor

# Change these values, then run the file.
texts = ["a cat", "a dog", "a red car", "a blue car", "a sunny beach", "a rainy city", "pizza", "salad"]
output = Path("outputs/embeddings/text_umap.png")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.inference_mode():
    features = model.get_text_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)

neighbors = min(5, len(texts) - 1)
points = umap.UMAP(n_neighbors=neighbors, min_dist=0.2, random_state=42).fit_transform(features.cpu().numpy())
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(points[:, 0], points[:, 1])
for point, label in zip(points, texts, strict=True):
    ax.annotate(label, point, xytext=(4, 4), textcoords="offset points")
ax.set(title="CLIP text embeddings (UMAP)", xticks=[], yticks=[])
fig.tight_layout()
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=150)
print(f"Saved {output}")
