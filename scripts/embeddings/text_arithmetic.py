"""Explore arithmetic between CLIP text embeddings and plot the result."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from transformers import CLIPModel, CLIPProcessor

words = ["king", "queen", "man", "woman", "boy", "girl", "prince", "princess"]
positive = "queen"
add = "king"
subtract = "man"
output = Path("outputs/embeddings/text_arithmetic.png")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
inputs = processor(text=words, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.inference_mode():
    features = model.get_text_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)

vectors = features.cpu().numpy()
word_vectors = dict(zip(words, vectors, strict=True))
result = word_vectors[positive] + word_vectors[add] - word_vectors[subtract]
result /= np.linalg.norm(result)
for word, score in sorted(zip(words, vectors @ result, strict=True), key=lambda item: item[1], reverse=True):
    print(f"{score:8.4f}  {word}")

all_vectors = np.vstack([vectors, result])
points = umap.UMAP(n_neighbors=5, min_dist=0.2, random_state=42).fit_transform(all_vectors)
labels = [*words, f"{positive} + {add} - {subtract}"]
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(points[:, 0], points[:, 1])
for point, label in zip(points, labels, strict=True):
    ax.annotate(label, point, xytext=(4, 4), textcoords="offset points")
ax.set(title="CLIP text arithmetic (UMAP)", xticks=[], yticks=[])
fig.tight_layout()
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=150)
print(f"Saved {output}")
