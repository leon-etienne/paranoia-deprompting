# Jupyter (optional):
# !pip install torch transformers umap-learn matplotlib
# %matplotlib inline

import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

words = ["king", "queen", "man", "woman", "boy", "girl", "prince", "princess"]
inputs = processor(text=words, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    features = model.get_text_features(**inputs)
# Normalize so dot product = cosine similarity
features = features / features.norm(dim=-1, keepdim=True)

# Map each word to its embedding vector
word_map = {w: e.cpu().numpy() for w, e in zip(words, features)}

# Final calculated embedding:
combo = word_map["queen"] + (word_map["king"] - word_map["man"])
combo = combo / np.linalg.norm(combo)

print("Embedding (queen + (king - man)):", combo.tolist())

sims = features.cpu().numpy() @ combo
ranked = sorted(zip(words, sims), key=lambda x: x[1], reverse=True)
print("Most similar words to (queen + (king - man)):")
for w, s in ranked[:5]:
    print(f"  {w}: {s:.3f}")

# UMAP plot with lines to show the analogy geometry
all_vectors = np.vstack([features.cpu().numpy(), combo])
labels = words + ["queen + (king - man)"]

reducer = umap.UMAP(n_neighbors=5, min_dist=0.2, random_state=42)
points = reducer.fit_transform(all_vectors)

plt.figure(figsize=(6, 5))
plt.scatter(points[:-1, 0], points[:-1, 1], s=30)
for i, label in enumerate(labels[:-1]):
    plt.text(points[i, 0], points[i, 1], label, fontsize=9)

# Draw lines: king -> man (subtract), then queen -> combo (add)
idx = {w: i for i, w in enumerate(words)}
plt.plot(
    [points[idx["king"], 0], points[idx["man"], 0]],
    [points[idx["king"], 1], points[idx["man"], 1]],
    linewidth=1,
)
plt.plot(
    [points[idx["queen"], 0], points[-1, 0]],
    [points[idx["queen"], 1], points[-1, 1]],
    linewidth=1,
)
plt.text(points[-1, 0], points[-1, 1], labels[-1], fontsize=9)
plt.title("CLIP Text Analogy (UMAP)")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("week6_text_add_subtract_umap.png", dpi=150)
plt.show()
