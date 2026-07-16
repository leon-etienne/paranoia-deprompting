"""Reduce the full CLIP token embedding table to searchable 2D coordinates."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

source = Path("outputs/token_map/token_embeddings.csv")
output = Path("outputs/token_map/umap_visualization.csv")
preview = Path("outputs/token_map/umap_preview.png")

with source.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
embeddings = np.asarray([[float(value) for value in row["embedding"].split()] for row in rows], dtype=np.float32)
points = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(embeddings)
with output.open("w", encoding="utf-8", newline="") as file:
    fields = ["token_id", "token", "clean_token", "token_type", "x", "y"]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    for row, point in zip(rows, points, strict=True):
        writer.writerow({**{field: row[field] for field in fields[:-2]}, "x": point[0], "y": point[1]})
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
ax.set(title="UMAP of CLIP token embeddings", xticks=[], yticks=[])
fig.tight_layout()
fig.savefig(preview, dpi=180)
print(f"Saved {output} and {preview}")
