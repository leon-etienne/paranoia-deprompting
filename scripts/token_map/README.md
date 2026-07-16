# Token dictionary · Cartography of AI

Embed every CLIP token, reduce the vectors with UMAP, and inspect the vocabulary
as a field of model residue.

```bash
python scripts/token_map/generate_embeddings.py
python scripts/token_map/generate_umap.py
```

Outputs:

- `outputs/token_map/token_embeddings.csv`
- `outputs/token_map/umap_visualization.csv`
- `outputs/token_map/umap_preview.png`

The full vocabulary is slow and memory-heavy. The notebook defaults to a
1,000-token sample.

[Notebook](../../notebooks/7_Token_Map.ipynb)
