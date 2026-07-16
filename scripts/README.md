# Example code

Edit the values near the top of a script, then run it from the project root.

| Area | Material | Start with |
|---|---|---|
| [CLIP](clip/README.md) | text ↔ image similarity | `python scripts/clip/image_to_text.py` |
| [Feedback loops](loops/README.md) | recursion and drift | `python scripts/loops/text_to_random_images.py` |
| [Tokens](tokens/README.md) | vocabulary and token IDs | `python scripts/tokens/count_tokens.py` |
| [Evolution](evolution/README.md) | search in token space | `python scripts/evolution/image_to_tokens.py` |
| [Embeddings](embeddings/README.md) | semantic vectors and UMAP | `python scripts/embeddings/text_umap.py` |
| [Classification](classification/README.md) | labels, overlays, camera | `python scripts/classification/rank_labels.py` |
| [Token map](token_map/README.md) | the whole CLIP vocabulary | `python scripts/token_map/generate_embeddings.py` |
| Complexity | repeated blur/sharpen | `python scripts/complexity.py` |

Interactive apps live in `clip/app.py`, `loops/app.py`, `tokens/app.py`, and
`evolution/app.py`.
