# CLIP · What does the machine see?

CLIP compares images and language in one embedding space.

| Lower match | Prompt | Higher match |
|---|---|---|
| <img src="../../docs/images/clip-happy-lower.jpg" width="180" alt="Lower CLIP match"><br>`20.67` | **Happy?**<br>Find the highest score | <img src="../../docs/images/clip-happy-higher.jpg" width="180" alt="Higher CLIP match"><br>`24.01` |

Portrait source: [`thispersondoesnotexist.com`](https://thispersondoesnotexist.com/)

```bash
python scripts/clip/image_to_text.py
python scripts/clip/text_to_images.py
python scripts/clip/text_to_text.py
python scripts/clip/image_to_images.py
python scripts/clip/app.py
```

- Cross-modal scripts rank a supplied candidate set.
- Same-modal scripts use cosine similarity.
- Scores describe relations, not objective labels.

[Notebook](../../notebooks/0_CLIP_Intro.ipynb) · [classification extensions](../classification/README.md)
