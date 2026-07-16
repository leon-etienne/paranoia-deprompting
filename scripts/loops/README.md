# Feedback · Run forever?

Use an output as the next input and observe the latent residue.

| state `n` | state `n + 1` | difference |
|---|---|---|
| ![Before](../../docs/images/latent-walk-before.jpg) | ![After](../../docs/images/latent-walk-after.jpg) | ![Difference](../../docs/images/latent-walk-difference.jpg) |

```bash
python scripts/loops/text_to_random_images.py
python scripts/loops/image_to_random_images.py
python scripts/loops/app.py
```

Frames and score extremes are saved under `outputs/text_loop/` or
`outputs/image_loop/`. Network input, model choice, and CLIP scoring all steer
the walk.

[Notebook](../../notebooks/1_CLIP_Loops.ipynb)
