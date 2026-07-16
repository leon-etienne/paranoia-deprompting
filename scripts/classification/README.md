# Classification · Happy?

Give CLIP labels, then inspect which relation scores highest.

```bash
python scripts/classification/rank_labels.py
python scripts/classification/tokenized_labels.py
python scripts/classification/overlay_labels.py
python scripts/classification/image_extremes.py
python scripts/classification/content_scan.py
python scripts/classification/camera.py
```

- Change the image and labels at the top of each file.
- Extremes turn scoring into a feedback/search mechanism.
- The default portrait stream comes from [`thispersondoesnotexist.com`](https://thispersondoesnotexist.com/).
- Webcam mode requires a camera and desktop session.
- Content scores are not calibrated moderation judgments.

[Notebook](../../notebooks/6_CLIP_Classification.ipynb)
