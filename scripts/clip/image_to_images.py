"""Rank candidate images against a reference image with CLIP cosine similarity."""

from clip_token_lab.clip import ClipEmbedder
from clip_token_lab.io import load_rgb_image

# Change these values, then run the file.
reference = "https://picsum.photos/seed/reference/512"
images = [
    "https://picsum.photos/seed/reference-like/512",
    "https://picsum.photos/seed/other-one/512",
    "https://picsum.photos/seed/other-two/512",
]

# Encode the reference and each candidate as normalized CLIP image vectors.
embedder = ClipEmbedder()
results = embedder.image_to_images(load_rgb_image(reference), [load_rgb_image(path) for path in images])

# Higher cosine similarity means closer in CLIP's image embedding space.
for result in results:
    index = int(result.label.split("_")[-1]) - 1
    print(f"{result.score:8.4f}  {images[index]}")
