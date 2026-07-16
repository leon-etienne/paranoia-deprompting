"""Rank text labels for one image with CLIP."""

from clip_token_lab.clip import ClipEmbedder
from clip_token_lab.io import load_rgb_image

# Change these values, then run the file.
image = "https://picsum.photos/seed/person/512"
texts = [
    "a portrait of a person",
    "a landscape photograph",
    "a close-up of food",
    "an abstract painting",
]

# CLIP compares one image against all candidate labels at once.
results = ClipEmbedder().image_to_text(load_rgb_image(image), texts)

# Scores are probabilities relative to the labels listed above.
for result in results:
    print(f"{result.score:8.4%}  {result.label}")
