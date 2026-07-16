"""Rank images for one text prompt with CLIP."""

from clip_token_lab.clip import ClipEmbedder
from clip_token_lab.io import load_rgb_image

# Change these values, then run the file.
text = "a red sports car"
images = [
    "https://picsum.photos/seed/car/512",
    "https://picsum.photos/seed/forest/512",
    "https://picsum.photos/seed/city/512",
]

# Load the candidate images before sending them through CLIP.
loaded_images = [load_rgb_image(path) for path in images]
results = ClipEmbedder().text_to_images(text, loaded_images)

# CLIP returns relative probabilities within this small candidate set.
for result in results:
    index = int(result.label.split("_")[-1]) - 1
    print(f"{result.score:8.4%}  {images[index]}")
