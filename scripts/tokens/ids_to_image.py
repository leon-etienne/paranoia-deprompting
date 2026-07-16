"""Generate one or more SDXL Turbo images from explicit token IDs."""

from pathlib import Path

from clip_token_lab.images import make_grid
from clip_token_lab.sdxl_tokens import SDXLTokenGenerator

# Change these values, then run the file.
ids = "320 1125 5390"
seed = 42
steps = 1
guidance = 0.0
width = 512
height = 512
count = 1
output = Path("outputs/token_images.png")

# The generator turns token IDs into prompt embeddings for SDXL Turbo.
images = SDXLTokenGenerator().generate(
    ids,
    seed=seed,
    steps=steps,
    guidance_scale=guidance,
    width=width,
    height=height,
    count=count,
)

# Save either the single image or a compact grid of results.
output.parent.mkdir(parents=True, exist_ok=True)
(images[0] if len(images) == 1 else make_grid(images)).save(output)
print(output)
