"""Generate SDXL Turbo images from evolved token IDs stored in JSON."""

import json
from pathlib import Path

from clip_token_lab.images import make_grid
from clip_token_lab.sdxl_tokens import SDXLTokenGenerator

# Change these values, then run the file.
result_json = Path("outputs/evolution.json")
seed = 0
steps = 2
count = 4
output = Path("outputs/evolved_images.png")

# Load the token IDs produced by image_to_tokens.py.
payload = json.loads(result_json.read_text(encoding="utf-8"))

# Generate images directly from those IDs.
images = SDXLTokenGenerator().generate(payload["token_ids"], seed=seed, steps=steps, count=count)

output.parent.mkdir(parents=True, exist_ok=True)
(images[0] if len(images) == 1 else make_grid(images)).save(output)
print(output)
