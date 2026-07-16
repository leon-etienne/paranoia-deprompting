"""Evolve a CLIP token sequence for a target image."""

import json
from pathlib import Path

from clip_token_lab.evolution import CLIPTokenOptimizer, EvolutionConfig
from clip_token_lab.io import load_rgb_image

# Change these values, then run the file.
image = "https://picsum.photos/seed/target/512"
tokens = 16
population = 1024
generations = 100
score_batch = 4096
seed = 0
output = Path("outputs/evolution.json")

# EvolutionConfig keeps the experiment parameters visible in one place.
config = EvolutionConfig(
    seed=seed,
    random_seed=False,
    n_tokens=tokens,
    population=population,
    generations=generations,
    score_batch_size=score_batch,
)

# The optimizer searches token IDs whose decoded text scores well for the image.
result = CLIPTokenOptimizer().optimize(
    load_rgb_image(image),
    config,
    progress=lambda current, total, best: print(f"{current:04d}/{total:04d} best={best:.6f}"),
)

payload = {"decoded_text": result.decoded_text, "score": result.score, "token_ids": result.token_ids}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(output)
