"""Evolve a CLIP token sequence for a target image."""

import json
import random
from pathlib import Path

import torch
from diffusers.utils import load_image
from transformers import CLIPModel, CLIPProcessor

# Change these values, then run the file.
image = "https://picsum.photos/seed/target/512"
tokens = 16
population = 1024
generations = 100
score_batch = 4096
seed = 0
output = Path("outputs/evolution.json")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for token evolution.")

random.seed(seed)
torch.manual_seed(seed)
device = "cuda"
model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id

pool = torch.arange(tokenizer.vocab_size, device=device)
for special in (bos, eos, tokenizer.pad_token_id):
    if special is not None:
        pool = pool[pool != special]

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    inputs = processor(images=load_image(image).convert("RGB"), return_tensors="pt").to(device)
    image_features = model.get_image_features(**inputs).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)


def score(sequences):
    scores = []
    with torch.inference_mode():
        for start in range(0, len(sequences), score_batch):
            batch = sequences[start : start + score_batch]
            input_ids = torch.cat(
                [
                    torch.full((len(batch), 1), bos, device=device),
                    batch,
                    torch.full((len(batch), 1), eos, device=device),
                ],
                dim=1,
            )
            with torch.autocast("cuda", dtype=torch.float16):
                features = model.get_text_features(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
            features = features.float()
            features /= features.norm(dim=-1, keepdim=True)
            scores.append((features @ image_features.T).squeeze(1))
    return torch.cat(scores)


sequences = pool[torch.randint(len(pool), (population, tokens), device=device)]
scores = score(sequences)
best_sequence = sequences[scores.argmax()].clone()
best_score = float(scores.max())
elite_count = max(1, round(population * 0.08))

for generation in range(1, generations + 1):
    elites = sequences[torch.argsort(scores, descending=True)[:elite_count]].clone()
    children = [elites]
    while sum(len(group) for group in children) < population:
        candidates = random.sample(range(population), min(4, population))
        parent_1 = sequences[max(candidates, key=lambda i: float(scores[i]))]
        candidates = random.sample(range(population), min(4, population))
        parent_2 = sequences[max(candidates, key=lambda i: float(scores[i]))]
        cut = random.randint(1, tokens - 1) if tokens > 1 else 1
        child = torch.cat([parent_1[:cut], parent_2[cut:]]).clone()
        mutation = torch.rand(tokens, device=device) < 0.08
        child[mutation] = pool[torch.randint(len(pool), (int(mutation.sum()),), device=device)]
        children.append(child.unsqueeze(0))

    sequences = torch.cat(children)[:population]
    scores = score(sequences)
    if float(scores.max()) > best_score:
        best_score = float(scores.max())
        best_sequence = sequences[scores.argmax()].clone()
    print(f"{generation:04d}/{generations:04d} best={best_score:.6f}")

token_ids = [int(value) for value in best_sequence.tolist()]
payload = {
    "decoded_text": tokenizer.decode(token_ids, skip_special_tokens=True),
    "score": best_score,
    "token_ids": token_ids,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(output)
