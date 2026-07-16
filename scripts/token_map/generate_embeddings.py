"""Generate one CLIP text embedding for every token in its vocabulary."""

import csv
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

model_name = "openai/clip-vit-base-patch32"
batch_size = 32
output = Path("outputs/token_map/token_embeddings.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CLIPTokenizer.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device).eval()
vocabulary = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["token_id", "token", "clean_token", "token_type", "embedding"])
    writer.writeheader()
    for start in tqdm(range(0, len(vocabulary), batch_size), desc="Embedding tokens"):
        batch = vocabulary[start : start + batch_size]
        clean_tokens = [tokenizer.decode([token_id]) for _, token_id in batch]
        inputs = tokenizer(clean_tokens, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.inference_mode():
            embeddings = model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        for (token, token_id), clean, embedding in zip(batch, clean_tokens, embeddings.cpu().tolist(), strict=True):
            token_type = "special" if token_id in tokenizer.all_special_ids else ("word_end" if token.endswith("</w>") else "subword")
            writer.writerow(
                {"token_id": token_id, "token": token, "clean_token": clean, "token_type": token_type,
                 "embedding": " ".join(map(str, embedding))}
            )
print(f"Saved {output}")
