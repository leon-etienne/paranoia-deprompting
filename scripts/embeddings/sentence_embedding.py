"""Print the normalized CLIP embedding of one sentence."""

import torch
from transformers import CLIPModel, CLIPProcessor

# Change this value, then run the file.
sentence = "A small cat sits on a mat."

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
inputs = processor(text=[sentence], return_tensors="pt", padding=True).to(device)

with torch.inference_mode():
    embedding = model.get_text_features(**inputs)[0]
    embedding = embedding / embedding.norm()

print("Sentence:", sentence)
print("Shape:", tuple(embedding.shape))
print("Embedding:", embedding.cpu().tolist())
