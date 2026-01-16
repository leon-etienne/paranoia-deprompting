# Jupyter (optional):
# !pip install torch transformers

import torch
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

sentence = "A small cat sits on a mat."
inputs = processor(text=[sentence], return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    embedding = model.get_text_features(**inputs)[0]

embedding = embedding / embedding.norm()

print("Sentence:", sentence)
print("Embedding:", embedding.cpu().tolist())
