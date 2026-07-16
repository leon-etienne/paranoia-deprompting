"""Show how candidate image labels are tokenized before CLIP ranks them."""

import torch
from diffusers.utils import load_image
from transformers import CLIPModel, CLIPProcessor

image = "http://images.cocodataset.org/val2017/000000039769.jpg"
labels = ["a cat", "a dog", "a person"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
inputs = processor(text=labels, images=load_image(image).convert("RGB"), return_tensors="pt", padding=True).to(device)
for label in labels:
    print(f"{label!r} -> {processor.tokenizer.tokenize(label)}")
print("Token IDs:\n", inputs["input_ids"].cpu())
with torch.inference_mode():
    probabilities = model(**inputs).logits_per_image[0].softmax(dim=0).cpu().tolist()
for label, probability in zip(labels, probabilities, strict=True):
    print(f"{probability:7.2%}  {label}")
