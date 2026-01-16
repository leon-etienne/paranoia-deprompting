from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
input_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

text_prompts = ["a cat", "a dog", "a person", "an animal", "a pet"]

model_inputs = processor(text=text_prompts, images=input_image, return_tensors="pt", padding=True)
model_outputs = model(**model_inputs)
logits = model_outputs.logits_per_image.squeeze().tolist()
probabilities = model_outputs.logits_per_image.softmax(dim=1).squeeze().tolist()

ranked = list(zip(text_prompts, logits, probabilities))
ranked.sort(key=lambda x: x[1], reverse=True)

print("Ranked results:")
for rank, (text, score, prob) in enumerate(ranked, 1):
    print(f"{rank}. {text:15s} | Score: {score:6.2f} | Prob: {prob*100:5.1f}%")

