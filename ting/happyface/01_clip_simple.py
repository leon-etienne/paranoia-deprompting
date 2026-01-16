from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Web image
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
input_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

model_inputs = processor(
    text=["a cat", "a dog"],
    images=input_image,
    return_tensors="pt",
    padding=True
)
model_outputs = model(**model_inputs)
probabilities = model_outputs.logits_per_image.softmax(dim=1)

print("Probabilities:", probabilities)

# Local image (commented out)
# image = Image.open("./path/to/image.jpg").convert("RGB")
# inputs = processor(text=["a cat", "a dog"], images=image, return_tensors="pt", padding=True)
# outputs = model(**inputs)
# probs = outputs.logits_per_image.softmax(dim=1)
# print("Probabilities:", probs)

