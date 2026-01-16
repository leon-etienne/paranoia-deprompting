from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)

output_dir = "./outputimage"
os.makedirs(output_dir, exist_ok=True)

max_score = float("-inf")
min_score = float("inf")
iteration = 0
while True:
    iteration += 1
    image_url = "https://thispersondoesnotexist.com/"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    model_inputs = processor(text=["happy"], images=image, return_tensors="pt", padding=True)
    model_outputs = model(**model_inputs)
    similarity_score = model_outputs.logits_per_image.item()

    print(f"{iteration}\tscore:{similarity_score}\tmax:{max_score}\tmin:{min_score}")
    if similarity_score > max_score:
        max_score = similarity_score
        image.save(os.path.join(output_dir, "happiestImage.jpg"))
    if similarity_score < min_score:
        min_score = similarity_score
        image.save(os.path.join(output_dir, "sadestImage.jpg"))
    image.save(os.path.join(output_dir, "this.jpg"))