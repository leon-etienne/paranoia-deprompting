from PIL import Image, ImageDraw, ImageFont
import os
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# List of NSFW keywords
nsfw_keywords = [
    "sexual",
    "nude",
    "sex",
    "18+",
    "naked",
    "nsfw",
    "porn",
    "dick",
    "vagina",
    "naked person (approximation)",
    "explicit content",
    "uncensored",
    "fuck",
    "nipples",
    "nipples (approximation)",
    "naked breasts",
    "areola",
]
output_dir = "./outputimage"
os.makedirs(output_dir, exist_ok=True)
iteration = 0
while True:
    image_url = "https://thispersondoesnotexist.com/"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    #inputKey = input("Enter the URL of the image: ")
    #url = inputKey
    #img = Image.open(requests.get(url, stream=True).raw)

    model_inputs = processor(text=nsfw_keywords, images=image, return_tensors="pt", padding=True)
    model_outputs = model(**model_inputs)
    logits_per_image = model_outputs.logits_per_image
    probabilities = logits_per_image.softmax(dim=1)
  
    # Get the NSFW keyword with the highest probability
    best_index = probabilities.argmax().item()
    best_label = nsfw_keywords[best_index]
    best_prob = probabilities[0, best_index].item()

    print(f"Highest probability NSFW: {best_label}\tProbability: {best_prob:.2f}")


    # Overlay top label on the image
    draw = ImageDraw.Draw(image)
    font_size = 80  # Increased font size even more
    font = ImageFont.truetype("NotoSans-Regular.ttf", font_size)
   
    draw.text((10, 50), f"{best_label}", stroke=(0, 0, 0), fill=(255, 255, 255), font=font)
    iteration += 1

    #print(f"{i}\tscore: {now:.2f}")
    for j, label in enumerate(nsfw_keywords):
        label_prob = probabilities[0, j].item()
        print(f"nsfw: {label}\tProbability: {label_prob:.2f}")


    image.save(os.path.join(output_dir, f"nsfw_{iteration}.jpg"))
