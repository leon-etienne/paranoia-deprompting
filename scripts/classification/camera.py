"""Classify webcam frames against editable labels and overlay the top results."""

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor

camera_index = 0
labels = ["pizza", "sushi", "ramen", "burger", "tacos", "salad"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)
camera = cv2.VideoCapture(camera_index)
if not camera.isOpened():
    raise RuntimeError(f"Could not open camera {camera_index}.")

font = ImageFont.load_default(size=20)
print("Press q to quit.")
try:
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            probabilities = model(**inputs).logits_per_image[0].softmax(dim=0).cpu().tolist()
        ranked = sorted(zip(labels, probabilities, strict=True), key=lambda row: row[1], reverse=True)[:3]
        draw = ImageDraw.Draw(image, "RGBA")
        for row, (label, probability) in enumerate(ranked):
            text = f"{probability:6.2%}  {label}"
            y = 12 + row * 28
            box = draw.textbbox((12, y), text, font=font)
            draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0, 180))
            draw.text((12, y), text, fill="white", font=font)
        cv2.imshow("CLIP camera classifier", cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
