import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPProcessor, CLIPModel

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

# List of food keywords
food_keywords = [
    "Smoked Salmon",
    "schnitzel",
    "Jollof rice",
    "tajine",
    "Macaroni and cheese",
    "Curry Wurst",
    "Pizza",
    "Nasi Lemak",
    "Sauce Mole",
    "Fried Rice",
    "Katsudon",
    "Crepe",
    "kimchi",
    "Sushi",
    "Pad Thai",
    "Pho",
    "Ramen",
    "Burger",
    "Tacos",
    "Paella"
]

camera_index = 1
camera = cv2.VideoCapture(camera_index)

# Check if the camera is opened successfully
if not camera.isOpened():
    fallback_index = 0
    camera.release()
    camera = cv2.VideoCapture(fallback_index)
    if not camera.isOpened():
        print(f"Failed to open the camera (tried {camera_index} and {fallback_index}).")
        exit()

print("Press 'q' to quit the program")

frame_count = 0

while True:
    # Capture frame from the camera
    ok, frame = camera.read()
    if not ok:
        print("Failed to capture frame.")
        break

    # Convert the captured frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    # Process the image with CLIP
    model_inputs = processor(text=food_keywords, images=image, return_tensors="pt", padding=True)
    model_outputs = model(**model_inputs)
    logits_per_image = model_outputs.logits_per_image
    probabilities = logits_per_image.softmax(dim=1)
    
    frame_count += 1
    
    # Get the highest probability food
    best_index = probabilities.argmax().item()
    best_food = food_keywords[best_index]
    best_prob = probabilities[0, best_index].item()
    
    # Print current stats
    print(f"\nFrame: {frame_count}")
    print(f"Highest probability food: {best_food}\tProbability: {best_prob:.2f}")
    
    # Print all foods and their probabilities
    for j, food in enumerate(food_keywords):
        food_prob = probabilities[0, j].item()
        print(f"{food}: {food_prob:.2f}")
    
    # Overlay scores on the camera image
    draw = ImageDraw.Draw(image)
    normal_font_size = 35
    top_font_size = 45  # Larger font for top 3
    try:
        font = ImageFont.truetype("NotoSans-Regular.ttf", normal_font_size)
        top_font = ImageFont.truetype("NotoSans-Regular.ttf", top_font_size)
    except OSError:
        font = ImageFont.load_default()
        top_font = ImageFont.load_default()
    
    # Get top 3 indices
    top_count = 3
    top_indices = torch.topk(probabilities[0], top_count).indices
    
    # Add text overlay for all foods
    y_offset = 10
    for i, food in enumerate(food_keywords):
        food_prob = probabilities[0, i].item()
        # Use larger font for top 3 foods
        current_font = top_font if i in top_indices else font
        # Add background rectangle
        text = f"{food}: {food_prob:.2f}"
        text_bbox = draw.textbbox((10, y_offset), text, font=current_font)
        # Make background more opaque for top 3
        bg_opacity = 180 if i in top_indices else 128
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                      fill=(0, 0, 0, bg_opacity))
        # Add text with different color for top 3
        text_color = (255, 255, 0) if i in top_indices else (255, 255, 255)  # Yellow for top 3
        draw.text((10, y_offset), text, fill=text_color, font=current_font)
        y_offset += 45 if i in top_indices else 40  # More spacing for top 3 items
    
    # Convert the PIL Image back to OpenCV format for display
    frame_with_text = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Display the frame in a window
    cv2.imshow('Food Detector', frame_with_text)
    
    # Handle key presses
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the window
camera.release()
cv2.destroyAllWindows() 