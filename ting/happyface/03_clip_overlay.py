import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPProcessor, CLIPModel

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

# Load the image from file
input_image_path = "shirt.png"
image = Image.open(input_image_path).convert("RGB")

# Process the image with CLIP
model_inputs = processor(text=food_keywords, images=image, return_tensors="pt", padding=True)
model_outputs = model(**model_inputs)
logits_per_image = model_outputs.logits_per_image
probabilities = logits_per_image.softmax(dim=1)  # Convert to probabilities
# Get the highest probability food
best_index = probabilities.argmax().item()
best_food = food_keywords[best_index]
best_prob = probabilities[0, best_index].item()

# Print current stats
print(f"Highest probability food: {best_food}\tProbability: {best_prob:.2f}")

# Print all foods and their probabilities
for j, food in enumerate(food_keywords):
    food_prob = probabilities[0, j].item()
    print(f"{food}: {food_prob:.2f}")

# Overlay scores on the image
image = image.convert("RGBA")
draw = ImageDraw.Draw(image)
normal_font_size = 18
top_font_size = 22  # Smaller font for top 3
font = ImageFont.truetype("NotoSans-Regular.ttf", normal_font_size)
top_font = ImageFont.truetype("NotoSans-Regular.ttf", top_font_size)

# Get top 3 indices
top_count = 3
top_indices = torch.topk(probabilities[0], top_count).indices

# Add text overlay for all foods
y_offset = 10
for i, food in enumerate(food_keywords):
    food_prob = probabilities[0, i].item()
    current_font = top_font if i in top_indices else font
    text = f"{food}: {food_prob:.2f}"
    text_bbox = draw.textbbox((10, y_offset), text, font=current_font)
    bg_opacity = 180 if i in top_indices else 128
    draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                  fill=(0, 0, 0, bg_opacity))
    text_color = (255, 255, 0) if i in top_indices else (255, 255, 255)
    draw.text((10, y_offset), text, fill=text_color, font=current_font)
    y_offset += 30 if i in top_indices else 24  # Adjusted spacing for smaller font

# Convert the PIL Image back to RGB and save
image = image.convert("RGB")
output_path = "outputimage/shirt_food_result.jpg"
image.save(output_path)
print(f"Result saved to {output_path}") 