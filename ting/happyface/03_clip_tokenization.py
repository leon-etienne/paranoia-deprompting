from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
input_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

text_prompts = ["a cat", "a dog", "a person"]
model_inputs = processor(text=text_prompts, images=input_image, return_tensors="pt", padding=True)

tokenizer = processor.tokenizer

print("Tokenization:")
for text in text_prompts:
    token_pieces = tokenizer.tokenize(text)
    print(f"'{text}' -> {token_pieces}")

print(f"\nInput shape: {model_inputs['input_ids'].shape}")
print(f"Token IDs:\n{model_inputs['input_ids']}")

model_outputs = model(**model_inputs)
probabilities = model_outputs.logits_per_image.softmax(dim=1)

print("\nSimilarities:")
for i, text in enumerate(text_prompts):
    print(f"{text}: {probabilities[0][i].item():.3f}")

