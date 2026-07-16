"""Encode a prompt into primary SDXL tokenizer IDs."""

from transformers import AutoTokenizer

# Change this value, then run the file.
prompt = "a cat astronaut"

# Token IDs are the small integers that the text encoder receives.
tokenizer = AutoTokenizer.from_pretrained("stabilityai/sdxl-turbo", subfolder="tokenizer")
ids = tokenizer(prompt, add_special_tokens=False).input_ids
print(" ".join(map(str, ids)))
