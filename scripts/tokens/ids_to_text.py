"""Decode primary SDXL tokenizer IDs."""

import re

from transformers import AutoTokenizer

# Change this value, then run the file.
ids = "320 1125 5390"

# Decoding shows the text implied by an explicit token sequence.
tokenizer = AutoTokenizer.from_pretrained("stabilityai/sdxl-turbo", subfolder="tokenizer")
text = tokenizer.decode([int(value) for value in re.findall(r"\d+", ids)])
print(text)
