"""Encode a prompt into primary SDXL tokenizer IDs."""

from clip_token_lab.tokens import TokenToolkit

# Change this value, then run the file.
prompt = "a cat astronaut"

# Token IDs are the small integers that the text encoder receives.
ids = TokenToolkit().encode(prompt)
print(" ".join(map(str, ids)))
