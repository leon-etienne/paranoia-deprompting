"""Decode primary SDXL tokenizer IDs."""

from clip_token_lab.tokens import TokenToolkit

# Change this value, then run the file.
ids = "320 1125 5390"

# Decoding shows the text implied by an explicit token sequence.
text = TokenToolkit().decode(ids)
print(text)
