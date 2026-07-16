"""Search the primary SDXL tokenizer vocabulary."""

from transformers import AutoTokenizer

# Change these values, then run the file.
query = "astronaut"
limit = 50

tokenizer = AutoTokenizer.from_pretrained("stabilityai/sdxl-turbo", subfolder="tokenizer")
matches = (
    (token, token_id)
    for token, token_id in tokenizer.get_vocab().items()
    if query.casefold() in token.casefold() or query in str(token_id)
)
for token, token_id in list(matches)[:limit]:
    print(f"{token_id:6d}  {token!r}")
