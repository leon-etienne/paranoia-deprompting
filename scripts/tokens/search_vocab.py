"""Search the primary SDXL tokenizer vocabulary."""

from clip_token_lab.tokens import TokenToolkit

# Change these values, then run the file.
query = "astronaut"
limit = 50

# TokenToolkit wraps the tokenizer so the script stays focused on the experiment.
toolkit = TokenToolkit()

for entry in toolkit.search(query, limit=limit):
    print(f"{entry.token_id:6d}  {entry.token!r}")
