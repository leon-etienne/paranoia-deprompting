"""Rank candidate texts against a reference text with CLIP cosine similarity."""

from clip_token_lab.clip import ClipEmbedder

# Change these values, then run the file.
reference = "a happy dog"
candidates = [
    "a joyful puppy",
    "a quiet office desk",
    "a sleeping cat",
    "a running animal",
]

# Text-to-text uses cosine similarity between CLIP text embeddings.
results = ClipEmbedder().text_to_text(reference, candidates)

for result in results:
    print(f"{result.score:8.4f}  {result.label}")
