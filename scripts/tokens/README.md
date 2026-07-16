# Tokens · Words are not tokens

Inspect the pieces between prompt and embedding.

![Images generated from single-token words](../../docs/images/single-token-words.jpg)

```bash
python scripts/tokens/count_tokens.py
python scripts/tokens/search_vocab.py
python scripts/tokens/prompt_to_ids.py
python scripts/tokens/ids_to_text.py
python scripts/tokens/ids_to_image.py
python scripts/tokens/app.py
```

- Start/end tokens count too.
- `</w>` marks a word ending in CLIP's vocabulary.
- Token IDs only mean something inside their tokenizer.

[Notebook](../../notebooks/2_Tokens.ipynb) · [full token map](../token_map/README.md)
