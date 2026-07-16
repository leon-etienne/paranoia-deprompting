"""Inspect the CLIP tokens used by an interactively entered prompt."""

from transformers import CLIPTokenizer

model_name = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(model_name)

while True:
    text = input("\nEnter text (or 'quit' to exit): ").strip()
    if text.lower() in {"quit", "exit", "q"}:
        break
    if not text:
        print("Please enter some text.")
        continue

    token_ids = tokenizer(text)["input_ids"]
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"\nToken count (including start/end tokens): {len(token_ids)}")
    for index, (piece, token_id) in enumerate(zip(pieces, token_ids, strict=True)):
        print(f"  [{index:2}] {piece!r:20} -> {token_id}")
    print(f"Decoded: {tokenizer.decode(token_ids)}")
