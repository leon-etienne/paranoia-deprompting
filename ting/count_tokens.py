from transformers import CLIPTokenizer

# Load CLIP tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Function to count tokens and show tokenization details
def count_tokens(text):
    """Count tokens and return tokenization details."""
    tokens = tokenizer(text, return_tensors="pt")
    token_ids = tokens['input_ids'][0].tolist()
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    token_count = len(token_ids)
    return token_count, token_strings, token_ids

# Interactive loop
while True:
    text = input("\nEnter text (or 'quit' to exit): ")
    if text.lower() in ['quit', 'exit', 'q']:
        break
    if text.strip():
        num_tokens, token_strings, token_ids = count_tokens(text)
        print(f"\nToken count: {num_tokens}")
        print(f"\nTokenization breakdown:")
        for i, (token_str, token_id) in enumerate(zip(token_strings, token_ids)):
            print(f"  [{i}] Token: '{token_str}' → ID: {token_id}")
        
        # Decode tokens back to UTF-8 text
        decoded_text = tokenizer.decode(token_ids)
        print(f"\nDecoded back to UTF-8: {decoded_text}")
        print(f"Original text: {text}")
        print(f"Match: {decoded_text == text}")
    else:
        print("Please enter some text.")

