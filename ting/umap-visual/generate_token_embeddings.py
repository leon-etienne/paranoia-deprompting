"""
Generate CLIP embeddings for all tokens.
"""

import csv
import os
import torch
from transformers import CLIPTokenizer, CLIPModel
from tqdm import tqdm


def generate_token_embeddings(input_csv_path, output_csv_path):
    """
    Read tokens from input_csv_path and write token_embeddings.csv.
    """
    print("Loading CLIP model and tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    print(f"Reading tokens from {input_csv_path}...")
    rows = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows.sort(key=lambda x: int(x['token_id']))
    total_tokens = len(rows)
    print(f"Total tokens: {total_tokens:,}")

    print("Generating embeddings...")
    output_rows = []
    batch_size = 32

    for i in tqdm(range(0, total_tokens, batch_size), desc="Processing batches"):
        batch = rows[i:i + batch_size]
        batch_tokens = []

        for row in batch:
            token_text = row.get('decoded_text') or row.get('token') or ''
            batch_tokens.append(token_text)

        with torch.no_grad():
            inputs = tokenizer(batch_tokens, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.get_text_features(**inputs)
            batch_embeddings = outputs.cpu().numpy()

        for row, embedding in zip(batch, batch_embeddings):
            embedding_str = ' '.join(map(str, embedding))
            output_rows.append({
                'token_id': int(row['token_id']),
                'token': row.get('token', ''),
                'clean_token': row.get('decoded_text') or row.get('token') or '',
                'token_type': row.get('token_type', ''),
                'embedding': embedding_str,
            })

    print(f"Saving embeddings to {output_csv_path}...")
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['token_id', 'token', 'clean_token', 'token_type', 'embedding']
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved {len(output_rows):,} embeddings")
    return output_rows


if __name__ == "__main__":
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Default input/output
    if len(sys.argv) > 1:
        input_csv_path = sys.argv[1]
    else:
        input_csv_path = os.path.join(script_dir, 'clip_token_dictionary_unicode.csv')

    output_csv_path = os.path.join(script_dir, 'token_embeddings.csv')

    print(f"Input CSV: {input_csv_path}")
    print(f"Output CSV: {output_csv_path}")

    generate_token_embeddings(input_csv_path, output_csv_path)
