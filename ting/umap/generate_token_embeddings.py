"""
Generate CLIP embeddings for all tokens and create UMAP visualization data.
"""

import csv
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPModel
from tqdm import tqdm
import os

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. UMAP coordinates will be set to 0,0")


def generate_token_embeddings(csv_path, output_dir="umap"):
    """
    Generate embeddings for all tokens and create UMAP coordinates.
    """
    print("Loading CLIP model and tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    # Read token data
    print(f"Reading tokens from {csv_path}...")
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # Sort by token_id
    data.sort(key=lambda x: int(x['token_id']))
    total_tokens = len(data)
    print(f"Total tokens: {total_tokens:,}")

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = []
    token_info = []

    batch_size = 32
    for i in tqdm(range(0, total_tokens, batch_size), desc="Processing batches"):
        batch = data[i:i + batch_size]
        batch_tokens = []
        batch_info = []

        for row in batch:
            token_text = row.get('clean_token', row.get('token', ''))
            batch_tokens.append(token_text)
            batch_info.append({
                'token_id': int(row['token_id']),
                'token': row.get('token', ''),
                'clean_token': token_text,
                'token_type': row.get('token_type', ''),
            })

        # Tokenize and get embeddings
        try:
            with torch.no_grad():
                inputs = tokenizer(batch_tokens, return_tensors="pt", padding=True, truncation=True, max_length=77)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.get_text_features(**inputs)
                batch_embeddings = outputs.cpu().numpy()

                embeddings.append(batch_embeddings)
                token_info.extend(batch_info)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Fallback: use zero embeddings for failed tokens
            for info in batch_info:
                embeddings.append(np.zeros((1, 512)))

    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings)
    print(f"Embedding shape: {all_embeddings.shape}")

    # Apply UMAP if available, otherwise use zeros
    if HAS_UMAP:
        # Normalize embeddings (L2 normalization for cosine similarity)
        print("Normalizing embeddings...")
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        normalized_embeddings = all_embeddings / (norms + 1e-8)

        # Apply UMAP
        print("Applying UMAP dimensionality reduction...")
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='cosine',
            random_state=42
        )
        umap_coords = reducer.fit_transform(normalized_embeddings)
        print(f"UMAP coordinates shape: {umap_coords.shape}")
    else:
        # Use zeros for UMAP coordinates if umap-learn is not available
        umap_coords = np.zeros((len(all_embeddings), 2))
        print("Skipping UMAP (not installed). Using zero coordinates.")

    # Create output data with full embedding vectors
    output_data = []
    viz_data = []

    for i, info in enumerate(token_info):
        # Convert embedding vector to space-separated string
        embedding_str = ' '.join(map(str, all_embeddings[i]))

        # Full data
        output_data.append({
            'token_id': info['token_id'],
            'token': info['token'],
            'clean_token': info['clean_token'],
            'embedding': embedding_str,
            'x': float(umap_coords[i, 0]),
            'y': float(umap_coords[i, 1]),
        })

        # Lightweight visualization data
        viz_data.append({
            'token_id': info['token_id'],
            'token': info['token'],
            'clean_token': info['clean_token'],
            'token_type': info['token_type'],
            'x': float(umap_coords[i, 0]),
            'y': float(umap_coords[i, 1]),
        })

    # Save as CSV for the visualization
    os.makedirs(output_dir, exist_ok=True)

    # Save full data (optional, but good to keep)
    csv_path = os.path.join(output_dir, 'token_embeddings.csv')
    print(f"Saving full data to {csv_path}...")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['token_id', 'token', 'clean_token', 'embedding', 'x', 'y'])
        writer.writeheader()
        writer.writerows(output_data)

    # Save lightweight visualization data
    viz_csv_path = os.path.join(output_dir, 'umap_visualization.csv')
    print(f"Saving visualization data to {viz_csv_path}...")
    with open(viz_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['token_id', 'token', 'clean_token', 'token_type', 'x', 'y'])
        writer.writeheader()
        writer.writerows(viz_data)

    print(f"Saved {len(output_data):,} tokens to CSVs")

    return output_data


if __name__ == "__main__":
    import sys

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Default CSV path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = os.path.join(project_root, 'data', 'clip', 'comprehensive_token_analysis.csv')

    # Output directory is the umap folder
    output_dir = script_dir

    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_dir}")

    generate_token_embeddings(csv_path, output_dir)
