"""
Simple script to get full embedding for a token ID.
Enter a token ID and it prints out the full embedding vector.
"""

import csv
import os

# Path to the token embeddings CSV
CSV_PATH = os.path.join(os.path.dirname(__file__), 'token_embeddings.csv')

def load_embeddings():
    """Load all embeddings from CSV into memory."""
    embeddings_dict = {}
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found!")
        return None
    
    print(f"Loading embeddings from {CSV_PATH}...")
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            token_id = int(row['token_id'])
            embeddings_dict[token_id] = {
                'token': row['token'],
                'clean_token': row['clean_token'],
                'embedding': row['embedding'],
                'x': row['x'],
                'y': row['y']
            }
    
    print(f"Loaded {len(embeddings_dict)} token embeddings")
    return embeddings_dict

def print_embedding(token_id, embeddings_dict):
    """Print full embedding for a given token ID."""
    if token_id not in embeddings_dict:
        print(f"Token ID {token_id} not found!")
        return
    
    data = embeddings_dict[token_id]
    embedding_str = data['embedding']
    embedding_values = [float(x) for x in embedding_str.split()]
    
    print(f"\n{'='*60}")
    print(f"Token ID: {token_id}")
    print(f"Token: {data['token']}")
    print(f"Clean Token: {data['clean_token']}")
    print(f"UMAP X: {data['x']}")
    print(f"UMAP Y: {data['y']}")
    print(f"Embedding dimension: {len(embedding_values)}")
    print(f"{'='*60}")
    print("\nFull embedding vector:")
    print(embedding_str)
    print(f"\nEmbedding as list:")
    print(embedding_values)

# Load embeddings
embeddings = load_embeddings()

if embeddings is None:
    exit(1)

# Interactive loop
print("\n" + "="*60)
print("Token Embedding Lookup")
print("="*60)
print("Enter a token ID to get its full embedding (or 'quit' to exit)")

while True:
    user_input = input("\nEnter token ID (or 'quit' to exit): ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    
    if not user_input:
        print("Please enter a token ID.")
        continue
    
    try:
        token_id = int(user_input)
        print_embedding(token_id, embeddings)
    except ValueError:
        print("Error: Please enter a valid integer token ID.")
    except Exception as e:
        print(f"Error: {e}")

