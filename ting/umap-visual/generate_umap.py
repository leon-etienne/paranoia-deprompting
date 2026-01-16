"""
Generate UMAP coordinates from token_embeddings.csv and show a quick plot.
"""

import csv
import os
import numpy as np

try:
    import umap
except ImportError:
    raise SystemExit("Please install umap-learn: pip install umap-learn")


def load_embeddings(embeddings_csv_path):
    rows = []
    embeddings = []

    with open(embeddings_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            embedding = [float(x) for x in row['embedding'].split()]
            rows.append(row)
            embeddings.append(embedding)

    return rows, np.array(embeddings)


def generate_umap(embeddings_csv_path, output_csv_path):
    print(f"Loading embeddings from {embeddings_csv_path}...")
    rows, embeddings = load_embeddings(embeddings_csv_path)
    print(f"Embedding matrix: {embeddings.shape}")

    print("Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42
    )
    coords = reducer.fit_transform(normalized)

    print(f"Saving UMAP coordinates to {output_csv_path}...")
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['token_id', 'token', 'clean_token', 'token_type', 'x', 'y']
        )
        writer.writeheader()
        for row, (x, y) in zip(rows, coords):
            writer.writerow({
                'token_id': row.get('token_id', ''),
                'token': row.get('token', ''),
                'clean_token': row.get('clean_token', ''),
                'token_type': row.get('token_type', ''),
                'x': float(x),
                'y': float(y),
            })

    print("Done.")
    show_umap_plot(coords)


def show_umap_plot(coords):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.5)
    plt.title("UMAP of CLIP token embeddings")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_csv_path = os.path.join(script_dir, 'token_embeddings.csv')
    output_csv_path = os.path.join(script_dir, 'umap_visualization.csv')

    generate_umap(embeddings_csv_path, output_csv_path)
