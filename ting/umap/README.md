# UMAP Token Visualization

Interactive UMAP visualization of CLIP token embeddings with search functionality.

## Files

### `index.html` & `index.js`
**Description:** Interactive web-based UMAP visualization with search functionality.

**Features:**
- Interactive scatter plot of token embeddings
- Zoom and pan functionality
- Search by token_id or token text
- Support for multiple search terms (separated by comma or space)
- Color-coded by token type
- Click on points to see token information

**Usage:**
1. Ensure `umap_visualization.csv` is present in this folder
2. Open `index.html` in a web browser
3. Use the search bar to find tokens

**Search Examples:**
- Token ID: `123` or `123, 456, 789`
- Token text: `cat` or `cat, dog, bird`
- Mixed: `123, cat, 456`

**Dependencies:**
- D3.js v5 (loaded from CDN)
- D3FC (loaded from CDN)

---

## Setup

1. **Open visualization:**
   - **IMPORTANT:** You must use a local server (not file://) due to CORS restrictions
   - Use a local server:
     ```bash
     cd umap
     python -m http.server 8000
     # Then open http://localhost:8000/index.html in your browser
     ```
   - Or from project root:
     ```bash
     python -m http.server 8000
     # Then open http://localhost:8000/umap/index.html
     ```

**Troubleshooting:**
- If you see "Error loading data", make sure:
  1. `umap_visualization.csv` exists in the `umap/` folder
  2. You're accessing via `http://localhost` (not `file://`)
  3. The CSV file is not empty and has valid `x`/`y` columns

## Notes

- The visualization uses WebGL for efficient rendering of large datasets
- Search results are highlighted in green
- Click on any point to see detailed token information

