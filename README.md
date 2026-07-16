# Paranoia De-Prompting: Artistic experiments on generative image

Small CLIP, token, SDXL, and image-process experiments for probing how prompts, tokens, images, and repeated transformations drift.

The repo is organized as annotated notebooks plus direct Python scripts. Scripts are intentionally plain: edit the lowercase variables near the top of a file, then run it.

## Install

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

`requirements.txt` includes the CUDA PyTorch wheel index for NVIDIA machines. If you need a different CUDA, CPU, or ROCm build, adjust the PyTorch lines using the selector at <https://pytorch.org/get-started/locally/>.

## Run The Interactive Scripts

```bash
python scripts/clip/app.py
python scripts/loops/app.py
python scripts/tokens/app.py
python scripts/evolution/app.py
```

To create a public Gradio link, explicitly change `.launch()` to `.launch(share=True)` or set `GRADIO_SHARE=True`. Do not expose an unauthenticated GPU app unless that is intentional.

## Run The Small Experiments

Each script has editable lowercase constants at the top.

### CLIP Similarity

```bash
python scripts/clip/image_to_text.py
python scripts/clip/text_to_images.py
python scripts/clip/text_to_text.py
python scripts/clip/image_to_images.py
```

### CLIP Loops

```bash
python scripts/loops/text_to_random_images.py
python scripts/loops/image_to_random_images.py
```

The loop scripts save `current.jpg`, `min.jpg`, and `max.jpg` inside `outputs/text_loop/` or `outputs/image_loop/`.

### Tokens

```bash
python scripts/tokens/search_vocab.py
python scripts/tokens/prompt_to_ids.py
python scripts/tokens/ids_to_text.py
python scripts/tokens/ids_to_image.py
```

### Evolution

```bash
python scripts/evolution/image_to_tokens.py
python scripts/evolution/tokens_to_image.py
```

### Complexity

```bash
python scripts/complexity.py
```

This repeats Gaussian blur and unsharp masking on the same image, then exports the intermediate frames to `out.mp4`.

## Notebooks

```text
notebooks/0_CLIP_Intro.ipynb
notebooks/1_CLIP_Loops.ipynb
notebooks/2_Tokens.ipynb
notebooks/3_Evolution.ipynb
notebooks/4_Complexity.ipynb
```

The uploaded originals are retained under `notebooks/original/`.

## Hardware Expectations

| Path | CPU | CUDA GPU |
|---|---:|---:|
| CLIP similarity | Works, slower | Recommended |
| CLIP random loops | Works, network-bound and slower inference | Recommended |
| Tokenizer inspect/encode/decode | Works | Not required |
| Blur/sharpen complexity video | Works | Not required |
| SDXL generation from IDs | Not supported by this repo | Required |
| Evolutionary token search | Not supported by this repo | Required; substantial VRAM recommended |

The first run downloads model weights from Hugging Face. `stabilityai/sdxl-turbo` may require accepting its model terms and authenticating with `huggingface-cli login` depending on the current Hub policy.

## Repository Map

```text
notebooks/               Annotated teaching notebooks
notebooks/original/      Uploaded notebooks, unchanged
scripts/                 Direct runnable experiments and Gradio launchers
src/clip_token_lab/      Reusable implementation
tests/                   Lightweight tests that do not download models
```

## Important Experimental Assumptions

- Cross-modal image/text ranking uses CLIP's scaled logits followed by softmax over the provided candidate set. These values are relative probabilities within that set, not calibrated real-world probabilities.
- Text/text and image/image compare normalized embeddings with cosine similarity. Negative similarities are preserved instead of clipped to zero.
- The evolution experiment treats CLIP token IDs as reusable SDXL tokenizer IDs. The vocabularies are closely related, but token semantics can differ between tokenizers.
- SDXL Turbo is designed for very few denoising steps. The defaults here preserve the notebook's experimental settings rather than claiming universal best settings.
- Repeated blur and sharpen operations do not cancel each other out; the residual changes accumulate into visible drift.

## Development Checks

```bash
python -m compileall src scripts
pytest
ruff check src scripts tests
```
