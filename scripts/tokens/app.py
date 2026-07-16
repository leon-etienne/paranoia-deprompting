"""Launch the tokenizer and token-to-image Gradio app."""

from functools import lru_cache

import gradio as gr
import pandas as pd

from clip_token_lab.images import make_grid
from clip_token_lab.sdxl_tokens import SDXLTokenGenerator
from clip_token_lab.tokens import TokenToolkit, parse_token_ids


@lru_cache(maxsize=1)
def toolkit() -> TokenToolkit:
    return TokenToolkit()


@lru_cache(maxsize=1)
def generator() -> SDXLTokenGenerator:
    return SDXLTokenGenerator()


def search_vocab(query):
    rows = toolkit().search(query, limit=500)
    data = [(row.token, row.token_id) for row in rows]
    return pd.DataFrame(data, columns=["Token", "ID"]), f"{len(rows)} shown"


def encode_prompt(text):
    ids = toolkit().encode(text)
    return " ".join(map(str, ids)), toolkit().highlighted_text(ids)


def inspect_ids(text):
    ids = parse_token_ids(text, vocab_size=toolkit().vocab_size)
    return toolkit().decode(ids), toolkit().highlighted_text(ids)


def generate_images(ids, seed, width, height, rows, columns):
    count = int(rows) * int(columns)
    images = generator().generate(ids, seed=int(seed), width=int(width), height=int(height), count=count)
    return make_grid(images, columns=int(columns))


initial = toolkit().search("", limit=500)
initial_df = pd.DataFrame([(row.token, row.token_id) for row in initial], columns=["Token", "ID"])

with gr.Blocks(title="Token-language toolkit") as demo:
    gr.Markdown("# Token-language toolkit")

    with gr.Accordion("Vocabulary", open=False):
        query = gr.Textbox(label="Token or ID search")
        stats = gr.Markdown("500 shown")
        table = gr.Dataframe(value=initial_df, interactive=False)
        query.change(search_vocab, query, [table, stats], trigger_mode="always_last")

    gr.Markdown("## Prompt to token IDs")
    prompt = gr.Textbox(value="a cinematic photo of a cat astronaut", label="Prompt")
    prompt_ids = gr.Textbox(label="Token IDs", interactive=False)
    prompt_preview = gr.HighlightedText(label="Token preview", combine_adjacent=True)
    prompt.change(encode_prompt, prompt, [prompt_ids, prompt_preview], trigger_mode="always_last")

    gr.Markdown("## Token IDs to text and image")
    ids = gr.Textbox(value="320 1125 5390 3310 539 320 2368", label="Token IDs")
    decoded = gr.Textbox(label="Decoded text", interactive=False)
    preview = gr.HighlightedText(label="Token preview", combine_adjacent=True)
    ids.change(inspect_ids, ids, [decoded, preview], trigger_mode="always_last")

    with gr.Row():
        seed = gr.Number(value=42, precision=0, label="Seed")
        width = gr.Slider(512, 1024, value=512, step=64, label="Width")
        height = gr.Slider(512, 1024, value=512, step=64, label="Height")
        rows = gr.Slider(1, 4, value=1, step=1, label="Rows")
        columns = gr.Slider(1, 4, value=4, step=1, label="Columns")

    button = gr.Button("Generate", variant="primary")
    output = gr.Image(type="pil", label="Generated grid")
    button.click(generate_images, [ids, seed, width, height, rows, columns], output)

demo.launch(theme=gr.themes.Soft())
