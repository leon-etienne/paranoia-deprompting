"""Inspect SDXL tokens and generate images from explicit token IDs."""

import math
import re
from functools import lru_cache

import gradio as gr
import pandas as pd
import torch
from diffusers import DiffusionPipeline
from PIL import Image
from transformers import AutoTokenizer

model_id = "stabilityai/sdxl-turbo"


@lru_cache(maxsize=1)
def tokenizer():
    return AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")


@lru_cache(maxsize=1)
def pipeline():
    if not torch.cuda.is_available():
        raise gr.Error("CUDA is required for image generation.")
    return DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")


def parse_ids(text):
    ids = [int(value) for value in re.findall(r"\d+", str(text))]
    if not ids:
        raise gr.Error("Enter at least one token ID.")
    return ids


def search_vocab(query):
    needle = str(query).casefold().strip()
    rows = [
        (token, token_id)
        for token, token_id in tokenizer().get_vocab().items()
        if not needle or needle in token.casefold() or needle in str(token_id)
    ][:500]
    return pd.DataFrame(rows, columns=["Token", "ID"]), f"{len(rows)} shown"


def preview(ids):
    values = parse_ids(ids)
    tokens = tokenizer().convert_ids_to_tokens(values)
    text = " ".join(tokens)
    entities = []
    cursor = 0
    for token, token_id in zip(tokens, values, strict=True):
        entities.append({"entity": str(token_id), "start": cursor, "end": cursor + len(token)})
        cursor += len(token) + 1
    return tokenizer().decode(values), {"text": text, "entities": entities}


def encode(prompt):
    ids = tokenizer()(prompt, add_special_tokens=False).input_ids
    text = " ".join(map(str, ids))
    return text, preview(text)[1]


def padded_ids(values, tok):
    max_length = tok.model_max_length
    if len(values) > max_length - 2:
        raise gr.Error(f"Use at most {max_length - 2} content tokens.")
    bos, eos = tok.bos_token_id, tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    ids = torch.tensor([[bos, *values, eos]], dtype=torch.long, device="cuda")
    return torch.nn.functional.pad(ids, (0, max_length - ids.shape[1]), value=pad)


def make_grid(images, columns):
    width, height = images[0].size
    rows = math.ceil(len(images) / columns)
    grid = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(images):
        grid.paste(image, ((index % columns) * width, (index // columns) * height))
    return grid


def generate(ids, seed, width, height, rows, columns):
    pipe = pipeline()
    values = parse_ids(ids)
    vocab_size = min(pipe.tokenizer.vocab_size, pipe.tokenizer_2.vocab_size)
    if any(value < 0 or value >= vocab_size for value in values):
        raise gr.Error(f"Token IDs must be between 0 and {vocab_size - 1}.")

    with torch.inference_mode():
        out_1 = pipe.text_encoder(padded_ids(values, pipe.tokenizer), output_hidden_states=True)
        out_2 = pipe.text_encoder_2(padded_ids(values, pipe.tokenizer_2), output_hidden_states=True)
        count = int(rows) * int(columns)
        images = pipe(
            prompt_embeds=torch.cat([out_1.hidden_states[-2], out_2.hidden_states[-2]], dim=-1),
            pooled_prompt_embeds=out_2[0],
            num_inference_steps=1,
            guidance_scale=0.0,
            width=int(width),
            height=int(height),
            num_images_per_prompt=count,
            generator=[torch.Generator("cuda").manual_seed(int(seed) + i) for i in range(count)],
        ).images
    return make_grid(images, int(columns))


initial, _ = search_vocab("")

with gr.Blocks(title="Token toolkit") as demo:
    gr.Markdown("# Token toolkit")

    with gr.Accordion("Vocabulary", open=False):
        query = gr.Textbox(label="Search token or ID")
        stats = gr.Markdown("500 shown")
        table = gr.Dataframe(value=initial, interactive=False)
        query.change(search_vocab, query, [table, stats], trigger_mode="always_last")

    prompt = gr.Textbox(value="a cinematic photo of a cat astronaut", label="Prompt")
    prompt_ids = gr.Textbox(label="Prompt token IDs")
    prompt_preview = gr.HighlightedText(label="Prompt tokens")
    prompt.change(encode, prompt, [prompt_ids, prompt_preview], trigger_mode="always_last")

    ids = gr.Textbox(value="320 1125 5390", label="Manual token IDs")
    decoded = gr.Textbox(label="Decoded text")
    token_preview = gr.HighlightedText(label="Token preview")
    ids.change(preview, ids, [decoded, token_preview], trigger_mode="always_last")

    with gr.Row():
        seed = gr.Number(value=42, precision=0, label="Seed")
        width = gr.Slider(256, 1024, value=512, step=64, label="Width")
        height = gr.Slider(256, 1024, value=512, step=64, label="Height")
        rows = gr.Slider(1, 4, value=1, step=1, label="Rows")
        columns = gr.Slider(1, 4, value=4, step=1, label="Columns")

    button = gr.Button("Generate", variant="primary")
    output = gr.Image(type="pil", label="Generated grid")
    button.click(generate, [ids, seed, width, height, rows, columns], output)

demo.launch(theme=gr.themes.Soft())
