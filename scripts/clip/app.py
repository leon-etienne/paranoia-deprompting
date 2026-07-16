"""Launch the CLIP similarity Gradio app."""

from functools import lru_cache

import gradio as gr

from clip_token_lab.clip import ClipEmbedder
from clip_token_lab.plots import image_ranking, ranked_bar


@lru_cache(maxsize=1)
def embedder() -> ClipEmbedder:
    return ClipEmbedder()


def lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def image_to_text(image, text):
    if image is None:
        raise gr.Error("Upload an image.")
    candidates = lines(text)
    if not candidates:
        raise gr.Error("Enter one candidate text per line.")
    results = embedder().image_to_text(image, candidates, probabilities=True)
    return ranked_bar(results, "Image to text", probability=True)


def text_to_images(text, *images):
    selected = [image for image in images if image is not None]
    if not text.strip() or not selected:
        raise gr.Error("Enter a prompt and upload at least one image.")
    results = embedder().text_to_images(text, selected, probabilities=True)
    return image_ranking(selected, results, "Text to images", probability=True)


def text_to_text(reference, candidates):
    selected = lines(candidates)
    if not reference.strip() or not selected:
        raise gr.Error("Enter a reference and at least one candidate.")
    results = embedder().text_to_text(reference, selected)
    return ranked_bar(results, "Text to text")


def image_to_images(reference, *images):
    selected = [image for image in images if image is not None]
    if reference is None or not selected:
        raise gr.Error("Upload a reference and at least one candidate image.")
    results = embedder().image_to_images(reference, selected)
    return image_ranking(selected, results, "Image to images")


with gr.Blocks(title="CLIP similarity pathways") as demo:
    gr.Markdown("# CLIP similarity pathways")

    with gr.Tab("Image to Text"):
        image = gr.Image(type="pil", label="Reference image")
        texts = gr.Textbox(lines=8, value="a dog\na cat\na car\na landscape", label="Candidate texts")
        button = gr.Button("Compare", variant="primary")
        output = gr.Plot()
        button.click(image_to_text, [image, texts], output)

    with gr.Tab("Text to Images"):
        prompt = gr.Textbox(value="a dog playing outside", label="Text prompt")
        candidate_images = [gr.Image(type="pil", label=f"Image {index}") for index in range(1, 6)]
        button = gr.Button("Compare", variant="primary")
        output = gr.Plot()
        button.click(text_to_images, [prompt, *candidate_images], output)

    with gr.Tab("Text to Text"):
        reference = gr.Textbox(value="a happy dog running through grass", label="Reference text")
        candidates = gr.Textbox(
            lines=8,
            value="a dog playing outside\na cat sleeping on a couch\na car driving on a road",
            label="Candidate texts",
        )
        button = gr.Button("Compare", variant="primary")
        output = gr.Plot()
        button.click(text_to_text, [reference, candidates], output)

    with gr.Tab("Image to Image"):
        reference_image = gr.Image(type="pil", label="Reference image")
        candidate_images = [gr.Image(type="pil", label=f"Candidate {index}") for index in range(1, 6)]
        button = gr.Button("Compare", variant="primary")
        output = gr.Plot()
        button.click(image_to_images, [reference_image, *candidate_images], output)

demo.launch(theme=gr.themes.Soft())
