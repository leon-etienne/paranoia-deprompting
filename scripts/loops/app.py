"""Launch the live CLIP random-image search Gradio app."""

from functools import lru_cache

import gradio as gr
import matplotlib.pyplot as plt

from clip_token_lab.clip import ClipEmbedder
from clip_token_lab.image_sources import SOURCES
from clip_token_lab.loops import iter_image_to_random_images, iter_text_to_random_images
from clip_token_lab.plots import loop_figure


@lru_cache(maxsize=1)
def embedder() -> ClipEmbedder:
    return ClipEmbedder()


def text_loop(prompt, source, delay):
    if not prompt.strip():
        raise gr.Error("Enter a text target.")
    for state in iter_text_to_random_images(prompt, embedder(), source=source, delay=float(delay), iterations=None):
        figure = loop_figure(state, title=f'Text target: "{prompt}"')
        status = (
            f"Iteration {state.iteration} | current {state.current_score:.4f} | "
            f"best {state.best_score:.4f} | worst {state.worst_score:.4f}"
        )
        yield figure, status
        plt.close(figure)


def image_loop(target, source, delay):
    if target is None:
        raise gr.Error("Upload a target image.")
    for state in iter_image_to_random_images(target, embedder(), source=source, delay=float(delay), iterations=None):
        figure = loop_figure(state, target_image=target, title="Image target")
        status = (
            f"Iteration {state.iteration} | current {state.current_score:.4f} | "
            f"best {state.best_score:.4f} | worst {state.worst_score:.4f}"
        )
        yield figure, status
        plt.close(figure)


source_choices = list(SOURCES)

with gr.Blocks(title="Live CLIP random image search") as demo:
    gr.Markdown("# Live CLIP random image search")

    with gr.Tab("Text to Random Images"):
        prompt = gr.Textbox(value="a smiling person", label="Text target")
        source = gr.Dropdown(source_choices, value="picsum", label="Image source")
        delay = gr.Slider(0, 5, value=0.25, step=0.05, label="Delay (seconds)")
        with gr.Row():
            start = gr.Button("Start", variant="primary")
            stop = gr.Button("Stop", variant="stop")
        plot = gr.Plot()
        status = gr.Textbox(label="Status")
        event = start.click(text_loop, [prompt, source, delay], [plot, status])
        stop.click(fn=None, cancels=[event])

    with gr.Tab("Image to Random Images"):
        target = gr.Image(type="pil", label="Target image")
        source = gr.Dropdown(source_choices, value="picsum", label="Image source")
        delay = gr.Slider(0, 5, value=0.25, step=0.05, label="Delay (seconds)")
        with gr.Row():
            start = gr.Button("Start", variant="primary")
            stop = gr.Button("Stop", variant="stop")
        plot = gr.Plot()
        status = gr.Textbox(label="Status")
        event = start.click(image_loop, [target, source, delay], [plot, status])
        stop.click(fn=None, cancels=[event])

    demo.queue()

demo.launch(theme=gr.themes.Soft())
