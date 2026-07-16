"""Run live CLIP searches over random images in Gradio."""

import time
from io import BytesIO

import torch
import gradio as gr
import requests
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

SOURCES = {
    "thispersondoesnotexist.com": "https://thispersondoesnotexist.com/",
    "https://picsum.photos/512": "https://picsum.photos/512",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}


def to_device(inputs):
    return {k: v.to(device) for k, v in inputs.items()}


def fetch_image(source_name):
    url = SOURCES[source_name]
    cache_buster = int(time.time() * 1000000)

    if "picsum" in url:
        url = f"{url}?random={cache_buster}"
    else:
        url = f"{url}?t={cache_buster}"

    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()

    return Image.open(BytesIO(r.content)).convert("RGB")


@torch.no_grad()
def encode_text(text):
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = to_device(inputs)

    text_outputs = model.text_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    features = model.text_projection(text_outputs.pooler_output)
    return features / features.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_image(image):
    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )
    inputs = to_device(inputs)

    image_outputs = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    features = model.visual_projection(image_outputs.pooler_output)
    return features / features.norm(dim=-1, keepdim=True)


def make_text_fig(prompt, current_img, current, best_img, best, worst_img, worst, step):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(current_img)
    ax[0].set_title(f"Current\n{current:.3f}")
    ax[0].axis("off")

    ax[1].imshow(best_img)
    ax[1].set_title(f"Max / best\n{best:.3f}")
    ax[1].axis("off")

    ax[2].imshow(worst_img)
    ax[2].set_title(f"Min / worst\n{worst:.3f}")
    ax[2].axis("off")

    fig.suptitle(f'Text target: "{prompt}"\nIteration {step}', fontsize=13)
    fig.tight_layout()
    return fig


def make_image_fig(target_img, current_img, current, best_img, best, worst_img, worst, step):
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))

    ax[0].imshow(target_img)
    ax[0].set_title("Target")
    ax[0].axis("off")

    ax[1].imshow(current_img)
    ax[1].set_title(f"Current\n{current:.3f}")
    ax[1].axis("off")

    ax[2].imshow(best_img)
    ax[2].set_title(f"Max / best\n{best:.3f}")
    ax[2].axis("off")

    ax[3].imshow(worst_img)
    ax[3].set_title(f"Min / worst\n{worst:.3f}")
    ax[3].axis("off")

    fig.suptitle(f"Image target similarity\nIteration {step}", fontsize=13)
    fig.tight_layout()
    return fig


def run_text_loop(prompt, source_name, delay):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter text.")

    target = encode_text(prompt.strip())

    best = -1.0
    worst = 999.0
    best_img = None
    worst_img = None
    step = 0

    while True:
        step += 1

        try:
            img = fetch_image(source_name)
            img_emb = encode_image(img)
            score = float((img_emb @ target.T).item())

            if score > best:
                best = score
                best_img = img.copy()
                best_img.save("best_text_match.jpg")

            if score < worst:
                worst = score
                worst_img = img.copy()
                worst_img.save("worst_text_match.jpg")

            fig = make_text_fig(
                prompt.strip(),
                img,
                score,
                best_img,
                best,
                worst_img,
                worst,
                step
            )

            status = (
                f"Iteration: {step} | "
                f"Current: {score:.4f} | "
                f"Max: {best:.4f} | "
                f"Min: {worst:.4f}"
            )

            yield fig, status
            plt.close(fig)
            time.sleep(delay)

        except Exception as e:
            yield None, f"Error on iteration {step}: {e}"
            time.sleep(delay)


def run_image_loop(target_image, source_name, delay):
    if target_image is None:
        raise gr.Error("Please upload a target image.")

    target_image = target_image.convert("RGB")
    target = encode_image(target_image)

    best = -1.0
    worst = 999.0
    best_img = None
    worst_img = None
    step = 0

    while True:
        step += 1

        try:
            img = fetch_image(source_name)
            img_emb = encode_image(img)
            score = float((img_emb @ target.T).item())

            if score > best:
                best = score
                best_img = img.copy()
                best_img.save("best_image_match.jpg")

            if score < worst:
                worst = score
                worst_img = img.copy()
                worst_img.save("worst_image_match.jpg")

            fig = make_image_fig(
                target_image,
                img,
                score,
                best_img,
                best,
                worst_img,
                worst,
                step
            )

            status = (
                f"Iteration: {step} | "
                f"Current: {score:.4f} | "
                f"Max: {best:.4f} | "
                f"Min: {worst:.4f}"
            )

            yield fig, status
            plt.close(fig)
            time.sleep(delay)

        except Exception as e:
            yield None, f"Error on iteration {step}: {e}"
            time.sleep(delay)


custom_css = """
footer {
    display: none !important;
}

.gradio-container {
    max-width: 1600px !important;
    width: 98% !important;
    margin: auto !important;
}

#start-btn {
    height: 48px;
    font-weight: 700;
}

#stop-btn {
    height: 48px;
    font-weight: 700;
}
"""


with gr.Blocks(
    title="Live CLIP Random Image Search",
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    gr.Markdown(
        """
        # Live CLIP Random Image Search

        Fetch random images in a loop, compare them with CLIP, and track the current, minimum, and maximum match.
        """
    )

    with gr.Tab("Text → Random Images"):
        text_prompt = gr.Textbox(
            label="Text target",
            value="a smiling person",
            lines=3
        )

        text_source = gr.Dropdown(
            choices=list(SOURCES.keys()),
            value="https://picsum.photos/512",
            label="Image source"
        )

        text_delay = gr.Slider(
            minimum=0.0,
            maximum=5.0,
            value=0.25,
            step=0.05,
            label="Delay between fetches, seconds"
        )

        with gr.Row():
            text_start = gr.Button("Start", variant="primary", elem_id="start-btn")
            text_stop = gr.Button("Stop", variant="stop", elem_id="stop-btn")

        text_plot = gr.Plot(label="Live comparison")
        text_status = gr.Textbox(label="Status")

        text_event = text_start.click(
            fn=run_text_loop,
            inputs=[text_prompt, text_source, text_delay],
            outputs=[text_plot, text_status]
        )

        text_stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[text_event]
        )

    with gr.Tab("Image → Random Images"):
        target_image = gr.Image(
            type="pil",
            label="Target image",
            height=360
        )

        image_source = gr.Dropdown(
            choices=list(SOURCES.keys()),
            value="https://picsum.photos/512",
            label="Image source"
        )

        image_delay = gr.Slider(
            minimum=0.0,
            maximum=5.0,
            value=0.25,
            step=0.05,
            label="Delay between fetches, seconds"
        )

        with gr.Row():
            image_start = gr.Button("Start", variant="primary", elem_id="start-btn")
            image_stop = gr.Button("Stop", variant="stop", elem_id="stop-btn")

        image_plot = gr.Plot(label="Live comparison")
        image_status = gr.Textbox(label="Status")

        image_event = image_start.click(
            fn=run_image_loop,
            inputs=[target_image, image_source, image_delay],
            outputs=[image_plot, image_status]
        )

        image_stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[image_event]
        )


demo.queue()

demo.launch(
    inline=False,
    inbrowser=False,
    share=True,
    debug=True
)
