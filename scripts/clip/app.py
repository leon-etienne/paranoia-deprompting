"""Compare image and text embeddings in a small Gradio UI."""
import textwrap
from io import StringIO

import torch
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display, IFrame, Javascript
from transformers import CLIPProcessor, CLIPModel

display(Javascript("""
try {
  google.colab.output.setIframeHeight(0, true, {maxHeight: 7000});
} catch (e) {}
"""))

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

def to_device(inputs):
    return {k: v.to(device) for k, v in inputs.items()}

def wrap_label(text, width=34, max_lines=3):
    lines = textwrap.wrap(str(text), width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ") + "…"
    return "\n".join(lines)

def split_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]

def get_text_embeddings(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = to_device(inputs)

    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        features = model.text_projection(text_outputs.pooler_output)
        features = features / features.norm(dim=-1, keepdim=True)

    return features

def get_image_embeddings(images):
    inputs = processor(images=images, return_tensors="pt")
    inputs = to_device(inputs)

    with torch.no_grad():
        image_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        features = model.visual_projection(image_outputs.pooler_output)
        features = features / features.norm(dim=-1, keepdim=True)

    return features

def make_bar(labels, scores, title, xlabel="Similarity"):
    labels = list(labels)
    scores = np.array(scores)

    order = scores.argsort()[::-1]
    labels_sorted = [labels[i] for i in order]
    scores_sorted = scores[order]
    wrapped = [wrap_label(x) for x in labels_sorted]

    fig_height = max(6, min(20, 1.5 + len(labels_sorted) * 0.75))
    fig, ax = plt.subplots(figsize=(15, fig_height))

    ax.barh(wrapped, scores_sorted)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, 1.0)

    for i, s in enumerate(scores_sorted):
        label = f"{s:.3f}" if xlabel == "Cosine similarity" else f"{s:.1%}"
        ax.text(min(float(s) + 0.015, 0.96), i, label, va="center", fontsize=10)

    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

def make_image_bar(reference_image, images, labels, scores, title, xlabel="Similarity", show_reference=False):
    scores = np.array(scores)
    order = scores.argsort()[::-1]

    images_sorted = [images[i] for i in order]
    labels_sorted = [labels[i] for i in order]
    scores_sorted = scores[order]

    n = len(images_sorted)
    fig_height = max(6, min(16, 1.4 + n * 1.2))

    if show_reference:
        fig = plt.figure(figsize=(17, fig_height))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 3.4], wspace=0.28)
        ax_ref = fig.add_subplot(gs[0])
        ax_imgs = fig.add_subplot(gs[1])
        ax_bar = fig.add_subplot(gs[2])

        ax_ref.imshow(reference_image)
        ax_ref.axis("off")
        ax_ref.set_title("Reference image", fontsize=13, pad=10)
    else:
        fig = plt.figure(figsize=(15, fig_height))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 3.4], wspace=0.28)
        ax_imgs = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

    thumb_size = 140
    canvas = np.ones((n * thumb_size, thumb_size, 3), dtype=np.uint8) * 255

    for i, img in enumerate(images_sorted):
        img = img.convert("RGB")
        img.thumbnail((thumb_size, thumb_size))

        thumb = np.ones((thumb_size, thumb_size, 3), dtype=np.uint8) * 255
        arr = np.array(img)

        y_offset = (thumb_size - arr.shape[0]) // 2
        x_offset = (thumb_size - arr.shape[1]) // 2
        thumb[y_offset:y_offset + arr.shape[0], x_offset:x_offset + arr.shape[1]] = arr

        row_start = i * thumb_size
        canvas[row_start:row_start + thumb_size] = thumb

    ax_imgs.imshow(canvas)
    ax_imgs.axis("off")
    ax_imgs.set_title("Compared images", fontsize=13, pad=10)

    y_pos = np.arange(n)

    ax_bar.barh(y_pos, scores_sorted)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels_sorted)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel(xlabel)
    ax_bar.set_title(title)

    for i, s in enumerate(scores_sorted):
        label = f"{s:.3f}" if xlabel == "Cosine similarity" else f"{s:.1%}"
        ax_bar.text(min(float(s) + 0.015, 0.96), i, label, va="center", fontsize=10)

    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig

def image_to_text(image, texts):
    if image is None:
        raise gr.Error("Please upload an image.")

    labels = split_lines(texts)
    if not labels:
        raise gr.Error("Please enter at least one text label.")

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True, truncation=True)
    inputs = to_device(inputs)

    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0].cpu().numpy()

    order = probs.argsort()[::-1]
    labels_sorted = [labels[i] for i in order]
    probs_sorted = probs[order]

    fig_height = max(7, min(20, 2 + len(labels_sorted) * 0.75))
    fig = plt.figure(figsize=(17, fig_height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 2.5], wspace=0.3)

    ax_img = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title("Input image", fontsize=14, pad=12)

    wrapped = [wrap_label(x) for x in labels_sorted]

    ax_bar.barh(wrapped, probs_sorted)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel("CLIP match probability")
    ax_bar.set_title(f"Best match: {wrap_label(labels_sorted[0], width=46, max_lines=2)}", pad=10)

    for i, p in enumerate(probs_sorted):
        ax_bar.text(min(float(p) + 0.015, 0.96), i, f"{p:.1%}", va="center", fontsize=10)

    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig

def text_to_images(text_prompt, image1, image2, image3, image4, image5):
    images = [img for img in [image1, image2, image3, image4, image5] if img is not None]

    if not text_prompt or not text_prompt.strip():
        raise gr.Error("Please enter a text prompt.")

    if not images:
        raise gr.Error("Please upload at least one image.")

    labels = [f"Image {i+1}" for i in range(len(images))]

    inputs = processor(
        text=[text_prompt.strip()],
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = to_device(inputs)

    with torch.no_grad():
        probs = model(**inputs).logits_per_text.softmax(dim=1)[0].cpu().numpy()

    return make_image_bar(
        reference_image=None,
        images=images,
        labels=labels,
        scores=probs,
        title=f"Text → Image matches for: {wrap_label(text_prompt, width=60, max_lines=2)}",
        xlabel="CLIP match probability",
        show_reference=False
    )

def text_to_text(reference_text, comparison_texts):
    if not reference_text or not reference_text.strip():
        raise gr.Error("Please enter a reference text.")

    labels = split_lines(comparison_texts)
    if not labels:
        raise gr.Error("Please enter at least one comparison text.")

    all_texts = [reference_text.strip()] + labels
    features = get_text_embeddings(all_texts)

    reference = features[0:1]
    candidates = features[1:]

    scores = (reference @ candidates.T)[0].cpu().numpy()
    scores = np.clip(scores, 0, 1)

    return make_bar(
        labels,
        scores,
        f"Text → Text similarity to: {wrap_label(reference_text, width=60, max_lines=2)}",
        xlabel="Cosine similarity"
    )

def image_to_images(reference_image, image1, image2, image3, image4, image5):
    images = [img for img in [image1, image2, image3, image4, image5] if img is not None]

    if reference_image is None:
        raise gr.Error("Please upload a reference image.")

    if not images:
        raise gr.Error("Please upload at least one comparison image.")

    all_images = [reference_image] + images
    labels = [f"Image {i+1}" for i in range(len(images))]

    features = get_image_embeddings(all_images)

    reference = features[0:1]
    candidates = features[1:]

    scores = (reference @ candidates.T)[0].cpu().numpy()
    scores = np.clip(scores, 0, 1)

    return make_image_bar(
        reference_image=reference_image,
        images=images,
        labels=labels,
        scores=scores,
        title="Image → Image similarity",
        xlabel="Cosine similarity",
        show_reference=True
    )

custom_css = """
footer {
    display: none !important;
}

.gradio-container {
    max-width: 1700px !important;
    width: 98% !important;
    margin: auto !important;
}

.main {
    max-width: 1700px !important;
}

.block {
    padding-top: 14px !important;
    padding-bottom: 14px !important;
}

#submit-btn {
    height: 48px;
    font-weight: 700;
}

#clear-btn {
    height: 48px;
}

textarea {
    min-height: 130px !important;
}
"""

with gr.Blocks(
    title="CLIP Similarity Comparison Tool",
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    gr.Markdown(
        """
        # CLIP Similarity Comparison Tool

        Compare image/text, text/image, text/text, and image/image similarity using CLIP.
        """
    )

    with gr.Tab("Image → Text"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="1. Upload image", height=320)

            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    lines=12,
                    label="2. Candidate texts",
                    info="One possible description per line.",
                    value="""a dog
a cat
a car
a landscape"""
                )

                with gr.Row():
                    clear_btn = gr.ClearButton([image_input, text_input], value="Clear", elem_id="clear-btn")
                    submit_btn = gr.Button("Compare", variant="primary", elem_id="submit-btn")

        output_plot = gr.Plot(label="Ranked CLIP matches")
        submit_btn.click(fn=image_to_text, inputs=[image_input, text_input], outputs=output_plot)

    with gr.Tab("Text → Images"):
        text_prompt = gr.Textbox(lines=3, label="1. Text prompt", value="a dog playing outside")

        with gr.Row():
            image_1 = gr.Image(type="pil", label="Image 1", height=220)
            image_2 = gr.Image(type="pil", label="Image 2", height=220)
            image_3 = gr.Image(type="pil", label="Image 3", height=220)
            image_4 = gr.Image(type="pil", label="Image 4", height=220)
            image_5 = gr.Image(type="pil", label="Image 5", height=220)

        with gr.Row():
            clear_btn = gr.ClearButton(
                [text_prompt, image_1, image_2, image_3, image_4, image_5],
                value="Clear",
                elem_id="clear-btn"
            )
            submit_btn = gr.Button("Compare", variant="primary", elem_id="submit-btn")

        output_plot = gr.Plot(label="Ranked image matches")

        submit_btn.click(
            fn=text_to_images,
            inputs=[text_prompt, image_1, image_2, image_3, image_4, image_5],
            outputs=output_plot
        )

    with gr.Tab("Text → Text"):
        reference_text = gr.Textbox(
            lines=3,
            label="1. Reference text",
            value="a happy dog running through grass"
        )

        comparison_texts = gr.Textbox(
            lines=12,
            label="2. Comparison texts",
            info="One comparison text per line.",
            value="""a dog playing outside
a cat sleeping on a couch
a car driving on a road
a person eating dinner"""
        )

        with gr.Row():
            clear_btn = gr.ClearButton([reference_text, comparison_texts], value="Clear", elem_id="clear-btn")
            submit_btn = gr.Button("Compare", variant="primary", elem_id="submit-btn")

        output_plot = gr.Plot(label="Text similarity scores")
        submit_btn.click(fn=text_to_text, inputs=[reference_text, comparison_texts], outputs=output_plot)

    with gr.Tab("Image → Image"):
        reference_image = gr.Image(type="pil", label="1. Reference image", height=320)

        with gr.Row():
            comp_image_1 = gr.Image(type="pil", label="Comparison Image 1", height=220)
            comp_image_2 = gr.Image(type="pil", label="Comparison Image 2", height=220)
            comp_image_3 = gr.Image(type="pil", label="Comparison Image 3", height=220)
            comp_image_4 = gr.Image(type="pil", label="Comparison Image 4", height=220)
            comp_image_5 = gr.Image(type="pil", label="Comparison Image 5", height=220)

        with gr.Row():
            clear_btn = gr.ClearButton(
                [reference_image, comp_image_1, comp_image_2, comp_image_3, comp_image_4, comp_image_5],
                value="Clear",
                elem_id="clear-btn"
            )
            submit_btn = gr.Button("Compare", variant="primary", elem_id="submit-btn")

        output_plot = gr.Plot(label="Image similarity scores")

        submit_btn.click(
            fn=image_to_images,
            inputs=[reference_image, comp_image_1, comp_image_2, comp_image_3, comp_image_4, comp_image_5],
            outputs=output_plot
        )

app, local_url, share_url = demo.launch(
    inline=False,
    inbrowser=False,
    prevent_thread_lock=True,
    share=True,
    debug=True
)

display(IFrame(
    src=share_url,
    width="100%",
    height=2200
))
