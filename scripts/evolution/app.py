"""Launch the CLIP-token evolution Gradio app."""

from functools import lru_cache

import gradio as gr

from clip_token_lab.evolution import CLIPTokenOptimizer, EvolutionConfig
from clip_token_lab.images import make_grid
from clip_token_lab.sdxl_tokens import SDXLTokenGenerator
from clip_token_lab.tokens import TokenToolkit


@lru_cache(maxsize=1)
def optimizer() -> CLIPTokenOptimizer:
    return CLIPTokenOptimizer()


@lru_cache(maxsize=1)
def token_toolkit() -> TokenToolkit:
    return TokenToolkit()


@lru_cache(maxsize=1)
def generator() -> SDXLTokenGenerator:
    return SDXLTokenGenerator()


def run_evolution(
    image,
    seed,
    random_seed,
    n_tokens,
    population,
    generations,
    score_batch_size,
    elite_fraction,
    tournament_size,
    crossover_probability,
    swap_mutation_probability,
    random_mutation_probability,
    progress=gr.Progress(),
):
    if image is None:
        raise gr.Error("Upload a target image.")

    config = EvolutionConfig(
        seed=int(seed),
        random_seed=bool(random_seed),
        n_tokens=int(n_tokens),
        population=int(population),
        generations=int(generations),
        score_batch_size=int(score_batch_size),
        elite_fraction=float(elite_fraction),
        tournament_size=int(tournament_size),
        crossover_probability=float(crossover_probability),
        swap_mutation_probability=float(swap_mutation_probability),
        random_mutation_probability=float(random_mutation_probability),
    )

    def report(current, total, best):
        progress(current / max(total, 1), desc=f"Generation {current}/{total}; best {best:.4f}")

    if generator.cache_info().currsize:
        generator().to_cpu()

    opt = optimizer()
    opt.to_cuda()
    try:
        result = opt.optimize(image, config, progress=report)
    finally:
        opt.to_cpu()

    ids_text = str(result.token_ids)
    preview = token_toolkit().highlighted_text(result.token_ids)
    return result.decoded_text, f"{result.score:.6f}", ids_text, preview


def generate(ids, seed, steps, guidance, count):
    if optimizer.cache_info().currsize:
        optimizer().to_cpu()

    gen = generator()
    gen.to_cuda()
    try:
        images = gen.generate(ids, seed=int(seed), steps=int(steps), guidance_scale=float(guidance), count=int(count))
    finally:
        gen.to_cpu()

    return images[0] if len(images) == 1 else make_grid(images)


with gr.Blocks(title="CLIP token evolution to SDXL") as demo:
    gr.Markdown("# CLIP token evolution to SDXL Turbo")

    image = gr.Image(type="pil", label="Target image")
    with gr.Row():
        seed = gr.Number(value=0, precision=0, label="Seed")
        random_seed = gr.Checkbox(value=True, label="Random seed")
        n_tokens = gr.Slider(1, 75, value=16, step=1, label="Tokens")

    with gr.Row():
        population = gr.Slider(16, 4096, value=1024, step=16, label="Population")
        generations = gr.Slider(1, 500, value=100, step=1, label="Generations")
        score_batch_size = gr.Slider(128, 32768, value=4096, step=128, label="Score batch")

    with gr.Accordion("Advanced GA", open=False):
        elite_fraction = gr.Slider(0.01, 0.5, value=0.08, step=0.01, label="Elite fraction")
        tournament_size = gr.Slider(2, 16, value=4, step=1, label="Tournament")
        crossover_probability = gr.Slider(0, 1, value=0.9, step=0.01, label="Crossover")
        swap_mutation_probability = gr.Slider(0, 1, value=0.25, step=0.01, label="Swap mutation")
        random_mutation_probability = gr.Slider(0, 1, value=0.08, step=0.01, label="Random mutation per token")

    run = gr.Button("Run evolution", variant="primary")
    decoded = gr.Textbox(label="CLIP-decoded sequence")
    score = gr.Textbox(label="Best cosine score")
    ids = gr.Textbox(label="Token IDs")
    preview = gr.HighlightedText(label="SDXL tokenizer preview", combine_adjacent=True)
    run.click(
        run_evolution,
        [
            image,
            seed,
            random_seed,
            n_tokens,
            population,
            generations,
            score_batch_size,
            elite_fraction,
            tournament_size,
            crossover_probability,
            swap_mutation_probability,
            random_mutation_probability,
        ],
        [decoded, score, ids, preview],
    )

    gr.Markdown("## Generate from the evolved IDs")
    with gr.Row():
        generation_seed = gr.Number(value=0, precision=0, label="Seed")
        steps = gr.Slider(1, 8, value=2, step=1, label="Steps")
        guidance = gr.Slider(0, 10, value=0, step=0.1, label="Guidance")
        count = gr.Slider(1, 9, value=4, step=1, label="Images")

    generate_button = gr.Button("Generate", variant="primary")
    output = gr.Image(type="pil", label="Output")
    generate_button.click(generate, [ids, generation_seed, steps, guidance, count], output)

    demo.queue(default_concurrency_limit=1, max_size=8)

demo.launch(theme=gr.themes.Soft())
