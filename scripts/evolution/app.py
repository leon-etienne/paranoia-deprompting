"""Evolve CLIP token IDs and render them with SDXL Turbo."""

import gc, random, time, ast, re, threading

import torch
import gradio as gr
from transformers import CLIPModel, CLIPProcessor
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, make_image_grid

if not torch.cuda.is_available():
    raise RuntimeError("CUDA/GPU required.")

device = "cuda"
use_fp16 = True
tokenizer_lock = threading.Lock()

clip_model_id = "openai/clip-vit-large-patch14"

clip_model = CLIPModel.from_pretrained(
    clip_model_id,
    torch_dtype=torch.float16,
).eval()

clip_proc = CLIPProcessor.from_pretrained(clip_model_id)
clip_tok = clip_proc.tokenizer
bos, eos = clip_tok.bos_token_id, clip_tok.eos_token_id

sdxl_pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

sdxl_pipe.set_progress_bar_config(disable=True)

sdxl_vocab = sdxl_pipe.tokenizer.get_vocab()
sdxl_inverse_dict = {int(v): k for k, v in sdxl_vocab.items()}
sdxl_vocab_size = sdxl_pipe.tokenizer.vocab_size


def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def clip_to_gpu():
    sdxl_to_cpu()
    clip_model.to("cuda")


def clip_to_cpu():
    clip_model.to("cpu")
    clear_cuda()


def sdxl_to_gpu():
    clip_to_cpu()
    sdxl_pipe.to("cuda")


def sdxl_to_cpu():
    sdxl_pipe.to("cpu")
    clear_cuda()


# Start with both off GPU until needed
clip_to_cpu()
sdxl_to_cpu()


@torch.no_grad()
def image_feature(pil):
    x = clip_proc(images=pil, return_tensors="pt").to("cuda")

    with torch.amp.autocast("cuda", dtype=torch.float16):
        vision_out = clip_model.vision_model(pixel_values=x["pixel_values"])
        cls = vision_out.last_hidden_state[:, 0, :]
        f = clip_model.visual_projection(cls)

    f = f.float()
    return (f / f.norm(dim=-1, keepdim=True)).squeeze(0)


def build_pool_full_vocab():
    vocab = clip_tok.vocab_size
    pool = torch.arange(vocab, device="cuda")

    banned = {
        t for t in (clip_tok.bos_token_id, clip_tok.eos_token_id, clip_tok.pad_token_id)
        if t is not None
    }

    mask = torch.ones(vocab, dtype=torch.bool, device="cuda")
    for t in banned:
        if 0 <= t < vocab:
            mask[int(t)] = False

    return pool[mask]


@torch.no_grad()
def score(pop_2d, img_f32, score_bs, img_f16=None):
    out = []

    for s in range(0, pop_2d.size(0), score_bs):
        seq = pop_2d[s:s + score_bs]
        B = seq.size(0)

        inp = torch.cat(
            [
                torch.full((B, 1), bos, device="cuda", dtype=torch.long),
                seq,
                torch.full((B, 1), eos, device="cuda", dtype=torch.long),
            ],
            dim=1,
        )

        attn = torch.ones_like(inp)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            hs = clip_model.text_model(
                input_ids=inp,
                attention_mask=attn,
            ).last_hidden_state[:, -1, :]
            txt = clip_model.text_projection(hs)

        txt = txt / txt.norm(dim=-1, keepdim=True)
        sims = txt @ img_f16
        out.append(sims.detach().float().cpu())

    return torch.cat(out, 0)


def run_ga(
    image_path,
    seed,
    random_seed,
    n_tokens,
    pop,
    gen,
    score_bs,
    elite_frac,
    tournament_k,
    crossover_p,
    swap_mut_p,
    rand_mut_p,
    progress=gr.Progress(track_tqdm=False),
):
    if image_path is None:
        raise gr.Error("Upload an image.")

    clip_to_gpu()

    image = load_image(image_path)

    seed = int(seed)
    n_tokens = int(n_tokens)
    pop = int(pop)
    gen = int(gen)
    score_bs = int(score_bs)
    tournament_k = int(tournament_k)

    if random_seed:
        seed = int(time.time() * 1e6) % (2**31 - 1)

    random.seed(seed)
    torch.manual_seed(seed)

    img_f32 = image_feature(image).float()
    img_f16 = img_f32.half()

    pool_ids = build_pool_full_vocab()
    K = int(pool_ids.numel())

    population = pool_ids[torch.randint(0, K, (pop, n_tokens), device="cuda")]
    elite_n = max(1, int(round(elite_frac * pop)))
    scores = score(population, img_f32, score_bs, img_f16)

    best_i = int(torch.argmax(scores))
    best_seq = population[best_i].detach().clone()
    best_sc = float(scores[best_i])
    eps = 1e-9

    def tsel(scores_cpu):
        idxs = random.sample(range(pop), k=min(tournament_k, pop))
        return max(idxs, key=lambda i: float(scores_cpu[i]))

    def crossover(a, b):
        if n_tokens < 2:
            return a.clone(), b.clone()
        cut = random.randint(1, n_tokens - 1)
        return torch.cat([a[:cut], b[cut:]]), torch.cat([b[:cut], a[cut:]])

    def swapmut(x):
        if n_tokens < 2:
            return x
        i, j = random.sample(range(n_tokens), 2)
        x[i], x[j] = x[j].clone(), x[i].clone()
        return x

    def randmut(x):
        m = torch.rand((n_tokens,), device="cuda") < rand_mut_p
        if m.any():
            x[m] = pool_ids[
                torch.randint(0, K, (int(m.sum().item()),), device="cuda")
            ]
        return x

    for g in range(1, gen + 1):
        progress(g / max(1, gen), desc=f"GA {g}/{gen}")

        order = torch.argsort(scores, descending=True)
        elites = population[order[:elite_n].to("cuda")].detach().clone()

        new = [elites]

        while sum(x.shape[0] for x in new) < pop:
            p1, p2 = population[tsel(scores)], population[tsel(scores)]

            if random.random() < crossover_p:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.clone(), p2.clone()

            if random.random() < swap_mut_p:
                c1 = swapmut(c1)

            if random.random() < swap_mut_p:
                c2 = swapmut(c2)

            c1 = randmut(c1)
            c2 = randmut(c2)

            new.append(c1.view(1, -1))

            if sum(x.shape[0] for x in new) < pop:
                new.append(c2.view(1, -1))

        population = torch.cat(new, 0)[:pop].to("cuda")
        scores = score(population, img_f32, score_bs, img_f16)

        gi = int(torch.argmax(scores))
        gbest = float(scores[gi])

        if gbest > best_sc + eps:
            best_sc = gbest
            best_seq = population[gi].detach().clone()

    token_ids = [int(x) for x in best_seq.tolist()]
    decoded = clip_tok.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    clip_to_cpu()

    return decoded, f"{best_sc:.6f}", str(token_ids)


def text_to_ids(text):
    ids = [int(num) for num in re.findall(r"\b\d+\b", str(text))]
    max_vocab = min(
        sdxl_pipe.tokenizer.vocab_size,
        sdxl_pipe.tokenizer_2.vocab_size,
    )
    return [i for i in ids if 0 <= i < max_vocab]


def tokens_to_colors(ids):
    decoded_tokens = [sdxl_inverse_dict.get(int(i), f"<missing:{i}>") for i in ids]
    text = " ".join(decoded_tokens)

    entities = []
    current_index = 0

    for token, token_id in zip(decoded_tokens, ids):
        token = str(token)
        entities.append(
            {
                "entity": str(token_id),
                "start": current_index,
                "end": current_index + len(token),
            }
        )
        current_index += len(token) + 1

    return {"text": text, "entities": entities}


def preview_tokens(input_text):
    ids = text_to_ids(input_text)
    return tokens_to_colors(ids)


def _add_bos_eos_and_pad(token_ids, tokenizer):
    B = token_ids.shape[0]
    max_len = tokenizer.model_max_length

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    ids = torch.cat(
        [
            torch.full((B, 1), bos_id, dtype=token_ids.dtype, device=token_ids.device),
            token_ids,
            torch.full((B, 1), eos_id, dtype=token_ids.dtype, device=token_ids.device),
        ],
        dim=1,
    )

    if ids.shape[1] > max_len:
        ids = ids[:, :max_len]
        ids[:, -1] = eos_id

    if ids.shape[1] < max_len:
        pad_len = max_len - ids.shape[1]
        ids = torch.cat(
            [
                ids,
                torch.full((B, pad_len), pad_id, dtype=ids.dtype, device=ids.device),
            ],
            dim=1,
        )

    return ids


@torch.no_grad()
def sdxl_prompt_embeds_from_token_ids(token_ids_1, token_ids_2=None):
    if token_ids_2 is None:
        token_ids_2 = token_ids_1

    ids1 = _add_bos_eos_and_pad(token_ids_1, sdxl_pipe.tokenizer).to("cuda")
    ids2 = _add_bos_eos_and_pad(token_ids_2, sdxl_pipe.tokenizer_2).to("cuda")

    out1 = sdxl_pipe.text_encoder(ids1, output_hidden_states=True)
    hs1 = out1.hidden_states[-2]

    out2 = sdxl_pipe.text_encoder_2(ids2, output_hidden_states=True)
    pooled2 = out2[0]
    hs2 = out2.hidden_states[-2]

    prompt_embeds = torch.cat([hs1, hs2], dim=-1)
    dtype = sdxl_pipe.text_encoder_2.dtype

    return (
        prompt_embeds.to(dtype=dtype, device="cuda"),
        pooled2.to(dtype=dtype, device="cuda"),
    )


def parse_token_ids(token_ids_text):
    ids = ast.literal_eval(token_ids_text.strip())

    if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
        raise gr.Error("Token IDs must be a list of integers.")

    max_vocab = min(
        sdxl_pipe.tokenizer.vocab_size,
        sdxl_pipe.tokenizer_2.vocab_size,
    )

    bad = [x for x in ids if x < 0 or x >= max_vocab]
    if bad:
        raise gr.Error(
            f"Some token IDs are outside the SDXL tokenizer range. "
            f"Max allowed ID is {max_vocab - 1}. Bad IDs: {bad[:20]}"
        )

    return ids


def generate_from_tokens(
    token_ids_text,
    sdxl_seed,
    sdxl_steps,
    sdxl_guidance,
    n_images,
    progress=gr.Progress(track_tqdm=False),
):
    if not token_ids_text.strip():
        raise gr.Error("Run GA first.")

    sdxl_to_gpu()

    ids = parse_token_ids(token_ids_text)
    token_ids = torch.tensor([ids], dtype=torch.long, device="cuda")

    prompt_embeds, pooled = sdxl_prompt_embeds_from_token_ids(token_ids)

    sdxl_seed = int(sdxl_seed)
    sdxl_steps = int(sdxl_steps)
    n_images = int(n_images)

    images = []

    for i in range(n_images):
        progress((i + 1) / max(1, n_images), desc=f"Image {i + 1}/{n_images}")

        generator = torch.Generator(device="cuda").manual_seed(sdxl_seed + i)

        img = sdxl_pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled,
            num_inference_steps=sdxl_steps,
            guidance_scale=float(sdxl_guidance),
            generator=generator,
            num_images_per_prompt=1,
        ).images[0]

        images.append(img)

    clear_cuda()

    if n_images == 1:
        return images[0]
    if n_images <= 4:
        return make_image_grid(images, 2, 2)
    return make_image_grid(images, 3, 3)


custom_css = """
footer { display: none !important; }

.gradio-container {
    max-width: 1320px !important;
    width: 98% !important;
    margin: auto !important;
}

#run-btn, #gen-btn {
    height: 44px;
    font-weight: 700;
}

textarea {
    min-height: 72px !important;
}

.mono textarea {
    font-family: monospace !important;
    font-size: 12px !important;
}
"""


with gr.Blocks(
    title="CLIP Token GA → SDXL Turbo",
    theme=gr.themes.Soft(),
    css=custom_css,
) as demo:

    gr.Markdown("# CLIP Token GA → SDXL Turbo")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=340):
            image_input = gr.Image(
                type="filepath",
                label="Target image",
                height=360,
            )

        with gr.Column(scale=2, min_width=620):
            with gr.Row():
                seed = gr.Number(label="Seed", value=0, precision=0)
                random_seed = gr.Checkbox(label="Random", value=True)
                n_tokens = gr.Slider(1, 77, value=16, step=1, label="Tokens")

            with gr.Row():
                pop = gr.Slider(16, 4096, value=1024, step=16, label="Population")
                gen = gr.Slider(1, 500, value=100, step=1, label="Generations")
                score_bs = gr.Slider(128, 32768, value=4096, step=128, label="Batch")

            with gr.Accordion("Advanced GA", open=False):
                with gr.Row():
                    elite_frac = gr.Slider(0.01, 0.5, value=0.08, step=0.01, label="Elite")
                    tournament_k = gr.Slider(2, 16, value=4, step=1, label="Tournament")
                with gr.Row():
                    crossover_p = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Crossover")
                    swap_mut_p = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Swap mut.")
                    rand_mut_p = gr.Slider(0.0, 1.0, value=0.08, step=0.01, label="Rand mut.")

            run_btn = gr.Button("Run GA", variant="primary", elem_id="run-btn")

            with gr.Row():
                best_prompt = gr.Textbox(label="CLIP decoded", lines=2)
                best_score = gr.Textbox(label="Score", max_lines=1)

            with gr.Row(equal_height=True):
                token_ids = gr.Textbox(label="Token IDs", lines=2, elem_classes=["mono"])
                token_preview = gr.HighlightedText(
                    label="Token preview",
                    combine_adjacent=True,
                )

    gr.Markdown("## Generate")

    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            with gr.Row():
                sdxl_seed = gr.Number(label="Seed", value=0, precision=0)
                sdxl_steps = gr.Slider(1, 8, value=2, step=1, label="Steps")
                sdxl_guidance = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="Guidance")
                n_images = gr.Slider(1, 9, value=4, step=1, label="Images")

            gen_btn = gr.Button("Generate from tokens", variant="primary", elem_id="gen-btn")

        with gr.Column(scale=1):
            sdxl_output = gr.Image(label="Output", type="pil", height=420)

    run_btn.click(
        fn=run_ga,
        inputs=[
            image_input,
            seed,
            random_seed,
            n_tokens,
            pop,
            gen,
            score_bs,
            elite_frac,
            tournament_k,
            crossover_p,
            swap_mut_p,
            rand_mut_p,
        ],
        outputs=[best_prompt, best_score, token_ids],
    )

    token_ids.change(
        fn=preview_tokens,
        inputs=token_ids,
        outputs=token_preview,
        trigger_mode="always_last",
    )

    gen_btn.click(
        fn=generate_from_tokens,
        inputs=[
            token_ids,
            sdxl_seed,
            sdxl_steps,
            sdxl_guidance,
            n_images,
        ],
        outputs=[sdxl_output],
    )


demo.queue(default_concurrency_limit=1, max_size=8)

try:
    demo.upload_file_set.is_tracked = lambda upload_id: False
except Exception:
    pass

app, local_url, share_url = demo.launch(
    inline=False,
    inbrowser=False,
    prevent_thread_lock=True,
    share=True,
    debug=True,
)
