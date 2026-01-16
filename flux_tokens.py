# %% [markdown]
# # FLUX (T5 patched out) — pooled CLIP from manual token ids
# This script:
# - loads FLUX.1-dev
# - keeps CLIP (for pooled embeddings)
# - patches out T5 by passing zero `prompt_embeds` (shape: B x 512 x 4096)
# - builds CLIP input ids from your *content token ids* (adds BOS/EOS + pads/truncates)
# - generates using pooled CLIP only

# %%
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from optimum.quanto import freeze, qfloat8, quantize

# %% [markdown]
# ## 1) config

# %%
bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16
device = torch.device("cuda")

# keep this link exactly as-is
fp8_url = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors"

guidance_scale = 5.5
num_inference_steps = 20
max_sequence_length = 512  # FLUX uses up to 512

# optional: VAE memory helpers
enable_vae_slicing = True
enable_vae_tiling = True

# %%
# -------------------------
# 2) inputs (edit here)
# -------------------------
# Your manual CLIP TOKEN IDS (content ids only, NO BOS/EOS)
# Batch size B = 1 example:
ids = [123, 456, 789]   # <-- replace with your ids

# Optional: if you want a *real* negative (still only affects pooled CLIP),
# provide negative content ids. Otherwise we'll use zeros for negative pooled.
ids_neg = None          # e.g. [10, 20, 30]

seed = 42               # set to an int for determinism

# %% [markdown]
# ## 3) load + quantize transformer (fp8 via quanto)

# %%
transformer = FluxTransformer2DModel.from_single_file(fp8_url, torch_dtype=dtype)
quantize(transformer, weights=qfloat8)
freeze(transformer)
transformer.to(device)

# %% [markdown]
# ## 4) load pipeline without T5, attach transformer

# %%
pipe = FluxPipeline.from_pretrained(
    bfl_repo,
    transformer=None,
    text_encoder_2=None,  # do NOT load T5
    torch_dtype=dtype,
)

pipe.transformer = transformer

pipe.to(device)
pipe.text_encoder.to(device)  # keep CLIP (needed for pooled embeddings)

# optional memory helpers
if enable_vae_slicing:
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

if enable_vae_tiling:
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

# %% [markdown]
# ## 5) build pooled CLIP embedding from manual ids

# %%
clip_content_ids_pos = torch.tensor([ids], dtype=torch.long, device=device)  # (B, L)
B = clip_content_ids_pos.shape[0]

clip_max = pipe.tokenizer.model_max_length  # typically 77
bos = pipe.tokenizer.bos_token_id
eos = pipe.tokenizer.eos_token_id
pad = pipe.tokenizer.pad_token_id


def make_clip_input_ids(content_ids: torch.LongTensor) -> torch.LongTensor:
    """
    Takes content token ids (no BOS/EOS), adds BOS/EOS, then pads/truncates to clip_max.
    Returns shape (B, clip_max).
    """
    ids_full = torch.cat(
        [
            torch.full((B, 1), bos, dtype=torch.long, device=device),
            content_ids.to(torch.long),
            torch.full((B, 1), eos, dtype=torch.long, device=device),
        ],
        dim=1,
    )

    if ids_full.shape[1] > clip_max:
        ids_full = ids_full[:, :clip_max]
        ids_full[:, -1] = eos

    if ids_full.shape[1] < clip_max:
        ids_full = torch.cat(
            [
                ids_full,
                torch.full((B, clip_max - ids_full.shape[1]), pad, dtype=torch.long, device=device),
            ],
            dim=1,
        )

    return ids_full


@torch.inference_mode()
def pooled_from_content_ids(content_ids: torch.LongTensor) -> torch.Tensor:
    """
    Returns pooled CLIP embedding (B, 768) in dtype on device.
    """
    clip_ids = make_clip_input_ids(content_ids)
    out = pipe.text_encoder(clip_ids)

    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        if isinstance(out, (tuple, list)) and len(out) > 1:
            pooled = out[1]
        else:
            raise RuntimeError("Could not get pooled CLIP output (pooler_output missing).")

    return pooled.to(device=device, dtype=dtype)


pooled_pos = pooled_from_content_ids(clip_content_ids_pos)  # (B, 768)

# %% [markdown]
# ## 6) patch out T5 (prompt_embeds = zeros), build CFG negatives

# %%
t5_dim = pipe.transformer.context_embedder.in_features  # should be 4096
prompt_embeds_pos = torch.zeros((B, max_sequence_length, t5_dim), device=device, dtype=dtype)

if guidance_scale > 1.0:
    negative_prompt_embeds = torch.zeros_like(prompt_embeds_pos)

    if ids_neg is None:
        negative_pooled = torch.zeros_like(pooled_pos)
    else:
        clip_content_ids_neg = torch.tensor([ids_neg], dtype=torch.long, device=device)
        negative_pooled = pooled_from_content_ids(clip_content_ids_neg)
else:
    negative_prompt_embeds = None
    negative_pooled = None

# %% [markdown]
# ## 7) sanity checks

# %%
print("prompt_embeds_pos:", tuple(prompt_embeds_pos.shape))  # (B, 512, 4096)
print("pooled_pos:", tuple(pooled_pos.shape))                # (B, 768)
if negative_prompt_embeds is not None:
    print("negative_prompt_embeds:", tuple(negative_prompt_embeds.shape))
    print("negative_pooled:", tuple(negative_pooled.shape))

# %% [markdown]
# ## 8) generate

# %%
gen = torch.Generator(device="cuda").manual_seed(seed)

result = pipe(
    prompt_embeds=prompt_embeds_pos,
    pooled_prompt_embeds=pooled_pos,
    negative_prompt_embeds=negative_prompt_embeds,
    negative_pooled_prompt_embeds=negative_pooled,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    max_sequence_length=max_sequence_length,
    output_type="pil",
    generator=gen,
)

image = result.images[0]

# %% [markdown]
# ## 9) save

# %%
image.save(f"flux_{seed}_{ids}.png")
print("saved:", f"flux_{seed}_{ids}.png")
