"""Generate one or more SDXL Turbo images from explicit token IDs."""

import math
import re
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from PIL import Image

# Change these values, then run the file.
ids = "320 1125 5390"
seed = 42
steps = 1
guidance = 0.0
width = 512
height = 512
count = 1
output = Path("outputs/token_images.png")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SDXL generation.")

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

token_ids = [int(value) for value in re.findall(r"\d+", ids)]
max_vocab = min(pipe.tokenizer.vocab_size, pipe.tokenizer_2.vocab_size)
if not token_ids or any(value >= max_vocab for value in token_ids):
    raise ValueError(f"Token IDs must be between 0 and {max_vocab - 1}.")


def padded(values, tokenizer):
    values = torch.tensor([values], dtype=torch.long, device="cuda")
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    values = torch.cat(
        [torch.tensor([[bos]], device="cuda"), values, torch.tensor([[eos]], device="cuda")],
        dim=1,
    )
    if values.shape[1] > tokenizer.model_max_length:
        values = values[:, : tokenizer.model_max_length]
        values[:, -1] = eos
    return torch.nn.functional.pad(values, (0, tokenizer.model_max_length - values.shape[1]), value=pad)


with torch.inference_mode():
    out_1 = pipe.text_encoder(padded(token_ids, pipe.tokenizer), output_hidden_states=True)
    out_2 = pipe.text_encoder_2(padded(token_ids, pipe.tokenizer_2), output_hidden_states=True)
    prompt_embeds = torch.cat([out_1.hidden_states[-2], out_2.hidden_states[-2]], dim=-1)
    pooled = out_2[0]
    images = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        num_images_per_prompt=count,
        generator=[torch.Generator("cuda").manual_seed(seed + i) for i in range(count)],
    ).images

columns = math.ceil(math.sqrt(count))
grid = Image.new("RGB", (columns * width, math.ceil(count / columns) * height), "white")
for i, image in enumerate(images):
    grid.paste(image, ((i % columns) * width, (i // columns) * height))

output.parent.mkdir(parents=True, exist_ok=True)
(images[0] if count == 1 else grid).save(output)
print(output)
