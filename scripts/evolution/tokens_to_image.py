"""Generate SDXL Turbo images from evolved token IDs stored in JSON."""

import json
import math
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from PIL import Image

# Change these values, then run the file.
result_json = Path("outputs/evolution.json")
seed = 0
steps = 2
count = 4
output = Path("outputs/evolved_images.png")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SDXL generation.")

token_ids = json.loads(result_json.read_text(encoding="utf-8"))["token_ids"]
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")


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
    images = pipe(
        prompt_embeds=torch.cat([out_1.hidden_states[-2], out_2.hidden_states[-2]], dim=-1),
        pooled_prompt_embeds=out_2[0],
        num_inference_steps=steps,
        guidance_scale=0.0,
        num_images_per_prompt=count,
        generator=[torch.Generator("cuda").manual_seed(seed + i) for i in range(count)],
    ).images

width, height = images[0].size
columns = math.ceil(math.sqrt(count))
grid = Image.new("RGB", (columns * width, math.ceil(count / columns) * height), "white")
for i, image in enumerate(images):
    grid.paste(image, ((i % columns) * width, (i // columns) * height))

output.parent.mkdir(parents=True, exist_ok=True)
(images[0] if count == 1 else grid).save(output)
print(output)
