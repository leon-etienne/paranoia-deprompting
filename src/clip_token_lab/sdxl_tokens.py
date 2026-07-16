"""Generate SDXL images by supplying token IDs directly to both text encoders."""

from __future__ import annotations

import gc
from typing import Iterable

import torch

from .config import SDXL_TURBO_MODEL
from .device import inference_dtype, require_cuda, resolve_device
from .tokens import parse_token_ids


def add_special_tokens_and_pad(token_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Add BOS/EOS and pad to the tokenizer's fixed CLIP context length."""
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence].")
    batch = token_ids.shape[0]
    max_length = int(tokenizer.model_max_length)
    bos_id = int(tokenizer.bos_token_id)
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id)

    if token_ids.shape[1] > max_length - 2:
        raise ValueError(f"At most {max_length - 2} content token IDs are supported.")

    ids = torch.cat(
        [
            torch.full((batch, 1), bos_id, dtype=token_ids.dtype, device=token_ids.device),
            token_ids,
            torch.full((batch, 1), eos_id, dtype=token_ids.dtype, device=token_ids.device),
        ],
        dim=1,
    )
    if ids.shape[1] < max_length:
        padding = torch.full(
            (batch, max_length - ids.shape[1]),
            pad_id,
            dtype=ids.dtype,
            device=ids.device,
        )
        ids = torch.cat([ids, padding], dim=1)
    return ids


class SDXLTokenGenerator:
    """Public-API alternative to copying and patching the whole SDXL pipeline."""

    def __init__(self, model_id: str = SDXL_TURBO_MODEL, device: str = "auto") -> None:
        from diffusers import DiffusionPipeline

        self.device = resolve_device(device)
        require_cuda(self.device, "SDXL token generation")
        self.dtype = inference_dtype(self.device)
        kwargs = {"torch_dtype": self.dtype, "use_safetensors": True}
        if self.device.type == "cuda":
            kwargs["variant"] = "fp16"
        self.pipeline = DiffusionPipeline.from_pretrained(model_id, **kwargs).to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.model_id = model_id
        self.vocab_size = min(self.pipeline.tokenizer.vocab_size, self.pipeline.tokenizer_2.vocab_size)

    def to_cuda(self) -> None:
        """Move the pipeline to CUDA before generation."""
        self.pipeline.to("cuda")
        self.device = torch.device("cuda")

    def to_cpu(self) -> None:
        """Offload the pipeline to system RAM."""
        self.pipeline.to("cpu")
        self.device = torch.device("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def prompt_embeddings(self, ids: str | Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
        parsed = parse_token_ids(ids, vocab_size=self.vocab_size)
        if not parsed:
            raise ValueError("At least one valid token ID is required.")
        content_ids = torch.tensor([parsed], dtype=torch.long, device=self.device)
        ids_1 = add_special_tokens_and_pad(content_ids, self.pipeline.tokenizer)
        ids_2 = add_special_tokens_and_pad(content_ids, self.pipeline.tokenizer_2)

        output_1 = self.pipeline.text_encoder(ids_1, output_hidden_states=True)
        output_2 = self.pipeline.text_encoder_2(ids_2, output_hidden_states=True)
        hidden_1 = output_1.hidden_states[-2]
        hidden_2 = output_2.hidden_states[-2]
        pooled_2 = output_2[0]
        prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)
        target_dtype = self.pipeline.text_encoder_2.dtype
        return (
            prompt_embeds.to(device=self.device, dtype=target_dtype),
            pooled_2.to(device=self.device, dtype=target_dtype),
        )

    @torch.inference_mode()
    def generate(
        self,
        ids: str | Iterable[int],
        *,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
        steps: int = 1,
        guidance_scale: float = 0.0,
        count: int = 1,
    ):
        if self.device.type != "cuda":
            self.to_cuda()
        if int(width) % 8 or int(height) % 8:
            raise ValueError("SDXL width and height must be divisible by 8.")
        if int(count) < 1:
            raise ValueError("count must be at least 1.")
        prompt_embeds, pooled_prompt_embeds = self.prompt_embeddings(ids)
        generators = [
            torch.Generator(device=self.device).manual_seed(int(seed) + index)
            for index in range(int(count))
        ]
        result = self.pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generators,
            width=int(width),
            height=int(height),
            num_images_per_prompt=int(count),
        )
        return result.images
