"""Genetic search over CLIP token sequences for a target image."""

from __future__ import annotations

import gc
import random
import time
from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image

from .config import CLIP_LARGE_MODEL
from .device import inference_dtype, require_cuda, resolve_device


@dataclass(frozen=True)
class EvolutionConfig:
    seed: int = 0
    random_seed: bool = True
    n_tokens: int = 16
    population: int = 1024
    generations: int = 100
    score_batch_size: int = 4096
    elite_fraction: float = 0.08
    tournament_size: int = 4
    crossover_probability: float = 0.9
    swap_mutation_probability: float = 0.25
    random_mutation_probability: float = 0.08


@dataclass(frozen=True)
class EvolutionResult:
    decoded_text: str
    score: float
    token_ids: list[int]


class CLIPTokenOptimizer:
    """Optimize fixed-length token sequences against a target image embedding."""

    def __init__(self, model_id: str = CLIP_LARGE_MODEL, device: str = "auto") -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = resolve_device(device)
        require_cuda(self.device, "CLIP token evolution")
        self.dtype = inference_dtype(self.device)
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.bos_id = int(self.tokenizer.bos_token_id)
        self.eos_id = int(self.tokenizer.eos_token_id)
        self.pool = self._build_token_pool()

    def to_cuda(self) -> None:
        """Move the optimizer model and token pool to CUDA."""
        self.model.to("cuda")
        self.device = torch.device("cuda")
        self.pool = self._build_token_pool()

    def to_cpu(self) -> None:
        """Offload the model after a search so SDXL can use the GPU."""
        self.model.to("cpu")
        self.device = torch.device("cpu")
        self.pool = torch.empty(0, dtype=torch.long)
        gc.collect()
        torch.cuda.empty_cache()

    def _build_token_pool(self) -> torch.Tensor:
        pool = torch.arange(self.tokenizer.vocab_size, device=self.device)
        banned = {
            token_id
            for token_id in (self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
            if token_id is not None
        }
        mask = torch.ones(self.tokenizer.vocab_size, dtype=torch.bool, device=self.device)
        for token_id in banned:
            mask[int(token_id)] = False
        return pool[mask]

    @torch.inference_mode()
    def image_embedding(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            vision_output = self.model.vision_model(pixel_values=inputs["pixel_values"])
            feature = self.model.visual_projection(vision_output.pooler_output)
        feature = feature.float()
        return feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.inference_mode()
    def _score(self, population: torch.Tensor, image_feature: torch.Tensor, batch_size: int) -> torch.Tensor:
        outputs = []
        for start in range(0, population.shape[0], batch_size):
            sequence = population[start : start + batch_size]
            batch = sequence.shape[0]
            input_ids = torch.cat(
                [
                    torch.full((batch, 1), self.bos_id, device=self.device, dtype=torch.long),
                    sequence,
                    torch.full((batch, 1), self.eos_id, device=self.device, dtype=torch.long),
                ],
                dim=1,
            )
            attention_mask = torch.ones_like(input_ids)
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                text_output = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
                text_features = self.model.text_projection(text_output.pooler_output)
            text_features = text_features.float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            outputs.append(text_features @ image_feature.T)
        return torch.cat(outputs, dim=0).squeeze(1)

    def optimize(
        self,
        image: Image.Image,
        config: EvolutionConfig = EvolutionConfig(),
        progress: Callable[[int, int, float], None] | None = None,
    ) -> EvolutionResult:
        require_cuda(self.device, "CLIP token evolution")
        seed = int(time.time_ns() % (2**31 - 1)) if config.random_seed else int(config.seed)
        random.seed(seed)
        torch.manual_seed(seed)

        image_feature = self.image_embedding(image)
        population_size = int(config.population)
        n_tokens = int(config.n_tokens)
        pool_size = int(self.pool.numel())
        population = self.pool[torch.randint(0, pool_size, (population_size, n_tokens), device=self.device)]
        scores = self._score(population, image_feature, int(config.score_batch_size))
        scores_cpu = scores.detach().cpu()
        elite_count = max(1, round(config.elite_fraction * population_size))

        best_index = int(torch.argmax(scores))
        best_sequence = population[best_index].clone()
        best_score = float(scores[best_index])

        def tournament() -> int:
            indices = random.sample(range(population_size), k=min(config.tournament_size, population_size))
            return max(indices, key=lambda index: float(scores_cpu[index]))

        for generation in range(1, int(config.generations) + 1):
            order = torch.argsort(scores, descending=True)
            elites = population[order[:elite_count]].clone()
            children = [elites]
            child_count = elite_count

            while child_count < population_size:
                parent_1 = population[tournament()]
                parent_2 = population[tournament()]
                child_1, child_2 = parent_1.clone(), parent_2.clone()

                if n_tokens > 1 and random.random() < config.crossover_probability:
                    cut = random.randint(1, n_tokens - 1)
                    child_1 = torch.cat([parent_1[:cut], parent_2[cut:]])
                    child_2 = torch.cat([parent_2[:cut], parent_1[cut:]])

                for child in (child_1, child_2):
                    if n_tokens > 1 and random.random() < config.swap_mutation_probability:
                        first, second = random.sample(range(n_tokens), 2)
                        tmp = child[first].clone()
                        child[first] = child[second]
                        child[second] = tmp
                    mutation_mask = torch.rand(n_tokens, device=self.device) < config.random_mutation_probability
                    if mutation_mask.any():
                        child[mutation_mask] = self.pool[
                            torch.randint(0, pool_size, (int(mutation_mask.sum()),), device=self.device)
                        ]
                    if child_count < population_size:
                        children.append(child.unsqueeze(0))
                        child_count += 1

            population = torch.cat(children, dim=0)[:population_size]
            scores = self._score(population, image_feature, int(config.score_batch_size))
            scores_cpu = scores.detach().cpu()
            generation_best_index = int(torch.argmax(scores))
            generation_best_score = float(scores[generation_best_index])
            if generation_best_score > best_score:
                best_score = generation_best_score
                best_sequence = population[generation_best_index].clone()
            if progress is not None:
                progress(generation, int(config.generations), best_score)

        token_ids = [int(token_id) for token_id in best_sequence.tolist()]
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return EvolutionResult(decoded, best_score, token_ids)
