"""Tokenizer inspection, parsing, encoding, decoding, and preview helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .config import SDXL_TURBO_MODEL

_TOKEN_ID_RE = re.compile(r"\d+")


def parse_token_ids(value: str | Iterable[int], *, vocab_size: int | None = None) -> list[int]:
    """Parse integer token IDs from flexible text or an integer iterable."""
    if isinstance(value, str):
        ids = [int(match) for match in _TOKEN_ID_RE.findall(value)]
    else:
        ids = [int(item) for item in value]
    if vocab_size is not None:
        bad = [token_id for token_id in ids if token_id < 0 or token_id >= vocab_size]
        if bad:
            raise ValueError(f"Token IDs outside [0, {vocab_size - 1}]: {bad[:20]}")
    return ids


@dataclass(frozen=True)
class TokenEntry:
    token: str
    token_id: int


class TokenToolkit:
    """Load the primary SDXL tokenizer and expose notebook-friendly operations."""

    def __init__(self, model_id: str = SDXL_TURBO_MODEL) -> None:
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.vocab = self.tokenizer.get_vocab()
        self.inverse_vocab = {int(token_id): token for token, token_id in self.vocab.items()}
        self.vocab_size = int(self.tokenizer.vocab_size)

    def encode(self, text: str, *, include_special_tokens: bool = False) -> list[int]:
        ids = self.tokenizer(text, padding=False, truncation=False).input_ids
        if include_special_tokens:
            return [int(token_id) for token_id in ids]
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        if ids and ids[0] == bos:
            ids = ids[1:]
        if ids and ids[-1] == eos:
            ids = ids[:-1]
        return [int(token_id) for token_id in ids]

    def decode(self, ids: str | Iterable[int], *, skip_special_tokens: bool = False) -> str:
        parsed = parse_token_ids(ids, vocab_size=self.vocab_size)
        return self.tokenizer.decode(parsed, skip_special_tokens=skip_special_tokens)

    def entries(self, ids: str | Iterable[int]) -> list[TokenEntry]:
        parsed = parse_token_ids(ids)
        return [TokenEntry(self.inverse_vocab.get(token_id, f"<missing:{token_id}>"), token_id) for token_id in parsed]

    def search(self, query: str, *, limit: int = 100) -> list[TokenEntry]:
        needle = query.casefold().strip()
        matches: list[TokenEntry] = []
        for token, token_id in self.vocab.items():
            if not needle or needle in token.casefold() or needle in str(token_id):
                matches.append(TokenEntry(token, int(token_id)))
                if len(matches) >= limit:
                    break
        return matches

    def highlighted_text(self, ids: str | Iterable[int]) -> dict[str, object]:
        entries = self.entries(ids)
        text_parts = [entry.token for entry in entries]
        text = " ".join(text_parts)
        entities = []
        cursor = 0
        for entry in entries:
            entities.append({"entity": str(entry.token_id), "start": cursor, "end": cursor + len(entry.token)})
            cursor += len(entry.token) + 1
        return {"text": text, "entities": entities}
