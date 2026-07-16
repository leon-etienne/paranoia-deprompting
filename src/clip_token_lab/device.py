"""Device and dtype selection helpers."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA, MPS, or CPU in that order."""
    requested = requested.lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inference_dtype(device: torch.device) -> torch.dtype:
    """Use fp16 on CUDA and fp32 elsewhere for broad compatibility."""
    return torch.float16 if device.type == "cuda" else torch.float32


def require_cuda(device: torch.device, feature: str) -> None:
    """Raise a useful error when a GPU-only experiment is requested."""
    if device.type != "cuda":
        raise RuntimeError(f"{feature} requires an NVIDIA CUDA GPU.")
