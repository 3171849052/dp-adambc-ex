"""Small reproducibility and device helpers."""

from __future__ import annotations

import random

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        if device.index not in (None, 0):
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES maps the selected physical GPU to cuda:0; "
                "runtime.device must be 'cuda' or 'cuda:0'"
            )
        return torch.device("cuda:0")
    return device


def validate_gpu_selection(gpu: int) -> None:
    """Fail early when the configured physical GPU cannot be selected."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"runtime.gpu={gpu} requested, but CUDA is not available"
        )
    device_count = torch.cuda.device_count()
    if gpu >= device_count:
        raise RuntimeError(
            f"runtime.gpu={gpu} is invalid: only {device_count} physical CUDA "
            "GPU(s) are available"
        )
