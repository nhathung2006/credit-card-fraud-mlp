"""Shared device, inference and file handling."""

from contextlib import contextmanager
from uuid import uuid4

import torch


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in this PyTorch environment. Use --device cpu.")
    return torch.device(value)


@torch.inference_mode()
def predict_logits(model, loader, device):
    model.eval()
    batches = []
    for dense, categorical in loader:
        logits = model(dense.to(device, non_blocking=True), categorical.to(device, non_blocking=True))
        batches.append(logits.cpu())
    return torch.cat(batches)


@contextmanager
def atomic_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
