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


class TensorBatchLoader:
    """Fast vectorized mini-batch iterator over aligned CPU tensors."""

    def __init__(self, *tensors, batch_size, shuffle=False):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(tensors[0])

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.size)
            for start in range(0, self.size, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                yield tuple(tensor[batch_idx] for tensor in self.tensors)
        else:
            for start in range(0, self.size, self.batch_size):
                end = start + self.batch_size
                yield tuple(tensor[start:end] for tensor in self.tensors)


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
