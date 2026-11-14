"""Batched evaluation from uint8 images, plus metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GPUImages:
    """Holds uint8 images on the device once; normalizes per batch."""

    def __init__(self, images: np.ndarray, mean, std, device: str):
        self.x = torch.from_numpy(np.ascontiguousarray(images)).to(device)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1) * 255

    def batches(self, bs: int):
        for i in range(0, len(self.x), bs):
            yield (self.x[i:i + bs].float() - self.mean) / self.std


@torch.no_grad()
def logits(model, data: GPUImages, bs: int = 256, amp: bool = True) -> torch.Tensor:
    dev = data.x.device
    outs = []
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
        for xb in data.batches(bs):
            outs.append(model(xb).float())
    return torch.cat(outs).cpu()


def ece(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    return total


def metrics(lg: torch.Tensor, labels: np.ndarray) -> dict:
    y = torch.from_numpy(labels)
    pred = lg.argmax(1)
    return dict(
        acc=pred.eq(y).float().mean().item(),
        loss=F.cross_entropy(lg, y).item(),
        ece=ece(lg.softmax(1), y),
        pred=pred.numpy().astype(np.int16),
    )


def paired_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray, n: int = 2000, seed: int = 0):
    """95% CI for acc(b) - acc(a) on the same examples."""
    rng = np.random.default_rng(seed)
    d = correct_b.astype(float) - correct_a.astype(float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
