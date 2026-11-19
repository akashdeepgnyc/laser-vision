"""LoRA (Hu et al. 2021) for timm ViTs: W x + (alpha/r) * B A x, with W frozen.

wrap() adds adapters to every block's qkv, proj, fc1 and fc2 (the 48 Linear layers holding the 72
LASER matrices). merged_state_dict() folds B A back into W, so the saved checkpoint has the plain
model's keys and every other script (sweep, delta_w) can read it unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def delta(self) -> torch.Tensor:
        return (self.B @ self.A) * self.scale


def _layers(model):
    for b in model.blocks:
        yield b.attn, "qkv"
        yield b.attn, "proj"
        yield b.mlp, "fc1"
        yield b.mlp, "fc2"


def wrap(model, r: int, alpha: float | None = None) -> list[nn.Parameter]:
    """Freeze everything except the classifier head, add adapters; return the trainable params."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    for parent, name in _layers(model):
        setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha if alpha else 2 * r))
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def merged_state_dict(model) -> dict:
    sd = {}
    for k, v in model.state_dict().items():
        if k.endswith(".A") or k.endswith(".B"):
            continue
        sd[k.replace(".base.", ".")] = v.detach().cpu().clone()
    for parent, name in _layers(model):
        m = getattr(parent, name)
        if isinstance(m, LoRALinear):
            key = next(k for k, mod in model.named_modules() if mod is m) + ".weight"
            sd[key] = (m.base.weight + m.delta()).cpu()
    return sd
