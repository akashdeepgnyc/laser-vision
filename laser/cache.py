"""Speed-ups for sweeps: mixed precision, and reusing the unedited early layers of a ViT.

An edit in block L leaves blocks 0..L-1 untouched, so their output is the same for every edit in
blocks >= L. BlockCache stores the token states entering block L for all images and only reruns
blocks L..end for each edit. Sweeps visit targets in increasing depth, so the cache only moves
forward (one extra block per new depth). Works for timm VisionTransformer models; other models
fall back to full forward passes.
"""
from __future__ import annotations

import contextlib

import torch

from .evaluate import GPUImages


def autocast(device: torch.device, enabled: bool):
    """bf16 on NVIDIA GPUs, fp16 on Apple-Silicon GPUs (MPS), nothing on CPU."""
    if not enabled or device.type not in ("cuda", "mps"):
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    try:
        return torch.autocast(device_type=device.type, dtype=dtype)
    except RuntimeError:  # older PyTorch without MPS autocast
        return contextlib.nullcontext()


def supported(model) -> bool:
    return all(hasattr(model, a) for a in ("patch_embed", "_pos_embed", "blocks", "norm", "forward_head"))


class BlockCache:
    def __init__(self, model, data: GPUImages, bs: int = 128, amp: bool = True):
        self.model, self.data, self.bs, self.amp = model, data, bs, amp
        self.dev = data.x.device
        self.store_dtype = torch.float16 if (amp and self.dev.type != "cpu") else torch.float32
        self.layer, self.states = None, None

    @staticmethod
    def bytes_needed(model, data: GPUImages, amp: bool) -> int:
        with torch.no_grad():
            t = BlockCache._stem(model, next(data.batches(1)))
        per = 2 if (amp and data.x.device.type != "cpu") else 4
        return len(data.x) * t.shape[1] * t.shape[2] * per

    @staticmethod
    def _stem(model, x):
        x = model.patch_embed(x)
        x = model._pos_embed(x)
        if hasattr(model, "patch_drop"):
            x = model.patch_drop(x)
        if hasattr(model, "norm_pre"):
            x = model.norm_pre(x)
        return x

    @torch.no_grad()
    def to_layer(self, L: int) -> None:
        """Make self.states hold the (unedited) token states entering block L."""
        if self.layer == L:
            return
        if self.layer is None or L < self.layer:
            out = []
            with autocast(self.dev, self.amp):
                for xb in self.data.batches(self.bs):
                    out.append(self.model.blocks[:L](self._stem(self.model, xb)).to(self.store_dtype))
        else:
            out = []
            with autocast(self.dev, self.amp):
                for i in range(0, len(self.states), self.bs):
                    out.append(self.model.blocks[self.layer:L](self.states[i:i + self.bs]).to(self.store_dtype))
        self.states, self.layer = torch.cat(out), L

    @torch.no_grad()
    def logits(self) -> torch.Tensor:
        """Run blocks layer..end (with whatever edit is currently applied) and the head."""
        out = []
        with autocast(self.dev, self.amp):
            for i in range(0, len(self.states), self.bs):
                x = self.model.blocks[self.layer:](self.states[i:i + self.bs])
                out.append(self.model.forward_head(self.model.norm(x)).float())
        return torch.cat(out).cpu()
