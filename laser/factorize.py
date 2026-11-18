"""Turn cut matrices into real savings: apply a plan of cuts, then store each low-rank
Linear layer as two thin layers (W ~ A @ B), so the model really does less work.

A plan is a list of {"target": "blocks.3.attn_q", "rho": 0.01, "mode": "top"}.
Storing an m x n matrix of rank r as two factors costs r*(m+n) numbers instead of m*n,
so it only pays off when r < mn/(m+n) (the break-even rank; 384 for a 768 x 768 matrix).

The fused qkv layer (2304 x 768) is factorized as a whole: its rank is the sum of the three
slices' ranks (an uncut slice counts as 768), so it only shrinks when all three are cut hard.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .edit import kept_rank, low_rank


def apply_plan(bundle, plan) -> None:
    """Permanently write every cut in `plan` into the model."""
    by_name = {t.name: t for t in bundle.targets}
    for p in plan:
        t = by_name[p["target"]]
        param = t.parameter()
        W = t.read(param.data)
        t.write(param.data, low_rank(W, p["rho"], p.get("mode", "top")))


def module_ranks(bundle, plan) -> dict:
    """Linear module -> rank of its weight after the plan (only 'top' cuts lower the rank)."""
    cut = {p["target"]: p for p in plan if p.get("mode", "top") == "top"}
    out = {}
    for t in bundle.targets:
        if not isinstance(t.module, nn.Linear):
            continue
        W = t.read(t.parameter().data)
        r = kept_rank(tuple(W.shape), cut[t.name]["rho"]) if t.name in cut else min(W.shape)
        out[t.module] = out.get(t.module, 0) + r  # qkv: three slices add up
    return {m: min(r, min(m.weight.shape)) for m, r in out.items()}


class Factorized(nn.Sequential):
    """x -> B x -> A (B x) + bias; drop-in for an nn.Linear."""


@torch.no_grad()
def factorize(bundle, plan) -> dict:
    """Replace each Linear layer that the plan made low-rank (below break-even) by two thin ones.
    Call after apply_plan. Returns counts of dense vs stored weights."""
    ranks = module_ranks(bundle, plan)
    parents = {m: (parent, name) for parent in bundle.model.modules()
               for name, m in parent.named_children()}
    dense = stored = replaced = 0
    for mod, r in ranks.items():
        m, n = mod.weight.shape
        dense += m * n
        if r * (m + n) >= m * n:
            stored += m * n
            continue
        U, S, Vh = torch.linalg.svd(mod.weight.data.float().cpu(), full_matrices=False)
        B = nn.Linear(n, r, bias=False)
        A = nn.Linear(r, m, bias=mod.bias is not None)
        B.weight.copy_(Vh[:r])
        A.weight.copy_(U[:, :r] * S[:r])
        if mod.bias is not None:
            A.bias.copy_(mod.bias.data)
        new = Factorized(B, A).to(mod.weight.device, mod.weight.dtype)
        parent, name = parents[mod]
        setattr(parent, name, new)
        stored += r * (m + n)
        replaced += 1
    return dict(edited_matrices_dense=dense, edited_matrices_stored=stored, layers_factorized=replaced)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def linear_macs(model, x) -> int:
    """Multiply-adds per image spent in Linear layers (the part cutting can shrink)."""
    total = [0]
    hooks = [m.register_forward_hook(lambda m, i, o: total.__setitem__(
        0, total[0] + i[0].numel() // m.in_features * m.in_features * m.out_features))
        for m in model.modules() if isinstance(m, nn.Linear)]
    model(x[:1])
    for h in hooks:
        h.remove()
    return total[0]
