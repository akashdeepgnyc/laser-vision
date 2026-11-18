"""LASER-style SVD edits and controls.

An edit replaces one weight matrix W (m x n) by an approximation built from its SVD.
k = max(1, floor(rho * min(m, n)))  -- rho is the fraction of the maximum rank kept,
as in Sharma et al. (2023).

Modes
-----
top     : keep the k largest singular components (LASER).
bottom  : keep only the components LASER would discard (W - W_top).   [control]
random  : project W onto a random k-dimensional input subspace (W Q Q^T). [control]
shrink  : scale the whole matrix, c * W, with c chosen so its size (Frobenius norm) equals
          LASER's W_top at the same rho. Keeps every direction, only makes W smaller.  [control]
          If shrink matches LASER, the effect is about size, not about which directions go.
prune   : magnitude pruning at LASER's parameter budget: keep the s = k(m+n) largest-magnitude
          entries (the parameters needed to store rank-k factors), zero the rest.  [comparison]
          Best s-sparse approximation, as top is the best rank-k one; usually stays near full rank.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

MODES = ("top", "bottom", "random", "shrink", "prune")


def kept_rank(shape: tuple[int, int], rho: float) -> int:
    m, n = shape
    return max(1, int(rho * min(m, n)))


@torch.no_grad()
def low_rank(W: torch.Tensor, rho: float, mode: str = "top", seed: int = 0) -> torch.Tensor:
    """Return the edited version of a 2-D matrix W (same shape, float32)."""
    if W.dim() != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {tuple(W.shape)}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    device = W.device
    W32 = W.float().cpu()  # SVD/QR on CPU: not all are implemented on Apple-Silicon GPUs (MPS)
    k = kept_rank(tuple(W32.shape), rho)
    if mode == "prune":
        m, n = W32.shape
        s = min(k * (m + n), m * n)
        if s >= m * n:
            return W32.to(device)
        thr = torch.topk(W32.abs().flatten(), s, sorted=False).values.min()
        return (W32 * (W32.abs() >= thr)).to(device)
    if mode == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        G = torch.randn(W32.shape[1], k, generator=g)
        Q, _ = torch.linalg.qr(G)  # n x k, orthonormal columns
        return (W32 @ Q @ Q.T).to(device)
    U, S, Vh = torch.linalg.svd(W32, full_matrices=False)
    if mode == "top":
        return ((U[:, :k] * S[:k]) @ Vh[:k]).to(device)
    if mode == "shrink":
        c = torch.sqrt((S[:k] ** 2).sum() / (S ** 2).sum())
        return (W32 * c).to(device)
    return ((U[:, k:] * S[k:]) @ Vh[k:]).to(device)


@contextmanager
def edited(target, rho: float, mode: str = "top", seed: int = 0):
    """Temporarily apply an edit to `target` (a laser.models.Target); restores on exit."""
    param = target.parameter()
    original = param.data.clone()
    try:
        W = target.read(param.data)
        target.write(param.data, low_rank(W, rho, mode, seed))
        yield kept_rank(tuple(W.shape), rho)
    finally:
        param.data.copy_(original)
