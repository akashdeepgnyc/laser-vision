"""Spectral statistics of weight matrices (no data needed).

For a matrix with singular values s_1 >= ... >= s_r (r = min(m, n)):
  stable_rank    ||W||_F^2 / s_1^2                     (how many directions carry the energy)
  eff_rank       exp(entropy of s / sum(s))             (Roy & Vetterli 2007, cited by LASER)
  gd_rank        #{s_i > tau}, tau = omega(beta) * median(s)
                 Gavish & Donoho (2014) optimal hard threshold for unknown noise level,
                 beta = min(m, n) / max(m, n). Treats W as low-rank signal + i.i.d. noise,
                 so it is a *prediction* of where a cut should stop, not ground truth.
  energy@rho     fraction of ||W||_F^2 kept by LASER's top floor(rho * r) components
Fractions are rank / r, so matrices of different sizes are comparable.
"""
from __future__ import annotations

import numpy as np
import torch

RHOS = (0.5, 0.2, 0.1, 0.05, 0.01)


def singular_values(W: torch.Tensor) -> np.ndarray:
    return torch.linalg.svdvals(W.detach().float().cpu()).numpy().astype(np.float64)


def gd_omega(beta: float) -> float:
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


def stats(s: np.ndarray, shape: tuple[int, int]) -> dict:
    m, n = shape
    r = len(s)
    fro2 = float((s**2).sum())
    p = s / s.sum()
    eff = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    tau = gd_omega(min(m, n) / max(m, n)) * float(np.median(s))
    gd = int((s > tau).sum())
    cum = np.cumsum(s**2) / fro2
    out = dict(rows=m, cols=n, max_rank=r, spectral_norm=float(s[0]), fro_norm=fro2**0.5,
               stable_rank_frac=fro2 / s[0] ** 2 / r, eff_rank_frac=eff / r,
               gd_rank=gd, gd_frac=gd / r)
    for rho in RHOS:
        k = max(1, int(rho * r))
        out[f"energy@{rho}"] = float(cum[k - 1])
    return out
