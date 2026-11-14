"""Spectrum atlas: how low-rank is every editable matrix, before vs after training? No data needed.

  uv run python scripts/spectrum_atlas.py                 # vit_b16, resnet50, clip_b32
  uv run python scripts/spectrum_atlas.py --models vit_b16 --out results/atlas

Writes to --out:
  atlas.csv                 one row per (model, matrix, trained|init) with laser.spectrum.stats
  spectra.npz               raw singular values, key "<model>|<target>|<trained|init>"
  <model>_gd_heatmap.png    Gavish-Donoho kept-rank fraction, matrix type x depth (trained)
  <model>_eff_rank.png      effective-rank fraction by depth, each matrix type, vs random init
  <model>_spectra.png       normalized spectra of mlp_in (or conv_b) at first / middle / last layer
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from laser import models, spectrum

# Fixed categorical order (validated palette: dataviz default slots 1-6). Never cycled.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
ORDER = {"vit": ["attn_q", "attn_k", "attn_v", "attn_out", "mlp_in", "mlp_out"],
         "resnet": ["conv_a", "conv_b", "conv_c", "fc"]}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="vit_b16,resnet50,clip_b32")
ap.add_argument("--out", default="results/atlas")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "legend.frameon": False})

rows, raw = [], {}
for name in a.models.split(","):
    for kind in ("trained", "init"):
        b = models.load(name, classnames=["x"], templates=[lambda c: c], random_init=(kind == "init"))
        for t in b.targets:
            W = t.read(t.parameter().data)
            s = spectrum.singular_values(W)
            raw[f"{name}|{t.name}|{kind}"] = s
            rows.append(dict(model=name, target=t.name, layer=t.layer, mtype=t.mtype, kind=kind,
                             **spectrum.stats(s, tuple(W.shape))))
        print(f"{name} {kind}: {len(b.targets)} matrices", flush=True)

df = pd.DataFrame(rows)
df.to_csv(out / "atlas.csv", index=False)
np.savez_compressed(out / "spectra.npz", **raw)

for name in a.models.split(","):
    d = df[df.model == name]
    tr, init = d[d.kind == "trained"], d[d.kind == "init"]
    order = [m for m in ORDER["resnet" if "conv_b" in set(d.mtype) else "vit"] if m in set(d.mtype)]
    colors = dict(zip(order, SLOTS))
    marks = dict(zip(order, MARKERS))

    # 1. Gavish-Donoho kept fraction heatmap (sequential blue: darker = more directions kept)
    piv = tr.pivot_table(index="mtype", columns="layer", values="gd_frac").reindex(order)
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * piv.shape[1] + 2), 0.45 * len(order) + 1.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=0, vmax=max(0.05, np.nanmax(piv.values)))
    ax.set_yticks(range(len(order)), order)
    ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns])
    ax.set_xlabel("Depth (block index)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Kept-rank fraction\n(Gavish–Donoho)", color=INK)
    ax.set_title(f"{name}: directions above the noise threshold", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / f"{name}_gd_heatmap.png", dpi=200)
    plt.close(fig)

    # 2. Effective-rank fraction by depth; init averaged over matrix types as one gray reference
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in order:
        s = tr[tr.mtype == m].sort_values("layer")
        ax.plot(s.layer, s.eff_rank_frac, color=colors[m], marker=marks[m], ms=6, lw=2, label=m)
    ref = init.groupby("layer").eff_rank_frac.mean()
    ax.plot(ref.index, ref.values, color=MUTED, ls="--", lw=1.5, label="random init (mean)")
    ax.set_xlabel("Depth (block index)")
    ax.set_ylabel("Effective rank / max rank")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=2, fontsize=9, loc="lower left")
    ax.set_title(f"{name}: how spread out is each matrix's spectrum?", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / f"{name}_eff_rank.png", dpi=200)
    plt.close(fig)

    # 3. Normalized spectra at first / middle / last layer, trained vs init
    probe = "conv_b" if "conv_b" in order else "mlp_in"
    layers = sorted(tr[tr.mtype == probe].layer.unique())
    pick = [layers[0], layers[len(layers) // 2], layers[-1]]
    fig, ax = plt.subplots(figsize=(7, 4))
    for shade, L in zip(["#86b6ef", "#2a78d6", "#0d366b"], pick):
        tgt = tr[(tr.mtype == probe) & (tr.layer == L)].target.iloc[0]
        for kind, ls in (("trained", "-"), ("init", "--")):
            s = raw[f"{name}|{tgt}|{kind}"]
            x = np.arange(1, len(s) + 1) / len(s)
            ax.plot(x, s / s[0], color=shade if kind == "trained" else MUTED, ls=ls, lw=2 if kind == "trained" else 1,
                    label=f"layer {L} trained" if kind == "trained" else (None if L != pick[-1] else "random init"))
    for rho in (0.1, 0.01):
        ax.axvline(rho, color=GRID, lw=1)
        ax.text(rho, 0.02, f" ρ={rho}", color=MUTED, fontsize=8, va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Singular value index / max rank (log)")
    ax.set_ylabel("σᵢ / σ₁ (log)")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title(f"{name}: {probe} spectra (LASER keeps the left of each ρ line)", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / f"{name}_spectra.png", dpi=200)
    plt.close(fig)

summary = (df.groupby(["model", "mtype", "kind"])[["gd_frac", "eff_rank_frac", "stable_rank_frac", "energy@0.01", "energy@0.1"]]
           .mean().round(3).unstack("kind"))
summary.to_csv(out / "summary.csv")
print(summary.to_string())
print("ATLAS_OK", out)
