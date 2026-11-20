"""Per-class view of the headline result: gain vs how many training images each class had.

Reads the *_perclass.csv files written by interpolate.py (report half), averages seeds, and plots
per-class accuracy gain against training count (log scale) for three comparisons:
  shrink - as is,   logit adjustment - as is,   both - logit adjustment
One row of panels per model. Prints Spearman correlations (gain vs training count).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--full", required=True)
ap.add_argument("--lora", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()

COMPARE = [("shrink only", "fine-tuned as is", "Shrink the update"),
           ("logit adjustment only", "fine-tuned as is", "Logit adjustment"),
           ("both", "logit adjustment only", "Shrink, on top of logit adjustment")]


def gains(files):
    df = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(files.split(","))])
    acc = df.pivot_table(index="cls", columns="config", values="acc", aggfunc="mean")  # mean over seeds
    cnt = df.groupby("cls").train_count.first()
    return acc, cnt


def spearman(x, y):
    rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


models = [("Full fine-tune", a.full)] + ([("LoRA rank 16", a.lora)] if a.lora else [])
fig, axes = plt.subplots(len(models), 3, figsize=(13, 3.8 * len(models)), squeeze=False, sharey="row")
for r, (mname, files) in enumerate(models):
    acc, cnt = gains(files)
    for c, (x, y, title) in enumerate(COMPARE):
        g = acc[x] - acc[y]
        ax = axes[r, c]
        ax.axhline(0, color="#888", lw=1)
        ax.scatter(cnt, g, s=18, color="#2a6f97", alpha=0.75, edgecolor="none")
        edges = np.logspace(np.log10(cnt.min()), np.log10(cnt.max() + 1), 8)
        idx = np.digitize(cnt, edges)
        mx = [cnt[idx == i].mean() for i in np.unique(idx)]
        my = [g[idx == i].mean() for i in np.unique(idx)]
        ax.plot(mx, my, color="#c44e52", lw=2, marker="o", ms=4, label="binned mean")
        rho = spearman(cnt.values, g.values)
        ax.set_xscale("log")
        ax.set_title(f"{title}\n{mname}: mean {g.mean():+.1f} pts, Spearman ρ = {rho:+.2f}", fontsize=10)
        ax.set_xlabel("training images in the class")
        if c == 0:
            ax.set_ylabel("per-class accuracy gain (points)")
        print(f"{mname:15s} | {title:38s} | mean {g.mean():+.2f} | rho {rho:+.2f} | "
              f"classes helped {int((g > 0).sum())}, hurt {int((g < 0).sum())}")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(a.out, dpi=160)
print("PLOT_OK", a.out)
