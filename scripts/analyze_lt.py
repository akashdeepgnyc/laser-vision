"""Rare vs common classes: does cutting help or hurt the long tail?

Reads a sweep run on a CIFAR-100-LT test set and splits every edit's accuracy into many-shot,
medium-shot and few-shot classes (by how many TRAINING images each class had).

  uv run python scripts/analyze_lt.py --results results/sweep_lt.jsonl \
      --data ~/data/laser-vision/cifar100lt_vitb16/test --split select
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from laser import cifar_lt, data
from laser.evaluate import paired_bootstrap

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--split", default="select")
ap.add_argument("--out", default=None, help="CSV of per-bin accuracy changes (default: next to results)")
a = ap.parse_args()

_, labels, meta = data.load_prepared(str(Path(a.data).expanduser()), a.split)
bins = np.array([cifar_lt.shot_bin(c) for c in meta["train_counts"]])[labels]
pred_dir = Path(a.results).parent / "preds"
rows = [json.loads(l) for l in open(a.results)]
rows = [r for r in rows if r["split"] == a.split]

def per_bin(pred):
    ok = pred == labels
    return ok, {b: float(ok[bins == b].mean()) * 100 for b in ("many", "medium", "few")} | {"all": float(ok.mean()) * 100}

base = next(r for r in rows if r["mode"] == "none")
base_ok, base_acc = per_bin(np.load(pred_dir / f"{base['key']}.npy"))
print("Baseline accuracy (%):", {k: round(v, 1) for k, v in base_acc.items()},
      "| test images per bin:", {b: int((bins == b).sum()) for b in ("many", "medium", "few")})

out = []
for r in rows:
    if r["mode"] == "none":
        continue
    ok, acc = per_bin(np.load(pred_dir / f"{r['key']}.npy"))
    d = {f"d_{k}": acc[k] - base_acc[k] for k in acc}
    few = bins == "few"
    _, lo, hi = paired_bootstrap(base_ok[few], ok[few])
    out.append(dict(target=r["target"], layer=r["layer"], mtype=r["mtype"], rho=r["rho"], mode=r["mode"],
                    **d, few_ci_lo=lo * 100, few_ci_hi=hi * 100))
df = pd.DataFrame(out)
path = a.out or str(Path(a.results).with_suffix("")) + "_bins.csv"
df.to_csv(path, index=False)

pd.set_option("display.width", 160)
fmt = lambda v: f"{v:+.1f}"
print("\nMean accuracy change (points) by edit type and ρ:")
print(df.groupby(["mode", "rho"])[["d_all", "d_many", "d_medium", "d_few"]].mean().round(2).to_string())
print("\nBest 10 edits for FEW-shot classes (with 95% CI on the few-shot change):")
cols = ["target", "rho", "mode", "d_all", "d_many", "d_medium", "d_few", "few_ci_lo", "few_ci_hi"]
print(df.sort_values("d_few", ascending=False)[cols].head(10).to_string(index=False, float_format=fmt))
sig = df[df.few_ci_lo > 0]
print(f"\nEdits whose few-shot gain has a 95% CI above zero: {len(sig)} of {len(df)}"
      " (expect ~2.5% by chance when testing many edits)")
print("ANALYZE_OK", path)
