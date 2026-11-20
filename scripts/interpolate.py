"""Shrink the fine-tuning update instead of cutting it: W = W_pretrained + c * dW.

Applied to all 72 ViT matrices at once (head, norms, biases stay fine-tuned). c = 1 is the
fine-tuned model, c = 0 the pretrained matrices. Scores every c on BOTH test halves (choose c on
'select', report on 'report'), per many/medium/few bin, plus loss and ECE.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from laser import cifar_lt, data, evaluate, models

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--scales", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
ap.add_argument("--splits", default="select,report")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--fake", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
root = Path(a.dir).expanduser()
dev = models.pick_device()

ck = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
import timm
if a.fake:
    torch.manual_seed(123)
pre = timm.create_model(ck["timm_name"], pretrained=not a.fake, num_classes=ck["num_classes"])
W0 = {t.name: t.read(t.parameter().data).float().clone() for t in models._vit_targets(pre)}
b = models.load_checkpoint(str(root / "model.pt"), dev)
dW = {t.name: t.read(t.parameter().data).float().cpu() - W0[t.name] for t in b.targets}

sets = {}
for sp in a.splits.split(","):
    images, labels, meta = data.load_prepared(str(root / "test"), sp, a.limit)
    bins = np.array([cifar_lt.shot_bin(c) for c in meta["train_counts"]])[labels]
    sets[sp] = (evaluate.GPUImages(images, b.preprocess.mean, b.preprocess.std, dev), labels, bins)

rows = []
for c in [float(x) for x in a.scales.split(",")]:
    for t in b.targets:
        t.write(t.parameter().data, (W0[t.name] + c * dW[t.name]).to(dev))
    for sp, (g, labels, bins) in sets.items():
        m = evaluate.metrics(evaluate.logits(b.model, g, a.batch), labels)
        ok = m.pop("pred") == labels
        row = dict(scale=c, split=sp, all=ok.mean() * 100, loss=m["loss"], ece=m["ece"] * 100,
                   **{bn: ok[bins == bn].mean() * 100 for bn in ("many", "medium", "few")})
        rows.append(row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
df = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
pd.set_option("display.width", 160)
for sp in sets:
    print(f"\n{sp} half:")
    print(df[df.split == sp].drop(columns="split").round(2).to_string(index=False))
sel = df[df.split == "select"]
best = sel.loc[sel["all"].idxmax(), "scale"]
print(f"\nbest scale on select: {best}")
if "report" in sets:
    r = df[df.split == "report"].set_index("scale")
    print(f"report half at that scale vs c=1: all {r.loc[best,'all']:.2f} vs {r.loc[1.0,'all']:.2f}, "
          f"few {r.loc[best,'few']:.2f} vs {r.loc[1.0,'few']:.2f}" if 1.0 in r.index else "")
print("INTERP_OK", a.out)
