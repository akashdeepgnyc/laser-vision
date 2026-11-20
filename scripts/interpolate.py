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
ap.add_argument("--ckpt", default="model.pt", help="checkpoint file inside --dir (e.g. model_epoch4.pt)")
ap.add_argument("--scales", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
ap.add_argument("--splits", default="select,report")
ap.add_argument("--taus", default="0", help="post-hoc logit adjustment (Menon et al. 2021): logits - tau*log(train prior)")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--fake", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
root = Path(a.dir).expanduser()
dev = models.pick_device()

ck = torch.load(root / a.ckpt, map_location="cpu", weights_only=False)
import timm
if a.fake:
    torch.manual_seed(123)
pre = timm.create_model(ck["timm_name"], pretrained=not a.fake, num_classes=ck["num_classes"])
W0 = {t.name: t.read(t.parameter().data).float().clone() for t in models._vit_targets(pre)}
b = models.load_checkpoint(str(root / a.ckpt), dev)
dW = {t.name: t.read(t.parameter().data).float().cpu() - W0[t.name] for t in b.targets}

sets = {}
for sp in a.splits.split(","):
    images, labels, meta = data.load_prepared(str(root / "test"), sp, a.limit)
    bins = np.array([cifar_lt.shot_bin(c) for c in meta["train_counts"]])[labels]
    sets[sp] = (evaluate.GPUImages(images, b.preprocess.mean, b.preprocess.std, dev), labels, bins)
counts = np.array(meta["train_counts"], dtype=float)
log_prior = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)

rows = []
for c in [float(x) for x in a.scales.split(",")]:
    for t in b.targets:
        t.write(t.parameter().data, (W0[t.name] + c * dW[t.name]).to(dev))
    for sp, (g, labels, bins) in sets.items():
        lg = evaluate.logits(b.model, g, a.batch)
        for tau in [float(x) for x in a.taus.split(",")]:
            m = evaluate.metrics(lg - tau * log_prior, labels)
            ok = m.pop("pred") == labels
            row = dict(scale=c, tau=tau, split=sp, all=ok.mean() * 100, loss=m["loss"], ece=m["ece"] * 100,
                       **{bn: ok[bins == bn].mean() * 100 for bn in ("many", "medium", "few")})
            rows.append(row)
            print({k: (round(float(v), 2) if not isinstance(v, str) else v) for k, v in row.items()}, flush=True)
df = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
pd.set_option("display.width", 160)
for sp in sets:
    print(f"\n{sp} half:")
    print(df[df.split == sp].drop(columns="split").round(2).to_string(index=False))
sel = df[df.split == "select"].reset_index(drop=True)
rep = df[df.split == "report"].set_index(["scale", "tau"]) if "report" in sets else None
def show(name, q):
    i = sel[q]["all"].idxmax() if q is not None else sel["all"].idxmax()
    c, tau = sel.loc[i, "scale"], sel.loc[i, "tau"]
    line = f"{name}: chosen on select -> scale {c}, tau {tau}"
    if rep is not None and (c, tau) in rep.index:
        r = rep.loc[(c, tau)]
        line += f" | report all {r['all']:.2f}, many {r['many']:.2f}, medium {r['medium']:.2f}, few {r['few']:.2f}"
    print(line)
print()
if rep is not None and (1.0, 0.0) in rep.index:
    r = rep.loc[(1.0, 0.0)]
    print(f"fine-tuned as is: report all {r['all']:.2f}, many {r['many']:.2f}, medium {r['medium']:.2f}, few {r['few']:.2f}")
show("shrink only", sel.tau == 0)
show("logit adjustment only", sel.scale == 1.0)
show("both", None)
print("INTERP_OK", a.out)
