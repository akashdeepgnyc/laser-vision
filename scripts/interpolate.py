"""Shrink the fine-tuning update instead of cutting it: W = W_pretrained + c * dW.

Applied to all 72 ViT matrices at once (head, norms, biases stay fine-tuned). c = 1 is the
fine-tuned model, c = 0 the pretrained matrices. Scores every c (and logit-adjustment tau) on BOTH
test halves (choose on 'select', report on 'report'), per many/medium/few bin, plus loss and ECE,
paired-bootstrap CIs between the chosen configs, and per-class accuracy for the chosen configs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from laser import cifar_lt, data, evaluate, models
from laser.evaluate import paired_bootstrap

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

rows, oks = [], {}
for c in [float(x) for x in a.scales.split(",")]:
    for t in b.targets:
        t.write(t.parameter().data, (W0[t.name] + c * dW[t.name]).to(dev))
    for sp, (g, labels, bins) in sets.items():
        lg = evaluate.logits(b.model, g, a.batch)
        for tau in [float(x) for x in a.taus.split(",")]:
            m = evaluate.metrics(lg - tau * log_prior, labels)
            ok = m.pop("pred") == labels
            oks[(c, tau, sp)] = ok
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
chosen = {}
def show(name, q):
    i = sel[q]["all"].idxmax() if q is not None else sel["all"].idxmax()
    c, tau = sel.loc[i, "scale"], sel.loc[i, "tau"]
    chosen[name] = (c, tau)
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
if rep is not None and (1.0, 0.0) in rep.index:
    bins_r = sets["report"][2]
    pairs = [("shrink only", "fine-tuned as is"), ("logit adjustment only", "fine-tuned as is"),
             ("both", "logit adjustment only"), ("both", "shrink only")]
    chosen["fine-tuned as is"] = (1.0, 0.0)
    print("\nPaired bootstrap on the report half (95% CI, points):")
    for x, y in pairs:
        ox, oy = oks[(*chosen[x], "report")], oks[(*chosen[y], "report")]
        d, lo, hi = paired_bootstrap(oy, ox)
        f = bins_r == "few"
        df_, flo, fhi = paired_bootstrap(oy[f], ox[f])
        print(f"  {x} - {y}: all {d*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]   few {df_*100:+.2f} [{flo*100:+.2f}, {fhi*100:+.2f}]")
    pc = []
    labels_r = sets["report"][1]
    counts_r = np.array(meta["train_counts"])
    for name, key in chosen.items():
        ok = oks[(*key, "report")]
        for k in range(len(counts_r)):
            msk = labels_r == k
            if msk.any():
                pc.append(dict(config=name, scale=key[0], tau=key[1], cls=k, train_count=int(counts_r[k]), acc=ok[msk].mean() * 100))
    pcp = str(Path(a.out).with_suffix("")) + "_perclass.csv"
    pd.DataFrame(pc).to_csv(pcp, index=False)
    print("per-class accuracy (report half) ->", pcp)
print("INTERP_OK", a.out)
