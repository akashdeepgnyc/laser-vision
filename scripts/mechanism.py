"""Why does shrinking the fine-tuning update help rare classes? A memorization test.

For several versions of the fine-tuned model (W = W_pretrained + c*dW for each --scales value, plus
'cut': dW truncated to its top --cut-rank directions), measure per class:
  train_acc   accuracy on the class's own long-tailed TRAINING images (no augmentation)
  test_acc    accuracy on the class's test images (report half)
  margin      test logit margin: true-class logit minus the best other logit
  tight       how tightly the class's training features cluster (mean cosine to their own mean)
  sim_own     test features' cosine to their class's training-feature mean
  sim_gap     sim_own minus cosine to the nearest OTHER class mean (how separable test images are)
Memorization signature: rare classes have train_acc ~100% and very tight training features, but test
features land far from them. If shrinking lowers train_acc/tightness and raises test_acc/sim_gap
for rare classes, it is undoing memorization.

Also saves a qualitative grid: rare-class test images FIXED by shrinking and BROKEN by cutting.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from laser import cifar_lt, data, models
from laser.cache import autocast

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--seed", type=int, default=0, help="seed the model was fine-tuned with (picks the LT subset)")
ap.add_argument("--scales", default="1.0,0.7,0.5,0.3")
ap.add_argument("--best", type=float, default=0.5, help="shrink scale used for the 'fixed' examples")
ap.add_argument("--cut-rank", type=int, default=64)
ap.add_argument("--data-root", default="~/data/laser-vision/cifar100")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--fake", action="store_true")
ap.add_argument("--out", required=True, help="output prefix (writes _perclass.csv, _bins.csv, _flips.png)")
a = ap.parse_args()
root = Path(a.dir).expanduser()
dev = models.pick_device()
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)

ck = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
import timm
if a.fake:
    torch.manual_seed(123)
pre = timm.create_model(ck["timm_name"], pretrained=not a.fake, num_classes=ck["num_classes"])
W0 = {t.name: t.read(t.parameter().data).float().clone() for t in models._vit_targets(pre)}
b = models.load_checkpoint(str(root / "model.pt"), dev)
model = b.model
dW = {t.name: t.read(t.parameter().data).float().cpu() - W0[t.name] for t in b.targets}
mean = torch.tensor(b.preprocess.mean, device=dev).view(1, 3, 1, 1) * 255
std = torch.tensor(b.preprocess.std, device=dev).view(1, 3, 1, 1) * 255
size = b.preprocess.size

meta = json.loads((root / "test" / "meta.json").read_text())
if a.fake:
    xtr, ytr, _, _, _ = cifar_lt.fake_cifar()
else:
    xtr, ytr, _, _, _ = cifar_lt.load_cifar100(str(Path(a.data_root).expanduser()))
keep = cifar_lt.lt_indices(ytr, meta.get("imbalance", 100.0), a.seed)
xtr, ytr = xtr[keep], ytr[keep]
assert np.array_equal(np.bincount(ytr, minlength=len(meta["train_counts"])), meta["train_counts"]), \
    "training subset does not match the checkpoint: check --seed"
xte, yte, _ = data.load_prepared(str(root / "test"), "report", a.limit)
names, counts = meta["classes"], np.array(meta["train_counts"])
n_cls = len(counts)


@torch.no_grad()
def run(x, is_train):
    feats, logits = [], []
    with autocast(torch.device(dev), True):
        for i in range(0, len(x), a.batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + a.batch])).to(dev)
            if is_train:
                xb = F.interpolate(xb.permute(0, 3, 1, 2).float(), size=(size, size), mode="bilinear", align_corners=False)
            xb = (xb.float() - mean) / std
            f = model.forward_head(model.forward_features(xb), pre_logits=True).float()
            feats.append(f.cpu())
            logits.append(model.get_classifier()(f.to(next(model.get_classifier().parameters()).dtype)).float().cpu())
    return F.normalize(torch.cat(feats), dim=1), torch.cat(logits)


def set_weights(cfg):
    for t in b.targets:
        d = dW[t.name]
        if cfg == "cut":
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            d = (U[:, :a.cut_rank] * S[:a.cut_rank]) @ Vh[:a.cut_rank]
        else:
            d = float(cfg) * d
        t.write(t.parameter().data, (W0[t.name] + d).to(dev))


configs = [float(s) for s in a.scales.split(",")] + ["cut"]
if a.best not in configs:
    configs.insert(1, a.best)
rows, preds = [], {}
for cfg in configs:
    set_weights(cfg)
    ftr, ltr = run(xtr, True)
    fte, lte = run(xte, False)
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
    mu = F.normalize(torch.stack([ftr[ytr_t == k].mean(0) for k in range(n_cls)]), dim=1)
    sims = fte @ mu.T
    own = sims[torch.arange(len(yte)), yte_t]
    other = sims.clone(); other[torch.arange(len(yte)), yte_t] = -2
    lt_true = lte[torch.arange(len(yte)), yte_t]
    lo = lte.clone(); lo[torch.arange(len(yte)), yte_t] = -1e9
    margin = lt_true - lo.max(1).values
    ptr, pte = ltr.argmax(1), lte.argmax(1)
    preds[str(cfg)] = pte.numpy()
    for k in range(n_cls):
        mtr, mte = ytr_t == k, yte_t == k
        rows.append(dict(config=str(cfg), cls=k, name=names[k], train_count=int(counts[k]),
                         bin=cifar_lt.shot_bin(counts[k]),
                         train_acc=float((ptr[mtr] == k).float().mean() * 100),
                         test_acc=float((pte[mte] == k).float().mean() * 100),
                         margin=float(margin[mte].mean()),
                         tight=float((ftr[mtr] @ mu[k]).mean()),
                         sim_own=float(own[mte].mean()),
                         sim_gap=float((own[mte] - other[mte].max(1).values).mean())))
    print(f"config {cfg}: train acc {(ptr == ytr_t).float().mean() * 100:.1f}, test acc {(pte == yte_t).float().mean() * 100:.1f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(str(out) + "_perclass.csv", index=False)
cols = ["train_acc", "test_acc", "margin", "tight", "sim_own", "sim_gap"]
summ = df.groupby(["config", "bin"], sort=False)[cols].mean().round(3)
summ.to_csv(str(out) + "_bins.csv")
pd.set_option("display.width", 160)
print("\nMean per class, by version of the model and class group (report half for test columns):")
print(summ.to_string())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
few = np.array([cifar_lt.shot_bin(counts[y]) == "few" for y in yte])
p1, pb, pc = preds["1.0"], preds[str(a.best)], preds["cut"]
fixed = np.flatnonzero(few & (p1 != yte) & (pb == yte))
broken = np.flatnonzero(few & (p1 == yte) & (pc != yte))
rng = np.random.default_rng(0)
pick = lambda ix: rng.choice(ix, size=min(8, len(ix)), replace=False) if len(ix) else []
groups = [(f"Fixed by shrinking (c={a.best}): wrong as fine-tuned, right after", pick(fixed), p1, pb),
          (f"Broken by cutting (top-{a.cut_rank} directions of the update): right as fine-tuned, wrong after", pick(broken), p1, pc)]
fig, axes = plt.subplots(2, 8, figsize=(16, 5.2))
for r, (title, ix, before, after) in enumerate(groups):
    for c in range(8):
        ax = axes[r, c]
        ax.axis("off")
        if c == 0 and len(ix) == 0:
            ax.text(0, 0.5, "none (nothing to cut: the update has fewer directions than the cut keeps)",
                    transform=ax.transAxes, fontsize=10, ha="left")
        if c < len(ix):
            i = ix[c]
            ax.imshow(np.transpose(xte[i], (1, 2, 0)))
            ax.set_title(f"{names[yte[i]]} ({counts[yte[i]]} train)\n{names[before[i]]} → {names[after[i]]}", fontsize=8)
    axes[r, 0].text(-0.1, 1.32, title, transform=axes[r, 0].transAxes, fontsize=11, ha="left")
fig.subplots_adjust(top=0.82, bottom=0.03, hspace=0.75, wspace=0.15)
fig.savefig(str(out) + "_flips.png", dpi=140)
print(f"\nrare-class test images fixed by shrinking: {len(fixed)}, broken by cutting: {len(broken)} "
      f"(of {few.sum()} rare-class test images)")
print("MECH_OK", out)
