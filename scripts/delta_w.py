"""What did fine-tuning change? SVD of dW = W_finetuned - W_pretrained for every matrix.

For each checkpoint (e.g. model_epoch2.pt ... model.pt) and each of the 72 ViT matrices:
  rel        ||dW|| / ||W_pretrained||   (how big the change is)
  r90        fraction of directions needed to hold 90% of dW's energy (low = LoRA-friendly)
  stable     stable rank of dW, ||dW||^2 / sigma_1^2 (as a fraction of the max rank)
  top8/32    share of dW's energy in its own top 8 / 32 directions
  overlap32  share of dW's energy inside W_pretrained's top-32 output directions,
             divided by what a random 32-dim subspace would get (1 = no preference, >1 = reuses
             the strong directions, <1 = writes into the weak ones)

No GPU needed. With --eval, also scores "pretrained + rank-r part of dW" (an after-the-fact LoRA)
on the LT test set, per many/medium/few bin (uses the GPU if there is one).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from laser import cifar_lt, data, evaluate, models

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="finetune_lt.py output folder (model.pt, model_epoch*.pt, test/)")
ap.add_argument("--out", required=True)
ap.add_argument("--eval", action="store_true")
ap.add_argument("--ranks", default="4,16,64")
ap.add_argument("--split", default="select")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--fake", action="store_true", help="pretrained = random init (tests only)")
a = ap.parse_args()
root = Path(a.dir).expanduser()

ckpts = sorted(root.glob("model_epoch*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1])) + [root / "model.pt"]
final = models.load_checkpoint(str(root / "model.pt"), "cpu")
ck0 = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
import timm
if a.fake:
    torch.manual_seed(123)
pre = timm.create_model(ck0["timm_name"], pretrained=not a.fake, num_classes=ck0["num_classes"])
pre_targets = {t.name: t for t in models._vit_targets(pre)}
W0 = {n: t.read(t.parameter().data).float() for n, t in pre_targets.items()}
U0 = {n: torch.linalg.svd(W, full_matrices=False)[0] for n, W in W0.items()}


def stats(name, dW):
    W = W0[name]
    m, n = dW.shape
    S = torch.linalg.svdvals(dW)
    e = S ** 2
    cum = torch.cumsum(e, 0) / e.sum()
    k = 32
    inside = (U0[name][:, :k].T @ dW).pow(2).sum() / e.sum()
    return dict(rel=float(dW.norm() / W.norm()), r90=float((cum < 0.9).sum() + 1) / min(m, n),
                stable=float(e.sum() / e[0] / min(m, n)), top8=float(cum[7]), top32=float(cum[31]),
                overlap32=float(inside / (k / m)))


rows = []
for p in ckpts:
    b = models.load_checkpoint(str(p), "cpu") if p != root / "model.pt" else final
    ep = int(re.findall(r"\d+", p.stem)[-1]) if "epoch" in p.stem else "final"
    for t in b.targets:
        dW = t.read(t.parameter().data).float() - W0[t.name]
        rows.append(dict(ckpt=ep, target=t.name, layer=t.layer, mtype=t.mtype, **stats(t.name, dW)))
    print(f"{p.name}: done", flush=True)
df = pd.DataFrame(rows)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

pd.set_option("display.width", 160)
cols = ["rel", "r90", "stable", "top8", "top32", "overlap32"]
fin = df[df.ckpt == "final"]
print("\nFinal model, mean over blocks, by matrix type:")
print(fin.groupby("mtype")[cols].mean().round(3).to_string())
print("\nFinal model, mean over matrix types, by block:")
print(fin.groupby("layer")[cols].mean().round(3).to_string())
if len(ckpts) > 1:
    print("\nOver training (mean over all 72 matrices):")
    print(df.groupby("ckpt", sort=False)[cols].mean().round(3).to_string())

if a.eval:
    dev = models.pick_device()
    images, labels, meta = data.load_prepared(str(root / "test"), a.split, a.limit)
    bins = np.array([cifar_lt.shot_bin(c) for c in meta["train_counts"]])[labels]

    def score(bundle):
        g = evaluate.GPUImages(images, bundle.preprocess.mean, bundle.preprocess.std, dev)
        ok = evaluate.metrics(evaluate.logits(bundle.model.to(dev), g, a.batch), labels)["pred"] == labels
        bundle.model.cpu()
        return {"all": ok.mean() * 100, **{bn: ok[bins == bn].mean() * 100 for bn in ("many", "medium", "few")}}

    res = [dict(config="fine-tuned (full dW)", **score(final))]
    for r in [int(x) for x in a.ranks.split(",")]:
        b = models.load_checkpoint(str(root / "model.pt"), "cpu")  # head, norms, biases stay fine-tuned
        for t in b.targets:
            param = t.parameter()
            dW = t.read(param.data).float() - W0[t.name]
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
            t.write(param.data, W0[t.name] + (U[:, :r] * S[:r]) @ Vh[:r])
        res.append(dict(config=f"pretrained + rank-{r} dW", **score(b)))
    b = models.load_checkpoint(str(root / "model.pt"), "cpu")
    for t in b.targets:
        t.write(t.parameter().data, W0[t.name])
    res.append(dict(config="pretrained matrices, new head only", **score(b)))
    ev = pd.DataFrame(res)
    ev.to_csv(out.with_name(out.stem + "_eval.csv"), index=False)
    print("\nAccuracy (%) with the fine-tuning change kept only at rank r (all 72 matrices):")
    print(ev.round(1).to_string(index=False))
print("DELTA_OK", out)
