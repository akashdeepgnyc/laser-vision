"""Calibration control: does a single softmax temperature explain the loss drop from edits?

Fits one temperature T on the 'select' split for the UNEDITED model (logits / T), then compares
with the edits already in a sweep results file. If T alone reaches the same loss and ECE as the
best edits, the edits are mostly fixing overconfidence, not removing harmful directions.

  uv run python scripts/temperature.py --model vit_b16 \
      --data ~/data/laser-vision/prepared/s224_c0.9_bicubic_pc4 --results results/pilot_mac.jsonl
"""
import argparse
import json

import pandas as pd
import torch
import torch.nn.functional as F

from laser import data, evaluate, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="select")
    ap.add_argument("--results", default=None, help="sweep JSONL to compare against")
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()

    dev = models.pick_device()
    b = models.load(a.model, device=dev)
    images, labels, _ = data.load_prepared(a.data, a.split)
    lg = evaluate.logits(b.model, evaluate.GPUImages(images, b.preprocess.mean, b.preprocess.std, dev), a.batch)
    y = torch.from_numpy(labels)

    temps = torch.linspace(0.5, 3.0, 251)
    losses = torch.stack([F.cross_entropy(lg / t, y) for t in temps])
    T = float(temps[losses.argmin()])
    base, cal = evaluate.metrics(lg, labels), evaluate.metrics(lg / T, labels)
    print(f"Unedited model:            acc {base['acc']:.4f}  loss {base['loss']:.4f}  ECE {base['ece']:.4f}")
    print(f"Unedited + temperature {T:.2f}: acc {cal['acc']:.4f}  loss {cal['loss']:.4f}  ECE {cal['ece']:.4f}")

    if a.results:
        d = pd.DataFrame([json.loads(l) for l in open(a.results)])
        d = d[(d["mode"] != "none") & (d["split"] == a.split)]
        best = d.sort_values("loss").groupby("mode").head(1)[["target", "rho", "mode", "acc", "loss", "ece"]]
        print("\nLowest-loss edit of each type:")
        print(best.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\nEdits beating temperature scaling on loss: {(d.loss < cal['loss']).sum()} of {len(d)}")
    print("TEMPERATURE_OK")


if __name__ == "__main__":
    main()
