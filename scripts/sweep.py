"""GPU job: evaluate the unedited model and every (matrix, rho, mode) edit. Resumable.

Each run appends one JSON line to --out and saves its per-example predictions to
<out_dir>/preds/<key>.npy (int16), for bootstrap CIs and per-class analysis later.

Example (pilot):
  python scripts/sweep.py --model vit_b16 --data $DATA/prepared/imagenet_val/s224_c0.9_bicubic \
      --split select --limit 5000 --rhos 0.5,0.1,0.01 --modes top --out results/pilot_vit.jsonl
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from laser import cache as lcache
from laser import data, evaluate, models
from laser.edit import edited
from laser.runlog import ResultsFile, environment

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help=f"one of {list(models.MODELS)}, or a name for --checkpoint")
ap.add_argument("--checkpoint", default=None, help="fine-tuned model.pt from finetune_lt.py")
ap.add_argument("--data", required=True, help="prepared dataset dir (from prep_images.py)")
ap.add_argument("--split", default="select", choices=["select", "report", "all"])
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--rhos", default="0.9,0.8,0.6,0.2,0.1,0.05,0.01")
ap.add_argument("--modes", default="top", help="comma list of top,bottom,random")
ap.add_argument("--mtypes", default=None, help="comma list; default all")
ap.add_argument("--layers", default=None, help="comma list of depth indices; default all")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--no-amp", action="store_true", help="full 32-bit precision (slower)")
ap.add_argument("--no-cache", action="store_true", help="always run the full model (no layer reuse)")
ap.add_argument("--cache-gb", type=float, default=4.0, help="max memory for cached layer states")
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = models.pick_device()
env = environment()
print("ENV", env, flush=True)

bundle = models.load_checkpoint(str(Path(a.checkpoint).expanduser()), dev) if a.checkpoint else models.load(a.model, device=dev)
images, labels, meta = data.load_prepared(str(Path(a.data).expanduser()), a.split, a.limit)
if meta["preprocess_key"] != bundle.preprocess.key:
    print(f"WARNING: data prepared with {meta['preprocess_key']}, model expects {bundle.preprocess.key}")
gpu = evaluate.GPUImages(images, bundle.preprocess.mean, bundle.preprocess.std, dev)

results = ResultsFile(a.out)
pred_dir = Path(a.out).parent / "preds"
pred_dir.mkdir(parents=True, exist_ok=True)
base_info = dict(model=a.model, data=a.data, split=a.split, n=len(labels), limit=a.limit, amp=not a.no_amp)
cache = None
if not a.no_cache and lcache.supported(bundle.model):
    need = lcache.BlockCache.bytes_needed(bundle.model, gpu, not a.no_amp) / 1e9
    if need <= a.cache_gb:
        cache = lcache.BlockCache(bundle.model, gpu, a.batch, not a.no_amp)
    print(f"layer cache: {'on' if cache else 'off'} ({need:.1f} GB needed, limit {a.cache_gb} GB)", flush=True)


def run(key, extra, layer=None):
    if key in results.done:
        return
    t0 = time.time()
    if cache is not None and layer is not None:
        lg = cache.logits()
    else:
        lg = evaluate.logits(bundle.model, gpu, a.batch, not a.no_amp)
    m = evaluate.metrics(lg, labels)
    np.save(pred_dir / f"{key}.npy", m.pop("pred"))
    results.append(dict(key=key, **base_info, **extra, **m, seconds=round(time.time() - t0, 2), env=env))
    print(f"{key}  acc={m['acc']:.4f} loss={m['loss']:.4f}", flush=True)


tag = f"{a.model}|{a.split}|{a.limit}"
run(f"{tag}|baseline", dict(target=None, layer=None, mtype=None, rho=1.0, k=None, mode="none", seed=None))

rhos = [float(r) for r in a.rhos.split(",")]
modes = a.modes.split(",")
mtypes = set(a.mtypes.split(",")) if a.mtypes else None
layers = {int(x) for x in a.layers.split(",")} if a.layers else None
targets = sorted((t for t in bundle.targets
                  if (mtypes is None or t.mtype in mtypes) and (layers is None or t.layer in layers)),
                 key=lambda t: t.layer)  # increasing depth, so the layer cache only moves forward
print(f"{len(targets)} targets x {len(rhos)} rhos x {len(modes)} modes", flush=True)

for t in targets:
    for rho in rhos:
        for mode in modes:
            seed = a.seed if mode == "random" else None
            key = f"{tag}|{t.name}|{rho}|{mode}|{seed}"
            if key in results.done:
                continue
            if cache is not None:
                cache.to_layer(t.layer)  # computed BEFORE the edit, from the unedited earlier blocks
            with edited(t, rho, mode, a.seed) as k:
                run(key, dict(target=t.name, layer=t.layer, mtype=t.mtype, rho=rho, k=k, mode=mode, seed=seed),
                    layer=t.layer)

print("SWEEP_OK", a.out)
