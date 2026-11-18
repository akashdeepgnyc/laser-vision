"""Is the cut model actually faster? Time the dense model vs cut models, per batch size.

Configs timed (each a row in --out):
  dense            the unedited model
  <plan> (top)     a compose.py plan with LASER cuts, stored as two thin layers per cut matrix
  <plan> (prune)   a compose.py plan with pruning: same parameter budget, but the zeros still
                   get multiplied (dense GPU kernels), so expect no speed-up
  int8             8-bit dynamic quantization of all Linear layers (CPU only, so compared
                   with the dense model on the CPU)
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from laser import data, evaluate, factorize, models
from laser.cache import autocast

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--plans", default="", help="comma list of compose.py outputs")
ap.add_argument("--int8", action="store_true")
ap.add_argument("--batches", default="1,32,256")
ap.add_argument("--cpu-batches", default="1,32", help="batch sizes for the CPU (int8) comparison")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--no-amp", action="store_true")
ap.add_argument("--data", default=None, help="prepared data: also measure accuracy of each config")
ap.add_argument("--split", default="select")
ap.add_argument("--limit", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
amp = not a.no_amp


def load(dev):
    return (models.load_checkpoint(str(Path(a.checkpoint).expanduser()), dev) if a.checkpoint
            else models.load(a.model, device=dev))


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def time_ms(model, dev, bs, use_amp):
    x = torch.randn(bs, 3, 224, 224, device=dev)
    ts = []
    with autocast(torch.device(dev), use_amp):
        for i in range(a.warmup + a.iters):
            sync(dev)
            t0 = time.perf_counter()
            model(x)
            sync(dev)
            if i >= a.warmup:
                ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


images = labels = None
if a.data:
    images, labels, _ = data.load_prepared(str(Path(a.data).expanduser()), a.split, a.limit)


def accuracy(bundle, dev, use_amp):
    if images is None:
        return None
    g = evaluate.GPUImages(images, bundle.preprocess.mean, bundle.preprocess.std, dev)
    return evaluate.metrics(evaluate.logits(bundle.model, g, 64, use_amp), labels)["acc"] * 100


outp = Path(a.out)
outp.parent.mkdir(parents=True, exist_ok=True)
log = outp.open("a")


def record(name, bundle, dev, batches, use_amp, extra=None):
    x1 = torch.randn(1, 3, 224, 224, device=dev)
    row = dict(config=name, device=dev, amp=use_amp, params=factorize.count_params(bundle.model),
               linear_macs=factorize.linear_macs(bundle.model, x1) if dev != "cpu" or not name.startswith("int8") else None,
               acc=accuracy(bundle, dev, use_amp), **(extra or {}))
    for bs in batches:
        ms = time_ms(bundle.model, dev, bs, use_amp)
        row[f"ms_b{bs}"] = round(ms, 2)
        row[f"img_per_s_b{bs}"] = round(bs / ms * 1000, 1)
    log.write(json.dumps(row) + "\n")
    log.flush()
    print(json.dumps(row), flush=True)


dev = models.pick_device()
batches = [int(b) for b in a.batches.split(",")]
record("dense", load(dev), dev, batches, amp)
for p in filter(None, a.plans.split(",")):
    spec = json.loads(Path(p).read_text())
    b = load(dev)
    factorize.apply_plan(b, spec["plan"])
    info = factorize.factorize(b, spec["plan"]) if spec["mode"] == "top" else {}
    record(f"{Path(p).stem} ({spec['mode']})", b, dev, batches, amp, dict(plan=p, cuts=spec["cuts"], **info))

if a.int8:
    eng = torch.backends.quantized.supported_engines
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in eng else eng[-1]
    cpu_b = [int(b) for b in a.cpu_batches.split(",")]
    record("dense_cpu", load("cpu"), "cpu", cpu_b, False)
    b = load("cpu")
    b.model = torch.ao.quantization.quantize_dynamic(b.model, {nn.Linear}, dtype=torch.qint8)
    record("int8_cpu", b, "cpu", cpu_b, False)
print("LATENCY_OK", outp)
