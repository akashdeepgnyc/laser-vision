"""Fine-tune a pretrained ViT on CIFAR-100-LT, then write everything the sweep needs.

  uv run python scripts/finetune_lt.py --out ~/data/laser-vision/cifar100lt_vitb16

Outputs in --out:
  model.pt        fine-tuned weights + how to rebuild the model (load with sweep.py --checkpoint)
  test/           the balanced test set in the sweep's prepared format (uint8, 224 px), with a
                  class-stratified select/report split and per-class training counts in meta.json
  train_log.jsonl one line per epoch: loss and test-select accuracy (overall, many, medium, few)

Images are kept as 32x32 uint8 on the device and augmented + resized on the fly (random crop with
4-pixel padding, horizontal flip, bilinear resize to 224), so no data-loading workers are needed.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from laser import cifar_lt, data, models
from laser.runlog import environment


def resize(x_uint8_nhwc: torch.Tensor, size: int) -> torch.Tensor:
    x = x_uint8_nhwc.permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """x: uint8 NHWC on device. Random 4-px-padded crop and horizontal flip, per image."""
    n = x.shape[0]
    xp = F.pad(x.permute(0, 3, 1, 2).float(), (4, 4, 4, 4), mode="reflect")
    ox = torch.randint(0, 9, (n,), generator=g).tolist()
    oy = torch.randint(0, 9, (n,), generator=g).tolist()
    flip = (torch.rand(n, generator=g) < 0.5).to(x.device)
    out = torch.stack([xp[i, :, oy[i]:oy[i] + 32, ox[i]:ox[i] + 32] for i in range(n)])
    out[flip] = out[flip].flip(-1)
    return out.permute(0, 2, 3, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="~/data/laser-vision/cifar100")
    ap.add_argument("--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k", help="timm name")
    ap.add_argument("--imbalance", type=float, default=100.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32, help="lower to 16 if the Mac runs out of memory")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0, help="also save weights every N epochs (for spectral dynamics)")
    ap.add_argument("--fake", action="store_true", help="tiny random data and random-init model (tests)")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = models.pick_device()
    env = environment()
    print("ENV", env, flush=True)

    # Data
    if a.fake:
        xtr, ytr, xte, yte, names = cifar_lt.fake_cifar()
    else:
        xtr, ytr, xte, yte, names = cifar_lt.load_cifar100(str(Path(a.data_root).expanduser()))
    keep = cifar_lt.lt_indices(ytr, a.imbalance, a.seed)
    xtr, ytr = xtr[keep], ytr[keep]
    n_cls = len(names)
    counts = np.bincount(ytr, minlength=n_cls)
    print(f"{len(ytr)} training images, {n_cls} classes, per-class {counts.max()}..{counts.min()}", flush=True)

    # Model: pretrained ViT with a fresh n_cls-way head
    import timm
    model = timm.create_model(a.model, pretrained=not a.fake, num_classes=n_cls).to(dev)
    cfg = timm.data.resolve_data_config({}, model=model)
    size = cfg["input_size"][-1]
    mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1) * 255
    std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1) * 255

    # Balanced test set in the sweep's prepared format (resized once, same bilinear resize as training)
    test_dir = out / "test"
    test_dir.mkdir(exist_ok=True)
    imgs = np.lib.format.open_memmap(test_dir / "images.npy", mode="w+", dtype=np.uint8, shape=(len(yte), 3, size, size))
    for i in range(0, len(yte), 500):
        imgs[i:i + 500] = resize(torch.from_numpy(xte[i:i + 500]), size).round().clamp(0, 255).byte().numpy()
    imgs.flush()
    np.save(test_dir / "labels.npy", yte.astype(np.int64))
    pre = models.Preprocess(size, 1.0, "bilinear", tuple(cfg["mean"]), tuple(cfg["std"]))
    (test_dir / "meta.json").write_text(json.dumps(dict(
        source="cifar100-test" if not a.fake else "fake", n=len(yte), preprocess=pre.__dict__,
        preprocess_key=pre.key, classes=names, train_counts=counts.tolist(), imbalance=a.imbalance)))
    data.write_splits(test_dir, yte)
    sel = np.array(json.loads((test_dir / "splits.json").read_text())["select"])
    bins = np.array([cifar_lt.shot_bin(c) for c in counts])

    xtr_d = torch.from_numpy(xtr).to(dev)
    ytr_d = torch.from_numpy(ytr).long().to(dev)
    xsel = torch.from_numpy(np.ascontiguousarray(imgs[sel])).to(dev)
    ysel = yte[sel]

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = math.ceil(len(ytr) / a.batch)
    total, warm = a.epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    g = torch.Generator().manual_seed(a.seed)

    def save(path):
        torch.save(dict(timm_name=a.model, num_classes=n_cls, preprocess=pre.__dict__,
                        state_dict={k: v.cpu() for k, v in model.state_dict().items()}), path)

    @torch.no_grad()
    def evaluate():
        model.eval()
        preds = []
        for i in range(0, len(xsel), 128):
            preds.append(model((xsel[i:i + 128].float() - mean) / std).argmax(1).cpu())
        p = torch.cat(preds).numpy()
        ok = p == ysel
        res = dict(acc=float(ok.mean()))
        for b in ("many", "medium", "few"):
            m = bins[ysel] == b
            res[f"acc_{b}"] = float(ok[m].mean()) if m.any() else None
        model.train()
        return res

    log = (out / "train_log.jsonl").open("a")
    model.train()
    for ep in range(a.epochs):
        t0, tot = time.time(), 0.0
        order = torch.randperm(len(ytr), generator=g).to(dev)
        for s in range(steps_per_epoch):
            idx = order[s * a.batch:(s + 1) * a.batch]
            x = (resize(augment(xtr_d[idx], g), size) - mean) / std
            loss = F.cross_entropy(model(x), ytr_d[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        ev = evaluate()
        row = dict(epoch=ep + 1, train_loss=tot / steps_per_epoch, minutes=round((time.time() - t0) / 60, 2), **ev)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)
        if a.save_every and (ep + 1) % a.save_every == 0 and ep + 1 < a.epochs:
            save(out / f"model_epoch{ep + 1}.pt")
    save(out / "model.pt")
    print("FINETUNE_OK", out)


if __name__ == "__main__":
    main()
