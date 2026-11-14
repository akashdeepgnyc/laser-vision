"""Decode images ONCE on CPU into a uint8 array, so GPU jobs never wait on JPEG decoding.

Cloud Bursting shuts down GPU instances with low utilization for 20 minutes, so all
decoding happens in a CPU job (scripts/prep_images.py). GPU jobs memory-map the result.

Layout of a prepared dataset directory:
  images.npy  uint8 (N, 3, H, W)   after resize + center crop, before normalization
  labels.npy  int64 (N,)
  meta.json   source, preprocessing recipe, class names
  splits.json {"select": [...], "report": [...]}  fixed, class-stratified halves
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .models import Preprocess


def eval_transform(p: Preprocess):
    interp = InterpolationMode.BICUBIC if p.interpolation == "bicubic" else InterpolationMode.BILINEAR
    return transforms.Compose([
        transforms.Resize(math.floor(p.size / p.crop_pct), interpolation=interp),
        transforms.CenterCrop(p.size),
        transforms.PILToTensor(),  # uint8, no normalization
    ])


def prepare_imagefolder(src: str, out: str, p: Preprocess, workers: int = 8, batch: int = 256,
                        limit: int | None = None) -> None:
    """src: an ImageFolder tree (one subfolder per class, ImageNet wnid order)."""
    ds = datasets.ImageFolder(src, transform=eval_transform(p))
    n = len(ds) if limit is None else min(limit, len(ds))
    if limit is not None:
        ds = torch.utils.data.Subset(ds, list(range(n)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8,
                                       shape=(n, 3, p.size, p.size))
    labels = np.empty(n, dtype=np.int64)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, shuffle=False)
    i = 0
    for x, y in loader:
        images[i:i + len(x)] = x.numpy()
        labels[i:i + len(x)] = y.numpy()
        i += len(x)
    images.flush()
    np.save(out / "labels.npy", labels)
    base = ds.dataset if limit is not None else ds
    meta = dict(source=str(src), n=n, preprocess=p.__dict__, preprocess_key=p.key, classes=base.classes)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    write_splits(out, labels)


def write_splits(out, labels: np.ndarray, seed: int = 0) -> None:
    """Class-stratified 50/50 split: 'select' picks edits, 'report' is touched once at the end."""
    rng = np.random.default_rng(seed)
    select, report = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        h = len(idx) // 2
        select += idx[:h].tolist()
        report += idx[h:].tolist()
    Path(out, "splits.json").write_text(json.dumps(
        {"seed": seed, "select": sorted(select), "report": sorted(report)}))


def load_prepared(path: str, split: str = "all", limit: int | None = None):
    """Returns (images uint8 memmap-backed array, labels int64 array, meta dict)."""
    path = Path(path)
    images = np.load(path / "images.npy", mmap_mode="r")
    labels = np.load(path / "labels.npy")
    meta = json.loads((path / "meta.json").read_text())
    idx = np.arange(len(labels)) if split == "all" else \
        np.array(json.loads((path / "splits.json").read_text())[split])
    if limit is not None:  # fixed random subset, so a pilot isn't biased toward the first classes
        idx = np.sort(np.random.default_rng(0).permutation(idx)[:limit])
    return images[idx], labels[idx], meta
