"""CIFAR-100-LT: a long-tailed version of CIFAR-100 (Cui et al., 2019).

Class c keeps n_c = round(500 * imbalance^(-c/99)) training images, so class 0 has 500 and
class 99 has 5 at imbalance 100 (about 10,847 images). The test set stays balanced (100 per
class). Classes are grouped the standard way: many-shot (>100 training images),
medium-shot (20-100) and few-shot (<20).
"""
from __future__ import annotations

import numpy as np


def lt_counts(n_classes: int = 100, n_max: int = 500, imbalance: float = 100.0) -> np.ndarray:
    c = np.arange(n_classes)
    return np.round(n_max * imbalance ** (-c / (n_classes - 1))).astype(int)


def lt_indices(targets: np.ndarray, imbalance: float = 100.0, seed: int = 0) -> np.ndarray:
    """Indices of a long-tailed subset of a balanced training set."""
    targets = np.asarray(targets)
    n_classes = int(targets.max()) + 1
    counts = lt_counts(n_classes, int(np.bincount(targets).min()), imbalance)
    rng = np.random.default_rng(seed)
    idx = [rng.permutation(np.flatnonzero(targets == c))[:counts[c]] for c in range(n_classes)]
    return np.sort(np.concatenate(idx))


def shot_bin(train_count: int) -> str:
    return "many" if train_count > 100 else ("medium" if train_count >= 20 else "few")


def load_cifar100(root: str):
    """Returns (train_images uint8 NHWC, train_labels, test_images, test_labels, class_names)."""
    from torchvision.datasets import CIFAR100
    tr = CIFAR100(root, train=True, download=True)
    te = CIFAR100(root, train=False, download=True)
    return tr.data, np.array(tr.targets), te.data, np.array(te.targets), tr.classes


def fake_cifar(n_classes: int = 4, per_class: int = 30, test_per_class: int = 6, seed: int = 0):
    """Tiny random stand-in with the same format, for tests."""
    rng = np.random.default_rng(seed)
    def make(k):
        y = np.repeat(np.arange(n_classes), k)
        x = rng.integers(0, 255, size=(len(y), 32, 32, 3), dtype=np.uint8)
        return x, y
    xtr, ytr = make(per_class)
    xte, yte = make(test_per_class)
    return xtr, ytr, xte, yte, [f"class{i}" for i in range(n_classes)]
