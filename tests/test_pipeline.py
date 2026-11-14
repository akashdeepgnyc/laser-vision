"""End to end on a fake ImageFolder: prep -> sweep -> resume."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _fake_imagefolder(root: Path, classes=3, per_class=6):
    rng = np.random.default_rng(0)
    for c in range(classes):
        d = root / f"n{c:08d}"
        d.mkdir(parents=True)
        for i in range(per_class):
            arr = rng.integers(0, 255, size=(rng.integers(200, 300), rng.integers(200, 300), 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i}.JPEG")


def _py(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_prep_sweep_resume(tmp_path):
    _fake_imagefolder(tmp_path / "val")
    _py("scripts/prep_images.py", "--src", str(tmp_path / "val"), "--model", "vit_tiny_test",
        "--out", str(tmp_path / "prep"), "--workers", "0")
    prep = tmp_path / "prep" / "s224_c0.9_bicubic"
    imgs = np.load(prep / "images.npy", mmap_mode="r")
    assert imgs.shape == (18, 3, 224, 224) and imgs.dtype == np.uint8
    splits = json.loads((prep / "splits.json").read_text())
    assert len(splits["select"]) == len(splits["report"]) == 9
    assert not set(splits["select"]) & set(splits["report"])

    out = tmp_path / "res" / "r.jsonl"
    args = ["scripts/sweep.py", "--model", "vit_tiny_test", "--data", str(prep), "--split", "select",
            "--rhos", "0.5,0.1", "--modes", "top,bottom,random", "--layers", "11",
            "--mtypes", "mlp_in,attn_q", "--batch", "4", "--out", str(out)]
    _py(*args)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1 + 2 * 2 * 3                  # baseline + targets x rhos x modes
    assert all(r["n"] == 9 for r in rows)
    assert len(list((out.parent / "preds").glob("*.npy"))) == len(rows)
    _py(*args)                                         # resume: nothing new appended
    assert len(out.read_text().splitlines()) == len(rows)
