import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from laser import cifar_lt

ROOT = Path(__file__).resolve().parents[1]


def test_lt_profile():
    c = cifar_lt.lt_counts()
    assert c[0] == 500 and c[-1] == 5 and 10_700 < c.sum() < 11_000
    assert [cifar_lt.shot_bin(x) for x in (500, 100, 20, 19)] == ["many", "medium", "medium", "few"]
    y = np.repeat(np.arange(100), 500)
    idx = cifar_lt.lt_indices(y)
    assert np.array_equal(np.bincount(y[idx]), c)


def _py(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_finetune_sweep_analyze(tmp_path):
    out = tmp_path / "lt"
    _py("scripts/finetune_lt.py", "--fake", "--model", "vit_tiny_patch16_224", "--epochs", "1",
        "--batch", "16", "--imbalance", "10", "--out", str(out))
    log = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert len(log) == 1 and "acc_few" in log[0]
    meta = json.loads((out / "test" / "meta.json").read_text())
    assert len(meta["train_counts"]) == 4
    res = tmp_path / "res" / "lt.jsonl"
    _py("scripts/sweep.py", "--model", "tiny_lt", "--checkpoint", str(out / "model.pt"), "--data", str(out / "test"),
        "--split", "select", "--layers", "11", "--mtypes", "mlp_in", "--rhos", "0.1", "--modes", "top,prune",
        "--batch", "8", "--out", str(res))
    s = _py("scripts/analyze_lt.py", "--results", str(res), "--data", str(out / "test"))
    assert "ANALYZE_OK" in s
