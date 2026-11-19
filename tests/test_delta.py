import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _py(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_delta_w(tmp_path):
    out = tmp_path / "lt"
    _py("scripts/finetune_lt.py", "--fake", "--model", "vit_tiny_patch16_224", "--epochs", "2", "--save-every", "1",
        "--batch", "16", "--imbalance", "10", "--out", str(out))
    s = _py("scripts/delta_w.py", "--dir", str(out), "--out", str(tmp_path / "d.csv"), "--fake",
            "--eval", "--ranks", "2", "--limit", "8", "--batch", "8")
    assert "DELTA_OK" in s
    df = pd.read_csv(tmp_path / "d.csv")
    assert len(df) == 2 * 72 and set(df.ckpt.astype(str)) == {"1", "final"}
    assert (df.r90 > 0).all() and (df.r90 <= 1).all()
    ev = pd.read_csv(tmp_path / "d_eval.csv")
    assert len(ev) == 3
