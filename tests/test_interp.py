import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_interpolate(tmp_path):
    def py(*args):
        r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout
    out = tmp_path / "lt"
    py("scripts/finetune_lt.py", "--fake", "--model", "vit_tiny_patch16_224", "--epochs", "1",
       "--batch", "16", "--imbalance", "10", "--out", str(out))
    s = py("scripts/interpolate.py", "--dir", str(out), "--fake", "--scales", "0.5,1.0", "--taus", "0,1", "--limit", "8",
           "--batch", "8", "--out", str(tmp_path / "i.csv"))
    assert "INTERP_OK" in s
    df = pd.read_csv(tmp_path / "i.csv")
    assert len(df) == 8 and set(df.split) == {"select", "report"}
