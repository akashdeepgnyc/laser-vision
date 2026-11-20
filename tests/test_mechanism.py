import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_mechanism(tmp_path):
    def py(*args):
        r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout
    out = tmp_path / "lt"
    py("scripts/finetune_lt.py", "--fake", "--model", "vit_tiny_patch16_224", "--epochs", "1",
       "--batch", "16", "--imbalance", "10", "--out", str(out))
    s = py("scripts/mechanism.py", "--dir", str(out), "--fake", "--scales", "1.0,0.5", "--best", "0.5",
           "--cut-rank", "4", "--batch", "16", "--out", str(tmp_path / "m"))
    assert "MECH_OK" in s
    df = pd.read_csv(tmp_path / "m_perclass.csv")
    assert set(df.config) == {"1.0", "0.5", "cut"} and len(df) == 3 * 4
    assert (tmp_path / "m_flips.png").exists()
