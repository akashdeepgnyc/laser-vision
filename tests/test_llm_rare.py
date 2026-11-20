import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_llm_rare_fake(tmp_path):
    r = subprocess.run([sys.executable, "scripts/llm_rare.py", "--fake", "--n", "60", "--target", "layers.3.mlp_in",
                        "--rho", "0.1", "--batch", "8", "--bins", "3", "--cache", str(tmp_path / "c.json"),
                        "--out", str(tmp_path / "r")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RARE_OK" in r.stdout
    assert len(pd.read_csv(tmp_path / "r_perprompt.csv")) == 60
    assert (tmp_path / "r_bins.png").exists()
