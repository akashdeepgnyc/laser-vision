import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from laser import spectrum

ROOT = Path(__file__).resolve().parents[1]


def test_gavish_donoho_recovers_planted_rank():
    g = torch.Generator().manual_seed(0)
    m, n, r = 300, 500, 10
    signal = torch.randn(m, r, generator=g) @ torch.randn(r, n, generator=g) * 0.5
    noise = torch.randn(m, n, generator=g) * 0.05
    st = spectrum.stats(spectrum.singular_values(signal + noise), (m, n))
    assert st["gd_rank"] == r


def test_rank_measures_on_simple_matrices():
    st = spectrum.stats(spectrum.singular_values(torch.eye(50)), (50, 50))
    assert abs(st["eff_rank_frac"] - 1) < 1e-6 and abs(st["stable_rank_frac"] - 1) < 1e-6
    one = torch.outer(torch.randn(40), torch.randn(60))
    st1 = spectrum.stats(spectrum.singular_values(one), (40, 60))
    assert st1["energy@0.01"] > 0.999           # rank 1: top component holds everything
    vals = [st["energy@0.01"], st["energy@0.1"], st["energy@0.5"]]
    assert vals == sorted(vals)


def test_atlas_script_runs(tmp_path):
    r = subprocess.run([sys.executable, "scripts/spectrum_atlas.py", "--models", "vit_tiny_test,resnet18_test",
                        "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for f in ["atlas.csv", "spectra.npz", "summary.csv", "vit_tiny_test_gd_heatmap.png",
              "resnet18_test_eff_rank.png", "resnet18_test_spectra.png"]:
        assert (tmp_path / f).exists(), f
