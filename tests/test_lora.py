import subprocess
import sys
from pathlib import Path

import pandas as pd
import timm
import torch

from laser import lora, models

ROOT = Path(__file__).resolve().parents[1]


def _py(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_merge_matches_wrapped():
    torch.manual_seed(0)
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=5).eval()
    keys = set(m.state_dict())
    lora.wrap(m, 4)
    for p in m.parameters():
        if p.requires_grad:
            p.data.normal_(0, 0.02)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = m(x)
    sd = lora.merged_state_dict(m)
    assert set(sd) == keys
    plain = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=5).eval()
    plain.load_state_dict(sd)
    with torch.no_grad():
        assert torch.allclose(plain(x), y, atol=1e-4)


def test_lora_finetune_then_delta(tmp_path):
    out = tmp_path / "lora"
    s = _py("scripts/finetune_lt.py", "--fake", "--model", "vit_tiny_patch16_224", "--epochs", "1", "--lora", "2",
            "--lr", "1e-3", "--batch", "16", "--imbalance", "10", "--out", str(out))
    assert "LoRA rank 2" in s
    models.load_checkpoint(str(out / "model.pt"))
    _py("scripts/delta_w.py", "--dir", str(out), "--out", str(tmp_path / "d.csv"), "--fake",
        "--eval", "--ranks", "2", "--modes", "top,shrink,random", "--limit", "8", "--batch", "8")
    ev = pd.read_csv(tmp_path / "d_eval.csv")
    assert len(ev) == 5 and ev.config.str.contains("random").any()
