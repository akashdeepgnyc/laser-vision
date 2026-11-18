import json
import subprocess
import sys
from pathlib import Path

import torch

from laser import factorize, models

ROOT = Path(__file__).resolve().parents[1]


def _py(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_factorized_model_matches_cut_model():
    torch.manual_seed(0)
    b = models.load("vit_tiny_test")
    plan = [dict(target=f"blocks.2.attn_{n}", rho=0.05) for n in "qkv"] + \
           [dict(target="blocks.3.mlp_in", rho=0.05), dict(target="blocks.4.attn_q", rho=0.05)]
    factorize.apply_plan(b, plan)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref = b.model(x)
    p0 = factorize.count_params(b.model)
    info = factorize.factorize(b, plan)
    with torch.no_grad():
        out = b.model(x)
    assert info["layers_factorized"] == 2            # qkv of block 2 and fc1 of block 3; block 4 qkv not (v, k uncut)
    assert factorize.count_params(b.model) < p0
    assert torch.allclose(ref, out, atol=1e-4)
    assert isinstance(b.model.blocks[2].attn.qkv, factorize.Factorized)


def test_compose_and_latency(tmp_path):
    out = tmp_path / "lt"
    _py("scripts/finetune_lt.py", "--fake", "--model", "vit_tiny_patch16_224", "--epochs", "1",
        "--batch", "16", "--imbalance", "10", "--out", str(out))
    plan = tmp_path / "res" / "compose.json"
    args = ["scripts/compose.py", "--model", "tiny", "--checkpoint", str(out / "model.pt"),
            "--data", str(out / "test"), "--limit", "16", "--mtypes", "attn_q,attn_k,attn_v,mlp_in",
            "--rhos", "0.05,0.5", "--tol", "100", "--batch", "8", "--report", "--out", str(plan)]
    _py(*args)
    res = json.loads(plan.read_text())
    assert res["cuts"] == res["matrices"] == 48 and all(p["rho"] == 0.05 for p in res["plan"])
    assert "report_acc" in res and res["matrix_params_after"] < res["matrix_params_dense"]
    n_trials = len(plan.with_suffix(".trials.jsonl").read_text().splitlines())
    _py(*args)                                          # resume: no new trials
    assert len(plan.with_suffix(".trials.jsonl").read_text().splitlines()) == n_trials
    lat = tmp_path / "res" / "lat.jsonl"
    _py("scripts/latency.py", "--model", "tiny", "--checkpoint", str(out / "model.pt"), "--plans", str(plan),
        "--int8", "--batches", "1,2", "--cpu-batches", "1", "--iters", "2", "--warmup", "1",
        "--data", str(out / "test"), "--limit", "8", "--out", str(lat))
    rows = [json.loads(l) for l in lat.read_text().splitlines()]
    assert [r["config"] for r in rows] == ["dense", "compose (top)", "dense_cpu", "int8_cpu"]
    assert rows[1]["params"] < rows[0]["params"] and rows[1]["linear_macs"] < rows[0]["linear_macs"]
