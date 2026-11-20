import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_llm_laser_fake(tmp_path):
    out = tmp_path / "llm.jsonl"
    args = [sys.executable, "scripts/llm_laser.py", "--fake", "--n", "40", "--rhos", "0.1", "--batch", "8",
            "--modes", "top,shrink,random", "--out", str(out)]
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "LLM_OK" in r.stdout
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1 + 2 * 2 * 3
    r2 = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    assert r2.returncode == 0 and len(out.read_text().splitlines()) == len(rows)
