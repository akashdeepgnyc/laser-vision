import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_semantics_script_runs(tmp_path):
    r = subprocess.run([sys.executable, "scripts/semantics.py", "--test", "--k", "2", "--out", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for f in ["report.md", "class_directions.csv", "clip_words.csv"]:
        assert (tmp_path / f).exists(), f
    assert "SEMANTICS_OK" in r.stdout
