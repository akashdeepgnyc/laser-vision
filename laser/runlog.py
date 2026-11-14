"""Append-only JSONL results with run metadata, so sweeps can resume after interruption."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch


def environment() -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    return dict(
        host=platform.node(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        slurm_account=os.environ.get("SLURM_JOB_ACCOUNT"),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available()
        else ("Apple MPS" if torch.backends.mps.is_available() else None),
        torch=torch.__version__,
        git_commit=git("rev-parse", "HEAD"),
        git_dirty=bool(git("status", "--porcelain")),
        time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )


class ResultsFile:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.done.add(json.loads(line)["key"])

    def append(self, row: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.done.add(row["key"])
