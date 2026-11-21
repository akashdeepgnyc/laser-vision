"""Do LASER's gains on a language model go to RARE facts? (LASER's own claim, tested with real counts.)

For each CounterFact prompt, count how often its subject and object appear together in the model's
training data (the Pile) via the public infini-gram API (index v4_piletrain_llama; "subject AND object"
within 100 tokens). Then compare the unedited model with the best LASER edit, per frequency bin, on the
report half only (the edit was chosen on the select half).

The edit is taken from a llm_laser.py results file (--sweep: best 'top' edit by select accuracy) or
given directly (--target layers.23.mlp_in --rho 0.1). Counts are cached in --cache (resumable).
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from laser.edit import edited
from laser.evaluate import paired_bootstrap
from laser.models import Target, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
ap.add_argument("--n", type=int, default=8000, help="must match the sweep's --n (same prompts and halves)")
ap.add_argument("--sweep", default=None)
ap.add_argument("--target", default=None)
ap.add_argument("--rho", type=float, default=None)
ap.add_argument("--cache", default="results/pile_counts.json")
ap.add_argument("--bins", type=int, default=5)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--fake", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = pick_device()

if a.sweep:
    rows = [json.loads(l) for l in open(a.sweep) if l.strip()]
    best = max((r for r in rows if r["mode"] == "top"), key=lambda r: r["select_acc"])
    a.target, a.rho = best["target"], best["rho"]
print(f"edit: {a.target} rho {a.rho}", flush=True)

if a.fake:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(0)
    model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=256, hidden_size=64, num_hidden_layers=4,
                                             num_attention_heads=4, intermediate_size=256))
    encode = lambda s: list(s.encode("utf-8")); eos = 0
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model)
    encode = lambda s: tok(s, add_special_tokens=False)["input_ids"]; eos = tok.eos_token_id or 0
model = model.eval().to(dev)
if dev != "cpu":
    model = model.half()
layers = model.gpt_neox.layers if hasattr(model, "gpt_neox") else model.transformer.h
i = int(a.target.split(".")[1]); kind = a.target.split(".")[2]
L = layers[i]
mod = {"mlp_in": getattr(L.mlp, "dense_h_to_4h", None) or getattr(L.mlp, "fc_in", None),
       "mlp_out": getattr(L.mlp, "dense_4h_to_h", None) or getattr(L.mlp, "fc_out", None)}[kind]
target = Target(a.target, i, kind, mod)

if a.fake:
    rng = np.random.default_rng(0)
    facts = [(f"Fact {k} is about", f" thing{rng.integers(0, 9)}", f"S{k}") for k in range(a.n)]
else:
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train").shuffle(seed=0).select(range(a.n))
    facts = [(r["prompt"], r["target_true"] if r["target_true"].startswith(" ") else " " + r["target_true"],
              r["subject"]) for r in ds]
enc = []
for p, t, _ in facts:
    pi, fi = encode(p), encode(p + t)
    if fi[:len(pi)] != pi or len(fi) == len(pi):
        fi = pi + encode(t)
    enc.append((fi, len(pi)))
half = len(enc) // 2


@torch.no_grad()
def per_prompt():
    acc, loss = np.zeros(len(enc)), np.zeros(len(enc))
    order = sorted(range(len(enc)), key=lambda k: len(enc[k][0]))
    for s in range(0, len(order), a.batch):
        bi = order[s:s + a.batch]
        Lm = max(len(enc[k][0]) for k in bi)
        ids = torch.full((len(bi), Lm), eos, dtype=torch.long); att = torch.zeros_like(ids)
        for r, k in enumerate(bi):
            ids[r, :len(enc[k][0])] = torch.tensor(enc[k][0]); att[r, :len(enc[k][0])] = 1
        lg_all = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float()
        for r, k in enumerate(bi):
            full, plen = enc[k]
            lg = lg_all[r, torch.arange(plen - 1, len(full) - 1, device=dev)]
            tgt = torch.tensor(full[plen:], device=dev)
            acc[k] = float(lg[0].argmax() == tgt[0])
            loss[k] = float(-F.log_softmax(lg, -1).gather(1, tgt[:, None]).mean())
    return acc, loss


acc0, loss0 = per_prompt()
with edited(target, a.rho, "top", 0):
    acc1, loss1 = per_prompt()
print(f"report half: acc {acc0[half:].mean()*100:.2f} -> {acc1[half:].mean()*100:.2f}", flush=True)

cache_p = Path(a.cache); cache_p.parent.mkdir(parents=True, exist_ok=True)
cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}


def count(q):
    if a.fake:
        return int(abs(hash(q)) % 10000)
    body = json.dumps({"index": "v4_piletrain_llama", "query_type": "count", "query": q,
                       "max_diff_tokens": 100}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request("https://api.infini-gram.io/", data=body,
                                         headers={"Content-Type": "application/json", "Accept": "application/json",
                                                  "User-Agent": "Mozilla/5.0 (Macintosh) laser-vision-research"})
            d = json.loads(urllib.request.urlopen(req, timeout=60).read())
            if "error" in d:
                raise RuntimeError(d["error"])
            return int(d["count"])
        except Exception as e:
            time.sleep(5 * 3 ** attempt)
            err = e
    print("count failed:", q, err, flush=True)
    return None


if count("Eiffel Tower AND Paris") is None:
    raise SystemExit("infini-gram API not reachable (see error above); counts skipped")
queries = [f"{s.strip()} AND {t.strip()}" for _, t, s in facts]
todo = sorted({q for q in queries if q not in cache})
print(f"Pile counts: {len(queries) - sum(q not in cache for q in queries)} cached, {len(todo)} to fetch", flush=True)
for s in range(0, len(todo), 200):
    chunk = todo[s:s + 200]
    with ThreadPoolExecutor(a.workers) as ex:
        for q, c in zip(chunk, ex.map(count, chunk)):
            if c is not None:
                cache[q] = c
    cache_p.write_text(json.dumps(cache))
    print(f"  {min(s + 200, len(todo))}/{len(todo)}", flush=True)
counts = np.array([cache.get(q, np.nan) for q in queries], dtype=float)

df = pd.DataFrame(dict(prompt=[f[0] for f in facts], object=[f[1] for f in facts], subject=[f[2] for f in facts],
                       split=["select"] * half + ["report"] * (len(enc) - half), count=counts,
                       acc0=acc0, acc1=acc1, loss0=loss0, loss1=loss1))
df.to_csv(str(a.out) + "_perprompt.csv", index=False)

rep = df[(df.split == "report") & df["count"].notna()].copy()
rep["bin"] = pd.qcut(np.log1p(rep["count"]), a.bins, labels=False, duplicates="drop")
print(f"\nReport half ({len(rep)} prompts with counts). Edit {a.target} rho {a.rho}. "
      f"Bins = quantiles of Pile co-occurrence count (0 = rarest).")
res = []
for b, g in rep.groupby("bin"):
    d, lo, hi = paired_bootstrap(g.acc0.values.astype(bool), g.acc1.values.astype(bool))
    res.append(dict(bin=int(b), n=len(g), count_min=int(g["count"].min()), count_max=int(g["count"].max()),
                    acc_before=g.acc0.mean() * 100, acc_after=g.acc1.mean() * 100, gain=d * 100,
                    ci_lo=lo * 100, ci_hi=hi * 100, loss_change=(g.loss1 - g.loss0).mean()))
res = pd.DataFrame(res)
res.to_csv(str(a.out) + "_bins.csv", index=False)
pd.set_option("display.width", 160)
print(res.round(2).to_string(index=False))
rho_s = float(np.corrcoef(pd.Series(np.log1p(rep["count"].values)).rank(), pd.Series(rep.acc1.values - rep.acc0.values).rank())[0, 1])
print(f"Spearman(log count, per-prompt accuracy change) = {rho_s:+.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
x = np.arange(len(res))
lab = [f"{r.count_min:,}–{r.count_max:,}" for r in res.itertuples()]
ax[0].bar(x - 0.2, res.acc_before, 0.4, label="unedited", color="#9aa5b1")
ax[0].bar(x + 0.2, res.acc_after, 0.4, label=f"LASER ({a.target}, ρ={a.rho})", color="#2a6f97")
ax[0].set_ylabel("fact recall (top-1, %)"); ax[0].legend(frameon=False, fontsize=8)
ax[1].errorbar(x, res.gain, yerr=[res.gain - res.ci_lo, res.ci_hi - res.gain], fmt="o", color="#c44e52", capsize=3)
ax[1].axhline(0, color="#888", lw=1); ax[1].set_ylabel("gain (points, 95% CI)")
for a_ in ax:
    a_.set_xticks(x); a_.set_xticklabels(lab, rotation=20, fontsize=7)
    a_.set_xlabel("subject–object co-occurrences in the Pile (rare → common)")
fig.suptitle(f"{a.model}: does LASER help rare facts?", fontsize=10)
fig.tight_layout()
fig.savefig(str(a.out) + "_bins.png", dpi=160)
print("RARE_OK", a.out)
