"""LASER on a language model, with the controls the original paper did not run.

Model: a Pythia (GPT-NeoX) or GPT-J causal LM from Hugging Face. Task: CounterFact facts
(NeelNanda/counterfact-tracing, 21,919 prompts such as "The Eiffel Tower is located in" -> " Paris").
Prompts are shuffled (seed 0); the first half of --n is 'select', the second half 'report'.

Metrics per prompt: acc = the true object's first token is the model's top-1 next token;
loss = mean negative log-probability of the true object's tokens. Loss is also computed at a grid
of softmax temperatures, so a temperature-scaled unedited model is a baseline for free.

Edits: every (layer, matrix type, rho, mode) in the grid, one at a time, as in LASER.
  top    = LASER (keep the strongest directions)
  shrink = scale the whole matrix to the same size as top (keeps every direction)
  random = project onto a random subspace of the same rank
Choose the best edit on 'select', report it on 'report', next to the temperature baseline.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from laser.edit import edited
from laser.models import Target, pick_device

TEMPS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
ap.add_argument("--n", type=int, default=4000, help="prompts used (half select, half report)")
ap.add_argument("--layers", default=None, help="comma list; default = the second half of the layers")
ap.add_argument("--mtypes", default="mlp_in,mlp_out")
ap.add_argument("--rhos", default="0.1,0.01")
ap.add_argument("--modes", default="top,shrink")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--fp32", action="store_true")
ap.add_argument("--fake", action="store_true", help="tiny random GPT-NeoX + synthetic prompts (tests)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = pick_device()

if a.fake:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(0)
    model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=256, hidden_size=64, num_hidden_layers=4,
                                             num_attention_heads=4, intermediate_size=256))
    encode = lambda s: list(s.encode("utf-8"))
    eos = 0
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model)
    encode = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id or 0
model = model.eval().to(dev)
if not a.fp32 and dev != "cpu":
    model = model.half()


def targets_of(m):
    t = []
    if hasattr(m, "gpt_neox"):
        for i, L in enumerate(m.gpt_neox.layers):
            t += [Target(f"layers.{i}.mlp_in", i, "mlp_in", L.mlp.dense_h_to_4h),
                  Target(f"layers.{i}.mlp_out", i, "mlp_out", L.mlp.dense_4h_to_h),
                  Target(f"layers.{i}.attn_qkv", i, "attn_qkv", L.attention.query_key_value),
                  Target(f"layers.{i}.attn_out", i, "attn_out", L.attention.dense)]
    elif hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        for i, L in enumerate(m.transformer.h):
            t += [Target(f"layers.{i}.mlp_in", i, "mlp_in", L.mlp.fc_in),
                  Target(f"layers.{i}.mlp_out", i, "mlp_out", L.mlp.fc_out),
                  Target(f"layers.{i}.attn_out", i, "attn_out", L.attn.out_proj)]
    else:
        raise ValueError("supported: Pythia (GPT-NeoX) and GPT-J models")
    return t


targets = targets_of(model)
n_layers = max(t.layer for t in targets) + 1

if a.fake:
    rng = np.random.default_rng(0)
    facts = [(f"Fact {i} is about", f" thing{rng.integers(0, 9)}") for i in range(a.n)]
else:
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train").shuffle(seed=0).select(range(a.n))
    facts = [(r["prompt"], r["target_true"] if r["target_true"].startswith(" ") else " " + r["target_true"]) for r in ds]
enc = []
for p, t in facts:
    pi, fi = encode(p), encode(p + t)
    if fi[:len(pi)] != pi or len(fi) == len(pi):
        pi, fi = pi, pi + encode(t)
    enc.append((fi, len(pi)))
half = len(enc) // 2
splits = {"select": np.arange(half), "report": np.arange(half, len(enc))}


@torch.no_grad()
def evaluate(idx):
    acc, loss, lossT = [], [], []
    order = sorted(idx, key=lambda i: len(enc[i][0]))
    for s in range(0, len(order), a.batch):
        bi = order[s:s + a.batch]
        L = max(len(enc[i][0]) for i in bi)
        ids = torch.full((len(bi), L), eos, dtype=torch.long)
        att = torch.zeros((len(bi), L), dtype=torch.long)
        for r, i in enumerate(bi):
            ids[r, :len(enc[i][0])] = torch.tensor(enc[i][0]); att[r, :len(enc[i][0])] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float()
        for r, i in enumerate(bi):
            full, plen = enc[i]
            pos = torch.arange(plen - 1, len(full) - 1, device=dev)
            tgt = torch.tensor(full[plen:], device=dev)
            lg = logits[r, pos]
            acc.append(float(lg[0].argmax() == tgt[0]))
            loss.append(float(-F.log_softmax(lg, -1).gather(1, tgt[:, None]).mean()))
            lossT.append([float(-F.log_softmax(lg / T, -1).gather(1, tgt[:, None]).mean()) for T in TEMPS])
    return np.mean(acc) * 100, np.mean(loss), np.mean(lossT, 0)


outp = Path(a.out)
outp.parent.mkdir(parents=True, exist_ok=True)
done = set()
if outp.exists():
    done = {json.loads(l)["key"] for l in outp.read_text().splitlines() if l.strip()}
log = outp.open("a")


def record(key, extra):
    if key in done:
        return
    t0 = time.time()
    row = dict(key=key, model=a.model, n=a.n, **extra)
    for sp, idx in splits.items():
        acc, loss, lT = evaluate(idx)
        row[f"{sp}_acc"], row[f"{sp}_loss"] = acc, loss
        row[f"{sp}_lossT"] = [round(float(x), 5) for x in lT]
    row["seconds"] = round(time.time() - t0, 1)
    log.write(json.dumps(row) + "\n"); log.flush()
    print(f"{key}: select acc {row['select_acc']:.2f} loss {row['select_loss']:.4f} | "
          f"report acc {row['report_acc']:.2f} loss {row['report_loss']:.4f} ({row['seconds']}s)", flush=True)


record("baseline", dict(target=None, layer=None, mtype=None, rho=1.0, mode="none"))
layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(n_layers // 2, n_layers))
grid = [t for t in targets if t.layer in layers and t.mtype in a.mtypes.split(",")]
print(f"{len(grid)} matrices x {len(a.rhos.split(','))} rhos x {len(a.modes.split(','))} modes", flush=True)
for t in grid:
    for rho in [float(r) for r in a.rhos.split(",")]:
        for mode in a.modes.split(","):
            key = f"{t.name}|{rho}|{mode}"
            if key in done:
                continue
            with edited(t, rho, mode, 0):
                record(key, dict(target=t.name, layer=t.layer, mtype=t.mtype, rho=rho, mode=mode))

rows = [json.loads(l) for l in outp.read_text().splitlines() if l.strip()]
base = next(r for r in rows if r["mode"] == "none")
ti = int(np.argmin(base["select_lossT"]))
print(f"\nUnedited: report acc {base['report_acc']:.2f}, loss {base['report_loss']:.4f}")
print(f"Unedited + temperature {TEMPS[ti]} (chosen on select): report loss {base['report_lossT'][ti]:.4f}, "
      f"acc {base['report_acc']:.2f} (unchanged)")
for mode in a.modes.split(","):
    rs = [r for r in rows if r["mode"] == mode]
    if not rs:
        continue
    for metric, better in (("loss", min), ("acc", max)):
        r = better(rs, key=lambda r: r[f"select_{metric}"])
        print(f"best {mode:6s} by select {metric:4s}: {r['target']} rho {r['rho']} -> report acc {r['report_acc']:.2f}, "
              f"loss {r['report_loss']:.4f}")
print("LLM_OK", outp)
