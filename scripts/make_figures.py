"""All report figures from the result files. Each figure is skipped (with a note) if its inputs are missing.

  uv run python scripts/make_figures.py --out results/figs
"""
import argparse
import json
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/figs")
a = ap.parse_args()
R, OUT = Path(a.results), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito (colour-blind safe), fixed meaning everywhere
C = dict(top="#0072B2", shrink="#E69F00", prune="#009E73", random="#999999", bottom="#CC79A7",
         none="#BBBBBB", la="#D55E00", both="#56B4E9")
NAME = dict(top="LASER (keep strongest)", shrink="shrink (all directions, same size)", prune="prune (same budget)",
            random="random directions", bottom="keep only weakest")
BINS = ["many", "medium", "few"]
BIN_LABEL = {"many": "common\n(>100 imgs)", "medium": "medium\n(20–100)", "few": "rare\n(<20)"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "figure.dpi": 160})
done = []


def fig(name):
    def deco(fn):
        try:
            fn()
            plt.savefig(OUT / f"{name}.png", bbox_inches="tight")
            plt.close("all")
            done.append(name)
            print("made", name)
        except FileNotFoundError as e:
            print(f"skip {name}: missing {Path(e.filename).name if e.filename else e}")
        except Exception:
            print(f"FAILED {name}:"); traceback.print_exc()
        plt.close("all")
        return fn
    return deco


def jl(p):
    return [json.loads(l) for l in open(R / p) if l.strip()]


# 1. Headline contrast: LASER in vision vs language
@fig("fig1_vision_vs_language")
def _():
    rows = jl("sweep_vit_full.jsonl")
    base = next(r for r in rows if r["mode"] == "none")["acc"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1, 1.3]))
    modes = ["top", "prune", "shrink"]
    data = [[(r["acc"] - base) * 100 for r in rows if r["mode"] == m] for m in modes]
    rng = np.random.default_rng(0)
    for i, (m, d) in enumerate(zip(modes, data)):
        ax[0].scatter(i + rng.uniform(-0.25, 0.25, len(d)), d, s=6, color=C[m], alpha=0.6)
        ax[0].plot([i - 0.3, i + 0.3], [max(d)] * 2, color="k", lw=1)
    ax[0].axhline(0, color="#666", lw=1)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["LASER", "prune", "shrink"])
    ax[0].set_ylabel("accuracy change (points)")
    ax[0].set_title(f"Vision (ViT-B/16, ImageNet): {len(data[0]) * 3} single edits\nnone improves accuracy (black line = best)")
    ax[0].set_ylim(-35, 3)
    models = [("410M", "llm_pythia410.jsonl"), ("1.4B", "llm_pythia14_l23.jsonl"), ("2.8B", "llm_pythia28.jsonl")]
    vals = {k: [] for k in ["none", "top", "shrink", "random", "bottom"]}
    for _, f in models:
        rs = jl(f)
        vals["none"].append(next(r for r in rs if r["mode"] == "none")["report_acc"])
        for m in ["top", "shrink", "random", "bottom"]:
            cand = [r for r in rs if r["mode"] == m]
            vals[m].append(max(cand, key=lambda r: r["select_acc"])["report_acc"] if cand else np.nan)
    x = np.arange(len(models)); w = 0.16
    labels = dict(none="unedited (= + temperature)", **NAME)
    for j, m in enumerate(["none", "top", "shrink", "random", "bottom"]):
        v = np.array(vals[m], float)
        if np.all(np.isnan(v)):
            continue
        ax[1].bar(x + (j - 2) * w, v, w, color=C[m], label=labels[m])
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"Pythia-{n}" for n, _ in models])
    ax[1].set_ylabel("fact recall, top-1 (%)")
    ax[1].set_title("Language (CounterFact, held-out half): LASER on the last MLP\nroughly doubles recall; controls do not")
    ax[1].legend(frameon=False, fontsize=7, ncol=2)


# 2. ViT heatmap: where cutting hurts
@fig("fig2_vit_heatmap")
def _():
    rows = jl("sweep_vit_full.jsonl")
    base = next(r for r in rows if r["mode"] == "none")["acc"]
    df = pd.DataFrame([r for r in rows if r["mode"] != "none" and r["rho"] == 0.01])
    df["d"] = (df.acc - base) * 100
    order = ["attn_q", "attn_k", "attn_v", "attn_out", "mlp_in", "mlp_out"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 2.8), sharey=True)
    for k, m in enumerate(["top", "prune", "shrink"]):
        p = df[df["mode"] == m].pivot_table(index="mtype", columns="layer", values="d").reindex(order)
        im = ax[k].imshow(p.values, cmap="RdBu", vmin=-30, vmax=30, aspect="auto")
        ax[k].set_title(f"{['LASER', 'prune', 'shrink'][k]}, keep 1%")
        ax[k].set_xticks(range(12)); ax[k].set_xlabel("block")
        ax[k].set_yticks(range(6)); ax[k].set_yticklabels(order)
    fig.colorbar(im, ax=ax, shrink=0.8, label="accuracy change (points)")


# 3. CIFAR-LT: cuts hurt rare classes most
@fig("fig3_lt_cuts_by_bin")
def _():
    df = pd.read_csv(R / "sweep_lt_bins.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for k, rho in enumerate([0.1, 0.01]):
        g = df[df.rho == rho].groupby("mode")[[f"d_{b}" for b in BINS]].mean()
        x = np.arange(3); w = 0.26
        for j, m in enumerate(["top", "prune", "shrink"]):
            if m in g.index:
                ax[k].bar(x + (j - 1) * w, g.loc[m].values, w, color=C[m], label=NAME[m])
        ax[k].set_xticks(x); ax[k].set_xticklabels([BIN_LABEL[b] for b in BINS])
        ax[k].set_title(f"keep {int(rho * 100)}% of each MLP matrix (mean over 24 matrices)")
        ax[k].axhline(0, color="#666", lw=1)
    ax[0].set_ylabel("accuracy change (points)")
    ax[0].legend(frameon=False, fontsize=7)


# 4. Fine-tuning update: cut vs shrink vs random
@fig("fig4_dW_controls")
def _():
    ev = pd.read_csv(R / "delta_w_ctrl_eval.csv").set_index("config")
    full = ev.loc["fine-tuned (full dW)"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for k, r in enumerate([16, 64, 256]):
        x = np.arange(4); w = 0.2
        ax[k].bar(x - 1.5 * w, [full[b] for b in ["all"] + BINS], w, color=C["none"], label="fine-tuned (full update)")
        for j, (m, lab) in enumerate([("top", ""), ("shrink", " (shrink)"), ("random", " (random)")]):
            key = f"pretrained + rank-{r} dW{lab}"
            if key in ev.index:
                ax[k].bar(x + (j - 0.5) * w, [ev.loc[key, b] for b in ["all"] + BINS], w, color=C[m],
                          label={"top": f"keep top-{r} directions", "shrink": "shrink to same size", "random": f"random {r} directions"}[m])
        ax[k].set_xticks(x); ax[k].set_xticklabels(["all", "common", "medium", "rare"])
        ax[k].set_title(f"rank {r}")
        ax[k].legend(frameon=False, fontsize=6.5)
    ax[0].set_ylabel("test accuracy (%)")
    fig.suptitle("CIFAR-100-LT: keep the fine-tuning update's strongest directions, shrink all of it, or random", fontsize=10, y=1.05)


# 5. Shrink curve with logit adjustment, 2 seeds
@fig("fig5_shrink_curve")
def _():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
    for k, (model, lbl) in enumerate([("full", "Full fine-tune"), ("lora", "LoRA rank 16")]):
        for s, ls in [(0, "-"), (1, "--")]:
            df = pd.read_csv(R / f"ci_{model}_s{s}.csv")
            rep = df[df.split == "report"]
            for tau, col, nm in [(0.0, C["shrink"], "shrink only"), (1.5, C["both"], "shrink + logit adj. (τ=1.5)")]:
                g = rep[rep.tau == tau].sort_values("scale")
                if len(g):
                    ax[k].plot(g.scale, g["all"], ls, color=col, marker="o", ms=3, label=f"{nm}, all, seed {s}")
                    ax[k].plot(g.scale, g["few"], ls, color=col, marker="^", ms=3, alpha=0.6, label=f"{nm}, rare, seed {s}")
        ax[k].axvline(1.0, color="#999", lw=0.8)
        ax[k].set_xlabel("c in  W = W_pretrained + c·ΔW   (1 = fine-tuned)")
        ax[k].set_title(f"{lbl} (held-out half)")
    ax[0].set_ylabel("test accuracy (%)  ● all classes  ▲ rare")
    h, l = ax[1].get_legend_handles_labels()
    ax[1].legend(h, l, frameon=False, fontsize=6, ncol=2, loc="lower right")


# 6. Mechanism: margins and separation
@fig("fig6_mechanism")
def _():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for model, best, mk in [("full", "0.7", "o"), ("lora", "0.5", "s")]:
        df = pd.read_csv(R / f"mech_{model}_s0_bins.csv")
        df["config"] = df["config"].astype(str)
        for cfg, col, nm in [("1.0", C["none"], "fine-tuned"), (best, C["shrink"], f"shrink c={best}"), ("cut", C["top"], "cut to top-64")]:
            g = df[df.config == cfg].set_index("bin").reindex(BINS)
            if model == "lora" and cfg == "cut":
                continue
            for k, col_ in enumerate(["train_acc", "margin", "sim_gap"]):
                ax[k].plot(range(3), g[col_], marker=mk, color=col, ls="-" if model == "full" else "--",
                           label=f"{'full' if model == 'full' else 'LoRA'}: {nm}")
    for k, t in enumerate(["accuracy on own TRAINING images", "test logit margin (true − best other)",
                           "test feature separation from nearest other class"]):
        ax[k].set_xticks(range(3)); ax[k].set_xticklabels([BIN_LABEL[b] for b in BINS]); ax[k].set_title(t)
    ax[0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Shrinking rebalances (common margins fall, rare classes separate); cutting deletes rare training knowledge", fontsize=10, y=1.05)


# 7. Efficiency: cuts compound
@fig("fig7_compose")
def _():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for f, col, nm in [("compose_top_all02.json", C["top"], "all 72 matrices, keep 20%"),
                       ("compose_top_attn02.json", C["prune"], "attention only, keep 20%"),
                       ("compose_top_tol5.json", C["shrink"], "greedy, ≤5-point budget")]:
        d = json.load(open(R / f))
        t = d["trajectory"]
        ax[0].plot([0] + [p["cuts"] for p in t], [d["baseline_acc"]] + [p["acc"] for p in t], color=col, marker=".", label=nm)
    ax[0].set_xlabel("matrices cut (cumulative)"); ax[0].set_ylabel("ImageNet accuracy (%)")
    ax[0].set_title("Single cuts look cheap; together they compound"); ax[0].legend(frameon=False, fontsize=7)
    lat = jl("latency2.jsonl")
    names = [r["config"].replace("compose_top_", "").replace(" (top)", "") for r in lat if r["device"] != "cpu"]
    rows = [r for r in lat if r["device"] != "cpu"]
    base = rows[0]
    sp = [r["img_per_s_b32"] / base["img_per_s_b32"] for r in rows]
    mac = [r["linear_macs"] / base["linear_macs"] for r in rows]
    x = np.arange(len(rows))
    ax[1].bar(x - 0.2, mac, 0.4, color="#999999", label="compute (relative)")
    ax[1].bar(x + 0.2, sp, 0.4, color=C["top"], label="throughput (relative, batch 32)")
    for i, r in enumerate(rows):
        ax[1].text(i, max(mac[i], sp[i]) + 0.05, f"acc {r['acc']:.0f}%", ha="center", fontsize=7)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=7)
    ax[1].set_title("Real speed-ups need big cuts (Mac GPU)"); ax[1].legend(frameon=False, fontsize=7)


# 8. Rare facts in language (if counts exist)
@fig("fig8_llm_rare_facts")
def _():
    fs = [(n, R / f"rare_{m}_bins.csv") for n, m in [("410M", "pythia410"), ("1.4B", "pythia14"), ("2.8B", "pythia28")]]
    fs = [(n, p) for n, p in fs if p.exists()]
    if not fs:
        raise FileNotFoundError(str(R / "rare_*_bins.csv"))
    fig, ax = plt.subplots(1, len(fs), figsize=(4 * len(fs), 3.2), sharey=True, squeeze=False)
    for k, (n, p) in enumerate(fs):
        d = pd.read_csv(p)
        x = np.arange(len(d))
        ax[0, k].errorbar(x, d.gain, yerr=[d.gain - d.ci_lo, d.ci_hi - d.gain], fmt="o", color=C["top"], capsize=3)
        ax[0, k].axhline(0, color="#666", lw=1)
        ax[0, k].set_xticks(x); ax[0, k].set_xticklabels([f"{a:,}–{b:,}" for a, b in zip(d.count_min, d.count_max)], rotation=20, fontsize=7)
        ax[0, k].set_title(f"Pythia-{n} (n={int(d.n.sum())})"); ax[0, k].set_xlabel("subject–object co-occurrences in the Pile")
    ax[0, 0].set_ylabel("LASER gain in fact recall (points, 95% CI)")


print(f"\n{len(done)} figures in {OUT}: {', '.join(done)}")
print("FIGS_OK")
