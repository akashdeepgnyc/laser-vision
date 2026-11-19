"""Cut MANY matrices at once, greedily, while accuracy stays within a tolerance.

Order: the matrices that single cuts hurt least go first (from a sweep's results, if given;
otherwise attention before MLP). For each matrix, try the hardest cut first (smallest rho)
and keep the first one that leaves accuracy within --tol points of the unedited model.
Rejected cuts are undone. Every trial is logged, so an interrupted run resumes.

Writes --out (the plan + accuracy trajectory) and --out with .trials.jsonl (every trial).
"""
import argparse
import json
import time
from pathlib import Path

from laser import data, evaluate, models
from laser.edit import kept_rank, low_rank

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--data", required=True)
ap.add_argument("--split", default="select")
ap.add_argument("--limit", type=int, default=2000, help="images used to accept/reject cuts")
ap.add_argument("--sweep", default=None, help="single-edit sweep results (JSONL) to order the matrices")
ap.add_argument("--rhos", default="0.01,0.05,0.1,0.2", help="tried hardest first")
ap.add_argument("--mode", default="top", choices=["top", "prune"])
ap.add_argument("--tol", type=float, default=1.0, help="allowed accuracy drop, in points")
ap.add_argument("--mtypes", default=None)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--report", action="store_true", help="also score the final model on the report split")
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = models.pick_device()
load = lambda: (models.load_checkpoint(str(Path(a.checkpoint).expanduser()), dev) if a.checkpoint
                else models.load(a.model, device=dev))
bundle = load()
images, labels, meta = data.load_prepared(str(Path(a.data).expanduser()), a.split, a.limit)
gpu = evaluate.GPUImages(images, bundle.preprocess.mean, bundle.preprocess.std, dev)
acc = lambda: evaluate.metrics(evaluate.logits(bundle.model, gpu, a.batch), labels)["acc"] * 100

targets = [t for t in bundle.targets if a.mtypes is None or t.mtype in a.mtypes.split(",")]
rhos = sorted(float(r) for r in a.rhos.split(","))
if a.sweep:  # least single-cut damage (at the hardest rho) first
    rows = [json.loads(l) for l in open(a.sweep)]
    avail = sorted({r["rho"] for r in rows if r["mode"] == a.mode})  # nearest rho the sweep has
    ref = min(avail, key=lambda x: abs(x - min(rhos))) if avail else None
    damage = {r["target"]: r["acc"] for r in rows if r["mode"] == a.mode and r["rho"] == ref}
    print(f"order: single-cut damage at rho={ref} from {a.sweep}", flush=True)
    targets.sort(key=lambda t: -damage.get(t.name, -1.0))
else:
    targets.sort(key=lambda t: (not t.mtype.startswith("attn"), -t.layer))

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
trials_path = out.with_suffix(".trials.jsonl")
done = {}
if trials_path.exists():
    for l in trials_path.read_text().splitlines():
        r = json.loads(l)
        done[(r["target"], r["rho"])] = r
trials = trials_path.open("a")

base = acc()
print(f"baseline {base:.2f}% on {len(labels)} images; tolerance {a.tol} points; {len(targets)} matrices", flush=True)
plan, traj = [], []
dense = stored = 0
t0 = time.time()
for t in targets:
    param = t.parameter()
    W = t.read(param.data)
    m, n = W.shape
    dense += m * n
    original = param.data.clone()
    chosen = None
    for rho in rhos:
        prev = done.get((t.name, rho))
        if prev is not None and not prev["accepted"]:
            continue
        t.write(param.data, low_rank(W, rho, a.mode))
        if prev is not None:      # resumed: replay the earlier decision without re-evaluating
            chosen, now = rho, prev["acc"]
            break
        now = acc()
        ok = now >= base - a.tol
        row = dict(target=t.name, rho=rho, acc=now, accepted=ok, cuts=len(plan) + ok)
        trials.write(json.dumps(row) + "\n")
        trials.flush()
        if ok:
            chosen = rho
            break
        param.data.copy_(original)
    if chosen is None:
        stored += m * n
        continue
    k = kept_rank((m, n), chosen)
    stored += min(k * (m + n), m * n)
    plan.append(dict(target=t.name, rho=chosen, mode=a.mode, k=k))
    traj.append(dict(cuts=len(plan), target=t.name, rho=chosen, acc=now))
    print(f"[{len(plan):3d}] keep {t.name} at rho={chosen}  acc {now:.2f}%  ({(time.time()-t0)/60:.1f} min)", flush=True)

final = acc()
res = dict(model=a.model, checkpoint=a.checkpoint, mode=a.mode, tol=a.tol, limit=a.limit, split=a.split,
           baseline_acc=base, final_acc=final, cuts=len(plan), matrices=len(targets),
           matrix_params_dense=dense, matrix_params_after=stored, plan=plan, trajectory=traj)
if a.report:  # honest estimate: images never used to choose the cuts
    imgs_r, lab_r, _ = data.load_prepared(str(Path(a.data).expanduser()), "report", None)
    g_r = evaluate.GPUImages(imgs_r, bundle.preprocess.mean, bundle.preprocess.std, dev)
    res["report_acc"] = evaluate.metrics(evaluate.logits(bundle.model, g_r, a.batch), lab_r)["acc"] * 100
    del g_r
    fresh = load()
    g_r = evaluate.GPUImages(imgs_r, fresh.preprocess.mean, fresh.preprocess.std, dev)
    res["report_baseline_acc"] = evaluate.metrics(evaluate.logits(fresh.model, g_r, a.batch), lab_r)["acc"] * 100
out.write_text(json.dumps(res, indent=1))
print(f"{len(plan)}/{len(targets)} matrices cut; edited matrices now {stored/dense:.1%} of their size; "
      f"acc {base:.2f} -> {final:.2f}" + (f"; report {res['report_baseline_acc']:.2f} -> {res['report_acc']:.2f}"
                                          if a.report else ""))
print("COMPOSE_OK", out)
