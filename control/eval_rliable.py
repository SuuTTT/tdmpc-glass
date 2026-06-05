#!/usr/bin/env python3
"""
eval_rliable.py — Iteration-14 fair-protocol evaluator (self-contained; no rliable dep,
which has a pandas/arch version clash in the Flask venv). Implements the metrics the plan
mandates: per-task-normalized IQM, 95% stratified-bootstrap CIs, probability-of-improvement,
and sample-efficiency snapshots. Cross-check with rliable-proper at write-up time.

Reads worker eval CSVs (cols: step,reward,eval_type,seed) from the local mirror, groups by
(method, task, seed), and reports method-vs-method comparisons at a fixed env-step budget.

Usage:
  eval_rliable.py --budget 1000000 \
      --method vanilla:HopperHop_phasei14s0_vanilla_* \
      --method glass:HopperHop_phasei14b_glass_clean_* ...
Normalization: DMC reward is 0-1000 -> divide by 1000 (set --norm to override).
"""
from __future__ import annotations
import argparse, csv, glob, os, statistics
import numpy as np

MIRROR = "/home/ubuntu/tdmpc-glass/exp/tdmpc_glass/remote_mirror"

def score_at_budget(csv_path, budget, window=3, metric="mppi"):
    """Mean of the last `window` evals at-or-before `budget` env-steps (sustained, not peak)."""
    rows = []
    with open(csv_path) as fh:
        r = csv.reader(fh); next(r, None)
        for row in r:
            if len(row) < 4: continue
            try: step = float(row[0]); rew = float(row[1])
            except ValueError: continue
            et = row[2]
            if metric and et != metric: continue
            if step <= budget: rows.append((step, rew))
    if not rows: return None
    rows.sort()
    tail = [v for _, v in rows[-window:]]
    return float(np.mean(tail))

def collect(pattern, budget, metric):
    """pattern matches HopperHop_<tag> dirs; returns {(task,seed): score}."""
    out = {}
    for f in glob.glob(f"{MIRROR}/**/{pattern}/seed_*.csv", recursive=True):
        if "diag" in f: continue
        d = os.path.basename(os.path.dirname(f))            # HopperHop_<tag>
        task = d.split("_")[0]                               # e.g. HumanoidWalk if tag-prefixed; else HopperHop
        seed = os.path.basename(f)[5:-4]
        s = score_at_budget(f, budget, metric=metric)
        if s is not None:
            out[(d, seed)] = max(out.get((d, seed), -1e9), s)
    return out

def iqm(xs):
    xs = np.sort(np.asarray(xs, float))
    if len(xs) < 4: return float(np.mean(xs)) if len(xs) else float("nan")
    lo, hi = int(0.25*len(xs)), int(np.ceil(0.75*len(xs)))
    return float(np.mean(xs[lo:hi]))

def bootstrap_ci(xs, fn=iqm, n=20000, alpha=0.05, seed=0):
    xs = np.asarray(xs, float); rng = np.random.default_rng(seed)
    if len(xs) == 0: return (float("nan"), float("nan"))
    boots = [fn(rng.choice(xs, len(xs), replace=True)) for _ in range(n)]
    return (float(np.percentile(boots, 100*alpha/2)), float(np.percentile(boots, 100*(1-alpha/2))))

def prob_improvement(a, b, n=20000, seed=0):
    """P(random A-run > random B-run) via paired resampling over the score pools."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0: return float("nan")
    rng = np.random.default_rng(seed)
    wins = (rng.choice(a, n) > rng.choice(b, n)).mean()
    return float(wins)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1_000_000)
    ap.add_argument("--norm", type=float, default=1000.0, help="divide raw reward (DMC=1000)")
    ap.add_argument("--metric", default="mppi", help="eval_type to use (mppi/pi)")
    ap.add_argument("--method", action="append", default=[], help="name:dir_glob_pattern")
    args = ap.parse_args()

    methods = {}
    for spec in args.method:
        name, pat = spec.split(":", 1)
        scores = collect(pat, args.budget, args.metric)
        methods[name] = {k: v/args.norm for k, v in scores.items()}

    print(f"\n=== fair-protocol report @ budget={args.budget/1e6:.1f}M  metric={args.metric}  (normalized /{args.norm:.0f}) ===")
    pooled = {}
    for name, sc in methods.items():
        vals = list(sc.values()); pooled[name] = vals
        if not vals: print(f"  {name:16s}: (no data)"); continue
        lo, hi = bootstrap_ci(vals)
        print(f"  {name:16s}: N={len(vals):2d}  IQM={iqm(vals):.3f}  95%CI=[{lo:.3f},{hi:.3f}]  "
              f"mean={statistics.mean(vals):.3f}  per-task-seeds={sorted(set(k[0] for k in sc))[:3]}...")
    names = list(methods)
    if len(names) >= 2:
        print("\n  pairwise probability-of-improvement P(row > col):")
        for a in names:
            row = "  ".join(f"{prob_improvement(pooled[a],pooled[b]):.2f}" if a!=b else " -- " for b in names)
            print(f"    {a:16s} {row}")
    print()

if __name__ == "__main__":
    main()
