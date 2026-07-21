#!/usr/bin/env python3
"""Aggregate D2 suite: per-task vanilla(mppi) vs jumpy(jumpy) peak & final + paired bootstrap CI.

Reads exp/tdmpc_glass/d2_{arm}_{task}_s{seed}/seed_{seed}.csv (cols step,reward,eval_type,seed).
Vanilla harvest eval_type=mppi; jumpy harvest eval_type=jumpy. final = mean reward over evals
with step >= FINAL_FROM. Prints a sign table and writes JSON.
"""
import numpy as np, glob, json, os, re
from pathlib import Path

ROOT = Path(os.environ.get("REPO", str(Path(__file__).resolve().parents[1])))
GLASS = ROOT / "exp" / "tdmpc_glass"
FINAL_FROM = int(os.environ.get("FINAL_FROM", "400000"))
RNG = np.random.default_rng(0)
TASKS = os.environ.get("TASKS", "PandaPickCube,PandaPickCubeOrientation,PandaOpenCabinet,"
                        "CheetahRun,HopperHop,CartpoleSwingupSparse").split(",")

def load(arm, task):
    """return dict seed -> (peak, final) using the arm's eval_type."""
    etype = "mppi" if arm == "van" else "jumpy"
    out = {}
    for d in glob.glob(str(GLASS / f"{task}_d2_{arm}_{task}_s*")):
        m = re.search(r"_s(\d+)$", d)
        if not m: continue
        seed = int(m.group(1))
        csvs = glob.glob(os.path.join(d, "seed_*.csv"))
        csvs = [c for c in csvs if "_diag" not in c and "_arbitration" not in c]
        if not csvs: continue
        rows = []
        for line in open(csvs[0]):
            p = line.strip().split(",")
            if len(p) < 4 or p[0] == "step": continue
            if p[2] != etype: continue
            rows.append((int(p[0]), float(p[1])))
        if not rows: continue
        rows.sort()
        steps = np.array([r[0] for r in rows]); rew = np.array([r[1] for r in rows])
        peak = float(rew.max())
        fin = rew[steps >= FINAL_FROM]
        final = float(fin.mean()) if len(fin) else float(rew[-1])
        out[seed] = (peak, final, int(steps.max()))
    return out

def boot_diff(a, b, n=10000):
    a, b = np.array(a), np.array(b)
    d = []
    for _ in range(n):
        d.append(RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean())
    return float(np.mean(d)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

res = {}
print(f"{'task':28s} {'n':>5} {'van_fin':>9} {'jum_fin':>9} {'Δfinal':>9} {'CI95':>20} sign")
for t in TASKS:
    v, j = load("van", t), load("jum", t)
    fin = lambda x: np.isfinite(x[0]) and np.isfinite(x[1])
    seeds = [s for s in sorted(set(v) & set(j)) if fin(v[s]) and fin(j[s])]
    dropped = [s for s in sorted(set(v) & set(j)) if s not in seeds]
    if dropped: print(f"  [{t}] dropped non-finite seeds: {dropped}")
    if len(seeds) < 2:
        print(f"{t:28s} {len(seeds):>5}  (waiting: van={len(v)} jum={len(j)})"); continue
    vf = [v[s][1] for s in seeds]; jf = [j[s][1] for s in seeds]
    vp = [v[s][0] for s in seeds]; jp = [j[s][0] for s in seeds]
    dmean, dci = boot_diff(jf, vf)
    pmean, pci = boot_diff(jp, vp)
    sign = "JUM_WIN" if dci[0] > 0 else ("VAN_WIN" if dci[1] < 0 else "null")
    _den = np.mean(vf)
    pct = (100 * dmean / _den) if abs(_den) > 1.0 else float("nan")  # van~0 → pct meaningless
    print(f"{t:28s} {len(seeds):>5} {np.mean(vf):9.1f} {np.mean(jf):9.1f} {dmean:9.1f} "
          f"[{dci[0]:8.1f},{dci[1]:8.1f}] {sign} ({pct:+.0f}%)")
    res[t] = {"n_seeds": len(seeds), "van_final_mean": float(np.mean(vf)),
              "jum_final_mean": float(np.mean(jf)), "delta_final": dmean, "delta_final_ci95": dci,
              "delta_final_pct": pct, "delta_peak": pmean, "delta_peak_ci95": pci, "sign": sign,
              "van_final_per_seed": vf, "jum_final_per_seed": jf, "max_step": max(v[s][2] for s in seeds)}
out = GLASS / "mechcheck" / "d2_suite_results.json"; out.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out, "w"), indent=2)
print("\nsaved", out)
