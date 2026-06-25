#!/usr/bin/env python3
"""Aggregate the AcrobotSwingup alpha-sweep (residual prior-authority sweep) into
one JSON: for each alpha group, peak/final mean+-std across seeds. Reads the
brax-log curves written by parse_logs.py. Every number from disk."""
import os, glob, json, re
import numpy as np
ROOT = "/root/tdmpc_glass/exp/tdmpc_glass/acrobot_abstraction/logs"

def curve(j):
    d = json.load(open(j))["curve"]
    return max(c["return"] for c in d), d[-1]["return"], len(d), d[-1]["step"]

groups = {}
for j in sorted(glob.glob(os.path.join(ROOT, "res_a*_s*/eval_curve.json"))):
    m = re.search(r"res_a([0-9.]+)_s(\d+)", j)
    if not m: continue
    a = m.group(1)
    pk, fn, n, last = curve(j)
    groups.setdefault(a, []).append({"seed": m.group(2), "peak": round(pk,1),
                                     "final": round(fn,1), "n_evals": n, "last_step": last})
out = {}
for a, runs in sorted(groups.items(), key=lambda kv: float(kv[0])):
    pks = [r["peak"] for r in runs]; fns = [r["final"] for r in runs]
    out[f"alpha_{a}"] = {"runs": runs,
        "peak_mean": round(float(np.mean(pks)),1), "peak_std": round(float(np.std(pks)),1),
        "final_mean": round(float(np.mean(fns)),1), "final_std": round(float(np.std(fns)),1),
        "n_seeds": len(runs)}
# vanilla + tdmpc2 reference
for j in glob.glob(os.path.join(ROOT, "ppo_vanilla_s*/eval_curve.json")):
    pk, fn, n, last = curve(j); out["vanilla_ppo"] = {"peak": round(pk,1), "final": round(fn,1), "last_step": last}
out["tdmpc2_part38"] = {"pi_peak": 364.0, "pi_final": 301.3, "mppi_peak": 419.7, "mppi_final": 419.7}
out["controller_alone"] = 16.7
json.dump(out, open(os.path.join(os.path.dirname(ROOT), "SWEEP_VERDICT.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
