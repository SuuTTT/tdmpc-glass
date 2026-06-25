#!/usr/bin/env python3
"""Parse brax PPO eval rewards from the training .log files into per-run curves.
The 'N: reward=X' lines are brax eval/episode_reward = Protocol A (n=128 eval envs,
1000-step, true task reward) -- the SAME metric TD-MPC2 phase CSV reports. N is the
true env-step. Writes eval_curve.json next to each log for make_verdict_acro.py."""
import os, re, json, glob, sys
ROOT = "/root/tdmpc_glass/exp/tdmpc_glass/acrobot_abstraction/logs"
pat = re.compile(r"^(\d+): reward=([\-\d.]+)")
for log in sorted(glob.glob(os.path.join(ROOT, "*.log"))):
    tag = os.path.basename(log)[:-4]
    curve = []
    for line in open(log):
        m = pat.match(line.strip())
        if m:
            curve.append({"step": int(m.group(1)), "return": float(m.group(2))})
    if not curve:
        continue
    # dedup by step keeping last
    d = {c["step"]: c["return"] for c in curve}
    curve = [{"step": s, "return": d[s]} for s in sorted(d)]
    outdir = os.path.join(ROOT, tag)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "eval_curve.json")
    json.dump({"source_log": log, "metric": "brax eval/episode_reward (Protocol A n=128, 1000-step)",
               "curve": curve}, open(out, "w"), indent=2)
    peak = max(c["return"] for c in curve); fin = curve[-1]["return"]
    print(f"{tag}: {len(curve)} evals, peak={peak:.1f} final={fin:.1f} -> {out}")
