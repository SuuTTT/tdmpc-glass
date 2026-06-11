#!/usr/bin/env python3
"""Persist the jumpy-vs-vanilla Franka anchor to JSON (fills TBD-persist-anchor-json).

Reads every per-seed CSV for the phasei27 jum/van arms (incl. ti27c continuation
runs, which share the PROBE_ID) from exp/tdmpc_glass/remote_mirror/, computes
per-seed peak and final (mean of mppi evals at step >= FINAL_FROM), aggregates
per (task, arm) with a paired-free bootstrap CI over seed means, and writes ONE
JSON: exp/tdmpc_glass/mechcheck/anchor_jumpy_vs_vanilla.json.

Dedup rule: the same (probe, task, seed) can appear under several box mirrors
(rsync mirrors every box). Keep the copy with the most rows (longest run).

Run on the control box (numpy only): python3 scripts/aggregate_anchor.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MIRROR = ROOT / "exp" / "tdmpc_glass" / "remote_mirror"
OUT = ROOT / "exp" / "tdmpc_glass" / "mechcheck" / "anchor_jumpy_vs_vanilla.json"

TASKS = ["PandaPickCube", "PandaPickCubeOrientation", "PandaOpenCabinet"]
ARMS = {"jum": "phasei27_jum", "van": "phasei27_van"}
FINAL_FROM = 400_000
N_BOOT = 10_000
SEED = 0


def load_seed_csv(path: Path):
    rows = list(csv.DictReader(open(path)))
    pts = [
        (int(float(r["step"])), float(r["reward"]))
        for r in rows
        if r.get("eval_type") == "mppi" and r.get("reward") not in (None, "")
    ]
    if not pts:
        return None
    pts.sort()
    peak = max(v for _, v in pts)
    fin = [v for st, v in pts if st >= FINAL_FROM]
    if not fin:  # run did not reach the final window -> exclude from final stats
        return {"rows": len(pts), "peak": peak, "final": None, "max_step": pts[-1][0]}
    return {"rows": len(pts), "peak": peak, "final": float(np.mean(fin)), "max_step": pts[-1][0]}


def collect(task: str, probe: str):
    """Dedup (seed)->best csv across box mirrors; return per-seed dicts."""
    best = {}
    for f in MIRROR.glob(f"*/{task}_{probe}_{task}_*/seed_*.csv"):
        if f.name.endswith("_diag.csv"):
            continue
        m = re.match(r"seed_(\d+)\.csv", f.name)
        if not m:
            continue
        s = int(m.group(1))
        d = load_seed_csv(f)
        if d is None:
            continue
        if s not in best or d["rows"] > best[s]["rows"]:
            best[s] = {**d, "file": str(f.relative_to(ROOT)), "seed": s}
    return [best[s] for s in sorted(best)]


def boot_ci(vals, rng):
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return [None, None]
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    means = v[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    rng = np.random.default_rng(SEED)
    out = {"final_from_step": FINAL_FROM, "n_boot": N_BOOT, "tasks": {}}
    for task in TASKS:
        per_arm = {}
        for arm, probe in ARMS.items():
            seeds = collect(task, probe)
            finals = [d["final"] for d in seeds if d["final"] is not None]
            peaks = [d["peak"] for d in seeds]
            per_arm[arm] = {
                "probe_id": probe,
                "n_seeds_total": len(seeds),
                "n_seeds_final": len(finals),
                "peak_mean": float(np.mean(peaks)) if peaks else None,
                "final_mean": float(np.mean(finals)) if finals else None,
                "final_per_seed": sorted(round(x, 1) for x in finals),
                "final_ci95": boot_ci(finals, rng),
                "peak_ci95": boot_ci(peaks, rng),
                "seeds": seeds,
            }
        j, v = per_arm["jum"], per_arm["van"]
        rel = None
        if j["final_mean"] and v["final_mean"]:
            rel = (j["final_mean"] - v["final_mean"]) / abs(v["final_mean"])
        # bootstrap CI of the final-mean DIFFERENCE (unpaired, independent seeds)
        diff_ci = [None, None]
        jf = [d["final"] for d in j["seeds"] if d["final"] is not None]
        vf = [d["final"] for d in v["seeds"] if d["final"] is not None]
        if len(jf) >= 2 and len(vf) >= 2:
            ji = rng.integers(0, len(jf), size=(N_BOOT, len(jf)))
            vi = rng.integers(0, len(vf), size=(N_BOOT, len(vf)))
            diffs = np.asarray(jf)[ji].mean(1) - np.asarray(vf)[vi].mean(1)
            diff_ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        out["tasks"][task] = {
            **per_arm,
            "final_rel_gain_jum_over_van": rel,
            "final_diff_ci95": diff_ci,
            "ci_separated_final": bool(diff_ci[0] is not None and diff_ci[0] > 0),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")
    for task, d in out["tasks"].items():
        print(
            f"{task}: jum final={d['jum']['final_mean']:.0f} (n={d['jum']['n_seeds_final']}) "
            f"van final={d['van']['final_mean']:.0f} (n={d['van']['n_seeds_final']}) "
            f"rel=+{100*d['final_rel_gain_jum_over_van']:.0f}% "
            f"diff CI95={[round(x) for x in d['final_diff_ci95']]} "
            f"CI-sep={d['ci_separated_final']}"
        )


if __name__ == "__main__":
    main()
