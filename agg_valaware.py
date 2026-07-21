#!/usr/bin/env python3
"""Aggregate the value-aware anti-collapse 3-arm comparison.

Reads phase CSVs straight from disk (no fabrication). For each (task, arm):
  return-AUC = np.trapezoid(mppi_return, step) / (step[-1]-step[0])   [trapezoid mean]
  late value_r2 = mean over the last `LATE_N` eval rows.
Reports mean +/- sd over seeds, n, and a crude CI-separation check vs default & unif.
"""
import os, glob, json
import numpy as np
import csv

ROOT = "/root/helios-rl/exp/tdmpc_glass"
TASKS = ["WalkerWalk", "CheetahRun"]
ARMS = ["default", "unif", "valunif"]
SEEDS = [1, 2, 3]
LATE_N = 3  # last 3 eval rows = late window


def load_csv(task, arm, seed):
    d = f"{ROOT}/{task}_unif_dmc_{task}_L16_{arm}_s{seed}"
    fs = glob.glob(f"{d}/seed_*_phase.csv")
    if not fs:
        return None
    step, mppi, vr2 = [], [], []
    with open(fs[0]) as f:
        for row in csv.DictReader(f):
            step.append(float(row["step"]))
            mppi.append(float(row["mppi_return"]))
            vr2.append(float(row["value_r2"]))
    if len(step) < 2:
        return None
    step = np.array(step); mppi = np.array(mppi); vr2 = np.array(vr2)
    auc = float(np.trapezoid(mppi, step) / (step[-1] - step[0]))
    late_vr2 = float(np.mean(vr2[-LATE_N:]))
    return auc, late_vr2, len(step)


def stat(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0)), len(a)


def ci_sep(m1, s1, n1, m2, s2, n2):
    """Crude 95% CI (mean +/- 1.96*sd/sqrt(n)) non-overlap check."""
    h1 = 1.96 * s1 / max(n1, 1) ** 0.5
    h2 = 1.96 * s2 / max(n2, 1) ** 0.5
    return (m1 - h1 > m2 + h2) or (m2 - h2 > m1 + h1)


results = {}
for task in TASKS:
    results[task] = {}
    for arm in ARMS:
        aucs, vr2s = [], []
        for s in SEEDS:
            r = load_csv(task, arm, s)
            if r is None:
                continue
            aucs.append(r[0]); vr2s.append(r[1])
        if not aucs:
            results[task][arm] = None
            continue
        am, asd, an = stat(aucs)
        vm, vsd, vn = stat(vr2s)
        results[task][arm] = dict(auc_mean=am, auc_sd=asd, n=an,
                                  vr2_mean=vm, vr2_sd=vsd,
                                  auc_raw=aucs, vr2_raw=vr2s)

print(json.dumps(results, indent=2))
print("\n=== RETURN-AUC (trapezoid mean of mppi_return), mean +/- sd (n) ===")
for task in TASKS:
    print(f"\n{task}:")
    for arm in ARMS:
        r = results[task][arm]
        if r is None:
            print(f"  {arm:9s}  (no data)"); continue
        print(f"  {arm:9s}  AUC {r['auc_mean']:7.1f} +/- {r['auc_sd']:5.1f} (n={r['n']})   "
              f"late_value_r2 {r['vr2_mean']:.3f} +/- {r['vr2_sd']:.3f}")
    d = results[task]["default"]; u = results[task]["unif"]; v = results[task]["valunif"]
    if all(x is not None for x in (d, u, v)):
        vu = ci_sep(v["auc_mean"], v["auc_sd"], v["n"], u["auc_mean"], u["auc_sd"], u["n"])
        vd = ci_sep(v["auc_mean"], v["auc_sd"], v["n"], d["auc_mean"], d["auc_sd"], d["n"])
        beats_unif = v["auc_mean"] > u["auc_mean"]
        ties_or_beats_default = v["auc_mean"] >= d["auc_mean"] or not vd
        print(f"  -> valunif beats unif (mean): {beats_unif} (CI-sep: {vu})")
        print(f"  -> valunif ties-or-beats default (mean): "
              f"{v['auc_mean'] >= d['auc_mean']} (CI-sep from default: {vd})")

json.dump(results, open("/root/helios-rl/exp/unif_dmc/valaware_results.json", "w"), indent=2)
print("\nwrote /root/helios-rl/exp/unif_dmc/valaware_results.json")
