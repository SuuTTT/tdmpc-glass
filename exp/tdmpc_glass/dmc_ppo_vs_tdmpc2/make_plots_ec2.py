#!/usr/bin/env python3
"""Plots for Part 18 from RESULTS.json. Saves PNGs to docs/images/."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(BASE, "..", "..", "..", "docs", "images"))
os.makedirs(IMG, exist_ok=True)
R = json.load(open(os.path.join(BASE, "RESULTS.json")))

TASKS = ["CheetahRun", "HopperHop", "AcrobotSwingup", "CartpoleSwingupSparse", "HumanoidRun"]
TASKS = [t for t in TASKS if t in R]
TD_C = "#1f77b4"; PP_C = "#d62728"

# ---- 1. return-vs-env-step curves (per task, log-x) ----
n = len(TASKS)
cols = 3; rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 4.2 * rows))
axes = np.array(axes).reshape(-1)
for i, t in enumerate(TASKS):
    ax = axes[i]
    d = R[t]
    fc = d["tdmpc2"]["fresh_curve"]
    if fc:
        xs = [s for s, _ in fc]; ys = [r for _, r in fc]
        ax.plot(xs, ys, "-o", ms=3, color=TD_C, label="TD-MPC2 (1 seed, same box)")
    pc = d["ppo"]["curve"]
    if pc:
        xs = [s for s, _ in pc]; ys = [r for _, r in pc]
        ax.plot(xs, ys, "-s", ms=3, color=PP_C, label=f"PPO (cfg {d['ppo']['num_timesteps_cfg']})")
    se = d["tdmpc2"]["sample_eff_500k"]
    if se:
        ax.axvline(se["step"], color="gray", ls=":", lw=1)
    ax.axhline(d["threshold"], color="green", ls="--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_title(t)
    ax.set_xlabel("env steps (log)")
    ax.set_ylabel("return")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.25)
for j in range(n, len(axes)):
    axes[j].axis("off")
fig.suptitle("TD-MPC2 vs PPO on DMC — return vs env-steps (note log x-axis: TD-MPC2 reaches "
             "comparable return ~100x fewer steps)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
p1 = os.path.join(IMG, "part18_return_vs_step.png")
fig.savefig(p1, dpi=110); plt.close(fig); print("wrote", p1)

# ---- 2. sample-efficiency bar: return@~500k ----
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(TASKS)); w = 0.38
td_se = [R[t]["tdmpc2"]["sample_eff_500k"]["mean"] if R[t]["tdmpc2"]["sample_eff_500k"] else 0 for t in TASKS]
td_err = []
for t in TASKS:
    se = R[t]["tdmpc2"]["sample_eff_500k"]
    if se and se["n_seeds"] > 1:
        td_err.append([se["mean"] - se["min"], se["max"] - se["mean"]])
    else:
        td_err.append([0, 0])
td_err = np.array(td_err).T
pp_se = [R[t]["ppo"]["ret_at_500k"][1] if R[t]["ppo"]["ret_at_500k"] else 0 for t in TASKS]
ax.bar(x - w / 2, td_se, w, yerr=td_err, capsize=4, color=TD_C, label="TD-MPC2 @~500k")
ax.bar(x + w / 2, pp_se, w, color=PP_C, label="PPO @500k")
ax.set_xticks(x); ax.set_xticklabels(TASKS, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("return at ~500k env-steps")
ax.set_title("Sample efficiency: return at ~500k env-steps (TD-MPC2 5-seed mean+-range where available)")
ax.legend(); ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
p2 = os.path.join(IMG, "part18_sample_eff_500k.png")
fig.savefig(p2, dpi=110); plt.close(fig); print("wrote", p2)

# ---- 3. wall-clock to 0.8*peak threshold ----
fig, ax = plt.subplots(figsize=(9, 4.5))
td_w = [R[t]["tdmpc2"]["wall_to_thr_s"] or 0 for t in TASKS]
pp_w = [R[t]["ppo"]["wall_to_thr_s"] or 0 for t in TASKS]
ax.bar(x - w / 2, td_w, w, color=TD_C, label="TD-MPC2 wall-clock to thr")
ax.bar(x + w / 2, pp_w, w, color=PP_C, label="PPO wall-clock to thr")
ax.set_xticks(x); ax.set_xticklabels(TASKS, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("seconds to 80% of better-algo peak (same box)")
ax.set_title("Wall-clock to reach 80%-of-peak threshold (same 3060 box)")
ax.legend(); ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
p3 = os.path.join(IMG, "part18_wallclock_to_thr.png")
fig.savefig(p3, dpi=110); plt.close(fig); print("wrote", p3)
