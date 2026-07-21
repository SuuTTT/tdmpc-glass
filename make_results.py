#!/usr/bin/env python3
"""Exp#6 RESULTS builder. Reads the two eval-curve JSONs produced by
eval_residual_curve.py (alpha=1.0, n=256, 1000-step) and emits a VALIDATED
RESULTS.json (net sizes, steps, peak/final real success per seed + across seeds)
with the verdict vs PPO 0.81 (standing bar), PPO-100M 0.832, and #5b 0.79.
Numbers are READ FROM FILES only; nothing fabricated."""
import json, glob, os, sys

BIG = "/root/tdmpc_glass/exp/tdmpc_glass/hl_beatppo_big"
ARMS = {"s1": f"{BIG}/curve_big_a1p0_s1.json",
        "s2": f"{BIG}/curve_big_a1p0_s2.json"}

PPO_BAR = 0.81           # standing PPO peak bar (reference)
PPO_100M = 0.832         # PPO at 100M (PPO_100_RESULT.json peak)
B5B = 0.79               # #5b/#5c abstraction-in-loop alpha=1.0 result

def netcfg(curve_json_dir_suffix):
    g = glob.glob(f"{BIG}/logs/PandaPickCube-*-big_a1p0_{curve_json_dir_suffix}/checkpoints/*/ppo_network_config.json")
    if not g:
        return None
    d = json.load(open(sorted(g)[0]))
    k = d["network_factory_kwargs"]
    return {"policy_hidden_layer_sizes": k["policy_hidden_layer_sizes"],
            "value_hidden_layer_sizes": k["value_hidden_layer_sizes"],
            "observation_size": d["observation_size"]["shape"]}

arms = {}
peaks = []
for name, path in ARMS.items():
    if not os.path.exists(path):
        arms[name] = {"status": "MISSING", "path": path}
        continue
    d = json.load(open(path))
    peak = d["peak_by_success"]
    final = d["final"]
    arms[name] = {
        "ckpt_root": d.get("ckpt_root"),
        "alpha_override": d.get("alpha_schedule", {}).get("override"),
        "net": netcfg(name),
        "peak_by_success": peak,
        "final": final,
        "steps_to_cross_0.66": d.get("steps_to_cross_0.66"),
        "n_curve_points": len(d.get("curve", [])),
    }
    peaks.append(peak["success_rate"])

best_peak = max(peaks) if peaks else None
mean_peak = round(sum(peaks)/len(peaks), 4) if peaks else None

verdict = {}
if best_peak is not None:
    verdict["best_peak_real_success"] = best_peak
    verdict["mean_peak_real_success"] = mean_peak
    verdict["vs_PPO_0.81"] = round(best_peak - PPO_BAR, 4)
    verdict["vs_PPO100M_0.832"] = round(best_peak - PPO_100M, 4)
    verdict["vs_5b_0.79"] = round(best_peak - B5B, 4)
    verdict["beats_PPO_0.81"] = bool(best_peak > PPO_BAR)
    verdict["clears_0.82_bar"] = bool(best_peak > 0.82)
    if best_peak > 0.82:
        verdict["conclusion"] = ("Bigger residual net + longer/keep-best BEATS the 0.82 bar "
                                 "-> the 0.79 result WAS capacity/budget-limited.")
    elif best_peak > PPO_BAR:
        verdict["conclusion"] = ("Bigger net edges past PPO 0.81 but not the 0.82 bar "
                                 "-> mild capacity/budget sensitivity.")
    elif best_peak >= B5B - 0.02:
        verdict["conclusion"] = ("Bigger residual net + longer/keep-best does NOT move peak above "
                                 "~0.79 -> the 0.79 abstraction-in-loop result is NOT capacity/budget-"
                                 "limited (valid negative finding). 0.79 ceiling is intrinsic to the "
                                 "in-loop-residual approach on PandaPickCube, not a small-net artifact.")
    else:
        verdict["conclusion"] = ("Bigger net UNDERPERFORMS #5b 0.79 (capacity hurt / optimization "
                                 "instability at this budget).")

out = {
    "exp": "#6 big-net capacity/budget test for the 0.79 abstraction-in-loop ceiling",
    "task": "PandaPickCube (mujoco_playground, state-based)",
    "metric": "real success = max over 1000-step rollout of metrics[box_target]>=0.9 (grasp-gated); n=256; deterministic policy; alpha=1.0 (RES_ALPHA_FIXED at train, --alpha_override 1.0 at eval)",
    "design": {
        "single_variable": "residual policy/value network capacity + training budget",
        "policy_net": "(512,512,512,512) vs default (32,32,32,32)",
        "value_net": "(512,512,512,512,512) vs default (256,256,256,256,256)",
        "budget_steps_target": 70_000_000,
        "num_evals": 35,
        "keep_best": "eval_residual_curve.py evaluates EVERY checkpoint -> peak = keep-best",
        "env": "ResidualPickCube (exp#5, env-var knobs only; residual_env.py NOT edited)",
    },
    "bars": {"PPO_peak_standing": PPO_BAR, "PPO_100M_peak": PPO_100M, "exp5b_alpha1.0": B5B,
             "controller_alpha0_sanity": 0.0781},
    "arms": arms,
    "verdict": verdict,
}
op = f"{BIG}/RESULTS.json"
json.dump(out, open(op, "w"), indent=2)
print(json.dumps(verdict, indent=2))
print("WROTE", op)
