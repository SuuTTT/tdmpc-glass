#!/usr/bin/env python3
"""EXP#13 finalizer: parse curve_*.json into RESULTS.json with peak/final uniform
+ far real success vs the 0.81 PPO ceiling, and a verdict. Reads from FILES only."""
import json, os
DIR = "/root/tdmpc_glass/exp/tdmpc_glass/hl_orient"
ARMS = {
    "orient_a1_s1": "curve_orient_a1_s1.json",
    "orient_a1_s2": "curve_orient_a1_s2.json",
    "topdown_a1_s1": "curve_topdown_a1_s1.json",
}
PPO_CEIL = 0.81


def summarize(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    curve = d.get("curve", [])
    if not curve:
        return None
    final = curve[-1]
    peak = max(curve, key=lambda r: r.get("success_rate", 0))
    peak_far = max(curve, key=lambda r: r.get("success_far", 0))
    cross = next((r["step"] for r in curve if r.get("success_rate", 0) >= 0.66), None)
    return {
        "peak_success": peak.get("success_rate"),
        "peak_success_step": peak.get("step"),
        "peak_success_far": peak.get("success_far"),
        "peak_far_arm_success_far": peak_far.get("success_far"),
        "peak_far_arm_step": peak_far.get("step"),
        "final_success": final.get("success_rate"),
        "final_success_far": final.get("success_far"),
        "final_success_near": final.get("success_near"),
        "steps_to_cross_0.66": cross,
        "n_curve_points": len(curve),
    }


res = json.load(open(os.path.join(DIR, "RESULTS.json")))
runs = {}
for arm, fn in ARMS.items():
    runs[arm] = summarize(os.path.join(DIR, fn))
res["residual_runs"] = runs
res["status"] = "DONE" if all(v for v in runs.values()) else "PARTIAL_eval"

# verdict
o1 = runs.get("orient_a1_s1") or {}
o2 = runs.get("orient_a1_s2") or {}
td = runs.get("topdown_a1_s1") or {}
orient_peaks = [v["peak_success"] for v in (o1, o2) if v and v.get("peak_success") is not None]
orient_far = [v["peak_success_far"] for v in (o1, o2) if v and v.get("peak_success_far") is not None]
best_orient = max(orient_peaks) if orient_peaks else None
best_orient_far = max(orient_far) if orient_far else None
td_peak = td.get("peak_success") if td else None
td_far = td.get("peak_success_far") if td else None

verdict = []
if best_orient is not None:
    verdict.append(f"orient peak success {best_orient} vs topdown {td_peak} vs PPO {PPO_CEIL}")
    if best_orient >= 0.82:
        verdict.append("BEATS 0.81 ceiling (uniform)")
    elif td_peak is not None and best_orient > td_peak + 0.02:
        verdict.append("beats top-down baseline but not the 0.81 ceiling")
    else:
        verdict.append("does NOT beat top-down/0.81 (uniform)")
if best_orient_far is not None:
    verdict.append(f"FAR-reach peak: orient {best_orient_far} vs topdown {td_far}")
res["verdict"] = "; ".join(verdict) if verdict else "EVAL INCOMPLETE"
json.dump(res, open(os.path.join(DIR, "RESULTS.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
