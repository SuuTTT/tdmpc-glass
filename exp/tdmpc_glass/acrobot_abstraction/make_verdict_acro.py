#!/usr/bin/env python3
"""Aggregate the abstraction-vs-TD-MPC2 AcrobotSwingup comparison into one honest
verdict JSON. Every number read from disk; nothing fabricated."""
import os, sys, glob, json, csv
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
NUM_TIMESTEPS = int(os.environ.get("VERDICT_STEPS", "10000000"))
TDMPC2_PHASE = os.environ.get(
    "TDMPC2_PHASE",
    "/root/tdmpc_glass/helios-rl/exp/tdmpc_glass/AcrobotSwingup_p38_AcrobotSwingup/seed_1_phase.csv")


def load_tdmpc2():
    f = TDMPC2_PHASE
    if not os.path.exists(f):
        return {"error": f"missing {f}"}
    rows = list(csv.DictReader(open(f)))
    steps = [int(float(r["step"])) for r in rows]
    pi = [float(r["pi_return"]) for r in rows]
    mppi = [float(r["mppi_return"]) for r in rows]
    def cross(vals, thr):
        return next((s for s, v in zip(steps, vals) if v >= thr), None)
    return {"source": f, "n_seeds": 1, "max_step": max(steps),
        "pi_final": round(pi[-1], 1), "pi_peak": round(max(pi), 1),
        "mppi_final": round(mppi[-1], 1), "mppi_peak": round(max(mppi), 1),
        "mppi_steps_to_300": cross(mppi, 300), "mppi_steps_to_400": cross(mppi, 400),
        "pi_steps_to_300": cross(pi, 300)}


def load_curve_run(curve_json):
    d = json.load(open(curve_json))
    curve = d["curve"]; labels = [c["step"] for c in curve]; rets = [c["return"] for c in curve]
    true_steps = [int(l) for l in labels]  # checkpoint/log step IS the true env-step
    def cross(thr):
        return next((s for s, v in zip(true_steps, rets) if v >= thr), None)
    return {"json": curve_json, "peak": round(max(rets), 1), "final": round(rets[-1], 1),
            "steps_to_300": cross(300), "steps_to_400": cross(400),
            "curve": [{"true_step": s, "return": round(r, 1)} for s, r in zip(true_steps, rets)]}


def collect(glob_pat):
    out = []
    for cj in sorted(glob.glob(os.path.join(ROOT, glob_pat))):
        try:
            out.append(load_curve_run(cj))
        except Exception as e:
            out.append({"json": cj, "error": str(e)})
    return out


def agg(runs, key):
    vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    if not vals: return None
    return {"mean": round(float(np.mean(vals)), 1), "std": round(float(np.std(vals)), 1),
            "vals": [round(v, 1) for v in vals], "n": len(vals)}


def main():
    ctrl = None
    cdj = os.path.join(ROOT, "controller_default.json")
    if os.path.exists(cdj): ctrl = json.load(open(cdj))
    tdmpc2 = load_tdmpc2()
    res = collect("logs/res_a*_s*/eval_curve.json")
    ppo = collect("logs/ppo_vanilla_s*/eval_curve.json")
    verdict = {"task": "AcrobotSwingup", "protocol": "A (n>=128, 1000-step, true reward)",
        "controller_alone": ctrl, "tdmpc2_part38": tdmpc2,
        "residual_alpha1": {"runs": res, "peak": agg(res, "peak"), "final": agg(res, "final"),
            "steps_to_300": agg(res, "steps_to_300"), "steps_to_400": agg(res, "steps_to_400")},
        "ppo_vanilla": {"runs": ppo, "peak": agg(ppo, "peak"), "final": agg(ppo, "final")}}
    out = os.path.join(ROOT, "VERDICT.json")
    json.dump(verdict, open(out, "w"), indent=2)
    print(json.dumps({
        "controller_alone_mean": ctrl["mean"] if ctrl else None,
        "controller_swingup_fail": f"{ctrl['swingup_fail']}/{ctrl['total']}" if ctrl else None,
        "tdmpc2": {k: tdmpc2.get(k) for k in ("pi_final","pi_peak","mppi_final","mppi_peak")},
        "residual_peak": verdict["residual_alpha1"]["peak"],
        "residual_final": verdict["residual_alpha1"]["final"],
        "ppo_peak": verdict["ppo_vanilla"]["peak"], "ppo_final": verdict["ppo_vanilla"]["final"]},
        indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
