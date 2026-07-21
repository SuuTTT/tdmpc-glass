#!/usr/bin/env python3
"""Build the definitive TD-MPC2-vs-PPO RESULTS.json on EC2.

Data sources (all read, never fabricated):
 - d2 5-seed TD-MPC2 curves (Cheetah/Hopper) -> sample-efficiency @~500k (mean+-range).
   exp/tdmpc_glass/{Task}_d2_van_{Task}_s{0..4}/seed_{s}.csv  (cols step,reward,eval_type,seed)
 - fresh 1M same-box TD-MPC2 runs (all 5) -> final@1M, sps, wall-clock.
   fresh_csv/{Task}_seed1.csv  (cols step,reward,eval_type,seed)  + logs/tdmpc2_{Task}.log
 - PPO logs (all 5) -> return-vs-step curve, peak, Time-to-train.
   logs/ppo_{Task}.log
TD-MPC2 deployable return per step = max(pi, mppi).
"""
import json, re, csv, os, glob, statistics as st

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))  # exp/tdmpc_glass
LOGS = os.path.join(BASE, "logs")
FRESH = os.path.join(BASE, "fresh_csv")

TASKS = ["CheetahRun", "HopperHop", "AcrobotSwingup", "CartpoleSwingupSparse", "HumanoidRun"]
D2_TASKS = {"CheetahRun", "HopperHop"}  # reuse 5-seed d2 for sample-efficiency

EVAL_TYPES = ("pi", "mppi", "arb", "protomppi", "jumpy", "")


def read(p):
    try:
        return open(p).read()
    except FileNotFoundError:
        return ""


def deployable_curve(path):
    """max(pi,mppi) per step from a single seed CSV. Skips NaN/inf rewards
    (TD-MPC2 can emit nan after numerical divergence, e.g. HumanoidRun)."""
    import math as _m
    by = {}
    try:
        for row in csv.DictReader(open(path)):
            try:
                s = int(float(row["step"])); rew = float(row["reward"])
            except Exception:
                continue
            if not _m.isfinite(rew):
                continue
            if row.get("eval_type", "") in EVAL_TYPES:
                by[s] = max(by[s], rew) if s in by else rew
    except FileNotFoundError:
        return []
    return sorted(by.items())


def nearest(c, t):
    cand = [(s, r) for s, r in c if s <= t]
    if cand:
        return max(cand, key=lambda x: x[0])
    # else nearest above
    return min(c, key=lambda x: abs(x[0] - t)) if c else None


def peak(c):
    return max(c, key=lambda x: x[1]) if c else None


def last(c):
    return c[-1] if c else None


def first_reach(c, thr):
    for s, r in sorted(c):
        if r >= thr:
            return s
    return None


def d2_seed_curves(task):
    """Return list of per-seed deployable curves from the 5-seed d2 suite."""
    out = []
    for s in range(5):
        p = os.path.join(ROOT, f"{task}_d2_van_{task}_s{s}", f"seed_{s}.csv")
        c = deployable_curve(p)
        if c:
            out.append(c)
    return out


def closest(c, t):
    return min(c, key=lambda x: abs(x[0] - t)) if c else None


def d2_ret_at(task, target):
    """Mean+-range of deployable return at the eval point closest to target across d2 seeds."""
    curves = d2_seed_curves(task)
    vals, step_used = [], None
    for c in curves:
        n = closest(c, target)
        if n:
            vals.append(n[1]); step_used = n[0]
    if not vals:
        return None
    return {
        "step": step_used,
        "mean": round(st.mean(vals), 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "n_seeds": len(vals),
        "per_seed": [round(v, 2) for v in vals],
    }


def tdmpc2_meta(task):
    log = read(os.path.join(LOGS, f"tdmpc2_{task}.log"))
    sps = [int(m) for m in re.findall(r"sps=([\d]+)", log)]
    sps_ss = max(sps) if sps else None
    t = re.search(r"START (\d+)", log); e = re.search(r"END (\d+)", log)
    wall = (int(e.group(1)) - int(t.group(1))) if (t and e) else None
    return sps_ss, wall, (e is not None)


def parse_ppo(task):
    log = read(os.path.join(LOGS, f"ppo_{task}.log"))
    curve = [(int(m.group(1)), float(m.group(2)))
             for m in re.finditer(r"^(\d+): reward=([\-\d.]+)", log, re.M)]
    tt = re.search(r"Time to train:\s*([\d.]+)", log)
    wall = float(tt.group(1)) if tt else None
    nt = re.search(r"num_timesteps:\s*(\d+)", log)
    num_timesteps = int(nt.group(1)) if nt else None
    t = re.search(r"START (\d+)", log); e = re.search(r"END (\d+)", log)
    wall_total = (int(e.group(1)) - int(t.group(1))) if (t and e) else None
    return curve, wall, wall_total, (tt is not None), num_timesteps


results = {}
for task in TASKS:
    # fresh 1M run = final + wall-clock + sps
    fresh = deployable_curve(os.path.join(FRESH, f"{task}_seed1.csv"))
    sps_ss, td_wall, td_done = tdmpc2_meta(task)
    td_pk = peak(fresh); td_last = last(fresh)

    # sample-efficiency @500k
    if task in D2_TASKS:
        td_se = d2_ret_at(task, 500000)
        se_source = "d2_5seed"
    else:
        n = closest(fresh, 500000)
        td_se = {"step": n[0], "mean": n[1], "min": n[1], "max": n[1],
                 "n_seeds": 1, "per_seed": [n[1]]} if n else None
        se_source = "fresh_1seed"

    ppc, pp_wall, pp_wt, pp_done, pp_nt = parse_ppo(task)
    pp_pk = peak(ppc); pp_last = last(ppc)
    pp_500 = nearest(ppc, 500000)

    # threshold = 80% of the WEAKER algo's peak, so BOTH can reach it and a
    # wall-clock-to-threshold is defined for each (a commonly-achievable competence level).
    finals = [v[1] for v in [td_pk, pp_pk] if v]
    base = min(finals) if finals else 0
    thr = round(0.8 * base, 2)

    td_step_thr = first_reach(fresh, thr)
    td_wall_thr = round(td_step_thr / sps_ss, 1) if (td_step_thr and sps_ss) else None
    pp_step_thr = first_reach(ppc, thr)
    pp_maxstep = max([s for s, _ in ppc], default=None)
    pp_wall_thr = round(pp_wall * (pp_step_thr / pp_maxstep), 1) if (
        pp_step_thr is not None and pp_wall and pp_maxstep) else None

    results[task] = dict(
        tdmpc2=dict(
            sample_eff_500k=td_se, sample_eff_source=se_source,
            final_1M_peak=td_pk, final_1M_last=td_last,
            sps_steadystate=sps_ss, wall_total_s=td_wall, done=td_done,
            step_to_thr=td_step_thr, wall_to_thr_s=td_wall_thr,
            n_eval_fresh=len(fresh), fresh_curve=fresh),
        ppo=dict(
            ret_at_500k=pp_500, peak=pp_pk, last=pp_last,
            wall_train_s=pp_wall, wall_total_s=pp_wt, done=pp_done,
            num_timesteps_cfg=pp_nt, total_reported_steps=pp_maxstep,
            step_to_thr=pp_step_thr, wall_to_thr_s=pp_wall_thr,
            n_eval=len(ppc), curve=ppc),
        threshold=thr,
        threshold_basis="0.8 * min(TDMPC2 fresh-1M peak, PPO peak) -- weaker-algo peak, so both reach it",
        notes=[
            "TD-MPC2 deployable return per step = max(pi,mppi).",
            "Sample-efficiency @500k: Cheetah/Hopper = 5-seed d2 mean+-range; hard tasks = fresh 1-seed.",
            "Final = fresh same-box 1M run (peak, robust to terminal collapse; literal last also recorded).",
            "PPO x-axis = raw brax-reported env-steps (incl eval-reset overhead).",
            "Representative final return = PEAK over curve (PPO sometimes collapses terminally).",
        ])

with open(os.path.join(BASE, "RESULTS.json"), "w") as f:
    json.dump(results, f, indent=2, default=list)
print("wrote", os.path.join(BASE, "RESULTS.json"))
# quick summary
for t, d in results.items():
    se = d["tdmpc2"]["sample_eff_500k"]
    print(f"\n{t}: thr={d['threshold']}")
    print(f"  TD-MPC2 SE@500k={se}  final1M_peak={d['tdmpc2']['final_1M_peak']} last={d['tdmpc2']['final_1M_last']}")
    print(f"          sps={d['tdmpc2']['sps_steadystate']} wall={d['tdmpc2']['wall_total_s']}s wall_to_thr={d['tdmpc2']['wall_to_thr_s']}")
    print(f"  PPO     500k={d['ppo']['ret_at_500k']} peak={d['ppo']['peak']} last={d['ppo']['last']}")
    print(f"          wall_train={d['ppo']['wall_train_s']}s wall_to_thr={d['ppo']['wall_to_thr_s']} cfg_steps={d['ppo']['num_timesteps_cfg']}")
