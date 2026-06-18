#!/usr/bin/env python3
"""Aggregate GHM/InFOM OGBench runs harvested to exp/tdmpc_glass/ghm/mirror/.
Reads each run's finetuning_eval.csv, takes PEAK episode.success (the headline metric;
final-step collapses), groups by (agent, env). Prints mean peak +- std over seeds.
No fabrication: numbers read straight from CSVs."""
import csv, glob, os, re, statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR = os.path.join(ROOT, "exp", "tdmpc_glass", "ghm", "mirror")

def peak_success(fp):
    best = float("nan")
    try:
        rows = list(csv.DictReader(open(fp)))
        col = "evaluation/episode.success"
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
        if vals: best = max(vals)
    except Exception:
        pass
    return best

def env_of(name):
    # run dir name like ghm_a12_cube_single_task3_s2 or cube_single_t1_s0_g3gpu0
    m = re.search(r"(antmaze[a-z0-9_]*|cube[a-z0-9_]*)", name)
    return m.group(1) if m else "?"

def agent_of(name):
    return "GHM" if name.startswith("ghm_") else "InFOM"

groups = defaultdict(list)  # (agent, env) -> [peak,...]
n_runs = 0
for fp in glob.glob(os.path.join(MIRROR, "*", "*", "sd*", "finetuning_eval.csv")):
    name = fp.split(os.sep)[-3]
    p = peak_success(fp)
    if p == p:
        groups[(agent_of(name), env_of(name))].append(p)
        n_runs += 1

print(f"{'agent':6} {'env':28} {'n':>3} {'peak_success(mean)':>18} {'std':>6} {'all'}")
for (ag, env), ps in sorted(groups.items()):
    m = st.mean(ps); s = st.pstdev(ps) if len(ps) > 1 else 0.0
    allv = " ".join(f"{x:.2f}" for x in sorted(ps, reverse=True))
    print(f"{ag:6} {env:28} {len(ps):>3} {m:>18.3f} {s:>6.3f}  {allv}")
print(f"\n{n_runs} runs total. Metric = PEAK episode.success (fraction of eval episodes reaching goal).")
print("Expectation: GHM≈InFOM on cube (short-horizon); antmaze is the long-horizon test that matters.")
