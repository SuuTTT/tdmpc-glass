#!/usr/bin/env python3
"""Aggregate the Paper-A discrimination matrix into the predictive-criterion table.

Reads harvested *_phase.csv from exp/tdmpc_glass/paperA/mirror/ and the queue map
exp/tdmpc_glass/paperA/state.tsv, then prints, per (task, config):
  final + peak  value_r2 (held-out linear value-decode R²)  and  pi/mppi return,
averaged over seeds. The criterion is PROVEN if:
  (1) distractors LOWER R² + return vs clean   (value-insufficient base exists), AND
  (2) bisim RAISES both back when distractors present  (abstraction restores sufficiency), AND
  (3) bisim is ~NULL when clean                 (redundant where R² already high).
No fabrication: every number is read from a CSV last/best row.
"""
import csv, glob, os, re, statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PA = os.path.join(ROOT, "exp", "tdmpc_glass", "paperA")
STATE = os.path.join(PA, "state.tsv")
MIRROR = os.path.join(PA, "mirror")

# name -> (task, distractor, bisim)
cfg = {}
if os.path.exists(STATE):
    for ln in open(STATE):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 5:
            cfg[p[0]] = (p[1], int(p[2 if False else 3]), float(p[4]))

def rows(fp):
    out = []
    try:
        for r in csv.DictReader(open(fp)):
            out.append(r)
    except Exception:
        pass
    return out

# collect per (name, seed)
res = {}  # (name, seed) -> dict
for fp in glob.glob(os.path.join(MIRROR, "*", "seed_*_phase.csv")):
    name = os.path.basename(os.path.dirname(fp)).rsplit("_s", 1)[0]
    m = re.search(r"seed_(\d+)_phase", os.path.basename(fp))
    seed = int(m.group(1)) if m else -1
    rs = rows(fp)
    if not rs:
        continue
    def col(r, k):
        try: return float(r[k])
        except Exception: return float("nan")
    r2 = [col(r, "value_r2") for r in rs]
    pir = [col(r, "pi_return") for r in rs]
    mpr = [col(r, "mppi_return") for r in rs]
    res[(name, seed)] = dict(
        n=len(rs),
        r2_final=r2[-1], r2_peak=max(r2),
        pi_final=pir[-1], pi_peak=max(pir),
        mppi_final=mpr[-1], mppi_peak=max(mpr),
        laststep=rs[-1].get("step"),
    )

# group by name -> over seeds
byname = defaultdict(list)
for (name, seed), d in res.items():
    byname[name].append(d)

def mean(xs):
    xs = [x for x in xs if x == x]
    return st.mean(xs) if xs else float("nan")

print(f"{'config':12} {'task':11} {'dist':>4} {'bisim':>5} {'seeds':>5} {'R2_fin':>7} {'R2_pk':>6} {'pi_fin':>7} {'mppi_fin':>8} {'laststep':>9}")
order = ["cleanNB","cleanB","distNB","distB","distNB64","distB64","distNB128","distB128",
         "wcleanNB","wcleanB","wdistNB","wdistB"]
for name in order:
    if name not in byname:
        continue
    ds = byname[name]
    task, dist, bisim = cfg.get(name, ("?", -1, -1))
    print(f"{name:12} {task:11} {dist:>4} {bisim:>5} {len(ds):>5} "
          f"{mean(d['r2_final'] for d in ds):>7.3f} {mean(d['r2_peak'] for d in ds):>6.3f} "
          f"{mean(d['pi_final'] for d in ds):>7.1f} {mean(d['mppi_final'] for d in ds):>8.1f} "
          f"{str(ds[0].get('laststep')):>9}")

print("\n--- CRITERION CHECK (CheetahRun, final, mean over seeds) ---")
def g(name, key):
    return mean(d[key] for d in byname.get(name, []))
for tag, nb, b in [("clean(d=0)", "cleanNB", "cleanB"),
                   ("dist d=32", "distNB", "distB"),
                   ("dist d=64", "distNB64", "distB64"),
                   ("dist d=128", "distNB128", "distB128")]:
    print(f"{tag:11}  R2: {g(nb,'r2_final'):.3f} -> {g(b,'r2_final'):.3f} (bisim)   "
          f"return(mppi): {g(nb,'mppi_final'):.1f} -> {g(b,'mppi_final'):.1f}")
print("\nProof needs: clean R2 high & bisim null there; distractor R2 LOW & bisim RAISES R2 AND return.")
