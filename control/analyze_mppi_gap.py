#!/usr/bin/env python3
"""
analyze_mppi_gap.py — Iteration 11 (B1): quantify the MPPI-vs-policy gap.

TD-MPC-Glass eval CSVs log two rows per step: eval_type=pi (actor policy) and
eval_type=mppi (planner). A recurring Iteration 8-10 observation is that MPPI is
sometimes WORSE than the actor — which caps G2 and means test-time planning can
hurt. This tool reads the per-seed CSVs and reports, per phase family and seed:

  gap = mppi_reward - pi_reward   (per eval, paired by step)

  - mean_gap / median_gap        : is MPPI systematically better or worse?
  - frac_mppi_worse              : fraction of evals where MPPI < pi
  - late_gap (last 25% of evals) : does the gap improve once a gait is learned?
  - best_pi / best_mppi          : which controller produced the run's ceiling?
  - winner                       : which controller owns best_any

Read-only. Skips collision side-cars (.collided/.run/.seed/.snapshot) and diag.
"""
from __future__ import annotations
import argparse, csv, glob, re, statistics as st
from collections import defaultdict

SKIP = (".collided", ".run", ".seed", ".snapshot", "_diag", "_arbitration")


def canon_family(path: str) -> tuple[str, int] | None:
    m = re.search(r"HopperHop_(.+?)/seed_(\d+)\.csv$", path)
    if not m:
        return None
    phase, seed = m.group(1), int(m.group(2))
    phase = re.sub(r"_auto_s\d+", "", phase)
    phase = re.sub(r"_[0-9a-f]{6,8}(-dirty)?$", "", phase)
    return phase, seed


def paired_evals(path: str):
    """Yield (step, pi_reward, mppi_reward) for steps having both rows."""
    by_step: dict[int, dict[str, float]] = defaultdict(dict)
    try:
        rows = list(csv.reader(open(path)))
    except Exception:
        return
    if not rows:
        return
    hdr = rows[0]
    try:
        si, ri, ei = hdr.index("step"), hdr.index("reward"), hdr.index("eval_type")
    except ValueError:
        si, ri, ei = 0, 1, 2
    for r in rows[1:]:
        try:
            step = int(float(r[si])); rew = float(r[ri]); et = r[ei].strip()
        except (ValueError, IndexError):
            continue
        if et in ("pi", "mppi"):
            by_step[step][et] = rew
    for step in sorted(by_step):
        d = by_step[step]
        if "pi" in d and "mppi" in d:
            yield step, d["pi"], d["mppi"]


def analyze_seed(path: str) -> dict | None:
    evals = list(paired_evals(path))
    if len(evals) < 2:
        return None
    gaps = [m - p for _, p, m in evals]
    pis = [p for _, p, _ in evals]
    mppis = [m for _, _, m in evals]
    k = max(1, len(gaps) // 4)
    late = gaps[-k:]
    best_pi, best_mppi = max(pis), max(mppis)
    return {
        "n": len(evals),
        "mean_gap": st.mean(gaps),
        "median_gap": st.median(gaps),
        "frac_mppi_worse": sum(1 for g in gaps if g < 0) / len(gaps),
        "late_gap": st.mean(late),
        "best_pi": best_pi,
        "best_mppi": best_mppi,
        "best_any": max(best_pi, best_mppi),
        "winner": "mppi" if best_mppi > best_pi else "pi",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="exp/tdmpc_glass")
    ap.add_argument("--min-seeds", type=int, default=1)
    ap.add_argument("--detail", help="print per-seed rows for this phase substring")
    args = ap.parse_args()

    fam: dict[str, dict[int, dict]] = defaultdict(dict)
    for f in glob.glob(args.root + "/**/seed_*.csv", recursive=True):
        if any(x in f for x in SKIP):
            continue
        cf = canon_family(f)
        if not cf:
            continue
        phase, seed = cf
        res = analyze_seed(f)
        if res is None:
            continue
        prev = fam[phase].get(seed)
        if prev is None or res["best_any"] > prev["best_any"]:
            fam[phase][seed] = res

    rows = []
    for phase, seeds in fam.items():
        if len(seeds) < args.min_seeds:
            continue
        rs = list(seeds.values())
        rows.append({
            "phase": phase, "n": len(rs),
            "mean_gap": st.mean(r["mean_gap"] for r in rs),
            "late_gap": st.mean(r["late_gap"] for r in rs),
            "frac_worse": st.mean(r["frac_mppi_worse"] for r in rs),
            "mppi_wins": sum(1 for r in rs if r["winner"] == "mppi"),
            "mean_best": st.mean(r["best_any"] for r in rs),
        })
    rows.sort(key=lambda r: -r["mean_best"])

    print("MPPI-vs-policy gap by phase family (gap = mppi - pi; negative = MPPI worse)\n")
    print(f"{'phase':46} {'n':>2} {'meanGap':>8} {'lateGap':>8} {'%worse':>7} "
          f"{'mppiWins':>8} {'meanBest':>8}")
    for r in rows:
        print(f"{r['phase'][:46]:46} {r['n']:>2} {r['mean_gap']:>8.1f} {r['late_gap']:>8.1f} "
              f"{r['frac_worse']*100:>6.0f}% {r['mppi_wins']:>3}/{r['n']:<4} {r['mean_best']:>8.1f}")

    if args.detail:
        print(f"\nper-seed detail for phases matching '{args.detail}':")
        for phase, seeds in sorted(fam.items()):
            if args.detail not in phase:
                continue
            for seed, r in sorted(seeds.items()):
                print(f"  {phase} s{seed}: n={r['n']} meanGap={r['mean_gap']:.1f} "
                      f"lateGap={r['late_gap']:.1f} %worse={r['frac_mppi_worse']*100:.0f} "
                      f"best_pi={r['best_pi']:.1f} best_mppi={r['best_mppi']:.1f} "
                      f"winner={r['winner']}")


if __name__ == "__main__":
    main()
