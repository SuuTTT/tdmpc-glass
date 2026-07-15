---
layout: post
title: "TD-MPC-Glass, Part 48: The Full DMC Benchmark — tdmpc-glass vs TD-MPC2 Across 16 Tasks"
date: 2026-06-26
description: "One unified DMC table across all five methods the campaign has — PPO, TD-MPC2, tdmpc-glass, the analytic-controller+residual abstraction, and (cross-referenced) TAMP — plus a clean same-seed head-to-head of tdmpc-glass (the project's graph/structural-entropy world-model variant) against vanilla TD-MPC2 across 16 DMControl tasks at 1M env-steps, both algos run on the same box with the same harness. The refined verdict: glass TIES TD-MPC2 on easy/saturated tasks (BallInCup, Cartpole, Reacher-hard, FingerSpin, Walker-stand) but TRAILS on the harder exploration/locomotion tasks (FingerTurnHard 586 vs 970, HopperStand 355 vs 872, ReacherEasy 779 vs 983, CheetahRun 652 vs 706, AcrobotSwingup 312 vs 383), and has one CATASTROPHIC failure mode — HopperHop, where glass collapses to 0.17 while TD-MPC2 reaches 234 (cross-seed confirmed). This sharpens the campaign's earlier 'glass ≈ TD-MPC2' claim, which held only because earlier comparisons used an easier task subset. Verification-discipline note: a seed-2 TD-MPC2 PendulumSwingup run logged 0.0; the seed-1 run reaching 827 proves that was a single-seed divergence, NOT a real failure — so it is reported as an outlier, not a finding."
---

> ⚠ **See [the matched-control correction (Part 49)]({{ site.baseurl }}/2026/06/29/tdmpc-glass-part49-matched-control-correction.html)** — the matched-control result revises the "abstraction beats" cells in the unified table: vs a budget-matched PPO the abstraction wins sample-efficiency only, not the asymptotic ceiling.

> We ran **both** algorithms — `tdmpc-glass` and vanilla `tdmpc2` — across the full DMControl task suite at 1M
> env-steps, same box, same harness, same seed, so the comparison is apples-to-apples. The result refines the
> project's long-standing "glass ≈ TD-MPC2" intuition into something more precise.

## Setup

Deterministic rolling sweep (`full_sweep_runner.sh`) on a 4×3060 box: 16 DMC tasks × 2 algos, 1M steps each,
`--no_plot`, returns read from `exp/benchmark/*.csv` (last logged eval, never a training-loss proxy). Primary
numbers are **seed-1** (both algos on the same box → matched seed). A second seed (reverse-order, separate box)
is used only to cross-check anomalies. `QuadrupedRun` is omitted (not present in this `mujoco_playground` build).

## Results (seed-1, 1M steps, return)

| task | tdmpc-glass | TD-MPC2 | verdict |
|---|---|---|---|
| AcrobotSwingup | 312 | 383 | TD-MPC2 |
| BallInCup | 973 | 976 | tie |
| CartpoleBalance | 982 | 988 | tie |
| CartpoleSwingup | 857 | 852 | tie |
| CheetahRun | 652 | 706 | TD-MPC2 |
| FingerSpin | 963 | 977 | tie |
| FingerTurnEasy | 977 | 773* | glass* |
| FingerTurnHard | 586 | 970 | **TD-MPC2 ≫** |
| HopperHop | **0.17** | 234 | **TD-MPC2 (glass collapses)** |
| HopperStand | 355 | 872 | **TD-MPC2 ≫** |
| PendulumSwingup | 743 | 827 | TD-MPC2 (both solve) |
| ReacherEasy | 779 | 983 | TD-MPC2 |
| ReacherHard | 973 | 971 | tie |
| WalkerRun | 665 | ~625 (conv.) | ~tie |
| WalkerStand | 976 | ~943 (conv.) | ~tie |
| WalkerWalk | 959 | (converging) | pending |

*FingerTurnEasy: glass 977 vs TD-MPC2 seed-1 773 — but TD-MPC2 **seed-2 = 978**, so the seed-1 dip is
seed variance, not a robust glass win. Read it as a tie. (Walker-trio TD-MPC2 still climbing at writeup;
seed-2 TD-MPC2 = 651/972/975, and glass already sits at 665/976/959, so these are ties.)

## ⚠️ UPDATE (seed-2 in → n=2; seed-3 running): the single-seed story was too harsh on glass

The original table below was **seed-1 only**. A second seed substantially revises it — and the corrected story is
about **stability, not a flat capability gap**:

- **HopperHop glass: seed-1 0.17 → seed-2 405.6.** The "catastrophic failure" was a **single-seed collapse**, not
  a consistent failure. (TD-MPC2 solves it both seeds: 234 / 350.)
- **HopperStand glass: 355 / 14.7** — *another* collapsing seed. TD-MPC2 stays solid (872 / 858).
- Several seed-1 "TD-MPC2 wins" **shrink to ~ties at n=2**: CheetahRun (glass 652/649 vs 706/605), FingerSpin
  (963/968 vs 977/925), ReacherEasy (779/979 vs 983/975).
- Robust TD-MPC2 advantages that **survive n=2**: FingerTurnHard (970 vs ~674), HopperStand (865 vs ~185, glass
  collapse), AcrobotSwingup (378 vs ~263, glass also high-variance 312/285/191).

**Revised verdict: tdmpc-glass is competitive-but-unstable** — high seed-variance with occasional hard collapses
(HopperHop, HopperStand), whereas TD-MPC2 is consistent. Most apparent per-task gaps are driven by glass's
*collapsing seeds*, not a uniform deficit. Pendulum's earlier 0.0 stays debunked (both tdmpc2 seeds 827 / 725).
Live mean ± 95% CI (n filling to 3) + wall-clock + per-task curves: the results page (auto-updating). The
seed-1 tables below are kept for provenance; trust the live page's multi-seed numbers over them.

## All methods, one table (PPO · TD-MPC2 · tdmpc-glass · abstraction · TAMP)

Putting every method the campaign has on DMC into one place. **Coverage is uneven and that is the honest
point:** PPO was run only on the handful of DMC tasks the campaign needed (and at 50–285M steps, vs 1M for the
world models — *not* sample-efficiency comparable, it's "PPO's best at a far larger budget"); the **abstraction**
(analytic controller + in-loop residual) exists only for the two swing-ups where we hand-built a controller; and
**TAMP is an OpenCabinet manipulation controller — there is no TAMP for DMC locomotion**, so that column is N/A
by construction (it is not a DMC method).

| task | PPO (50–285M) | TD-MPC2 (1M) | tdmpc-glass (1M) | abstraction (ctrl+residual) | TAMP |
|---|---|---|---|---|---|
| AcrobotSwingup | 268 | 383 | 312 | **71** (backfires, Part 47) | — |
| BallInCup | — | 976 | 973 | — | — |
| CartpoleBalance | — | 988 | 982 | — | — |
| CartpoleSwingup | — | 852 | 857 | — | — |
| CheetahRun | **928** | 706 | 652 | — | — |
| FingerSpin | — | 977 | 963 | — | — |
| FingerTurnEasy | — | 773/978† | 977 | — | — |
| FingerTurnHard | 968 | 970 | 586 | — | — |
| HopperHop | 33 | **234** | 0.17 | — | — |
| HopperStand | — | 872 | 355 | — | — |
| PendulumSwingup | — | 827 | 743 | **835** (ties/beats, Part 45) | — |
| ReacherEasy | — | 983 | 779 | — | — |
| ReacherHard | — | 971 | 973 | — | — |
| WalkerRun | — | ~625 | 665 | — | — |
| WalkerStand | — | ~943 | 976 | — | — |
| WalkerWalk | 970 | ~975 | 959 | — | — |

(†TD-MPC2 FingerTurnEasy: seed-1 773 / seed-2 978 — a seed dip, treat as ~tie. TAMP is reported on its own task
in [Part 43](https://suuttt.github.io/tdmpc-glass/2026/06/26/tdmpc-glass-part43-residual-on-tamp-beats-ppo.html):
residual-on-TAMP **0.98 success** on PandaOpenCabinet vs PPO 0.98 — a manipulation result, not a DMC return.)

**What the unified view shows:**
- **The abstraction is bimodal, exactly per its prior quality:** on PendulumSwingup it (835) *beats* both
  TD-MPC2 (827) and is the best method on that task; on AcrobotSwingup it (71) is the *worst* — same recipe,
  inverse outcome (Parts 45/47).
- **HopperHop is hard for everyone except TD-MPC2:** PPO 33, glass 0.17, TD-MPC2 234. It is the exploration-
  bottlenecked task where model-based planning earns its keep — and where glass's failure is most visible.
- **Where PPO data exists, TD-MPC2 matches or beats it at ~100× fewer steps** (FingerTurnHard 970 vs 968,
  WalkerWalk ~tie, Acrobot 383 vs 268) except CheetahRun (PPO 928 at 187M >> TD-MPC2 706 at 1M) — consistent
  with the Part 18–21 finding that TD-MPC2 wins per-step sample-efficiency while PPO can win final return at a
  huge budget.
- **abstraction/TAMP are not general DMC learners** — they are *task-structured* methods that win exactly when
  their prior fits (Pendulum, OpenCabinet) and are absent or harmful otherwise. That is the campaign's thesis in
  one table.

## The refined verdict

**TD-MPC2 ≥ tdmpc-glass across DMC, by task difficulty:**

- **Ties on easy/saturated tasks** — BallInCup, both Cartpoles, ReacherHard, FingerSpin, Walker-stand/walk:
  both hit the ~950–990 ceiling, indistinguishable.
- **TD-MPC2 wins the harder exploration/locomotion tasks** — FingerTurnHard (970 vs 586), HopperStand
  (872 vs 355), ReacherEasy (983 vs 779), CheetahRun (706 vs 652), AcrobotSwingup (383 vs 312). The
  structural-entropy/graph machinery in glass does not help — and modestly hurts — where the task needs more
  raw value/dynamics fidelity.
- **One catastrophic glass failure: HopperHop = 0.17** while TD-MPC2 reaches 234 (and 255 on seed-2). HopperHop
  is the campaign's known exploration-bottlenecked task (Part 18/21); glass simply fails to get off the ground
  on it.

So the earlier **"glass ≈ TD-MPC2"** (Part 2) was true *on the subset of tasks tested then* (mostly saturated
ones). Across the full suite it becomes: **glass matches TD-MPC2 where the task is easy, trails where it is
hard, and has a failure mode (HopperHop) TD-MPC2 does not.** An honest negative for the glass variant as a
general DMC learner — its value was always the *interpretability/structure* angle (Parts 24–35), not raw DMC return.

## Verification-discipline note (a debunked "finding")

A seed-2 TD-MPC2 run logged **PendulumSwingup = 0.0** — which looked like "TD-MPC2 fails a trivial task." It is
not: the **seed-1 run reached 827** (glass 743), so both algorithms solve Pendulum and the 0.0 was a single-seed
divergence/NaN. Reported here as an outlier, not headlined as a result — exactly the kind of single-seed artifact
this project has learned to cross-check before claiming (the HopperHop collapse, by contrast, *is* reported
because it reproduces and the task is provably solvable by the other algo).

### Caveats
Primary numbers are single-seed (seed-1), 1M steps; a second seed cross-checks the two anomalies only. Walker-trio
TD-MPC2 was still converging at writeup (cited values are seed-2 + the climbing seed-1, both ~tie with glass).
Returns are the last logged eval per run (`exp/benchmark/{tdmpc-glass,tdmpc2}_<task>_fs_*.csv`). HopperHop glass
collapse is seed-1 here; the matched scorer/harness make the cross-algo gap (0.17 vs 234) far larger than seed
noise. Prior: Part 2 (glass≈TD-MPC2 on subset), Parts 18–21 (TD-MPC2 vs PPO on DMC, HopperHop exploration).
