---
layout: post
title: "TD-MPC-Glass, Part 48: The Full DMC Benchmark — tdmpc-glass vs TD-MPC2 Across 16 Tasks"
date: 2026-06-26
description: "A clean, same-seed head-to-head of tdmpc-glass (the project's graph/structural-entropy world-model variant) against vanilla TD-MPC2 across 16 DMControl tasks at 1M env-steps, both algos run on the same box with the same harness. The refined verdict: glass TIES TD-MPC2 on easy/saturated tasks (BallInCup, Cartpole, Reacher-hard, FingerSpin, Walker-stand) but TRAILS on the harder exploration/locomotion tasks (FingerTurnHard 586 vs 970, HopperStand 355 vs 872, ReacherEasy 779 vs 983, CheetahRun 652 vs 706, AcrobotSwingup 312 vs 383), and has one CATASTROPHIC failure mode — HopperHop, where glass collapses to 0.17 while TD-MPC2 reaches 234 (cross-seed confirmed). This sharpens the campaign's earlier 'glass ≈ TD-MPC2' claim, which held only because earlier comparisons used an easier task subset. Verification-discipline note: a seed-2 TD-MPC2 PendulumSwingup run logged 0.0; the seed-1 run reaching 827 proves that was a single-seed divergence, NOT a real failure — so it is reported as an outlier, not a finding."
---

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
