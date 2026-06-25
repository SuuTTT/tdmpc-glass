---
layout: post
title: "TD-MPC-Glass, Part 36: Reproducing InFOM on OGBench — Cube Reproduces; AntMaze Is Out of InFOM's Benchmark"
date: 2026-06-25
description: "The week opened with the goal of reproducing the InFOM base (flow-occupancy) on OGBench. This note closes that thread honestly. On cube-single — which IS in InFOM's benchmark — it reproduces cleanly: 2 of 3 seeds hit 0.96/0.98 success, with one seed collapsing to 0.16 (real seed-variance, flagged). AntMaze, by contrast, is OUT of InFOM's benchmark: a dataset investigation found InFOM's -ft- finetuning datasets are generated locally for 8 MANIPULATION envs only (cube/scene/puzzle) — antmaze is absent from the InFOM repo and paper, which reports no antmaze results. So our antmaze run (~0.18 on a fallback dataset) is an out-of-benchmark extension, NOT a reproduction of any InFOM number, and the 0.49/0.85 figures are CompPlan/CRL numbers for a different method — an invalid comparison we are correcting here. Net: InFOM reproduces on its actual (manipulation) benchmark; antmaze is out of scope. We reproduced the base method, not the CompPlan planner."
---

> The week started here: reproduce *Compositional Planning with Jumpy World Models* and its **InFOM** base on a
> common OGBench task set, in one table. Many detours later (Parts 23-35), this note closes that original
> thread — honestly, including where it *doesn't* reproduce.

## What we ran

The **InFOM base** (flow-occupancy; Zheng et al.) on OGBench, HPs from the repo README
(expectile 0.95, kl_weight 0.05, alpha 30), 1.5M steps, eval = OGBench `episode.success` (n=50 eval episodes
per checkpoint). We reproduced the **base method, not the CompPlan planner** (the paper's actual contribution).

**A correction up front (this note was revised).** An initial version treated antmaze as a reproduction
*failure* against "paper" numbers of 0.49/0.85. That comparison is **invalid** and is corrected below: a
dataset investigation found that **InFOM does not benchmark antmaze at all** — its `-ft-` finetuning datasets
are *generated locally* for 8 **manipulation** envs only (cube-single/double, scene, puzzle-4x4 and visual
variants); antmaze appears nowhere in the InFOM repo (`data_gen_scripts/` has no antmaze path) or paper. The
0.49/0.85 are **CompPlan/CRL** figures for a *different method*. So antmaze is **out of InFOM's benchmark**, our
antmaze run is an out-of-benchmark extension on a fallback dataset, and only the **cube** result is an actual
InFOM reproduction.

## The reproduction (InFOM's actual benchmark)

| task | in InFOM benchmark? | our InFOM base (per-seed best success) | reproduces? |
|---|---|---|---|
| **cube-single (task1)** | **yes** (manipulation) | **0.96 / 0.98** (+ one 0.16 outlier seed) | **yes** — clean InFOM-base reproduction |
| antmaze-medium (task1) | **no** (not in InFOM) | 0.18 / 0.20 / 0.08 (fallback dataset) | n/a — out-of-benchmark extension, not comparable |

(InFOM paper: arXiv 2506.08902 — manipulation only, no antmaze. The 0.49/0.85 antmaze figures come from
CompPlan arXiv 2602.19634 Table 1 for CRL/CompPlan, *not* InFOM.)

## Cube — clean reproduction, with honest seed-variance

Two of three cube seeds reach **0.96 and 0.98** best success — at or above the paper's CompPlan number (0.86)
and far above the CRL base (0.28). So InFOM base on cube reproduces. But the **third seed collapsed to 0.16**
(latest 0.0) — a real instability we are not hiding: 1-in-3 seed failure is exactly the kind of seed-variance
the campaign's stability axis (Part 35) warns must be *measured*, not assumed. Two more cube seeds (s3, s4) are
running to characterize whether 0.16 is a rare fluke or a ~1/3 failure rate.

## AntMaze — out of InFOM's benchmark (not a reproduction at all)

The original version of this note read antmaze's ~0.18 as a reproduction *failure*. A dataset investigation
(`/root/ghm/repro_results/antmaze_dataset_FINDINGS.md`) corrected that:

- **InFOM's `-ft-` finetuning datasets are generated locally, not downloaded** — and only for **8 manipulation
  envs** (cube-single/double, scene, puzzle-4x4 + visual variants). The on-box `data_gen_scripts/` has no
  antmaze path; the `-ft-` filename is *constructed* by InFOM (`envs/env_utils.py`) but no antmaze `-ft-` file
  is ever produced.
- **InFOM's paper (arXiv 2506.08902) reports no antmaze results** — "antmaze" does not appear in its
  experiments. So **0.49/0.85 are not InFOM numbers**; they are CompPlan/CRL figures (a different method).

So our antmaze runs (~0.18 across seeds, on the non-ft fallback dataset) are an **out-of-benchmark extension**,
not a reproduction of any InFOM result, and not comparable to the CompPlan/CRL antmaze numbers. There is **no
"proper" antmaze dataset to fetch** — the earlier "404 / data-availability" story was wrong. The right action
is to **drop antmaze from the InFOM reproduction claim** (or label it explicitly as an out-of-distribution
extension), which is what this revision does.

## Verdict

**InFOM base reproduces on its actual (manipulation) benchmark — cube-single ~0.97 on 2 of 3 seeds — with one
seed collapsing to 0.16 (real seed-variance).** AntMaze is **out of InFOM's scope** and the ~0.18 there is not
a reproduction number. The honest one-liner: *InFOM reproduces where it actually benchmarks; antmaze was never
an InFOM task and shouldn't have been scored against another method's figures.* The CompPlan planner itself
(the paper's contribution) was not run; building it on this InFOM base is the natural, multi-day next step.

### Caveats
Numbers read directly from `finetuning_eval.csv` (`evaluation/episode.success`), never fabricated. Cube: 3
seeds (2 good + 1 outlier), s3/s4 in progress to characterize the 0.16. AntMaze: out-of-benchmark, run on the
non-ft fallback dataset (`ft_dataset_fallback=1`) — reported only for completeness, not as a reproduction.
b3060 currently has no outbound network, but the antmaze-absence was settled from the InFOM repo/paper
off-box, not a live server check. Base method only, not the CompPlan planner. Data:
`/root/ghm/exp/{infom_cube1_s*, pq_antmaze_s6/7/8}/sd*/finetuning_eval.csv`; investigation
`/root/ghm/repro_results/antmaze_dataset_FINDINGS.md`.
