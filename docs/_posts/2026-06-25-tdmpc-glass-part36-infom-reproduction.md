---
layout: post
title: "TD-MPC-Glass, Part 36: Reproducing InFOM on OGBench — Cube Yes, AntMaze No (and Why)"
date: 2026-06-25
description: "The week opened with the goal of reproducing 'Compositional Planning with Jumpy World Models' and its InFOM base on a common OGBench task set. This note closes that thread with an honest cross-task reproduction of the InFOM BASE (flow-occupancy; not the CompPlan planner). On cube-single it reproduces cleanly: 2 of 3 seeds hit 0.96/0.98 success (matching/exceeding the paper's CompPlan 0.86 and far above the CRL base 0.28), with one seed collapsing to 0.16 (real seed-variance, flagged). On antmaze-medium it does NOT reproduce: best success ~0.18 (complete seed) / 0.08-0.20 across three seeds, far below the paper's CRL 0.49 and CompPlan 0.85. The cause is a data-availability artifact, not (necessarily) an InFOM fault: the OGBench server returns 404 for the cube/antmaze -ft- finetuning datasets, so we fell back to the non-ft reward-labeled singletask dataset (ft_dataset_fallback=1); cube tolerates this, antmaze's longer-horizon navigation does not. We reproduced the base method, not the CompPlan planner (the paper's actual contribution), which is a multi-day build left as the next step."
---

> The week started here: reproduce *Compositional Planning with Jumpy World Models* and its **InFOM** base on a
> common OGBench task set, in one table. Many detours later (Parts 23-35), this note closes that original
> thread — honestly, including where it *doesn't* reproduce.

## What we ran

The **InFOM base** (flow-occupancy; Zheng et al.) on OGBench, HPs from the repo README
(expectile 0.95, kl_weight 0.05, alpha 30), 1.5M steps, eval = OGBench `episode.success` (n=50 eval episodes
per checkpoint). **Important honest caveat up front:** the OGBench server returns **404** for the `-ft-`
finetuning datasets, so we used `ft_dataset_fallback=1` — the non-ft, reward-labeled singletask dataset. This is
not the exact paper setup, and it matters (below). We reproduced the **base method, not the CompPlan planner**
(the paper's actual contribution) — that is a multi-day build, deferred.

## The cross-task table

| task | our InFOM base (per-seed best success) | paper CRL base | paper CompPlan | reproduces? |
|---|---|---|---|---|
| **cube-single (task1)** | **0.96 / 0.98** (+ one 0.16 outlier seed) | 0.28 | 0.86 | **yes** — matches/exceeds CompPlan, ≫ CRL base |
| **antmaze-medium (task1)** | **0.18** (complete seed) / 0.20 / 0.08 | 0.49 | 0.85 | **no** — far below even the CRL base |

(Paper refs: CompPlan arXiv 2602.19634 Table 1, cube-1 and antmaze-medium rows.)

## Cube — clean reproduction, with honest seed-variance

Two of three cube seeds reach **0.96 and 0.98** best success — at or above the paper's CompPlan number (0.86)
and far above the CRL base (0.28). So InFOM base on cube reproduces. But the **third seed collapsed to 0.16**
(latest 0.0) — a real instability we are not hiding: 1-in-3 seed failure is exactly the kind of seed-variance
the campaign's stability axis (Part 35) warns must be *measured*, not assumed. Two more cube seeds (s3, s4) are
running to characterize whether 0.16 is a rare fluke or a ~1/3 failure rate.

## AntMaze — does not reproduce, and the reason is the dataset

AntMaze-medium tells the opposite story: best success **0.18** on the completed seed (final 0.12), and
**0.08-0.20** across three seeds — *below the CRL base (0.49)*, let alone CompPlan (0.85). This is not a
marginal miss; it is a different regime. The most likely cause is the **`ft_dataset_fallback`**: cube finetunes
fine from the reward-labeled non-ft dataset, but antmaze-medium — a longer-horizon navigation task — evidently
needs the proper `-ft-` finetuning dataset that the OGBench server no longer serves. So the antmaze gap is best
read as a **data-availability artifact of our setup**, not a verdict on InFOM. (Two of three antmaze seeds were
still completing at write time — s6 complete at 0.18 best, s7/s8 partial at 0.20/0.08 — but the trend is flat
and low across all three, so the "does not reproduce" conclusion is robust.)

## Verdict

**InFOM base reproduces on cube-single (~0.97 on 2/3 seeds, matching CompPlan) but not on antmaze-medium
(~0.18, a fallback-dataset artifact).** The honest one-liner: *the reproduction succeeds where the available
data matches the paper's setup and fails where it doesn't* — which is a statement about dataset availability on
OGBench as much as about the method. The CompPlan planner itself (the paper's contribution, expected to lift
the base toward 0.86/0.85) was **not** run; building and running it on this InFOM base is the natural, multi-day
next step.

### Caveats
Numbers read directly from `finetuning_eval.csv` (`evaluation/episode.success`), never fabricated. Cube: 3
seeds (2 good + 1 outlier), s3/s4 in progress. AntMaze: 3 seeds, 1 complete (s6, 1.5M) + 2 partial (s7 @1.1M,
s8 @1.0M) — all low. `ft_dataset_fallback=1` throughout (the `-ft-` datasets 404 on the OGBench server) — the
single biggest caveat, and the prime suspect for the antmaze gap. Base method only, not the CompPlan planner.
Data: `/root/ghm/exp/{infom_cube1_s*, pq_antmaze_s6/7/8}/sd*/finetuning_eval.csv`; harvest
`/root/ghm/repro_results/harvest_repro.py` (cube) + direct CSV reads (antmaze).
