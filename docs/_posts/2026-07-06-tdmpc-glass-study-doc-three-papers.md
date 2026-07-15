---
layout: post
title: "TD-MPC-Glass Study Document: The Three Papers, Explained From Scratch"
date: 2026-07-06
description: ""
---

This is the audit document for the TD-MPC-Glass research program's papers. It explains each paper from
scratch: the question it asks, how the evidence was produced, every headline number and where it comes from,
and what is honestly weak. All numbers are harvested from disk artifacts (CSV/JSONL eval logs), recorded in
the append-only ledger (`bet2_null_results.md` in the paper repo), and cross-posted to this blog as they
landed. Nothing here is from memory.

## 0. The program in one paragraph

We built a **matched-environment benchmark**: TD-MPC2 (a model-based RL agent with SimNorm latents and MPPI
planning), tuned brax PPO, and tuned brax SAC, all running on **byte-identical MuJoCo-MJX environments**
(`registry.load(env, impl=jax)`), so that no comparison is contaminated by simulator or observation
differences. On top of that benchmark we ran a discipline of budget-indexed rates (never "method X fails" —
always "0/5 seeds ≥ threshold at N steps"), per-loss ablations, causal discriminator tasks, and honest
revision whenever new seeds moved a number. The program produced three papers plus supporting artifacts.

---

## Paper A — "The Redundancy Criterion" (complete; awaiting author block)

**Question.** When does adding structure to a world model — entity factorization, graphs, structural-entropy
(SE) regularization, hierarchy — actually buy performance, and when is it redundant?

**Method.** For each proposed structure: build it into TD-MPC2, run matched-budget comparisons against the
plain latent on the same tasks, and — crucially — run a *mechanism check*: verify whether the structure's
extra coupling is actually used by the value/policy pathway, not just present.

**Claims and evidence.**
1. *SE/"glass" state abstraction is redundant for control*: identical returns to vanilla TD-MPC2 across a
   16-task DMControl matched benchmark (n=3–4/task) at 1.35× wall-clock cost.
2. *Entity/graph world models don't beat the plain latent at matched budgets* (multi-object MJX suite;
   compositional-OOD controls included) — the graph's couplings go unused by the value pathway.
3. *A practical criterion*: structure is worth adding only if the mechanism check shows the new couplings
   carry gradient that the value pathway consumes. In our program, none did.

**Weaknesses.** A negative-results program: real and useful, but its citations will come from people building
structured world models, not from a headline effect. One codebase.

**Status.** Draft complete; needs only the author block to submit.

---

## Paper 3 — "The Wall and the Mechanism" (the dissection paper; submission-shaped)

File: `paper_wall_mechanism.tex` — abstract, 8 sections, 2 figures built from raw seed curves, embedded
bibliography; compiles clean. This is the program's capstone. Four linked claims:

### Claim 1 — PPO hits a *categorical* exploration wall on contact-critical hopper tasks

- HopperHop: tuned PPO **0/5 seeds ≥ 200 return at 472M steps/seed** (peak 53.8). SAC crosses 200 in 6/12
  seeds by 5M and **6/9 by 8M** (the 20M cohort crossed 5/5 at 4.1–7.7M, but four fresh direct-8M seeds went
  1/4 — SAC's crossing by 8M is a ~2/3 rate, honestly revised twice). TD-MPC2: **6/6 by ~1M**.
- The wall **survives the standard exploration knob**: entropy cost ×3 and ×10 leave PPO at peaks 4–74
  (n=2 each at 150M) — under-exploration hyperparameters are not the explanation.
- HopperStand is a *graded, near-categorical* barrier: PPO escapes 2/16 (eight fresh 285M seeds all walled at
  105–195). Entropy ×3 put 1/4 seeds over (627) — a rate consistent with the baseline lottery, not a repair;
  ×10 walled 0/4. The knob can ride a graded barrier's variance but never dents the categorical wall.
- Final gradient on Stand: **TD-MPC2 5/5 (943–962, ≤1M) ≫ SAC 7/10 @5M (+4/4 direct at 8M) ≫ PPO 2/16**.
- Five no-wall control tasks (e.g. FingerTurnHard PPO ≈973 3/3) show the wall does not generalize to
  ordinary tasks.

### Claim 2 — The wall is causally scoped: it requires *contact-criticality*, not instability

- AcrobotSwingup is unstable but contact-free. There, **PPO learns fine (267–344, n=4)** — no wall — while
  SAC is slow and inconsistent even at 20M (42/66/123/207, n=4) and TD-MPC2 leads both classes (422–454
  within 1M, n=4). One discriminator task-pair is suggestive, not a law — this is the claim we plan to
  harden with pre-registered predictions on more tasks.

### Claim 3 — The mechanism: TD-MPC2's engine is the TD value pathway, not the world model

The 5-loss ablation (mask one loss at a time), **4 tasks × n≥4 per arm**, MPPI-best per seed:

| Arm removed | CheetahRun | HopperHop | WalkerRun | HopperStand |
|---|---|---|---|---|
| none (full) | 721–795 | 287–570 | 680–731 | 911–946 |
| value | 16–58 dead | ~0 dead (n=6) | 28–56 dead | 6–13 dead |
| policy | 123–192 dead | ~0 dead | 53–83 dead | 9–34 dead |
| reward | 5–31 (π **681–805 full**) | ~0 (π full) | 44 (π full) | 265–542 (π **943–944**) |
| consistency | 367–558 mild | 185–245 mild | 483–674 mild | **816–898 near-full** |

Reading: value and policy losses are individually fatal everywhere (without the TD value loss the agent
cannot even learn to *stand*); the reward head only feeds the planner (the policy still reaches full
strength); the consistency loss — the actual "world model" objective — is the mildest cut on all four tasks.
This **contradicts the TD-MPC2 paper's own ablation story**, which makes it the most contentious and
checkable claim we have.

### Claim 4 — Method orderings invert by task class; efficiency claims must be task-class-indexed

- Hopper (contact-critical): TD-MPC2 ≫ SAC ≫ PPO.
- Acrobot (contact-free unstable): TD-MPC2 > PPO ≫ SAC.
- Humanoid (21-DoF): **only SAC survives** (Walk 625–909 4/5, Stand 918/922 2/2); PPO nan under every config
  we could build (no official humanoid config exists — partly a tooling boundary, stated honestly); TD-MPC2
  diverges loss=nan in 7/7 default runs, and the one stable knob variant (k_update 64) never learns (21.8).
- Efficiency anchor at final n: TD-MPC2 HopperHop 5M bests **295–610, mean 420 ± 113 (n=12)**. A fresh
  vanilla 1M cohort meaning ~408 vs an older cohort's ~282 also showed why we report budget-indexed rates
  instead of cohort means.

**Weaknesses.** One codebase/implementation; the discriminator is n=1 task-pair; humanoid is partly tooling.
The consolidation plan makes Claims 1, 2 and 4 supporting evidence for Claim 3's thesis.

---

## Paper 4 — "The World Model is Scaffolding?" (constructive sequel; core experiment finishing now)

**Question.** Paper 3's ablations prove *necessity* one loss at a time. The flip side is *sufficiency*: train
the stripped agent (consistency loss OFF from scratch) at full budget — does it match the full agent?

**Evidence so far (the 2×2, second row finishing at time of writing).**
- **HopperHop (exploration-bound): sufficiency GO.** Stripped at 5M: **165 / 475 / 481 / 511** (n=4) vs full
  420 ± 113 (n=12). Three seeds land at the *top* of the full band and reached 440+ by ~2M — faster than
  typical full seeds; one low seed (165) below the full floor. The consistency loss's residual job on this
  task is worst-seed insurance.
- **WalkerRun (dense): sufficiency FAILS.** Stripped plateaus at **537/541/542/564** vs full
  **709/705/753/782** — a ~26% gap, rock-stable across seeds. (Formal marker minutes away; numbers are
  final-eval.)
- **Emerging verdict**: the world-model loss is **task-class-indexed** — removable where directed exploration
  is the bottleneck, load-bearing where dense state-tracking is. This mirrors every ordering result in the
  program and is a sharper claim than either "removable" or "necessary."
- **Task 3 (CheetahRun) is in flight on both boxes** (full baseline + stripped) to decide whether the split
  is a law or a two-point coincidence.

**Also tested and honestly closed:** novelty-directed MPPI (Part 5's proposal A) — Q-head-disagreement and
RND bonuses inside the planner, β ∈ {0.3, 1.0}, HopperHop 1M, then a **matched-seed vanilla control**:
novelty was **worse on 4/4 same-seed comparisons** (vanilla 442/509/321/359 vs 5.6/285/259/274), with one
catastrophic break. Closed null in the regime vanilla solves; any residual value would be on a frontier task
where vanilla fails, which is future work.

---

## Supporting artifacts (context, not main-line)

- **Speed-of-learning paper** (assembled earlier): abstraction and priors buy learning speed, not ceiling;
  hierarchy's value comes from competent primitives (2-level 0.215 vs 1-level 0.184 on Panda; mechanism =
  place-phase 0.94 vs 0.57), with a NULL on raising asymptotes. H-JEPA proper: multi-seed NULL on
  PandaPickCube (n=5).
- **Paper B (manipulation methods benchmark)** and **honest-rl-bench** (public toolkit + tutorial site).
- **Part 9 blog post** — the living capstone with every number and revision:
  [Anatomy of Beating PPO](https://suuttt.github.io/projects/2026-07-05-tdmpc-glass-part9-anatomy-of-beating-ppo/).

## The verification discipline (why you can trust the numbers)

Every number above was read from disk (CSV/JSONL) at harvest time, with n stated; every revision (SAC's 8M
cell twice, the wall framing three times, the entropy "rescue" that dissolved under the full control) was
recorded in the ledger and pushed the same tick. The program's earlier history included ~7 fabrication
incidents from log-eyeballing — the ledger-first discipline exists because of them.

## Where this goes next

1. **Consolidate** Paper 3 (+ Paper 4's split) into one dissection paper around the mechanism thesis.
2. **Harden the discriminator** with pre-registered contact-criticality predictions on new tasks.
3. Paper A needs only an author block. Venue selection for the dissection paper is the user's call.
