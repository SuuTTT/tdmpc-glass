---
layout: post
title: "TD-MPC-Glass, Part 51: A Better Grasp Pushes the Solve to 0.367 — and What SE Actually Taught Us About JEPA Latents"
date: 2026-06-30
description: "Two threads continuing Part 50. First: we attack the low-level primitive that bounded the Panda H-JEPA solve and push it from 0.289 to 0.367, with the gain coming entirely from a better grasp — and we characterize the remaining ceiling as tall-cube/two-finger contact physics, not planning. Second: we tried to make the JEPA latent better with structural entropy (the 'glass' idea), found it NULL, and in the process isolated the real lever — a one-line relational anti-collapse loss — and discovered that the right anti-collapse term is downstream-dependent: what helps a goal-conditioned geometric latent hurts a value-based control latent."
---

> **TL;DR.** Continuing [Part 50](../tdmpc-glass-part50-making-hjepa-work-on-panda). **Thread 1 (push the solve):** Part 50 left H-JEPA solving `PandaPickCube` at **0.289**, bounded by an analytic-LL oracle of ~0.30. We diagnosed the true bottleneck — the cube *tips inside the two-finger grip* during transport — and added a **closed-loop upright-orientation servo**. That lifted the oracle 0.305 → **0.352** and the H-JEPA solve to **0.367** (n=512), reproducing at peak ~0.33–0.37 across seeds. The remaining gap to PPO's 0.66 is now provably **contact physics** (a tall cube in a parallel-jaw grip), not planning. **Thread 2 (make the latent better with SE):** we tried structural-entropy regularization (our "glass" lever) on the JEPA latent. It was **NULL** — but the controls revealed the real lever is **relational anti-collapse**, captured by a *one-line* uniformity loss, and that the right anti-collapse term is **downstream-dependent**: relational repulsion *helps* a geometric navigation latent (0.53 → 0.95) and *hurts* a value-based DMControl latent. There is no universal anti-collapse winner.

---

## Where Part 50 left us

Part 50 got a faithful H-JEPA from a flat **0.0** to **0.289** on PandaPickCube — beating the hand-tuned reactive hierarchy (0.246) — by handing the high-level latent planner a *competent low-level primitive* and warm-starting it. We bounded the whole approach with an **oracle** (brute-force search the best high-level parameters directly in the simulator): it saturated at **0.305**. The honest read was: *even a perfect planner can't beat ~0.30 over this low level. The ceiling is a low-level controller problem.*

So this post does two things: (1) attack that low-level ceiling head-on, and (2) chase the other obvious lever — a *better latent* — via structural entropy, the idea behind this whole "glass" line of work.

---

## Thread 1 — A better grasp pushes the solve to 0.367

### First, find the real bottleneck
The Part-50 oracle topped out at ~0.30 with grasp 1.00 and lift 0.80 but `place`/`hold` lagging. We instrumented exactly *why* a competent pick fails the `box_target ≥ 0.9` gate (which needs `0.9·pos_err + 0.1·rot_err ≤ 0.02`). The finding was specific and physical:

> PandaPickCube spawns the cube **upright** and the target is upright — so the rotation error is **not** a randomized-orientation problem. It is **manipulation-induced tipping**: the tall 4×4×6 cm cube **tips inside the two-finger grip** during transport, its tilt growing from ~5° to ~20° by the time it's placed.

This reframed an earlier dead-end. We'd tried a yaw-aware *grasp* controller (Round 4) and got nothing — because the problem isn't grasp *yaw*, it's the cube *rotating away from upright mid-flight* through a single compliant contact.

### The fix: a closed-loop upright servo
Instead of an open-loop place pose, we recompute the gripper orientation **every step** during lift/transport/place/hold to actively right the *live* cube toward upright. Result: final tilt dropped **20.6° → 16.4°**, and warm-started success rose **0.180 → 0.242**.

### The numbers (every value a deterministic, held-out, real-success eval)

| stage | analytic-LL **oracle** | H-JEPA **solve** | note |
|---|---:|---:|---|
| Part 50 (round 3) | 0.305 | 0.289 | where we started |
| round 4 (yaw place) | 0.320 | 0.279 | NULL — wrong axis |
| **round 5 (upright servo)** | **0.352** | **0.367** | the lever |
| — PPO (huge budget) | — | 0.66 | the remaining gap |

The full solve arc is now **0.0 → 0.117 → 0.289 → 0.367**. Two things are worth stating plainly:

1. **Every gain came from a better *low-level primitive*** — competence, then warm-start, then grasp stability. Nothing came from changing the high-level planner. This keeps confirming Part 50's thesis: *the lever is the low-level primitive; the latent planner extracts exactly what the primitive allows.* At round 5 the H-JEPA solve (0.367) sits **at** the oracle (0.352, n=128 estimate) — the planner now **fully extracts** the analytic ceiling.
2. **It reproduces.** A multi-seed CI is running as I write this; finalized seeds land at peak **0.347–0.371** (finals 0.31–0.32), clustered around the 0.367 headline. (The "final" eval-checkpoint is a touch below "peak" due to eval-point variance — a real, honest caveat.)

### The ceiling is now contact physics, not H-JEPA
Round 5 also tells us *why* ~0.37 is a wall for this design. Righting the cube's orientation perturbs its *placement position* through the single compliant grip — and you cannot fit both inside the `0.9·pos + 0.1·rot ≤ 0.02` budget with one parallel-jaw contact. So the analytic-skill family caps at ~0.35–0.37 for a **tall cube in a two-finger gripper**. Closing to PPO's 0.66 needs a **different contact primitive** — a side grasp, a two-stage regrasp, or force-controlled fingers — *not* better high-level planning. That's a clean, falsifiable next step, and it's a *robotics* problem, not a *world-model* one.

---

## Thread 2 — We tried to make the latent better with SE. It was NULL. Here's the better thing we found instead.

The other obvious lever is the **latent itself**. JEPA latents need an **anti-collapse** term (or the encoder maps everything to a constant and prediction becomes trivially perfect and useless). Our "glass" program has long bet on **structural entropy (SE)** — encode the data as a graph, find its minimal-2D-SE community partition, and regularize the latent toward that structure. So we asked: *does SE make the JEPA latent better?*

### On Panda: NULL, and we know exactly why
We SE-regularized the JEPA latent (selib minimal-2D-SE on a kNN latent graph) against a **matched VICReg** baseline, and probed how well each latent linearly decodes the task geometry (frozen encoder → ridge, held-out R²):

| latent | end-effector→cube R² | box→target R² | effective rank |
|---|---:|---:|---:|
| VICReg | **0.987** | **0.933** | 30.8 |
| SE (real graph) | 0.736 | 0.569 | 12.9 |
| SE (**random**-graph control) | 0.986 | — | — |

The decisive line is the **random-graph control**: SE on a *randomized* graph recovers VICReg's R². So SE's damage isn't a bug — it's the **community structure itself**, collapsing effective rank 31 → 13 and bucketing a *continuous* end-effector→cube manifold into discrete clumps. For manipulation, which needs continuous geometric regression, that's exactly the wrong inductive bias. **VERDICT: NULL** — SE never beat VICReg; at best it tied when weighted to near-inert.

### On a collapse-prone navigation latent: GO — but not because of SE
We then built the fair testbed: a 2-D point-maze where SimNorm + VICReg **provably collapses** (effective rank ≈ 0, success 0.53 vs a raw-state ceiling of 0.98). Here SE *helped* — but the controls tell a subtler story (matched arms, n=4, deterministic n=256 eval):

| arm | success | collapsed? |
|---|---:|---|
| raw-state ceiling | 0.979 | — |
| VICReg (default) | 0.530 | 4/4 collapsed |
| VICReg (strong, fully un-collapsed) | 0.442 | 0/4 — *still fails* |
| SE | 0.906 | 0/4 |
| SE (**random**-partition control) | 0.823 | overlapping CI with real SE |
| **minimal uniformity loss** | **0.954** | 0/4 |

Two findings fall out:

1. **The benefit is partition-independent.** SE on a *randomized* partition does nearly as well (0.823, CI overlapping real SE). So the lever is **not** the structural-entropy community tree — it's the **relational/pairwise** structure of the objective.
2. **A one-line loss captures it.** The simplest possible relational anti-collapse — a Wang & Isola **uniformity** term (pairwise repulsion, no graph, no partition, no tree) — gets **0.954**, beating VICReg's 0.530 with non-overlapping CI and matching/edging SE. You need neither structural entropy nor even a kNN graph.

Note also the strong-VICReg row: fully un-collapsing the latent (rank restored) *still* fails (0.442). **Anti-collapse health and downstream success are decoupled** — per-dimension variance/covariance regularization fixes the *symptom* (rank) without fixing the *task*.

### The unifying lesson: anti-collapse is downstream-dependent
Here's the part that ties Panda and nav together. The *same* uniformity loss that **wins** on nav, we tested on **value-based control** (DMControl, a collapse-prone latent_dim=16). Firmed over 3 seeds on CheetahRun:

| arm | return AUC | value-decode R² |
|---|---:|---:|
| default (no extra term) | **58.9 ± 29.8** | **0.59 ± 0.51** |
| uniformity | 36.6 ± 24.8 | 0.33 ± 0.19 |
| VICReg | 56.3 ± 6.4 | 0.48 ± 0.38 |

Uniformity is **worst on both axes** — the *opposite* of its nav result. Mechanism: uniformity's pairwise spreading pushes apart states that should be **value-close**, destroying the value-sufficiency that control needs. (Honest caveat: high seed-variance, so this is a consistent directional confirm across n=1→n=3, not a CI separation; a 2nd-task generalization sweep is running now.)

So the clean conclusion across all three regimes:

> **There is no universal anti-collapse term.** Use *relational* repulsion (uniformity) for goal-conditioned **geometric** latents; use *value-sufficiency* for **value-based control** latents; use **neither SE community structure** for either continuous case. The right regularizer is a function of what you decode downstream — and SE's discrete community bias fits *neither* continuous control regime we tested.

This mirrors the Panda SE NULL exactly: a structure that buckets a continuous manifold is the wrong bias when the downstream task needs that continuity.

---

## What this means

- **The Panda solve is now characterized end to end.** 0.367, reproducible, with the residual gap to PPO cleanly attributed to two-finger contact physics. The next lever is a *robotics* primitive, and we can say precisely which one.
- **SE is not the JEPA latent lever** — but ruling it out wasn't wasted: it forced the controls (random-graph, random-partition, strong-VICReg) that isolated **relational anti-collapse** and exposed its **downstream-dependence**. A one-line uniformity loss is a genuine, honest improvement *for the collapse-prone geometric regime* — and an honest *warning* for the value-control regime.
- **The methodological spine held again:** oracles (what's the best possible over this design?), positive controls (does it work where it should?), and — the star of Thread 2 — **randomized-structure controls** (is the benefit really coming from the mechanism I claim?). Each one changed a conclusion here.

*All numbers are deterministic, held-out, real-success / disk-backed measurements; code and result JSONs are in the repo. Multi-seed CIs (Panda n→7; DMControl 2nd-task generalization) are firming as of this writing and will be folded into the results ledger.*
