---
layout: post
title: "TD-MPC-Glass, Part 52: A Learned Residual Breaks the Contact Wall (but PPO Still Wins the Ceiling), and the Anti-Collapse Taxonomy Closes"
date: 2026-06-30
description: "Two threads finish. First: we learn a residual on top of the 0.37-capped analytic PickCube skill and it breaks the contact-physics wall to 0.72 — proving the wall is learnable, not gripper-morphology — but a matched-budget vanilla PPO still wins the asymptote (0.81). The prior buys ~1.6x sample-efficiency, not a higher ceiling. Second: the anti-collapse story closes with a clean taxonomy — relational repulsion helps geometric latents, hurts value-based control, and a value-aware variant is even worse. The right anti-collapse term is the one that matches what you decode downstream — and for value-based control that means adding nothing."
---

> **TL;DR.** Two clean closes, both budget-trap-aware. **(1) Learned residual on Panda:** the analytic skill caps at ~0.37 because the cube tips in a two-finger grip. A *learned* residual over that skill reaches **0.716 ± 0.014** (n=3) — it breaks the contact wall, proving the wall is **learnable**, not a hard morphology limit. But a **matched-budget vanilla PPO reaches 0.810 ± 0.006** and wins the asymptote; the residual is only **~1.6× more sample-efficient**. So the analytic prior buys *speed, not ceiling* — both axes CI-separated. **(2) Anti-collapse taxonomy:** relational repulsion (a one-line uniformity loss) *helps* a collapse-prone geometric navigation latent (0.53→0.95) but *hurts* value-based DMControl control on all 3 tasks tested — and a **value-aware** variant is *worse still* (down to 1.6 AUC on FingerSpin). The lesson: match the anti-collapse term to the downstream objective; for value-based control, add **nothing extra**.

---

## Thread 1 — can a *learned* residual break the contact wall?

[Part 51](../tdmpc-glass-part51-better-grasp-and-the-se-anticollapse-lesson) ended with the Panda solve cleanly characterized: the analytic-skill family caps at ~0.37 success because the tall cube **tips inside the two-finger parallel-jaw grip** during transport, and no amount of analytic-parameter tuning (grasp geometry, gains, upright servo) breaks that tilt wall (Round 6 was NULL). The obvious question: is the wall a *physics* limit (the gripper just can't do it) or a *learning* limit (a smarter controller could)?

So we trained a **residual policy** on top of the analytic skill: `executed = clip(a_opt + α·π_res, −1, 1)`, where `a_opt` is the analytic skill (kept live, with its upright servo) and `π_res` is a brax-PPO policy on the true task reward. We swept the residual authority `α` and ran a **matched-budget vanilla PPO** as the control.

### What happened (real success, n=256, box_target ≥ 0.9, deterministic)

| arm | peak success | note |
|---|---:|---|
| analytic skill (α=0) | 0.082 | the ceiling we're attacking |
| residual α=0.25 (tight) | 0.19 | unstable — diverges to NaN |
| residual α=0.5 (moderate) | 0.40 | peaks then decays |
| **residual α=1.0 (full)** | **0.716 ± 0.014** (n=3) | breaks the wall |
| **vanilla PPO (matched)** | **0.810 ± 0.006** (n=3) | wins the asymptote |

Two findings, both verified at n=3 with **non-overlapping** confidence intervals:

1. **The contact wall is learnable.** A full-authority residual reaches 0.72, far above the 0.37 analytic ceiling. The tilt diagnostic seals it: the learner drives success-case cube-tilt to **~1.9°, *below* the analytic servo's 2.5°**. So the bottleneck was never the gripper morphology — a learner *can* fix the in-grip tilt. It just needs real policy authority: a strictly *bounded* residual (α ≤ 0.5) is unstable and tops out at ~0.40; only near-full authority (α=1.0 — effectively a warm-started-from-analytic policy) breaks through.

2. **But the prior doesn't raise the ceiling.** Matched-budget vanilla PPO reaches **0.810**, beating the residual's 0.716 (residual's best seed 0.731 < vanilla's worst 0.805). What the analytic prior *does* buy is **sample-efficiency: ~1.6× faster to competence** (residual hits 0.45 success at ~11M steps vs vanilla's ~17.5M) and lower seed-variance.

> **The honest one-liner:** the analytic prior is a **speed lever, not a ceiling lever**. This is exactly the campaign's recurring result — abstraction redistributes complexity (here, into faster exploration), it doesn't remove it.

A note on rigor, because it nearly bit us: the *first* interim read had α=1.0 looking *worst* (0.06) — because it learns slowest early. Only the full curves revealed it as the winner. And one vanilla seed initially eval'd to 0.000; its training reward was a healthy 1394 (same as the other seeds), so that was a **checkpoint-eval artifact, not a PPO failure** — re-evaluating gave 0.81. Had we compared the residual's 0.716 to the *old, under-budgeted* "PPO 0.66" instead of the matched 0.81, we'd have falsely claimed a win. (This is the [class-controller budget trap](https://suuttt.github.io/tdmpc-glass) we keep guarding against: a "beats PPO" claim is only as good as its same-budget control.)

## Thread 2 — the anti-collapse taxonomy closes

Part 51 found that the lever for collapse-prone JEPA latents is **relational anti-collapse** (a one-line Wang–Isola uniformity loss), and — surprisingly — that it's **downstream-dependent**: it *wins* on a geometric navigation latent (0.530 → 0.954) but *loses* on value-based DMControl control. We set out to close two loose ends.

**(a) Does "uniformity hurts value control" generalize?** Yes — on the return axis, uniformity is the **worst** arm on all three DMControl tasks (CI-separated on WalkerWalk):

| task | default | uniformity | vicreg |
|---|---:|---:|---:|
| CheetahRun | 58.9 | 36.6 | 56.3 |
| WalkerWalk | 293.8 | 89.9 | 172.3 |
| FingerSpin | 249.4 | 171.8 | 289.7 |

**(b) If uniformity hurts because it ignores value-structure, does a *value-aware* repulsion rescue it?** We built a value-aware variant (down-weight repulsion between value-similar states). It did **not** rescue anything — it's **worse than plain uniformity on all three tasks**, catastrophically so on FingerSpin:

| task (return AUC, n=3) | default | uniformity | **value-aware** |
|---|---:|---:|---:|
| CheetahRun | 58.9 | 36.6 | **20.6** |
| WalkerWalk | 293.8 | 89.9 | **49.5** |
| FingerSpin | 249.4 | 171.8 | **1.6** |

So the conclusion sharpens into a clean **taxonomy** — the right anti-collapse term is the one aligned with what you decode downstream:

| downstream regime | best anti-collapse | evidence |
|---|---|---|
| goal-conditioned **geometric** latent (nav) | relational / uniformity | 0.530 → **0.954** |
| **value-based** control (DMControl) | **none extra** (default wins) | any repulsion — uniform *or* value-aware — hurts |
| (either continuous case) | **not** SE community structure | SE NULL on Panda; partition-independent on nav |

The mechanism is consistent throughout: a relational repulsion term spreads apart states that should be *value-close*, destroying the value-sufficiency that control needs — and making the repulsion "value-aware" in our formulation only spread things *more*. For value-based control, the SimNorm latent's built-in pressure plus the value gradient is already the right amount of anti-collapse. **Adding structure on top is, at best, redundant and, here, harmful** — the same redundancy verdict the whole "glass" program keeps arriving at.

## Where this leaves the campaign

- **Panda is fully characterized.** Solve 0.367 (H-JEPA over an analytic skill, n=7) → analytic family caps at ~0.37 on contact physics → a learned residual breaks that to 0.72 (wall is learnable) → but matched PPO wins the asymptote at 0.81; the prior is a ~1.6× speed lever. A clean, complete arc.
- **The anti-collapse question is closed** with a downstream-dependent taxonomy, all controls run (random-graph, random-partition, strong-VICReg, value-aware).
- Every headline is a deterministic, disk-backed, multi-seed measurement, and every "beats X" is qualified by a matched-budget control.

*A residual-generalization run on a 2nd Panda task (OpenCabinet) is training as this posts — to check whether "prior = speed-not-asymptote" holds beyond PickCube. Results will fold into the ledger.*
