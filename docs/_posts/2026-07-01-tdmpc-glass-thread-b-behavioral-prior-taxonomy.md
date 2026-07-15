---
layout: post
title: "Thread B — When Does a Behavioral Prior Help RL? (taxonomy paper, living log)"
date: 2026-07-01
description: "The clean empirical contribution: a 2-axis taxonomy (prior-fit × exploration-difficulty) plus the escape-difficulty frontier. A fitting prior is a speed lever; a mismatched one is dead weight; on exploration-hard tasks it's an escape. Drafted and honest; this log tracks it to submission."
---

> **Living log.** Updated when the draft moves. Context: [Part 5 method map](../2026-07-01-tdmpc-glass-part5-beat-ppo-reality-check),
> [Part 6 five bets](../2026-07-02-tdmpc-glass-part6-five-bets-next-phase).

## The bet

Not a method — a **taxonomy**. The question "when does injecting a behavioral prior (an analytic skill, a
hand-controller, a structured latent) actually help RL?" has a clean 2-axis answer we've measured across Panda +
DMControl:

- **Axis 1 — prior-fit.** Does the prior match the task's solution manifold?
- **Axis 2 — exploration-difficulty.** Can on-policy sampling find the solution unaided?

The three regimes that fall out:

| regime | prior-fit | exploration | outcome |
|---|---|---|---|
| **speed lever** | fits | easy | prior buys sample-efficiency (~1.6–7× faster to competence) |
| **anchor (dead weight)** | mismatched | easy | vanilla wins; the prior only slows you down |
| **escape** | fits | hard | *only* the prior survives — vanilla never escapes |

The **escape-difficulty frontier** operationalizes axis 2: progressively weaken a task's actuation until it
crosses from "PPO solves it" to "only the prior-equipped agent survives." That crossing is the figure.

## Status board

| item | state |
|---|---|
| Draft (`paper_speed_of_learning.tex`, ~12pp, compiles) | ✅ done |
| Anchor case (locomotion CPG — vanilla wins 5/5) | ✅ verified |
| Speed-lever case (Reacher, OpenCabinet) | ✅ verified |
| Escape frontier (actuation sweep) | ✅ verified |
| F1–F5 hierarchy figures, related-work sweep, author block | ⏳ finalize |
| Submit (workshop → conference) | ⏳ pending |

## Progress log

### 2026-07-01 — status: drafted, honest, finishable now
The paper is written and compiles. It's the *safest* of the five bets: a clean empirical/analysis contribution
that doesn't depend on any pending experiment. Reframed after the supervisor review to lead with **learning
speed** as the axis abstraction actually buys (not ceiling), which is the honest through-line of the whole
campaign. Remaining work is presentation, not science: finalize the F1–F5 figures, tighten related work, add the
author block. No GPU required. **This is the paper to land first.**

### Supporting results already in the ledger
- **Structured prior (analytic skill + learned residual)** = sample-efficiency lever, *not* a ceiling lever, vs
  matched-budget PPO (Panda PickCube 0.716 < 0.810 final; OpenCabinet tie; ~1.6× / ~7× faster to competence).
- **Anchor case:** locomotion CPG prior — vanilla TD-MPC2 wins 5/5 (mismatched prior = dead weight).
- **Escape case:** on exploration-hard tasks (HopperHop, sparse swing-ups) the planning/prior agent is the only
  one that reaches competence at a practical budget.
