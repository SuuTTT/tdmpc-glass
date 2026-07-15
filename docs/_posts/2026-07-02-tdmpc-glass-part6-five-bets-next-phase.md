---
layout: post
title: "TD-MPC-Glass, Part 6: Five Bets for the Next Phase — Planning-as-Exploration, and Doing JEPA Right"
date: 2026-07-02
description: "After a supervisor review, two corrections reshape the program: (1) the real reason planning beats PPO on exploration-hard tasks is exploration, not asymptote or even learning speed — planning is a directed-exploration operator; (2) anti-collapse is a JEPA property, and our DMControl study measured it on TD-MPC2 (value-anchored), which tests redundancy, not JEPA. From these we lay out five concrete proposals with experiments — the flagship being planning-as-exploration with structural-entropy-discovered subgoals — and the week's schedule to run them in parallel on our GPU boxes."
---

> **⚠ Correction (2026-07-02).** This post asserts as its central correction-of-record that "**planning is a
> directed-exploration operator**." That thesis was subsequently **refuted three ways** by the controlled,
> noise-matched plan-vs-π ablations ([Part 7](../2026-07-03-tdmpc-glass-part7-five-bets-resolved): no coverage
> gain, discovery *earlier* without planning), and the exploration advantage was relocated first to the world
> model and finally — after the SAC control — to an **on-policy pathology of PPO** with the world model as a
> *sample-efficiency* lever ([Part 8](../2026-07-04-tdmpc-glass-part8-chasing-positives)). The five bets below
> mostly resolved null (Part 7/8 scorecards). Also: the "R²≈0.999 value-probe" cited below was retracted as an
> instrument by the [R²-criterion postmortem](../2026-06-17-tdmpc-glass-r2-criterion-postmortem) — the
> redundancy conclusion rests on the direct nulls, not that probe. Kept unedited as the planning document of
> record.

> A planning document. The [Part 5 method map](../2026-07-01-tdmpc-glass-part5-beat-ppo-reality-check) mapped
> where four families (state/behavioral/planning/PPO) win across Panda + DMControl. A supervisor review then
> reframed *why*, and pointed at what to do next. This post is the honest reframe + five proposals we're now
> running in parallel.

## Two corrections that reshape the program

**1. The breakthrough is *exploration*, not learning speed.** Our sharpest result — TD-MPC2 beats PPO on
`HopperHop` **367 vs 33** — is not a ceiling story and not even mainly a "speed" story. It's that **planning is a
directed-exploration operator**: a learned model + lookahead reaches gaits (and, generally, hard-to-reach states)
that PPO's on-policy sampling never finds. On dense/explorable tasks PPO's throughput wins; on
**exploration-bottlenecked** tasks, planning explores its way to a policy PPO can't reach at any budget. That is
the interesting claim, and it subsumes the "speed-not-ceiling" one.

**2. Anti-collapse is a JEPA property, and we tested it on the wrong object.** A *pure JEPA* has no reward/value
loss — anti-collapse is the only thing preventing latent collapse, so it's load-bearing. **TD-MPC2 is not a pure
JEPA**: its value/reward loss already anchors the latent (value-sufficient, linear value-probe \(R^2\approx0.999\)).
So our DMControl "anti-collapse" study actually measured **redundancy** (a relational regularizer fighting an
already-anchored latent), not JEPA anti-collapse. Only the **nav H-JEPA** result — where the latent is genuinely
self-predictive and collapses to effective-rank ≈ 0 — is a real JEPA test. To carry the JEPA question onto
DMControl honestly, you strip the value loss (Proposal D).

## The five bets

### A — Planning as a directed-exploration operator  ★ flagship
**Hypothesis.** On exploration-bottlenecked tasks, model-based planning explores more effectively than model-free
RL, and *that exploration* is the win. Make it explicit and steer it with abstraction.
- **A1 — Isolate.** On `HopperHop` and the actuation-weakened *escape frontier*, measure **state-space coverage**
  (occupancy entropy; number of distinct structural-entropy communities visited) for TD-MPC2 planning vs PPO vs a
  no-plan (\(\pi\)-only) ablation. Show planning → coverage → success.
- **A2 — Amplify.** Add an **intrinsic-novelty bonus to the MPPI objective** (model-disagreement / latent count) —
  Plan2Explore *inside* TD-MPC2. Does it push the escape frontier to even weaker actuation?
- **A3 — Direct (SE).** Run min-2D structural entropy on the replay-buffer latent graph → **communities are
  abstract states, boundaries are bottleneck subgoals** → a high level plans toward *under-visited* communities.
  This uses SE for what SE is actually for — structure discovery — not as a latent regularizer.

**White space.** Plan2Explore / LEXA / MAX / Go-Explore explore via curiosity or disagreement, but nobody uses a
**learned abstraction to define the exploration targets a planner pursues**. We already own SE, a planner, and a
controlled exploration-difficulty axis. That intersection is the paper.

### B — "When does a behavioral prior help RL?" (taxonomy paper) — finalize
The 2-axis taxonomy (**prior-fit × exploration-difficulty**) with the escape-difficulty frontier is drafted and
honest: a fitting prior is a *speed lever*; a mismatched one is *dead weight (an anchor)*; on exploration-hard
tasks it's an *escape*. Finish the figures and submit — it's a clean empirical/analysis contribution.

### C — Abstraction as variance-reduction
On the 16-task benchmark, the SE "glass" latent beats TD-MPC2 on 6/16 tasks — a *wash on the mean* but with
visibly **lower seed-variance** on several (ReacherHard 976±3 vs 883±151). Possible real signal: SE structure
doesn't raise the mean, it **reduces collapse-seed variance / improves worst-case**. Cheap per-seed check first;
report honestly either way (it's a section, not a headline).

### D — JEPA anti-collapse, done right
- **D1** *(running)* — the SE arm on TD-MPC2 DMControl: does SE also fight the value structure like uniformity,
  and does SE+uniformity combine? (A *redundancy* datapoint, correctly scoped.)
- **D2** — the real test: a **pure self-predictive JEPA latent** on DMControl (no reward/value loss), probed on a
  **geometric** readout *and* a **value/return** readout. Does the downstream-dependent taxonomy hold on a genuine
  JEPA?
- **D3** — **pixels**, where the information term is load-bearing: non-generative JEPA vs generative Dreamer world
  model, matched budget. On low-dim state, anti-collapse is redundant (we showed that); on pixels it should matter.

### E — Structural entropy for *structure discovery*, not regularization
We used SE to regularize the latent — redundant, and the wrong bias for continuous geometry. SE's real strength is
community/hierarchy detection → **subgoal / option discovery** and hierarchical planning (this merges into A3).
That's where "glass" belongs after SE-as-representation proved a dead end.

## Running them in parallel (this week)

| box | thread |
|---|---|
| **b3060** (DMControl, 4×3060) | SE arm finishing (D1) → **A1** coverage → **A2** intrinsic-novelty MPPI → **A3** SE-subgoals |
| **b3060b** (Panda/pixels, 4×3060) | beat-PPO tail → **D3** pixel JEPA-vs-Dreamer → **D2** pure-JEPA-on-DMControl |
| no-GPU / writing | **C** per-seed robustness · **B** finalize taxonomy paper · planning-exploration proposal |

Priority if time-boxed: **A1 (de-risk the flagship) > C > B > D2/D3 > A2/A3.** Everything is
deterministic, disk-backed, and reported with matched controls — including the nulls, of which this program has
produced many on purpose. Results will land in the ledger as they complete; a Part 7 will report them.
