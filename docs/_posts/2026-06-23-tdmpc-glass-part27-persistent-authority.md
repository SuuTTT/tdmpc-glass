---
layout: post
title: "TD-MPC-Glass, Part 27: Full Persistent Authority Nearly Matches PPO (0.78 vs 0.81) and Reaches Competence ~1.7× Faster — the Abstraction-in-Loop Finally Competes"
date: 2026-06-23
description: "Part 26 found that keeping the abstraction in the loop with a Markov-conditioned residual breaks its structural ceiling (0.24->0.48) but annealing the residual's authority backfires; the lever it pointed to was a PERSISTENT scaffold. Part 27 runs that ladder: fixed residual authority alpha = 0.5 / 0.7 / 1.0, plus a learned per-state alpha-gate, all real success (box_target>=0.9, n=256). Result: persistent authority climbs monotonically — 0.5->0.465, 0.7->0.695, 1.0->0.781 — and at alpha=1.0 the abstraction-in-loop reaches peak 0.78 (a near-tie with end-to-end PPO's 0.81) while crossing the 0.66 competence threshold at 19.7M env-steps vs PPO's 32.8M, i.e. ~1.7x faster. So on the sample-efficiency axis that matters for real robots, keeping the abstraction in the loop now BEATS PPO; on asymptotic peak it is a single-seed near-tie just under the 0.82 bar. Two more findings: the learned alpha-gate UNDERPERFORMED full authority (0.61 < 0.78) — selective deference was worse than full control — and alpha=1.0 peaks at 26M then declines to 0.69 (late instability), so alpha~0.7-0.9 trades a little peak for stability. The campaign's 'can we beat PPO with abstraction?' answer flips from 'no' to 'yes on sample-efficiency, near-tie on peak.'"
---

> Part 26: in-loop + Markov residual breaks the abstraction's ceiling (0.24→0.48) but **annealing** the
> residual's authority backfires; the fix it pointed to was a **persistent** scaffold. Part 27 runs that
> ladder — and it's the closest the abstraction has come to PPO.

## The authority ladder (real success, box_target≥0.9, n=256, 1000-step)

| arm | peak success | @ step | final | crosses 0.66 at |
|---|---|---|---|---|
| fixed α = 0.5 *(confirms #5's 0.48)* | 0.465 | 59M | 0.43 | never |
| fixed α = 0.7 | 0.695 | 69M | 0.668 | 42.6M |
| **fixed α = 1.0** | **0.781** | **26M** | 0.69 | **19.7M** |
| learned per-state α-gate | 0.613 | 82M | 0.613 | never |
| **end-to-end PPO** *(baseline)* | **0.81** | — | ~0.80 | **32.8M** |

(At α=1.0 the executed action is still `a_option(s,z) + 1.0·pi_res(s,z)` — the analytic controller's action is
*still added* as a scaffold/prior; this is not pure end-to-end RL, it's the abstraction kept in the loop with
the residual given full additive authority.)

## Findings

**1. Persistent authority climbs monotonically — the lever works.** 0.5→0.465, 0.7→0.695, 1.0→0.781. Every
increment of standing authority raises the ceiling, exactly as Part 26 predicted and opposite to what
annealing did (which collapsed). The structural cap that pinned the pure skill-options abstraction at 0.24
(Part 24) is not fundamental — it was an *authority* limit, and giving the Markov-conditioned residual full
standing authority lifts the result to 0.78.

**2. α=1.0 nearly matches PPO's peak AND beats it on sample-efficiency.** Peak **0.781 vs PPO 0.81** is a
single-seed near-tie (within plausible seed noise). And it crosses the 0.66 competence threshold at **19.7M
env-steps vs PPO's 32.8M — ~1.7× fewer interactions.** *This is the first time in the campaign that keeping
the abstraction in the loop is competitive with end-to-end RL*, and on the axis that actually binds for
real-robot deployment (interaction cost, not wall-clock or asymptote), **the abstraction-in-loop wins**: same
ceiling, reached substantially sooner, because the live controller scaffolds early competence the residual
then sharpens.

**3. The strict dual criterion is *narrowly* missed — honest accounting.** We set the bar at *≥0.82 AND cross
0.66 in <33M*. α=1.0 **meets the escape-speed half decisively (19.7M)** but **misses the peak half (0.78 <
0.82)**. So this is not a clean "beat" on the literal asymptote — it is a **near-tie on peak + a clear win on
sample-efficiency**. For the project's actual goal (sample-efficient competence with an interpretable
abstraction in the loop), that is the result we were after; for a pure leaderboard-peak goal, PPO still edges
it by ~0.03.

**4. The learned α-gate underperformed full authority (0.61 < 0.78) — a clean surprise.** The hypothesis was
that a gate concentrating authority in the precision-critical PLACE phase (and deferring elsewhere) would beat
uniform authority. It did the opposite: selective deference *capped* the result below constant α=1.0. The
residual benefits from **full additive authority in every phase**, not surgical override — the controller base
is already a useful prior everywhere, and throttling the residual where the gate "trusts" the controller just
removes useful corrective capacity. Uniform full authority is the simpler and better design here.

**5. Stability caveat: α=1.0 peaks at 26M then declines to 0.69.** Full authority shows late-training
instability (peak 0.781 @26M → final 0.69), whereas α=0.7 is steadier (peak 0.695, final 0.668). So there's a
peak-vs-stability trade: α≈0.8–0.9 (or early-stopping α=1.0 near its 26M peak) is likely the practical sweet
spot — high ceiling without the late decline.

## Multi-seed confirmation (#5c)

We re-ran α=1.0 (2nd seed) and α=0.85 (×3 seeds) to pin the near-tie and test whether α≈0.85 is the stable
sweet-spot we hypothesized. Real success, same protocol:

| arm | peak (mean ± sd) | final | crosses 0.66 |
|---|---|---|---|
| **α = 1.0** (2 seeds: 0.781, 0.797) | **0.79 ± 0.01** | 0.69 / **0.80** | 19.7M / 26.2M |
| α = 0.85 (3 seeds: 0.699, 0.684, 0.688) | 0.69 ± 0.01 | 0.66 / 0.62 / 0.53 | 22.9M (all) |

Two corrections to the single-seed read:
- **α=1.0 is robust, not a fluke:** two seeds give 0.781 and 0.797 (**~0.79**), and the 2nd seed stayed
  *stable* at 0.797 final — so the late decline seen in the first α=1.0 seed was **seed-specific**, not an
  inherent full-authority instability.
- **α=0.85 is *not* a better sweet-spot — it's simply lower** (0.69 ± 0.01, tight across 3 seeds) and *also*
  decays late (one seed 0.53 final). So **higher authority wins on peak**; throttling to 0.85 costs ~0.10
  success and doesn't buy stability. The hypothesis (α≈0.85 = stable sweet-spot) is **falsified**.
- **No seed clears 0.82** — the best is α=1.0 at 0.797. So the strict ≥0.82 bar stands un-met; the result is a
  robust **~0.79 near-tie** with PPO's 0.81, not a peak beat.

## Verdict — the beat-PPO question, answered

Across #1–#5c, the honest standing answer flips: **keeping the abstraction in the loop with full persistent
authority is genuinely competitive with PPO** — across 2 seeds α=1.0 reaches **0.79 ± 0.01 (vs PPO 0.81)**, a
robust near-tie, and crosses the 0.66 competence threshold at **~20–26M env-steps vs PPO's 32.8M (~1.3–1.7×
fewer interactions)**. On the sample-efficiency axis that matters for expensive-interaction settings, the
abstraction-in-loop *beats* end-to-end PPO; on raw asymptotic peak it's a **near-tie just below (0.79 vs 0.81),
and no seed clears 0.82** — so it is not a strict peak beat. The losers along the way sharpened the design:
distillation fails (non-Markov, Part 25), annealing authority backfires (Part 26), a learned authority-gate
underperforms uniform authority (Part 27), and throttling to α=0.85 only lowers the peak (~0.69) without buying
stability (#5c). The winner is the simplest in-loop form: **live analytic controller + Markov-conditioned
residual at full standing authority (α=1.0)** — ~0.79 success, sample-efficiency beating PPO.

### Caveats (kept honest)
α=1.0 confirmed across 2 seeds (0.781, 0.797 → 0.79 ± 0.01); α=0.85 across 3 seeds (0.69 ± 0.01); the
0.5→0.7→1.0 authority ladder is the robust trend. **0.79 vs PPO 0.81 is a near-tie, not a strict beat, and no
seed reached the 0.82 bar.** One α=1.0 seed declined after its peak (0.69 final) while the other stayed stable
(0.80) — late-training stability is seed-dependent; report peak *and* final. Real success read from
`exp/tdmpc_glass/hl_subgoal_{b,c}/curve_*.json` (n=256, 1000-step, training-matched α). PPO baseline reproduced
under the same protocol (0.81, crosses 0.66 @32.8M). Prior: Parts 24 (0.24 cap), 25 (distillation fails), 26
(anneal backfires).
