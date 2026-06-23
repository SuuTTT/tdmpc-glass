---
layout: post
title: "TD-MPC-Glass, Part 26: Keeping the Abstraction In the Loop Breaks Its Ceiling (0.24→0.48) — but Doesn't Beat PPO, and Annealing Authority Backfires"
date: 2026-06-23
description: "Part 25 showed you cannot beat PPO by distilling the abstraction OUT (it's non-Markov; BC/warm-start/DAPG all failed). Part 26 tests the opposite: keep the analytic option-controller LIVE and learn a residual on top, made Markov-correct by feeding the controller's phase z into the observation (the #4 fix), with the residual's authority alpha either fixed or annealed 0->1. Three findings, all real success (box_target>=0.9, n=256): (1) POSITIVE — keeping the abstraction in-loop with Markov conditioning DOES break its structural ceiling: a fixed-authority residual (alpha=0.5) reaches 0.48 real success, 2x the pure skill-options abstraction's 0.24 and ~4x the raw residual's 0.11. (2) NO-GO vs PPO — 0.48 is still well below end-to-end PPO's 0.81 and never crosses 0.66; at alpha=0.5 the boxes reach mean box_target 0.84 but not the 0.9 success bar (a final-precision cap). (3) NEGATIVE MECHANISM — annealing alpha 0->1 backfires: handing full authority to a half-trained residual is a non-stationary target that collapses the controller-inherited competence (success falls to ~0.01 as alpha->1). The smoke check (alpha=0 -> 0.094, the controller baseline, not 0) confirms the Markov fix worked. Next lever: a persistent scaffold — higher fixed alpha or a learned per-state alpha-gate, not a global time-anneal."
---

> Part 25's lesson: the abstraction's competence is **non-Markov** and cannot be distilled into a flat
> policy (BC/warm-start/DAPG all failed to beat from-scratch PPO). Part 26 takes the only remaining route —
> **don't distill it out; keep it in the loop** — and asks whether that can finally beat PPO.

## Design (the #4 fix, applied)

A brax-PPO-trainable wrapper of PandaPickCube whose executed action is
```
a_t = clip( a_option(s_t, z_t)  +  alpha(t) · pi_res(s_t, z_t) ,  -1, 1 )
```
- `a_option` = the analytic skill-options controller, kept **live** (its phase machine runs every step).
- **Markov fix:** the controller's phase/option `z_t` is fed **into the observation** (one-hot phase + sub-goal
  offset + grip intent → obs 66→77), so `pi_res` is Markov in `(s, z)`. No distillation.
- `alpha(t)` = the residual's authority: either **fixed** or **annealed 0→1**.
- Milestone shaping (+0.2 grasp, +0.3 lift) on top of task reward; **selection/report on true success only**.
- `pi_res` trained with brax PPO; `z` reset on episode boundary (handles autoreset).

**Validation that the fix is real:** at `alpha=0` with zero residual, the system reproduces the controller's
**0.094 real success (not 0)** — confirming `z` is threaded and the controller is genuinely live (this is
exactly what Part 25's memoryless clone could not do).

## Results (real success, n=256, 1000-step, box_target≥0.9)

| variant | peak success | note |
|---|---|---|
| **fixed alpha = 0.5** | **0.48** @36M (mean box_target 0.84, reached 0.98) | **2× the #2 abstraction (0.24), ~4× #1 (0.11)** |
| anneal 0→1 by 10M | 0.07 → collapses to **0.008** | fails as alpha→1 |
| anneal 0→1 by 20M (×2 seeds) | ~0.10 → declines to ~0.04 | fails as alpha→1 |
| *(anchors)* | scratch PPO **0.81**, crosses 0.66 @32.8M · #2 0.24 · #1 0.11 · controller 0.063 | |

## The three findings

**1. (Positive) In-loop + Markov conditioning breaks the structural ceiling.** A fixed-authority residual
over the live controller reaches **0.48** — *double* the pure skill-options abstraction's 0.24, which was
capped by the analytic controller's structure. This is the direct confirmation of Part 25's diagnosis: the
ceiling was the *non-Markov, can't-be-overridden* controller, and giving a **Markov-conditioned residual
standing authority** lifts it. Keeping the abstraction in the loop is the right move.

**2. (No-go vs PPO) It still doesn't beat end-to-end RL.** 0.48 is well short of PPO's 0.81, and it never
crosses the 0.66 threshold. The mechanism is visible in the telemetry: at `alpha=0.5` the boxes get to
**mean box_target 0.84 with reached 0.98** — the residual reliably grasps and lifts and *almost* places, but
the half-authority cap limits the final-placement precision needed to clear the 0.9 success bar. The
abstraction-in-loop buys a much higher ceiling than before, but not PPO's.

**3. (Negative mechanism) Annealing authority backfires — and this was our own hypothesis, falsified.** The
plan was `alpha: 0→1` so the policy inherits the controller early then takes over. It does the opposite:
**handing full authority to a half-trained residual is a non-stationary control target**, and success
*collapses* (to ~0.01) precisely as `alpha→1`. The controller-inherited competence is destroyed faster than
the residual can replace it. **Fixed authority strictly beat annealed authority** here — a clean,
counter-intuitive result worth recording.

## Verdict and next lever

**Keeping the abstraction in the loop is the best abstraction result yet (0.48, 2× the prior ceiling) — but
it does not beat PPO (0.81), and time-annealing the residual's authority is harmful.** The honest standing of
the whole "beat PPO *with* abstraction" question after #1–#5: every route improves on the heuristic and on
prior abstractions, none beats unconstrained end-to-end RL; the abstraction's value remains *fast, structured,
interpretable competence at low budget*, not peak.

The result points to one specific untested lever: a **persistent scaffold** rather than a vanishing one —
either a **higher fixed authority** (`alpha` 0.7–1.0, since 0.5 was still placement-limited and not clearly
asymptoted) or, better, a **learned per-state `alpha`-gate** that lets the policy decide *where* to override
the controller (full authority for the precision-critical place phase, deference elsewhere) instead of a
global time schedule. That is launched as the follow-up (#5b).

### Caveats (kept honest)
Single seed for the fixed-`alpha` arm (annealing has 2 seeds, both fail consistently); the fixed-0.5 curve is
noisy in ~0.35–0.48 and was stopped before a clear asymptote, so 0.48 is a lower-ish bound on that
configuration. Real success read from `exp/tdmpc_glass/hl_subgoal/RESULTS.json` (n=256, 1000-step,
training-matched alpha). Scratch-PPO baseline reproduced under the same protocol. Prior: Parts 24 (the 0.24
abstraction), 25 (distillation fails).
