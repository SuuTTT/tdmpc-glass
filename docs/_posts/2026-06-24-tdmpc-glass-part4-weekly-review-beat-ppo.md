---
layout: post
title: "TD-MPC-Glass Weekly Review #4 (Jun 17–24): The Beat-PPO Campaign — Abstraction-in-Loop Ties PPO and Wins on Sample-Efficiency"
date: 2026-06-24
description: "Second weekly campaign review, covering Parts 18–27. Last week ended with PPO repeatedly out-competing our world-model/abstraction work and the R² 'redundancy criterion' falsified. This week answered the sharper question head-on: can a learned abstraction actually BEAT PPO on a real manipulation task? The arc: (a) a four-part honest field report (Parts 18–22) — TD-MPC2 beats PPO only on sample-efficiency and exploration-bottlenecked tasks, loses wall-clock and high-dim, and is 5× over-parameterized; (b) the HL-loop campaign (#1–#5c, Parts 24–27) — a ladder from the heuristic (0.063) through value-aware skill-options (0.24, the first positive abstraction) to a persistent in-loop residual at full authority (0.79 ± 0.01), which near-ties end-to-end PPO's 0.81 and crosses competence ~1.3–1.7× faster; (c) a 5-seed shrink-robustness confirmation (Part 23b). Verdict: keeping the abstraction in the loop matches PPO on peak and beats it on sample-efficiency — the answer to 'beat PPO with abstraction' flipped from no to a qualified yes. Plus process: a meaningful-only GPU policy and a recovered silent eval failure."
---

> Second weekly review (the [first](https://suuttt.github.io/projects/2026-06-17-tdmpc-glass-part3-campaign-review/)
> closed with PPO winning and the R² criterion falsified). This week we stopped circling and asked the blunt
> question: **can a learned abstraction beat PPO on a real contact task?** Here's the week, honestly.

## Where we started (last week's close)
PPO kept out-competing the world-model/abstraction work on commodity GPUs, and the "redundancy criterion"
(R²≈1 + nulls) was over-claimed and dropped. Open question: *is the abstraction idea dead, or were we testing
the wrong kind?*

## Part 1 of the week — the honest field report (Parts 18–22)
Before chasing abstraction we pinned down the PPO-vs-world-model baseline truthfully, all read from logs:

- **TD-MPC2 beats PPO only on sample-efficiency** (28–160× fewer env-steps) and **only where exploration is
  the bottleneck** (HopperHop 283 vs 33; Acrobot 420 vs 268). It **ties** on non-exploration "hard" tasks
  (FingerTurnHard 975≈968), **loses wall-clock** (~9×), and its **default config diverges on high-dim Humanoid**.
- **The training bottleneck is the gradient updates, not the planner** (~99% of step time) — so "cheaper
  planning" buys deploy speed, not training speed; ~2–4× is the realistic speed-up, not PPO's ~100×.
- **TD-MPC2 is 5× over-parameterized on standard DMC** — a `tiny` model (0.72M) keeps a median ~99% of return.

Takeaway: model-based/abstraction buys *sample-efficiency and structure*; model-free buys *the ceiling and the
clock*. That framed the real test.

## Part 2 of the week — can abstraction beat PPO? (the HL-loop campaign, Parts 24–27)
We used the hand-coded PickCube controller (HL, ~6% real success) as a scaffold and tried, in sequence, to
beat PPO's **0.81** real success. The ladder (real success = box_target ≥ 0.9, n=256):

| step | approach | real success |
|---|---|---|
| baseline | HL heuristic | 0.063 |
| #1 | raw-action residual (no abstraction) | 0.13 |
| #2 | **value-aware skill-options abstraction** | **0.24** — first positive abstraction |
| #3 | demo-seeded TD-MPC2 | 0 (clean negative) |
| #4 | distill abstraction out (BC / warm-start / DAPG) | 0 / no-beat |
| #5 | in-loop residual, α=0.5 | 0.48 — breaks the structural ceiling |
| **#5b/#5c** | **in-loop residual, persistent α=1.0** | **0.79 ± 0.01** (2 seeds) |
| reference | end-to-end PPO | 0.81 |

**The verdict:** keeping the abstraction *in the loop* — the live controller plus a Markov-conditioned residual
at full standing authority — reaches **0.79, a robust near-tie with PPO's 0.81, and crosses the competence
threshold ~1.3–1.7× faster (≈20–26M vs 32.8M env-steps).** So **on sample-efficiency the abstraction-in-loop
beats PPO**; on raw peak it's a near-tie (no seed cleared 0.82). The "beat PPO with abstraction" question
flipped from *no* to a **qualified yes**.

The negatives along the way did the real work — each ruled out a wrong idea:
- **Distillation fails** because the abstraction is *non-Markov* (its action depends on hidden phase state) —
  cloning it into a flat policy gives 0% despite low action error. *Keep it in the loop; don't distill it out.*
- **Annealing authority backfires** (hand authority to a half-trained residual → non-stationary → collapse).
  *Persistent authority, not a schedule.*
- **A learned per-state authority-gate underperformed uniform full authority**, and **throttling to α=0.85
  only lowered the peak** (0.69). *Simplest in-loop form wins.*

## Part 3 of the week — shrink robustness (Part 23b)
5-seed confirmation that the `tiny` (5×-smaller) model's loss on high-velocity locomotion is real, not noise:
tiny retains **88.7 / 87.5 / 89.2%** on Cheetah/Walker/Hopper (~11–12% loss), with a thin *instability tail*
(one tiny seed peaked then collapsed at its final eval). Rule: `tiny` is the right default everywhere except
locomotion/high-dim and where the *final* checkpoint ships.

## Process notes
- **Meaningful-only GPU policy.** We retired the throwaway keep-busy filler: GPUs now run a *meaningful*
  experiment queue and **idle rather than burn compute** when the backlog is empty.
- **Recovered a silent failure.** One experiment's evaluation died with zero results; tracing the intact
  checkpoints and re-running it recovered the verdict instead of losing a day of compute.

## Where it stands & next week
The headline result is settled and honest: **abstraction-in-loop matches PPO (0.79 vs 0.81) and beats it on
sample-efficiency.** Two things remain open and are next week's targets: (1) **clear PPO's peak** (the last
~0.03), and (2) the real prize — **push real success toward ~0.95** by attacking the ~20% hard-config tail
(neither PPO at 0.81 nor we at 0.79 solve it). Likely levers: a larger residual/world-model (test if 0.79 is
capacity-limited), a hard-config curriculum, and — the abstraction-in-loop's natural advantage — **closed-loop
retry** (detect a failed grasp and re-attempt within the episode). Published this week: Parts 18–27.
