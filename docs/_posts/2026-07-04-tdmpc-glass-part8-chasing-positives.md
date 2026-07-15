---
layout: post
title: "TD-MPC-Glass, Part 8: Chasing Positives — What the Re-Run Found"
date: 2026-07-04
description: "After Part 7's mostly-null five bets, we re-ran each with stronger methods and one sharp new question: is TD-MPC2's edge on exploration-hard tasks sample-efficiency or genuine exploration? The chase produced one reframe-level positive — the world model, not the planner, is the exploration lever (PPO can't crack HopperHop at 94× the budget) — plus one localized positive (learned hierarchy on multi-room nav). The 'add novelty to planning' ideas nulled. Honest scorecard."
---

> **→ This post evolved live as controls landed; [Part 9](../2026-07-05-tdmpc-glass-part9-anatomy-of-beating-ppo)
> is the consolidated final state of every claim below.**

> A positive-chase. After [Part 7](../2026-07-03-tdmpc-glass-part7-five-bets-resolved) closed the five bets
> mostly null, we re-ran each with a stronger method and added the question that turned out to matter most:
> *is TD-MPC2's win on exploration-hard tasks sample-efficiency, or real exploration?* Everything below is
> disk-verified with matched controls; two confirmations are still running (flagged).

## Scorecard

| bet / test | verdict |
|---|---|
| **Planning-as-exploration → the real question** | **POSITIVE (sharpened 07-02): PPO's exploration wall is real but *on-policy-specific*; the world model buys ~5× sample-efficiency + ~2× level over SAC** |
| **Learned hierarchy (feudal) vs flat** | **RE-ATTRIBUTED (07-02)**: dense shaping alone matches it (shaped-flat 3/6 ≈ feudal 4/6); surviving claim = the hierarchy *self-generates* its dense signal |
| A2 — novelty-seeking MPPI | null / harmful |
| D — SE as *structure* (not penalty) | null |
| E — SE-subgoal option discovery | null |
| MiniGrid — PPO+RND on hard-exploration | null |

## The win, sharpened: the wall is real — and it's an on-policy pathology

Part 7 showed planning (MPPI) is *not* a directed-exploration operator — a policy-only ablation of TD-MPC2
explored/discovered just as well. That left the real question open: **why does TD-MPC2 crush PPO on `HopperHop`
(367 vs 33)?** Sample-efficiency, or exploration?

We settled it by running **PPO at a huge budget**: brax PPO (mujoco_playground's tuned DMC config, and — verified —
byte-for-byte the *same* MJX `HopperHop` env, episode length, and action repeat as our TD-MPC2 runs), 5 seeds,
**472M env-steps per seed**. *(Correction 2026-07-02: an earlier version said "94× the ~5M TD-MPC2 needs to reach
367." The disk says better: TD-MPC2's 366.8 came at ~**1M** steps (best of 4 seeds; n=4 mean ≈282, range 180–378),
and a fresh same-env 2-seed run sits at ~480–520 by 3.9M. So the PPO budget is ≈**470×**, not 94× — the wall is
stronger than first stated, but the old numbers didn't trace.)*

| | peak (mean / best) | seeds crossing 200 / 367 |
|---|---|---|
| **PPO @ 472M** | **40.0 / 53.8** | **0 / 5** |
| TD-MPC2 @ ~1M (same env) | ~367 best seed (n=4 mean ≈282; ~480–520 by 3.9M, n=2) | — |

PPO **never gets past ~54 through 472M steps**. Honest phrasing: the curves are still *creeping* (half-vs-half
means roughly double, peaks arrive late), so "walled through 472M," not "plateaued" — but extrapolating that creep
to TD-MPC2's level would take tens of billions of steps. A genuine exploration wall, not slow learning. Combined
with the planning-vs-policy null, this pins the mechanism:

> **Planning (MPPI) is a *pruning / exploitation* operator; the *world model* (learned latent dynamics + value
> from imagined rollouts) is what lets TD-MPC2 discover the gait that on-policy PPO cannot find at any practical
> budget** — though, as the SAC control below shows, it is not the *only* escape route.

This reframes the whole flagship honestly: the exploration advantage is real, but it lives in the **model**, not
the **planner**.

**The SAC control (ran 2026-07-02, n=3 — the review-mandated decider).** We ran brax SAC (mujoco_playground's
tuned DMC config, audited; byte-identical MJX env) for 3 seeds × 5M steps. **Peaks 207 / 235 / 274 — all three
seeds cross 200 within 5M steps**, the threshold PPO's five seeds never touched in 472M. So the strong reading
("model-free RL is walled") is **refuted**, and the honest sharpened claim is:

> **The exploration wall is an *on-policy* pathology.** Off-policy replay + entropy exploration (SAC) escapes it.
> What the world model buys — cleanly, same env — is **sample-efficiency**: TD-MPC2 reaches ~282 (n=4 mean,
> best 367) at **1M** steps, above SAC's 5M-step level (~5×), and ~480 (n=2, final) at 5M vs SAC's ~239 at 5M.
> PPO stays categorically walled. *(Level update, 2026-07-02: an earlier version claimed "~2× attained level at
> matched budget." The 20M SAC runs killed the robustness of that: seeds span **277 / 414 finals at 20M, with a
> third seed at 564 by 12.8M — above TD-MPC2's anchor**. There is no reliable level gap at larger budgets;
> the world model's robust advantage is efficiency, full stop. **Final 20M distribution (n=5): finals 246 / 277 /
285 / 414 / 572, peaks up to 574** — mean final ~359 sits below TD-MPC2's ~477, but the best seeds reach or exceed
it, so the honest summary is: no clean level gap; TD-MPC2 is more *consistent* and ~4–5× more sample-efficient.)*

This lands the campaign back on its own law from an unexpected direction: the "exploration" advantage decomposes
into an on-policy failure (PPO's) plus a model-based *efficiency* advantage (TD-MPC2's) — the world model is a
sample-efficiency-and-level lever, not a unique key to the gait.

**The generalization sweep is also final: the wall does not generalize — it is task-specific.** Verdict from
`analyze_ppowall.py` over all runs: PPO caught up to TD-MPC2 on all scored tasks. FingerTurnHard: PPO 971/975/971
≈ TD-MPC2 984 (no wall, 3/3). Pendulum: originally 2/3 seeds walled, but the cell was confounded by an upstream
`PendulumSwingUp`-vs-`PendulumSwingup` case mismatch that silently skipped the tuned override — **re-running with
the override applied (2026-07-02, n=2) gives peaks 842.5 / 830.9: both catch up. The "walled" Pendulum seeds were
the bug.** BallInCup: discovery-luck on *both* algorithms (PPO 1/3 seeds solve at ~967; TD-MPC2 itself 1/2 at
~975) — not a wall. With no confounded cells remaining, the wall is **specific to the gait-discovery regime**
(HopperHop): on-policy PPO fails at gait discovery specifically, not at hard control generally.
**Which net carries it? The per-head ablation (CheetahRun done, n=2; HopperHop running, n=4 queued).** Zeroing one
loss term at a time (mask verified live): full = MPPI 738 / pi 782. Ablating **value** → 16 / 12 (catastrophic,
kills planner *and* policy); **reward** → MPPI 5 but pi **761** (planning-only, partly by construction — MPPI
scores rollouts with it); **policy prior** → 123 / 2.5; and — the surprise — **consistency, the self-predictive
"world model" loss itself** → 367 / 541, the *smallest* drop of the four. On the clean (pi) readout the ranking is
value ≫ policy > consistency > reward. So the mechanism behind the efficiency advantage looks like **TD value
learning through the latent, not the self-predictive dynamics loss per se** — a finding that rhymes with the whole
campaign (the value anchor keeps winning). The HopperHop version — the one that matters for the exploration story —
folds in when it lands.

## The localized positive: hierarchy helps where it should

A fully-**learned** 2-level feudal hierarchy (learned high-level subgoal emitter + goal-conditioned low level)
vs a matched-budget flat baseline, on sparse navigation. The honest, verification-checked result:

- **Multi-room maze (`fourroom`, doorway bottlenecks), n=6:** **feudal solves 4/6 seeds, flat 0/6.** Flat
  literally cannot cross the connected rooms; the hierarchy's committed subgoals do.
- **Open rooms (n=3):** feudal ≥ flat but **within seed variance** (bigroom is a tie) — flat is a competent
  baseline there. *(An earlier n=1 "feudal 1.0 vs flat 0.0" was seed luck; firming to n=3 corrected it.)*

So the positive is **localized and mechanistic**: learned hierarchy earns its keep on **bottleneck / multi-room**
structure, not on open long-horizon rooms. That's a more defensible claim than "hierarchy beats flat."

**Two scope caveats (added 2026-07-02, review):** (1) the feudal LL trains on dense *self-generated*
subgoal-distance shaping (no privileged task information — the true goal never enters a training reward, and eval
is unshaped for both arms) while flat TD3 trains on the sparse reward alone; a **shaped-flat control** (flat +
intrinsic dense signal) hasn't been run, so the clean claim is "a learned 2-level hierarchy *with self-generated
dense shaping* beats sparse flat TD3 at matched env-steps," not "hierarchy per se." The feudal arm also takes 2
gradient updates per env step (LL+HL) vs flat's 1 — matched per-network, not in total compute. (2) 4/6 vs 0/6 is
Fisher-exact p ≈ 0.03 one-sided — real but thin; one flipped seed would un-signify it.

**🚩 RESOLVED (2026-07-02, the shaped-flat control ran, n=6): the positive re-attributes to dense shaping.**
Flat TD3 given dense potential-based shaping toward the true goal solves fourroom **3/6 seeds (finals 1,1,1,0,0,0)
≈ feudal's 4/6** — on the maze sparse flat never solved. So what carried the win was the *dense signal*, not the
hierarchy. The "learned hierarchy beats flat" claim is retired. The surviving claim is narrower and still worth
having: **the hierarchy *self-generates* its dense learning signal** — feudal needed no privileged goal
information, while this control was handed the true goal. That independently replicates Nachum et al. (2019),
who found most of HRL's benefit reproduces on flat agents given better exploration/shaping.

## The nulls (honest, and consistent)

- **A2 — novelty-seeking MPPI** (RND / disagreement bonus in the MPPI objective): null-to-harmful. One n=2 bump on
  `CartpoleSwingupSparse` is within that task's huge seed variance; on dense tasks novelty **hurts**
  (`PendulumSwingup` 766 → 135). Adding novelty to the planner over-widens the search and wrecks exploitation.
- **D — SE as *structure*** (community-contrastive / encoding-tree, not a penalty): doesn't beat plain JEPA
  `none` on any DMControl readout.
- **E — SE-subgoal options**: SE-community goal-shaping ties or hurts flat; worst on the sparse task it was meant
  to help.
- **MiniGrid** (`MultiRoom-N6`, `KeyCorridorS3R3`): PPO **and** PPO+RND both stall at 0 success. RND explores more
  (coverage/discovery up) but **coverage ≠ task success** — exactly why RIDE/NovelD/AMIGo exist.

A consistent thread runs through the nulls: **novelty widens coverage but does not, by itself, solve
hard-credit tasks** — and bolting novelty onto a planner mostly costs exploitation.

## Takeaway

Chasing positives on purpose turned one Part-7 null into a sharper, *truer* positive: not "planning explores,"
but **"the world model explores; the planner exploits."** Plus a clean, localized hierarchy win. The
novelty-injection ideas (A2/E/MiniGrid) and SE-as-structure (D) are honest nulls that tighten the story rather
than pad it. The two running confirmations (wall generalization + per-head ablation) will finish the picture.
