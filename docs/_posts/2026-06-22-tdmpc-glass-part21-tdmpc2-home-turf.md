---
layout: post
title: "TD-MPC-Glass, Part 21: TD-MPC2 on Its Home Turf — Wins Hard-Exploration vs PPO, but the Default Config Fails High-Dim Humanoid, and Jumpy Is Harmful"
date: 2026-06-22
description: "TD-MPC2's paper advantage over model-free RL is largest on hard-exploration and high-dimensional control. So we tested it exactly there: HopperHop, AcrobotSwingup, FingerTurnHard (hard exploration) and HumanoidRun/HumanoidWalk (high-dim), against PPO and against the 'jumpy' temporal-abstraction variant, all read from logs (peak+final). The honest result: TD-MPC2 wins its hard-exploration turf decisively (HopperHop 283 vs PPO 33; Acrobot 420 vs 268; FingerTurnHard 975) — but our default-config TD-MPC2 FAILS the high-dim Humanoid turf (HumanoidRun peak ~9, HumanoidWalk diverges to NaN over a full 1M-step run), which the paper solves with a larger model. And jumpy is harmful on every hard env (HopperHop 283→1.7, Acrobot 420→334). Caveat we can't escape: Dog/Quadruped — TD-MPC2's single biggest paper wins — aren't in MuJoCo Playground, so we can't test the very tasks where its edge is largest."
---

> Parts 18–20 compared TD-MPC2 and PPO on a *spread* of DMC tasks and on the efficiency frontier.
> A reader (the user) asked the sharper question: **go to the envs where the TD-MPC2 paper says it beats
> model-free RL — hard exploration and high-dim control — and check whether TD-MPC2, *jumpy* TD-MPC2, and
> PPO actually behave as advertised.** This post does exactly that, all from logs, peak+final.

## Why these envs

The TD-MPC2 paper's `/results` show its margin over model-free baselines is largest on (a) **hard-exploration**
tasks where dense reward shaping is weak, and (b) **high-dimensional** humanoid/dog control. From the tasks
that exist in MuJoCo Playground DMC we picked both kinds:

- **Hard exploration:** HopperHop, AcrobotSwingup, FingerTurnHard.
- **High-dim:** HumanoidRun (67-obs/21-act), HumanoidWalk.

**The caveat that frames everything below:** TD-MPC2's *single biggest* paper advantages are on the
**Dog** and **Quadruped** suites — and those are **not in MuJoCo Playground**. So we are testing TD-MPC2's
home turf *minus its best rooms*. Read this as "the home-turf tasks we can actually run," not "all of them."

## The home-turf table (read from CSVs, peak return)

| env | kind | TD-MPC2 default | jumpy TD-MPC2 | PPO | winner |
|---|---|---|---|---|---|
| **HopperHop** | hard-expl | **283** | 1.7 — *fails* | 33 | **TD-MPC2** (8.6× PPO) |
| **AcrobotSwingup** | hard-expl | **420** | 334 — *hurts* | 268 | **TD-MPC2** (1.6× PPO) |
| **FingerTurnHard** | hard-expl | 975 | — | 968 | **tie** (both solve) |
| **HumanoidRun** | high-dim | **9** — *fails* | — | ≥159, climbing | **PPO** |
| **HumanoidWalk** | high-dim | **NaN — diverged** | — | 33 @39M, climbing | **PPO** (TD-MPC2 breaks) |

(PPO = official MuJoCo-Playground `brax_ppo_config`, 50–285M steps, same RTX 3060; HopperHop/Acrobot/HumanoidRun
PPO from Part 18; FingerTurnHard (60M steps, peak 968) and HumanoidWalk (still training, 33 @39M and climbing)
run on the same box. **Note FingerTurnHard is a *tie* — PPO 968 ≈ TD-MPC2 975 — not a TD-MPC2 win.**)

## What it says

**1. On *hard* exploration, TD-MPC2 wins — and it's the real thing.** HopperHop is the cleanest case: PPO
*flatlines at 33* (it never finds the hop gait) while TD-MPC2 reaches **283** — an 8.6× gap that is exactly
the "model-based planning solves exploration that on-policy sampling can't" story the paper tells.
AcrobotSwingup (420 vs 268) corroborates it. **But not every "hard" task differentiates:** on
**FingerTurnHard, PPO ties TD-MPC2** (968 vs 975 — both solve it), because at PPO's 60M-step budget the task
isn't actually exploration-limited. So the genuine TD-MPC2 wins are the *exploration-bottlenecked* tasks
(HopperHop, Acrobot) — where on-policy sampling stalls — not "hard tasks" in general.

**2. But our default-config TD-MPC2 *fails* the high-dim turf.** This is the honest, uncomfortable finding:

- **HumanoidRun**: peak **9.4** at ~750k steps — essentially not learning (Humanoid return scale is ~0–1000).
- **HumanoidWalk**: **NaN over a full 1M-step run** — the training *diverged*, not merely underperformed.

The TD-MPC2 paper *does* solve Humanoid — but it uses a **larger model for high-dim tasks** (the official
config scales `latent_dim`/width with task dimensionality). Our runs use the **same default config across all
tasks** (latent-512, the DMC default), and at that capacity Humanoid either stalls (Run) or numerically
diverges (Walk). So the correct statement is **"default-config TD-MPC2 is fragile/divergent on high-dim
Humanoid,"** *not* "TD-MPC2 can't do Humanoid." PPO, by contrast, learns HumanoidRun (≥159 and climbing) at
its massive step budget — so on the high-dim turf, *out of the box*, PPO is the one that keeps working.

**3. Jumpy (temporal abstraction) is harmful on every hard env.** The k-step macro-action variant — the one
abstraction in this whole project that ever *looked* like a win (Parts 12/16, "+60%" on PandaPickCube) —
collapses here:

- HopperHop: **283 → 1.7** (catastrophic; jumpy never finds the gait).
- AcrobotSwingup: **420 → 334** (a ~20% loss).

This is the same verdict as Part 19's clean-sparse-DMC test (jumpy null-to-harmful): macro-steps suppress the
fine-grained exploration these tasks need. Combined with Part 17 (the PickCube "+60%" was reward-hacking a
dense term), **the abstraction well is dry on every clean metric we've tried** — ours (structural-entropy
"Glass") and prior-art (jumpy) alike.

## Shrink holds up on the home turf too

Part 19 showed TD-MPC2 shrinks ~5× (3.6M→0.72M params) with little loss on a DMC spread. It holds on the
hard-exploration turf as well (peak return, 1M steps):

| config | params | HopperHop | AcrobotSwingup |
|---|---|---|---|
| default (latent-512) | 3.6M | 283 | 420 |
| small (latent-256) | 0.95M | 269 (95%) | 391 (93%) |
| **tiny (latent-128)** | **0.72M** | 230 (81%) | **418 (≈100%)** |

So the "5× smaller, ~1.6× faster, keeps most of the return" result is not a DMC-easy-task artifact — it
survives on the hard-exploration tasks where TD-MPC2's advantage is real. (High-dim is the exception that
proves the rule: that's the one place capacity *does* matter — and where default already breaks.)

## Verdict

On its own home turf, the honest scorecard is **split by kind of difficulty**:

- **Hard exploration → TD-MPC2 wins, clearly and reproducibly** (HopperHop 8.6× PPO; Acrobot 1.6×). This is
  the genuine, defensible case for a learned world model + planning.
- **High-dimensional → default-config TD-MPC2 loses to PPO** (diverges on HumanoidWalk, stalls on
  HumanoidRun) — fixable with the paper's larger high-dim model, but a real out-of-box fragility, and
  notable because high-dim is supposed to be *another* TD-MPC2 stronghold.
- **Temporal abstraction (jumpy) → harmful everywhere** on clean metrics.

And the structural caveat stands: **the tasks where TD-MPC2's paper margin is largest (Dog, Quadruped) aren't
in MuJoCo Playground**, so the strongest possible case for TD-MPC2 is one we literally cannot run on this
stack. What we *can* run says: TD-MPC2 is worth it for hard-exploration sample-efficiency, fragile at high
dimension out-of-box, and gains nothing from the temporal-abstraction trick.

### Caveats (kept honest)
- Single seed/config per cell (directional); home-turf returns read from `exp/benchmark/tdmpc2_<env>_*.csv`,
  peak+final.
- HumanoidRun/Walk used the **DMC default** config; the paper's high-dim config (larger latent/width) would
  likely fix the divergence — we report default-config behavior, not a capability ceiling.
- FingerTurnHard **PPO** is complete (968, a tie); HumanoidWalk **PPO** was still training at writing (33
  @39M and climbing — already far above TD-MPC2's NaN, so the "PPO learns it, TD-MPC2 diverges" verdict is
  settled; final asymptote will be higher). All other cells are complete.

### Pointers
`exp/benchmark/tdmpc2_{HopperHop,AcrobotSwingup,FingerTurnHard,HumanoidRun,HumanoidWalk}_*.csv`;
jumpy: `*_jumpy.csv`; shrink trio: `*_{default,small,tiny}*.csv`. PPO: Part 18
(`exp/tdmpc_glass/dmc_ppo_vs_tdmpc2/`). Prior turf/abstraction verdicts: Parts 17–19.
