---
layout: post
title: "TD-MPC-Glass, Part 32: Why TD-MPC2 Wins HopperHop but Fails PandaPickCube — Exploration in Learning vs Execution"
date: 2026-06-24
description: "TD-MPC2 is a model-based planner that crushes DMC HopperHop (366.8 return, vs PPO's 33.5 after 265M steps) yet scores ZERO real success on PandaPickCube where PPO reaches 0.83. Why does the same algorithm win one and fail the other? We separate the two channels TD-MPC2 actually has — LEARNING-time exploration (does training discover the real objective?) and EXECUTION-time planning (does MPPI at test time improve over the policy prior?) — and measure each from files. EXECUTION: on HopperHop planning helps (mppi > pi at convergence); on PandaPickCube planning HURTS (pi 2520 vs mppi 701 at 1.5M) — but over a reward-hacked value. DISCOVERY (decisive): TD-MPC2 real success = 0.0 on PandaPickCube across vanilla + 6 reward-engineering variants vs PPO 0.83 → the primary failure is LEARNING-time: it never explores its way to grasp→place and instead converges to a hover that maximizes the shaped reward without grasping. MODEL-QUALITY (new k-step leg): phase-binned k-step latent-prediction error tests whether the contact model ALSO degrades — does the error spike at contact/grasp on Panda while staying flat across HopperHop's smooth gait? Verdict: model-based planning helps at execution ONLY when learning has already discovered the real objective; when learning fails first (sparse/long-horizon → reward-hack hover), planning has nothing good to plan over and even hurts. All numbers read from JSON/CSV on the b3060/b3060b workers; cited inline."
---

> TD-MPC2 is a model-based RL agent: it learns a latent world model + value, and at test time it *plans*
> (MPPI) through that model. On dense-reward DMC locomotion it is state-of-the-art. On a sparse, long-horizon
> manipulation task — `PandaPickCube` — it scores **zero real success** while plain PPO reaches ~0.83. Part 32
> asks the mechanism question directly: *why does the same algorithm win HopperHop and fail PandaPickCube?* The
> trap is to assume "it's the planner." TD-MPC2 has **two** channels, and they fail in different places.

## Two channels, measured separately

TD-MPC2's benefit can come from either:

1. **LEARNING-time exploration** — during training, does the agent *discover* the trajectory that achieves the
   real task objective? (For Hopper: a stable hopping gait. For Panda: reach → grasp → lift → place.)
2. **EXECUTION-time planning** — at test time, does MPPI planning through the learned model improve over just
   running the policy prior `pi`? (Δ = `mppi_return − pi_return`.)

We measure all three legs from files. (1) is the *discovery* leg; (2) is the *execution* leg; and a third
**model-quality** leg asks whether the learned model itself is the bottleneck on contact.

## Leg 1 — EXECUTION: planning helps Hopper, hurts Panda

`run_benchmark.py` evaluates BOTH the planner-off policy prior (`eval_pi`: `act = pi(z)`, no planning) and the
planner-on MPPI controller at every eval, logging both to `*_phase.csv`. So the planner-on/off contrast is
already on disk for both tasks.

**HopperHop** (`exp/tdmpc_glass/dmc_ppo_vs_tdmpc2/RESULTS.json`, b3060): TD-MPC2 peaks at **366.8** return at
1M steps. For reference, **PPO on the same task peaks at only 33.5** (after 265M env-steps) and ends at 5.2.
TD-MPC2 wins HopperHop by an order of magnitude. At convergence the planner gives a small consistent positive
lift (the earlier converged b3060 run: pi 338.7 → mppi 366.8, ~+8%, and the first non-zero return at 500k comes
from MPPI only — planning *bootstraps* learning earlier). The fresh b3060b 2-seed reruns
(`HopperHop/seed_{1,2}_phase.csv`) are noisier per-eval (e.g. seed 2 @1.5M: pi 312.0, mppi 321.4) but agree
that planning is neutral-to-positive — never harmful.

**PandaPickCube** (`PandaPickCube_realsucc_vanilla_s1/seed_1_phase.csv`, b3060): the policy prior consistently
**≥** the planner. At 1.0M: pi 2280.1 vs mppi 1793.4; at 1.5M: **pi 2519.7 vs mppi 700.9** — planning *hurts*,
sometimes badly. But read the caveat: these returns are the **reward-hack hover** (see Leg 2). The Δ here is
"planning over a *mis-specified* value landscape," not planning over the true task. MPPI faithfully optimizes a
value that points at the wrong attractor, so more planning = more committed to the wrong thing.

> First-order answer, already on file: **planning helps where the learned value/model is good (Hopper) and is
> neutral-to-harmful where the value points at the wrong attractor (Panda hover).**

## Leg 2 — DISCOVERY: TD-MPC2 never finds the grasp (the decisive failure)

This is the leg that settles it. *Real* success on PandaPickCube = `max over the 1000-step rollout of
box_target ≥ 0.9, gated by a grasp` (identical scoring for all algos).

- **PPO** (`baselines_ppo_sac/PPO_100_RESULT.json` + `.md`, b3060): crosses the success threshold at ~19.7M
  steps and **plateaus at 0.80–0.83** (peak **0.832** @ 108.1M; 24.6M→0.789, 34.4M→0.816, 88.5M→0.820), with
  reached_rate → 1.0 and mean max box_target 0.95. (An earlier 20M-budget PPO run peaked 0.66 @ 32.8M —
  longer budget + variance lifts the ceiling to ~0.83, not 100%.)
- **SAC** (same file): 0.0 success at 20M (reaches the box, reached 0.99, but box_target only 0.077 — never
  places).
- **TD-MPC2 = 0.0 real success, always** (`reward_eng/RESULTS.json`, b3060):
  - vanilla original reward, 3 seeds @1.5M → **peak_success 0.0**, peak_box_target 0.0845 ("success NEVER > 0;
    reached/box_target only flickered after ~1.4M").
  - **6 reward-engineering variants** (v1 reweight; v2 +ungated lift; v3 +lift+grasp; v4 strong lift; v5 low
    gripper-box + strong lift+grasp; v6 strong grasp+box_target) → **ALL peak_success 0.0**, peak box_target
    ≤ 0.06. Shaping the reward to bribe a grasp did not work.
  - The b3060b vanilla TD-MPC2 reruns confirm it: `whyhopper_eval_seed1.json` reports final & peak
    pi/mppi success = 0.0.

> DISCOVERY answer, already on file and decisive: **TD-MPC2 never discovers the grasp→place trajectory.** It
> converges to a *hover* that maximizes the shaped reward (the high `full_reward_rate=1.0` in the diag CSV is
> this hover artifact, not task success) without ever grasping. This is a **LEARNING-time exploration failure**,
> not an execution/planning failure. PPO, with no model and no planning, simply explores its way to the place
> and wins 0.83-to-0.0.

## Leg 3 — MODEL QUALITY: does the contact model also degrade?

The discovery leg already explains the failure. The model-quality leg asks a *contributing-factor* question
(H1): even setting aside exploration, **is TD-MPC2's learned dynamics model worse at contact than at
free-space?** If contact is a representational cliff for the latent model, that compounds the problem — and it
would show up as a *spike* in k-step latent-prediction error at contact/grasp phases on Panda, while HopperHop
(smooth, continuous dynamics) stays flat across its gait cycle.

We measured it directly. For each task we load the trained best-`pi` checkpoint, roll out the policy for 12
episodes, and at every step compute the **iterated k-step latent error**
`e_t = || iter_dyn(z_t, a_{t:t+k}) − enc(obs_{t+k}) ||` (k=4) using the trained 1-step dynamics, then bin it by
a *physical* phase signal:

- **PandaPickCube**: gripper-box proximity `1 − tanh(5·‖box − gripper‖)` (from the obs (obj−gripper) vector).
  Phase = **contact/grasp** (proximity > 0.5, gripper at/near the box) vs **free/approach** (proximity ≤ 0.5).
- **HopperHop**: the foot touch sensors (last two obs dims = `log1p(toe, heel)`). Phase = **stance** (foot in
  contact) vs **flight**.

Tooling: an env-gated `WHYHOPPER_KSTEP=1` path added to `run_benchmark.py` (reuses the existing encoder/policy/
dynamics apply functions; CPU-cheap analysis but GPU-only rollout because the MJX env needs CUDA). Results in
`exp/tdmpc_glass/tdmpc_whyhopper/kstep_{HopperHop,PandaPickCube}_s1.json`.

**Result** (`kstep_HopperHop_s1.json`, `kstep_PandaPickCube_s1.json`, k=4, 12 episodes, ~11,952 steps each,
best-`pi` ckpts):

| task | phase | mean k-step latent err | frac of steps |
|---|---|---|---|
| HopperHop | flight (no contact) | **0.160** | 0.28 |
| HopperHop | stance (foot contact) | **0.112** | 0.72 |
| PandaPickCube | free/approach | **0.146** | 0.18 |
| PandaPickCube | contact/grasp | **0.165** | 0.82 |

- **HopperHop: error does NOT spike at contact — it is *lower*.** Stance (foot-on-ground) error 0.112 vs flight
  0.160, a contact/free ratio of **0.70**, and the error-vs-contact-proximity correlation is slightly
  **negative** (−0.21). The Hopper world model is smooth across the whole gait; the contact-rich stance phase
  (which dominates, 72% of steps, so it's heavily seen in training) is actually the *better-predicted* regime.
  No contact cliff.
- **PandaPickCube: error DOES rise at contact/grasp, but modestly.** Contact/grasp error 0.165 vs free/approach
  0.146 — a contact/free ratio of **1.13** (~+13%), with a small *positive* error-vs-proximity correlation
  (+0.10). So there is a measurable contact-model degradation, in the predicted direction (H1), but it is a
  **bump, not a wall** — the overall Panda k-step error (0.162) is comparable to Hopper's (0.125), i.e. the
  learned dynamics model is *not* catastrophically broken at contact.

> Model-quality answer: H1 (contact-model-error) is a **real but secondary contributing factor** on Panda — the
> error is ~13% higher at contact/grasp than in free space, and the sign is opposite to Hopper (where contact is
> the *easier* regime). It is not the primary cause: a 1.13× bump cannot explain a 0.83→0.0 success gap. The
> primary cause remains the learning-time exploration failure (Leg 2).

## Verdict — exploration in learning vs execution, in what circumstance

Putting the three legs together:

- **Model-based planning helps at EXECUTION only when learning has already discovered the real objective.** On
  **HopperHop** the reward is *dense* and the dynamics are *smooth*, so learning finds a good gait and a good
  value/model; MPPI then plans over a faithful landscape and adds ~+8% (and bootstraps earlier). TD-MPC2 wins.
- **On PandaPickCube the learning fails first.** The task is *sparse* and *long-horizon* (place is gated behind
  a grasp the agent never stumbles into), so TD-MPC2 settles into a reward-hack **hover**. The learned value
  now points at the wrong attractor — so planning has nothing good to plan over, and MPPI *hurts* (pi 2520 vs
  mppi 701) because it optimizes the wrong objective harder. Real success stays 0.0 across vanilla + 6 reward
  variants while PPO reaches 0.83.
- **The decisive axis is *exploration in learning*, not planning at execution.** Planning is a multiplier on
  whatever the learned objective is: a positive multiplier when learning succeeded (Hopper), a *negative* one
  when learning converged to the wrong attractor (Panda). And the new k-step leg shows the **contact model is a
  secondary contributor, not the cause**: Panda's latent error is ~13% higher at contact/grasp than in free
  space (ratio 1.13, vs HopperHop's 0.70 where contact is the *easier* regime), enough to mildly compound the
  problem but nowhere near enough to explain a 0.83→0.0 gap.

The honest one-line summary: **a model-based planner can only plan as well as its learning explored.** Dense
reward + smooth dynamics → learning discovers the objective → planning helps. Sparse, long-horizon, contact-
gated reward → learning reward-hacks a hover → planning has nothing good to optimize and can make it worse.

### Caveats and data

MODEL-QUALITY leg (new): `exp/tdmpc_glass/tdmpc_whyhopper/kstep_{HopperHop,PandaPickCube}_s1.json` (b3060b) —
iterated k-step (k=4) latent-prediction error of the trained 1-step dynamics, 12 pi-rollout episodes per task,
binned by a physical phase signal (foot-touch sensors for Hopper; gripper-box proximity for Panda). Produced by
the `WHYHOPPER_KSTEP=1` path in `run_benchmark.py` (GPU-only rollout, the un-JITted per-step loop is slow —
~15-25 min/task on one RTX 3060 — but cheap; no training, no full-state). Single seed (contributing-factor
probe). HopperHop ckpt = the b3060b 1.5M-step best-pi (mean episode return ~82 on this short eval rollout, a
lower-variance subset of the converged policy); PandaPickCube ckpt = best-pi at 1.4M (mean shaped return ~2813,
the hover).
EXECUTION leg: `exp/tdmpc_glass/dmc_ppo_vs_tdmpc2/RESULTS.json` (HopperHop 366.8 vs PPO 33.5),
`PandaPickCube_realsucc_vanilla_s1/seed_1_phase.csv` (pi≥mppi; 1.5M pi 2519.7 / mppi 700.9), and the b3060b
reruns `HopperHop/seed_{1,2}_phase.csv` + `whyhopper_eval_seed1.json` (all on the b3060/b3060b workers).
DISCOVERY leg: `baselines_ppo_sac/PPO_100_RESULT.{json,md}` (PPO peak 0.832, plateau 0.80–0.83; SAC 0.0),
`reward_eng/RESULTS.json` (TD-MPC2 vanilla + 6 variants all peak_success 0.0). The PandaPickCube returns in the
phase CSV are over the *shaped* reward (reward-hack hover), NOT task success — real success is the gated
box_target≥0.9 metric, which is 0.0 for TD-MPC2. Single seed for the k-step model-quality leg (it is a
contributing-factor probe; the discovery leg is the decisive, multi-seed result). Prior: Parts 25 (can we beat
PPO), 27–31 (the abstraction-in-loop arc on this same Panda family).
