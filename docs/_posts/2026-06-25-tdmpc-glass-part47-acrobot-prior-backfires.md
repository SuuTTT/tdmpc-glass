---
layout: post
title: "TD-MPC-Glass, Part 47: The Analytic Prior BACKFIRES on AcrobotSwingup"
date: 2026-06-25
description: "Part 45 showed an analytic energy-shaping prior + in-loop residual ties-to-beats TD-MPC2 on PendulumSwingup. The obvious next test is AcrobotSwingup — the task with the LARGEST planning advantage (+39%, Part 38), the hardest underactuated swing-up. We built the Spong collocated-PFL energy-shaping controller (verified: it pumps the two-link chain to upright energy in ~80% of episodes given enough time, but the +-2 N*m elbow torque CANNOT fully swing up + settle in the standard 1000-step episode, so controller-alone return is only 16.7). Then the same in-loop residual recipe (executed = clip(a_ctrl + residual), reward unchanged, phase exposed, zero-residual reproduces controller-alone to 0.0000). The result inverts the pendulum story: the residual-on-prior (peak ~68, 3 seeds, n=256) LEARNS SLOWER than vanilla PPO (~212) and plateaus far below it; both are crushed by TD-MPC2 mppi (420). The weak analytic prior is not a head start here — it is an anchor PPO must fight (residual loses to plain PPO by ~3x). Honest negative: the abstraction does NOT generalize to the harder swing-up; on Acrobot a bad prior hurts."
---

> Part 45: on PendulumSwingup the analytic energy-shaping prior + learned residual ties-to-beats TD-MPC2 on
> final/mean/stability/sample-efficiency (it loses only on peak). The natural question: does it generalize to
> **AcrobotSwingup** — the task Part 38 measured as having the *largest* planning advantage (+39%), the hardest
> underactuated swing-up in the suite? This post is the honest answer: **no — and not in a boring way. The
> analytic prior actively BACKFIRES.**

## The task and the bar

AcrobotSwingup (dm_control_suite, `mujoco_playground`): a two-link pendulum, torque only on the **elbow**
(gear 2 → ±2 N·m), shoulder is unactuated. Swing both links to vertical (tip at z=4). Dense `tolerance` reward,
1000-step episode. Underactuated, long-horizon, act-against-the-reward — exactly where Part 38 found planning
earns +39%.

**TD-MPC2 baseline (Part 38, 1 seed, read from `AcrobotSwingup_p38_AcrobotSwingup/seed_1_phase.csv`):**

| | peak | final |
|---|---|---|
| pi (reactive) | 364.0 | 301.3 |
| **mppi (planning)** | **419.7** | **419.7** |

That is the bar: ~420 return for the strong (planning) baseline, ~300 for the reactive policy.

## The analytic controller (Spong collocated PFL)

We built the classic Spong acrobot swing-up: total mechanical energy `E` (0 at upright, ~−38 hanging), and a
**collocated partial-feedback-linearization** law that commands the elbow acceleration
`v = −kp·q2 − kd·q̇2 + kE·(E−E_des)·q̇1` — the last term injects energy through the unactuated shoulder DOF —
then solves the 2×2 manipulator equation for the elbow torque that realizes it. Near the top, a PD/LQR balance
catches. Phase ∈ {PUMP, BALANCE} is exposed for the residual. Sign/obs/energy conventions were all measured
empirically on the real env first (`diag_acrobot.py`).

**What the controller can and cannot do (verified, `validate_acrobot.py`, n=256):**

- It DOES pump energy to the top: at KE=8, ~80% of episodes reach `maxE > −2` (essentially upright energy)
  given a 3000-step horizon.
- It does NOT complete the swing-up + settle within the standard **1000-step (10 s)** episode — the ±2 N·m
  elbow torque is simply too weak; the chain reaches the top region but flies through it.
- **Controller-alone return (1000 steps, n=256): 16.7 ± 1.9, swing-up "success" (return ≥ 100) in 17/768
  episodes.** A weak prior. (Contrast pendulum, where the analytic prior alone scored ~823.)

This is itself consistent with Part 38: Acrobot is brutally hard — even TD-MPC2 only reaches ~420/1000.

## The residual recipe (identical to Part 45)

Executed elbow command = `clip(a_ctrl(obs) + α·π_res(obs_aug), −1, 1)`, α=1.0, reward UNCHANGED, obs augmented
with the controller phase one-hot. **Plumbing verified exactly: zero-residual reproduces controller-alone to
max per-episode diff = 0.0000** (`smoke_match.py`). 3 residual seeds + 1 vanilla PPO, identical brax-PPO config
(2048 envs), same env-step budget, all read from the same `eval/episode_reward` (Protocol A, n=128, 1000-step)
TD-MPC2 reports.

## The result — the prior is an anchor

All numbers below are read from disk (`VERDICT.json` + per-run `eval_curve.json` /
`eval_curve_n256.json`); none is hand-typed. Return = Protocol A (1000-step episode reward sum), the same
metric the TD-MPC2 phase CSV reports. Residual/PPO are taken at the matched converged region (all runs
≥ 196M env-steps; the runs were left training and continue past this snapshot).

| method | peak return | final return | n |
|---|---|---|---|
| analytic controller ALONE | — | **16.7** (swing-up 17/768) | n=256 |
| **residual on prior, α=1 (3 seeds)** | **68.1** (62.2 / 81.8 / 60.4) | **66.7** (61.0 / 81.8 / 58.4) | n=256, 3 seeds |
| vanilla PPO (from scratch) | **211.7** | **210.9** | n=256 |
| TD-MPC2 pi (reactive) | 364.0 | 301.3 | Part 38, 1 seed |
| **TD-MPC2 mppi (planning)** | **419.7** | **419.7** | Part 38, 1 seed |

(n=128 brax-log curves agree with the n=256 checkpoint eval: residual 65.7/82.4/63.4 peak, vanilla 216.2 peak.)

The ordering is unambiguous and inverts the pendulum result:

**residual-on-prior (≈68) ≪ vanilla PPO (≈212) ≪ TD-MPC2 mppi (420).**

**The inversion vs pendulum:** vanilla PPO, learning from scratch, climbs steadily to ~212 and keeps inching up;
the residual-on-prior is pinned in the analytic controller's weak basin at ~60–80 across all three seeds — it
loses to **vanilla PPO by ~3×** and to **TD-MPC2 by ~6×**. With α=1.0 the residual *can* fully override the
prior (the executed action is re-clipped to ±1), yet PPO's exploration is anchored to the controller's pumping
behavior and learns the good swing-up far slower than it does with no prior at all. On pendulum the analytic
prior alone already scored ~823 and the residual polished it to ~835; here the prior alone scores **16.7** and
the residual cannot dig out of it.

## Why pendulum generalized and Acrobot did not

The lever is **prior quality**, not task family:

- **Pendulum:** the analytic prior alone already scored ~823 (near TD-MPC2 pi). The residual started in a good
  basin and only had to polish. Good prior → head start → ties-to-beats.
- **Acrobot:** the analytic prior alone scores ~17 — the ±2 N·m torque can't finish the swing-up in 10 s. The
  residual starts in a *bad* basin that biases every rollout toward the weak pumping behavior. Bad prior →
  anchor → slower than scratch.

This is the dual of Part 38's planning rule: **abstraction-in-the-loop helps exactly when the abstraction is
already near-competent on the task; when the analytic prior is weak (because the task's actuation/horizon is
the binding constraint, not the policy class), wiring it into the loop HURTS.** AcrobotSwingup is the worst case
for the recipe precisely because it is the best case for planning (+39%): the thing planning buys — multi-step
lookahead under a tight torque budget — is exactly what a fixed analytic pump cannot supply, and what a learned
residual cannot cheaply add on top of a misleading prior.

## Honest caveats / pending

- TD-MPC2 baseline is 1 seed (Part 38). The gap (residual/PPO ≪ 420) is large enough that seed noise does not
  change the verdict, but it is not a 3-seed bar.
- Runs use the brax default budget; the comparison is taken at a **matched env-step count** across all four runs
  (cited in the table). Every number is read from a JSON/CSV on disk (`VERDICT.json`); none is hand-typed.
- The controller is the best of a gain grid; a stronger analytic acrobot controller (e.g. true LQR balance with
  a longer pump horizon) might raise the prior — but it cannot beat the 1000-step torque-budget wall, which is
  the root cause.

## Verdict

On AcrobotSwingup the analytic-prior + in-loop residual **does NOT beat TD-MPC2** (mppi 420, pi 301) — it does
not even beat **vanilla PPO**. The recipe that ties-to-beats on pendulum **backfires** on the harder swing-up.
The honest, transferable finding: *abstraction-in-the-loop is only as good as the abstraction; a weak analytic
prior on an actuation-limited task is a liability, not a curriculum.*
