---
layout: post
title: "Thread D — JEPA Anti-Collapse, Done Right (living log — a reversal)"
date: 2026-07-01
description: "Anti-collapse is a JEPA property, so test it on a pure JEPA — not on value-anchored TD-MPC2. Result reverses our earlier story: a pure self-predictive JEPA does NOT collapse on DMControl (state or pixels); anti-collapse ranges neutral to harmful; the BYOL predictor+EMA asymmetry is the load-bearing ingredient; the downstream-dependent taxonomy was nav-specific."
---

> **Living log.** Context: [Part 5 method map](../2026-07-01-tdmpc-glass-part5-beat-ppo-reality-check),
> [Part 6 five bets](../2026-07-02-tdmpc-glass-part6-five-bets-next-phase).

## The bet — and the correction that motivated it

"Anti-collapse" (VICReg, uniformity, structural entropy) is a **JEPA** property. A *pure JEPA* has no
reward/value loss, so anti-collapse is the only thing preventing latent collapse — load-bearing by construction.
But **TD-MPC2 is not a pure JEPA**: its value/reward losses already anchor the latent. So our earlier DMControl
"anti-collapse" study actually measured **redundancy** (a relational regularizer fighting an already-anchored
latent), not JEPA anti-collapse. To ask the JEPA question honestly you strip the value loss. That is Thread D.

Four arms: **D1** SE-arm on TD-MPC2 (a redundancy datapoint, correctly scoped); **D2** a pure self-predictive
JEPA on DMControl state; **D2-ext** firming D2 at fixed λ across more tasks; **D3** pixels, where the information
term should finally matter.

## Status board

| arm | state | verdict |
|---|---|---|
| D2 — pure JEPA on state | ✅ done | **premise falsified — no collapse** |
| D2-ext — fixed-λ, 3 tasks + online | ✅ done | **reversal FIRMED** |
| D3 — pixel JEPA | ✅ done | **NULL — same picture as state** |
| D1 — SE-arm on TD-MPC2 (redundancy) | ✅ done | **SE hurts like uniformity; SE+unif no synergy** |
| open cell — narrow/on-policy data collapse? | ✅ done | **NULL — narrow data doesn't collapse pure JEPA** |
| final cell — is closed-loop feedback the trigger? | ✅ done | **PARTIAL — a soft amplifier, not a clean trigger** |

**Net: Thread D is fully resolved. It *reverses* the Part-5 anti-collapse story; the collapse regime is narrower
than any data-distribution story; and the one setting that did collapse (closed-loop nav) is only *softly*
explained by the loop — the latent is chronically low-rank there regardless, and, either way, representation
collapse stays decoupled from control (success is identical with or without a dead latent).**

## The finding (D2 + D2-ext + D3)

**A pure self-predictive JEPA does not collapse on DMControl — state or pixels.** The `none` arm (zero
anti-collapse) has the *best* readouts; adding an anti-collapse term ranges from neutral to actively harmful. The
load-bearing ingredient is the **predictor + EMA-target (BYOL) asymmetry**, not any explicit repulsion — and that
asymmetry is present in both TD-MPC2 *and* a pure JEPA. LeCun's collapse concern does not materialize even in the
high-dim pixel regime his argument targets. The "downstream-dependent taxonomy" (uniformity helps geometric
readouts, hurts value readouts) was **specific to the narrow nav-collapse regime**, not a general law.

## Progress log

### 2026-07-01 — the final cell: closed-loop feedback is a *soft amplifier*, not a clean trigger
The one setting that ever collapsed was the closed-loop online nav H-JEPA. Direct controlled A/B on the *same*
architecture (matched 6094 updates, \(n{=}3\)), varying only the data-collection policy: **online** (closed-loop,
policy co-evolves with the representation) vs **offline** (fixed random-policy buffer, loop broken).

- **Robust signals** (mid-training eff-rank floor + a \(z\to\)state readout-R²): online hits a dead latent
  (eff-rank ≈ \(10^{-7}\)) on **3/3** seeds with readout R² ≈ 0.19; offline on only **1/3**, keeping 2× more
  decodable structure (R² ≈ 0.41). Breaking the loop **mitigates** collapse — so closed-loop feedback *is* a real
  driver.
- **But it's not a clean trigger, and the metric fights itself.** The *final-eval* eff-rank (noisy, one online
  seed bounces to 1.66) actually reads online 0.55 > offline 0.42 — backwards from the trajectory floor. Both arms
  are *chronically* low-rank on the 4-dim nav task (offline nowhere near the eff-rank 10–22 of broad DMControl),
  so much of the low rank is architecture/task, not the loop. The strong binary hypothesis is a **null**; the weak
  "closed-loop deepens the transient collapse" gets **partial, fragile** support.
- **And it doesn't matter for control:** task success is **identical (~0.64)** with or without a dead latent —
  representation health is decoupled from performance, the same thing every nav probe has found.

So the honest close on Thread D: the nav collapse is not cleanly explained by any single lever — closed-loop
feedback is at most a soft amplifier of a latent that is low-rank anyway, and the collapse is invisible to the
policy. (Caveats: offline used a random, not frozen-expert, buffer; a ≥300k budget with more seeds would sharpen
whether the residual rank-loss is coverage or architecture.) That is the last open question in the thread, closed.

### 2026-07-01 — D1 done + the open cell closed: the collapse regime is *narrower* than "on-policy data"
Two harvests land together and finish the thread.

**D1 — SE arm on TD-MPC2 (the correctly-scoped redundancy datapoint), n=3, L16 collapse config.** Return-AUC vs
the known default/uniformity baselines:

| task | default | uniformity | **SE** | **SE+unif** |
|---|---|---|---|---|
| CheetahRun | 58.9 | 36.6 | 23.4 | 29.0 |
| WalkerWalk | 293.8 | 89.9 | 110.3 | 62.5 |
| FingerSpin | 249.4 | 171.8 | 214.4 | 63.8 |

**SE hurts value-based control just like uniformity** (default > SE on all three), and **SE + uniformity is
*worse* than either alone** — no synergy. Importantly SE is *not* catastrophically over-weighted here (FingerSpin
SE 214 ≈ default 249, not nuked) — so the D2 grad-match artifact does **not** apply on a value-anchored latent,
exactly as D2 predicted (‖∇se‖≈‖∇unif‖ there). Confirms the redundancy law: on a value-sufficient latent the
right anti-collapse is *nothing extra*; any relational term (uniformity, SE, or both) hurts.

**Open cell — does narrow / on-policy data collapse a pure JEPA where broad data doesn't? NULL** (WalkerWalk,
n=2, partial). A distribution-width sweep (broad-random → broad-smooth → on-policy(ARS) → narrow top-return
2–25%) on the same `none` arm. eff_rank never approaches 0 (stays 10–22); geometric decodability holds
(0.5–0.83). Value-R² *does* fall as the buffer narrows (0.11 → −0.05), but that mirrors the narrow buffers'
intrinsically low value variance (raw-obs value-R² is also ~0.04–0.11 there) — a property of the *data*, not
encoder collapse. So the nav H-JEPA collapse (eff_rank≈0) is **not** reproduced by data-distribution width alone.
(Honest caveats: two pipeline bugs — ReacherEasy width buffers never generated, broad-random train arm hit a
shell word-split; "on-policy" here is an ARS refit, not a true closed-loop value-optimizing agent. That last,
genuinely-on-policy closed-loop cell is the only condition still untested.)

**Thread D, closed.** Pure self-predictive JEPA does not collapse on broad DMControl (state or pixels) *or* on
narrow/on-policy-ARS data; anti-collapse is neutral-to-harmful throughout; on a value-anchored TD-MPC2 latent all
relational terms (uniformity, SE, SE+unif) hurt. The one setting that *did* collapse — the nav closed-loop online
H-JEPA — is not explained by data width, so the true collapse trigger is the value/goal closed loop itself, not
the breadth of the data. That is the honest remaining open question.

### 2026-07-01 — D3 (pixels): NULL — anti-collapse not load-bearing on pixels either
Rendered 64×64×3 pixel DMControl (via mujoco_playground MJX dynamics + an EGL renderer, since dm_control's loader
is broken against mujoco 3.8), DrQ-style 4-conv CNN encoder into the D2 harness, latent self-prediction only,
frozen-encoder ridge probes. 2 tasks × 4 fixed-λ arms × n=3 (30 runs).

- **Pure JEPA does not collapse on pixels:** `none` reaches eff-rank 33–34/64 and the best/tied-best readouts on
  both tasks (Walker geom 0.314 / value 0.216; Cheetah geom 0.421 / value 0.168). No arm collapsed.
- **Anti-collapse rescues nothing:** uniformity *maximizes* eff-rank (61/64) but gives the *worst* readouts
  (Walker 0.175/0.109) — the same inverse eff-rank↔readout pattern as state. vicreg ≈ neutral; SE mildly hurts
  value. Order neutral→harmful is identical to the state result.
- The only shift vs state is *lower absolute* geom-R² (0.31 vs 0.77) — decoding qpos from a single frame (no
  velocity, no frame-stack) is simply harder. That is task difficulty, not collapse.

Caveats (honest): single frame caps absolute value-R²; fixed-λ only; small CNN, 12k steps, random buffer; strict
dimensional collapse never observed on any arm. `b3060b:exp/proposal_D3_pixel_jepa/`.

### 2026-07-02 — D2-ext: reversal FIRMED (3 tasks, fixed λ, n=3)
Firmed D2 across `WalkerWalk`/`CheetahRun`/`ReacherEasy` at **fixed λ** (no grad-matching) + a WalkerWalk online
probe. (1) Pure JEPA doesn't collapse — `none` best-or-tied on 6/8 readouts; `ReacherEasy none` value-R² 0.257
≈ 7× raw-obs; WalkerWalk-online `none` beats raw-obs on both readouts (latent *enrichment*). (2) No
downstream-dependent taxonomy — uniformity hurts *both* readouts everywhere; geom and value move together.
(3) Grad-match = artifact confirmed — at fixed λ nothing nukes training. **Two honest self-corrections** to D2's
own claims: `none` is best-or-*tied* (not strictly dominant), and **vicreg** is the most-neutral regularizer, not
SE. Only open cell: truly reward-optimizing on-policy data (the nav regime) — the smooth-correlated probe did not
induce collapse. `b3060b:exp/proposal_D2_pure_jepa/ext/`.

### 2026-07-02 — D2: the pure-JEPA test REVERSES the story (WalkerWalk, n=3)
Correctly-scoped JEPA: encoder + jumpy predictor + EMA target, **no reward/value/policy loss**; frozen-encoder
ridge probes for geometric *and* value(RTG) readouts (differentiable-2D-SE validated vs selib to 3e-7). Three
surprises: **(1) premise falsified** — the `none` arm has the *highest* decodability (geom 0.795, value 0.304,
above raw-obs), so collapse is prevented by the **BYOL predictor+EMA asymmetry**, not the value anchor (my own
"value loss prevents collapse" correction was itself wrong). **(2)** The downstream-dependent taxonomy does *not*
hold on a pure JEPA — uniformity hurts *both* readouts while raising eff-rank 5→40. **(3)** SE ≠ uniformity, and
**grad-norm-matching is a trap**: it inflated SE's weight ~74× and destroyed the readouts — a catastrophic
over-weighting artifact, not intrinsic to SE. This flagged that D1 (SE-arm on TD-MPC2, grad-matched) must be
re-checked with a fixed-λ control before any "SE hurts value-control" claim. `b3060b:exp/proposal_D2_pure_jepa/`.

### Open / next
- **D1** (SE-arm on TD-MPC2) still grinding on b3060 (~4/18) as a redundancy datapoint — harvest and **re-read
  through the fixed-λ lens** (its grad-matched setup may over-weight SE exactly as D2 showed).
- **One open cell:** does a pure JEPA collapse under *truly* reward-optimizing on-policy data (the nav regime)?
  The smooth-policy probe didn't trigger it; this is the honest remaining question for the JEPA story.
