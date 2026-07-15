---
layout: post
title: "TD-MPC-Glass, Part 12: Does TD-MPC2's Hopper Win Actually Come From the World Model? A Critique"
date: 2026-07-08
description: "HopperHop is the one DMControl environment where TD-MPC2 clearly beats both PPO and SAC — and it is also the only task in our five-task grid where TD-MPC2's world-model (consistency) loss can be ablated off with no loss. Those two facts are in tension: the hardest environment for PPO is the one where the 'world model' is dispensable. This post explains the consistency loss in detail, argues that TD-MPC2's Hopper advantage is therefore mis-attributed to the world model and is really an off-policy TD-learning + planning win, and lays out the competing hypotheses (off-policy replay, planning, contact-criticality, credit assignment) with the decisive experiments that would settle which one is doing the work."
---

> A puzzle at the heart of the dissection. **HopperHop is essentially the only DMControl environment where TD-MPC2
> clearly beats both PPO and SAC** — and it is also **the only task in our five-task sufficiency grid where TD-MPC2's
> "world model" loss can be turned off with no loss of performance.** So on the one env where the method most
> convincingly wins, the component it is named after is *dispensable*. This post takes that tension seriously and
> asks: is TD-MPC2's Hopper win actually a *world-model* win — or is the world model getting credit that belongs to
> TD-learning and planning?

## The two facts, precisely

**Fact 1 — TD-MPC2 owns HopperHop.** Tuned on-policy PPO reaches **0/5 seeds ≥ 200 return at 472M steps/seed**
(peak 53.8) and survives the entropy knob (×3, ×10) with no change. SAC (off-policy) crosses 200 in 6/12 seeds by
5M and ~6/9 by 8M. **TD-MPC2 crosses 6/6 by ~1M.** Across the 16-task benchmark this is the cleanest, largest win
TD-MPC2 has over PPO on *both* axes (sample-efficiency and final capability) — most other envs are ties or
sample-efficiency-only.

**Fact 2 — the world-model loss is removable on HopperHop.** Train TD-MPC2 with the consistency loss OFF from
scratch and HopperHop still reaches full performance: stripped seeds **165/475/481/511** (original) +
**306/449/443/477** (replication) = **n=8, with 7/8 seeds inside the full-model band 420 ± 113.** On WalkerRun,
CheetahRun, AcrobotSwingup the same ablation loses **−23% / −38% / −44%** (non-overlapping). Hopper is the outlier.

## What the consistency ("world model") loss actually is

TD-MPC2 encodes an observation to a SimNorm-normalized latent \\(z_t=\text{enc}(o_t)\\) and trains four heads. The
**consistency loss** is the one that makes the model a *world model*: roll the latent dynamics forward,
\\(\hat z_{t+1}=\text{dyn}(z_t,a_t)\\), and pull it toward the *encoded* next observation
\\(\bar z_{t+1}=\text{enc}(o_{t+1})\\) (stop-grad target):

$$\mathcal{L}_{\text{cons}} = \big\lVert \text{dyn}(z_t,a_t) - \text{sg}[\text{enc}(o_{t+1})]\big\rVert^2 .$$

This is the JEPA-style self-prediction objective — the only loss that trains the latent dynamics to *predict the
future*, i.e. the part you can roll forward and plan over. The other three heads are RL/planning machinery:
**value (Q)** — a TD-bootstrapped critic; **policy** — an actor trained to maximize Q and to seed the planner;
**reward** — a head the planner uses to score rollouts. Our five-loss ablation (4 tasks, n≥4) is unambiguous about
which of these is the engine:

| Arm removed | effect |
|---|---|
| value | **fatal, every task** (without the TD value loss it cannot even learn to *stand*) |
| policy | **fatal, every task** |
| reward | planner degrades, but the **policy still reaches full strength** |
| **consistency (world model)** | **mildest cut** — removable on Hopper, load-bearing on dense tasks |

## The critique: Hopper's win is a TD-learning + planning win, not a world-model win

TD-MPC2 is branded and cited as a *world-model* / model-based method. But put the two facts together with the
ablation and a sharper reading emerges:

1. The **engine** is the off-policy **TD actor-critic** (value + policy losses are individually fatal on every
   task). The world model is a *rollout-quality regularizer* on top of it.
2. On **HopperHop specifically**, that regularizer is **dispensable** (removable, n=8). So whatever gives TD-MPC2
   its decisive Hopper edge over PPO and SAC **cannot be the world model** — the win survives deleting it.
3. Therefore TD-MPC2's most convincing win is, mechanistically, a **TD-learning (+ planning) win that the
   "world-model" framing over-claims.** The self-predictive loss earns its keep on *dense* tasks (Walker/Cheetah/
   Acrobot), where precise multi-step rollouts set the return level — not on the env the method is most celebrated
   for.

This is a real, publishable critique of the world-model narrative: *the component doing the winning on the flagship
task is not the component the method is named after.*

## But then what *does* beat PPO on Hopper? Four hypotheses

If not the world model, the Hopper advantage must come from the rest of the stack. Candidates, with the observation
that rules each in or out:

- **H1 — off-policy TD-learning + replay.** PPO is *on-policy*: it learns from fresh rollouts and discards them, so
  a rare successful hop is seen once and forgotten; its policy-gradient/GAE credit assignment is high-variance on a
  contact-gated reward. Off-policy replay (SAC, TD-MPC2) revisits rare success and bootstraps value via TD.
  *Partly true, but insufficient:* SAC is also off-policy yet **much slower** (6/9 by 8M vs TD-MPC2 6/6 by 1M) — so
  off-policy replay explains beating PPO, not beating SAC.
- **H2 — planning (MPPI) on top of value.** TD-MPC2's extra over SAC is the policy-prior-seeded MPPI planner. *But
  the crux:* on Hopper the world model the planner rolls is *removable* — so is planning even helping there, and if
  so, via the model's predictive accuracy or via the value/reward heads it uses to score? **Decisive experiment:**
  policy-only vs MPPI at matched weights on *stripped* HopperHop. If MPPI still helps with a non-self-supervised
  dynamics model, planning's Hopper value is value-scoring, not world-model fidelity.
- **H3 — contact-criticality breaks the on-policy gradient.** The wall is *contact-critical* (contact-free-but-
  unstable Acrobot has no wall). Discontinuous contact makes on-policy advantage estimates unreliable; TD
  bootstrapping + planning are more robust to it. *Consistent with H1/H2, and testable* by grading contact
  stiffness and watching the PPO wall move.
- **H4 — credit assignment on the sparse-ish hop reward.** The value ensemble + policy prior may simply assign
  credit better than PPO's GAE on a reward that only pays when the hop succeeds. Overlaps H1.

## The synthesis, and how to settle it

Our current best account: **HopperHop is exploration-hard but execution-simple.** The hard part is *discovering*
the hopping limit cycle (a contact-critical exploration problem that PPO's on-policy search fails and TD-MPC2's
off-policy value+planning solves); once found, the gait is a low-dimensional periodic behavior the *policy* executes
without needing accurate multi-step latent rollouts — which is exactly why the world model is removable there. On
Walker/Cheetah/Acrobot the *return level* is set by fine continuous control that the planner's rollouts provide, so
the world model is load-bearing.

Three experiments settle which hypothesis carries the Hopper win — all reuse existing harnesses:

1. **Isolation:** SAC vs TD-MPC2 vs *stripped* TD-MPC2 on HopperHop — does stripped-TD-MPC2 still beat SAC's speed?
   If yes, the win is TD-value + planning, not the world model (confirms the critique).
2. **π-only vs MPPI on stripped-Hop:** does planning help once the world model is off? Separates planning-the-
   operator from world-model-fidelity.
3. **Contact-stiffness sweep:** does the PPO wall (and TD-MPC2's margin) track contact-criticality, pinning H3?

## Update (2026-07-09): experiment 2 ran, and it confirms the critique

We ran the decisive probe — **π-only (raw policy) vs MPPI, full-WM vs stripped-WM, on HopperHop at 5M** (n=2 each),
logging both the planner return and the policy return at every eval:

| arm | MPPI return | π-only return | does planning help? |
|---|---|---|---|
| **full world model** | **571** (554/587) | 542 (506/577) | **yes** — MPPI > π on both seeds |
| **stripped world model** | 421 (386/455) | **448** (386/511) | **no** — π ≥ MPPI (tie, then 511 > 455) |

The pattern is exactly what the critique predicts: **MPPI planning beats the raw policy only when the world model is
present; strip the world model and planning stops adding value — the policy alone is as good or better.** So the
planner's contribution on Hopper is *world-model-rollout-quality*, and since the world model is removable there
(stripped ≈ 421–448, within the earlier removable-Hop band ~420 ± 113), **the win that survives is the TD value +
policy, not planning-over-the-world-model.** (Honest caveat: the stripped-vs-full *absolute* gap is n = 2-noisy —
this full pair drew high seeds; the robust signal is the *within-arm* MPPI-vs-π comparison, which is unambiguous.)

Until the remaining probes (isolation vs SAC, contact-stiffness sweep) run, the honest headline stands: **TD-MPC2's
signature win is a TD-learning-and-planning win on a contact-critical exploration task; the world model it is named
after is, on that very task, along for the ride.**
That is not a knock on TD-MPC2 — it is a correction of *why* it wins, and it sharpens where a better world model
could actually matter (the dense/planner-led tasks, and the goal-conditioned/long-horizon regimes outside dense
value-based control).
