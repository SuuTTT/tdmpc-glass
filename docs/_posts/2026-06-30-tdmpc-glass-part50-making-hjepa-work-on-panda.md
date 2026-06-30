---
layout: post
title: "TD-MPC-Glass, Part 50: Making H-JEPA Work on a Robot — from a flat 0% to solving PandaPickCube"
date: 2026-06-30
description: "A from-scratch explainer of JEPA and Hierarchical JEPA (H-JEPA), and the honest engineering story of getting a faithful H-JEPA to actually solve a manipulation task. We start at 0.0 success, diagnose exactly why, validate the machinery on a toy maze, then climb to 0.289 real success — beating the hand-tuned hierarchy — by giving the high-level latent planner a competent low-level primitive and warm-starting it. Along the way we catch (and fix) a critic bug and an XLA-nondeterminism confound, and we bound the whole approach with an oracle. The recurring lesson: the lever is a competent low-level primitive, not learned hierarchical abstraction by itself."
---

> **TL;DR.** We built a faithful, minimal **H-JEPA** (LeCun's hierarchical world-model architecture) and asked it to solve `PandaPickCube`. From scratch it scores **0.0**. We show *why* (the contact primitive is unlearnable from scratch), prove the architecture is sound on a toy task (**0.94**), then climb to **0.289** real success — **above the hand-tuned hierarchy (0.246)** — by handing the high-level latent planner a competent low-level skill and warm-starting it. The honest punchline: H-JEPA's hierarchy/planning works, but the thing that actually moves the needle is a *competent low-level primitive*, not learned abstraction by itself.

---

## 1. What is JEPA, in one picture

A **world model** predicts what happens next. The dominant recipe (Dreamer-style) is **generative**: encode the observation, predict the *next observation*, and train by reconstructing pixels/state. That works, but it spends enormous capacity predicting details that don't matter (exact textures, every background pixel).

**JEPA — Joint-Embedding Predictive Architecture** (Yann LeCun's program) makes one decisive change: **predict in representation space, not observation space.** Concretely:

- An **encoder** maps an observation to a latent: `z = enc(o)`.
- A **predictor** predicts the *future latent* from the current latent and the action: `ẑ_{t+k} = pred(z_t, a_t…a_{t+k-1})`.
- The loss is in **latent space**: make `ẑ_{t+k}` match `sg(enc_target(o_{t+k}))` — the encoded *real* future, where `enc_target` is an EMA (slow-moving) copy of the encoder and `sg` is stop-gradient (the BYOL trick).
- **No decoder. No reconstruction.** The model never tries to draw the future; it only has to get the *abstract* future right.

There's one catch that defines the whole field: **collapse.** If you only ask "predict your own future embedding," the trivial winner is to map *everything* to a constant — then prediction is perfect and the representation is useless. JEPA prevents this with an **anti-collapse term** (here, VICReg-style variance + covariance regularizers that force the latent dimensions to stay spread out and decorrelated). LeCun's framing: you need an *information* term to keep the representation rich, and that term is the hard part — you can't measure "information content" exactly, you can only regularize a proxy.

> **The JEPA bet, stated plainly:** intelligence is about predicting the *consequences of actions in an abstract space*, not rendering the world. Don't model what you can't (and needn't) predict.

## 2. What H-JEPA adds: hierarchy and planning

A single JEPA predicts one step (or one *k*-step jump) ahead in latent space. **Hierarchical JEPA (H-JEPA)** stacks them across timescales:

- The **low level (LL)** acts at every timestep.
- The **high level (HL)** runs slowly — every *k* steps it emits a **subgoal**, expressed as a point/direction in the *learned latent space*, for the low level to achieve.
- **Hierarchical planning:** the HL doesn't just react — it can *plan* over its learned latent world model (roll the predictor forward over candidate subgoals, score them with a value, pick the best). Each level plans coarse-to-fine: the HL sketches "where to go next" in abstract terms, the LL fills in the motor details.

This is the architecture LeCun argues is the path to agents that plan and handle novelty. **Our question for this whole campaign: is *learned hierarchical abstraction* the lever that unlocks hard tasks — or is something else doing the work?**

## 3. Our faithful minimal H-JEPA

We implemented H-JEPA keeping all four defining properties (so the test is fair):

1. **Non-generative** — no decoder, ever. Only latent-space losses.
2. **Latent-predictive with anti-collapse** — a *jumpy* predictor `pred(z, a…)→ẑ` trained against an **EMA-target encoder** + **VICReg** variance/covariance. (We use a SimNorm latent — a simplex-normalized code that has built-in anti-collapse pressure — and *still* verify it doesn't collapse.)
3. **Two levels** — the HL emits a latent subgoal every *k≈150* steps; the LL is goal-conditioned and acts every step.
4. **Latent-space planning at the HL** — CEM/MPPI over the jumpy predictor, scored by a learned value.

Then we pointed it at **`PandaPickCube`**: a 7-DoF Franka arm must reach a small cube, grasp it, lift it, and place it upright at a target (`box_target ≥ 0.9`). It's the regime where a hierarchy *should* shine: long-horizon, multi-phase, sparse credit.

## 4. The honest journey: 0.0 → 0.289

Here's the whole arc, every number a deterministic, held-out, real-success eval (`box_target ≥ 0.9`, n = 256–512):

| stage | success | what it taught us |
|---|---:|---|
| Flat / faithful H-JEPA, from scratch | **0.000** | the NULL we started with |
| + competent low-level primitive | 0.117 | crosses competence; grasp/lift solved |
| + planning fixes (stability, anchor) | 0.117 | better sample-eff, **same ceiling** |
| + warm-start + anchor (final) | **0.289** | **beats the hand-tuned hierarchy** |
| — hand-tuned reactive hierarchy | 0.227–0.246 | the baseline we beat |
| — analytic-LL **oracle** ceiling | 0.305 | hard cap of *this* low level |
| — PPO (huge budget) | 0.66 | the gap = low-level precision |

### Step 1 — from scratch, it's 0.0. *Why?*
We didn't just accept the NULL. We instrumented it. The wall is the **low-level reach primitive**: reaching the cube is joint-space inverse kinematics to a target smaller than **1.2 cm**, from ~30–40 cm away. A *random* policy triggers the contact event **0 times out of 512**. Even with a dominant dense "get closer" reward, the goal-conditioned LL — acting on a JEPA latent that doesn't preserve fine end-effector–cube geometry — never reliably closes the gap. So the HL has no achievable subgoal to plan over. **It's a low-level motor-primitive wall, not an abstraction failure.**

### Step 2 — prove the machinery actually works
Maybe our H-JEPA was just broken? We ran it as a **positive control** on a trivial-low-level task: a 2-D point-maze where "move toward a subgoal" is easy and the only challenge is long-horizon temporal abstraction. With proper care it reaches **0.94 ± 0.14 (8 seeds), tied with the ceiling.** So the hierarchy machinery is sound — confirming the Panda 0.0 is the contact primitive, not the architecture.

*(This step also caught two bugs, which is the point of positive controls: a shape error that made the low-level critic train against other samples' targets, and an XLA/TF32 nondeterminism that made "identical" runs disagree. Both fixed; same-seed runs are now bit-identical. A result you can't reproduce isn't a result.)*

### Step 3 — give it a competent primitive
The fix that follows directly: stop asking the latent LL to *learn* the impossible contact, and **hand it a competent analytic skill** (a reach/grasp/lift/place controller that reliably executes). Now the HL's job is the *interesting* one — plan, in latent space, which skill-parameters to deploy. Immediately: grasp **1.00**, lift **0.80**, and success crosses competence to **0.117**. The new bottleneck becomes the **place** phase.

### Step 4 — make the planner trustworthy, and bound the problem
Planning over a *learned* model invites "model exploitation" — the planner chases predictor errors. We stabilized the predictor and added a **real-rollout anchor** (keep the better of {plan, base} measured by the *true* reward). That bought sample-efficiency but not a higher ceiling. So we ran an **oracle**: brute-force-search the best high-level parameters *directly in the simulator*. It saturates at **0.305**. That's a hard ceiling of the analytic low-level itself — **even a perfect planner can't beat 0.305 over this primitive.** The remaining gap to PPO's 0.66 is *low-level place-position precision*, full stop.

### Step 5 — warm-start the planner → solve it
The last gap was that the latent planner, starting from scratch, *under-extracted* the primitive (its plan-only success was ~0.05 while the competent base was ~0.20 — it was wasting search rediscovering competence). The fix: **warm-start** the latent planner from the competent solution (seed the search, the anchor, and the replay buffer with it). The planner's *own* contribution then climbs **0.05 → 0.207** — it now extracts essentially the full primitive value — and the anchor adds another +0.04–0.06. Final: **0.289 (n=512), beating the hand-tuned reactive hierarchy (0.227–0.246) and 2.5× the un-warm-started H-JEPA.**

## 5. What this means (the honest version)

- **Yes, we solved Panda with H-JEPA** — competitively, beating the hand-tuned hierarchy, with the latent planner genuinely contributing.
- **But the lever was a competent low-level primitive + warm-start, not learned hierarchical abstraction by itself.** From scratch, the same architecture is 0.0. This is a gentle but real pushback on the strong form of LeCun's "Bet 2": on this task, learned hierarchy *helps* once the primitive exists, but it is *not* what bridges the hard gap — the primitive is.
- **The ceiling is now a low-level controller problem, not an H-JEPA problem.** Pushing past 0.30 toward PPO's 0.66 needs a more precise place-position primitive, then re-plan. That's the next experiment.

Three habits made this trustworthy and are worth stealing: **positive controls** (does it work where it *should*?), **oracles** (what's the best possible over this design?), and **reproducibility checks** (do identical runs agree?). Each one changed a conclusion here.

*All numbers are deterministic, held-out, real-success evaluations read from disk; code and result JSONs are in the repo. Next up (Part 51): can a better low-level place-position controller raise the 0.305 oracle and let H-JEPA climb toward PPO?*
