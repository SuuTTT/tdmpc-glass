---
layout: post
title: "TD-MPC-Glass, Part 22: Budgets, the *Why*, and How to Put Planning & Abstraction Into RL Correctly"
date: 2026-06-22
description: "A synthesis capstone for Parts 18–21. First, the actual budgets we ran (steps AND wall-clock, from the logs): TD-MPC2 at 1M env-steps takes ~50–60 min/task; PPO at 78–285M steps takes ~22 min–2.2 hr — PPO does up to 285× more steps in *less* wall-clock. Second, the mechanistic *why* behind every pro and con of PPO, TD-MPC2, and jumpy TD-MPC2 — not just what we observed but the cause. Third, the prescriptive part the whole campaign was building toward: eight design rules for adding planning and abstraction to RL correctly and efficiently — match the algorithm to the binding constraint, plan only where exploration is the bottleneck, scale model capacity with dimensionality, guard planning against shaping local-optima, and make abstraction value-aware (not merely predictive), because jumpy proved 'better world model ≠ better policy.'"
---

> Parts 18–21 measured TD-MPC2 vs PPO vs jumpy across sample-efficiency, wall-clock, home turf, and
> abstraction. This capstone does three things people asked for: (1) the **exact budgets** we ran — steps
> *and* time; (2) the **mechanistic why** behind each pro/con; (3) the **design rules** that fall out — how
> to add planning and abstraction to RL correctly and efficiently.

## 1. The budgets we actually ran (from the logs)

Same RTX 3060 throughout. PPO = official MuJoCo-Playground per-task tuned config; TD-MPC2 = default config,
1M env-steps. The asymmetry is the whole story: **PPO ran 78–285M steps, TD-MPC2 ran 1M — yet wall-clock is
comparable, because PPO is 50–700× faster per second.**

| Task | TD-MPC2 steps | TD-MPC2 time | PPO steps | PPO time | PPO throughput |
|---|---|---|---|---|---|
| CheetahRun | 1.0M | 3,695 s (62 min) | 285M | 1,950 s (33 min) | ~146k sps |
| HopperHop | 1.0M | 2,866 s (48 min) | 285M | 2,891 s (48 min) | ~99k sps |
| AcrobotSwingup | 1.0M | 3,314 s (55 min) | 285M | 1,313 s (22 min) | ~217k sps |
| CartpoleSwingupSparse | 1.0M | 3,332 s (56 min) | 285M | 1,293 s (22 min) | ~220k sps |
| WalkerWalk | 1.0M | 3,196 s (53 min) | 78.6M | — | — |
| HumanoidRun | 1.0M | 3,128 s (52 min, *diverged*) | 285M | 7,877 s (131 min) | ~36k sps |
| FingerTurnHard | 1.0M | ~3,300 s | 88.5M | 1,877 s (31 min) | ~47k sps |
| HumanoidWalk | 1.0M | (*NaN*) | 88.5M | 5,842 s (97 min) | ~15k sps |

- **TD-MPC2: ~270–350 sps, near-constant across tasks** (gradient-block-dominated, so task dimension barely
  changes throughput).
- **PPO: ~15k–220k sps, swings ~15× with dimensionality** (Humanoid is expensive even for PPO).
- **The one number that says it all:** on CheetahRun, PPO did **285× more env-steps in *half* the wall-clock**
  of TD-MPC2's 1M. TD-MPC2 wins per-sample; PPO wins per-second.

(Manipulation reference: PPO solved PandaPickCube at ~33M steps → 66% (108M → 83%); TD-MPC2 we ran to a max
of 24.3M and it never left 0% real success.)

## 2. The pros and cons — and *why*

### PPO — fast & robust, sample-hungry & exploration-blind

| | observation | **mechanism (why)** |
|---|---|---|
| **+** | 50–700× higher throughput | 2048 parallel envs + a *few* batched updates over huge on-policy batches → GPU stays saturated, no per-step sync; compute is amortized over millions of collected steps. |
| **+** | robust, never diverges | no learned model to compound error; the policy-gradient update is a stable supervised-ish step. It learned Humanoid where TD-MPC2 NaN'd. |
| **+** | scales with compute | throughput is just "more envs"; nothing in the algorithm fights parallelism. |
| **−** | needs 78–285M steps | **on-policy**: data is discarded after each update (no replay), and the gradient is a high-variance Monte-Carlo estimate that needs many samples to average down. No model to extract extra signal per transition. |
| **−** | flatlines on hard exploration (HopperHop 33) | explores *only* via action-noise/entropy on the current policy. If the reward landscape rarely rewards random perturbations (the hop gait is a thin target), it never stumbles in. No directed lookahead. |
| **−** | slow to *first* success on long-horizon | the same blind search must chain many steps before any reward (PickCube: 0% until ~28M). |

### TD-MPC2 — sample-efficient & cracks exploration, slow & fragile & hackable

| | observation | **mechanism (why)** |
|---|---|---|
| **+** | 28–160× more sample-efficient | **off-policy replay** (K_UPDATE=64 updates/step = replay ratio 64 — every transition reused ~64×) *plus* MPPI **planning** that extracts behavior by lookahead instead of by sampling it. The model is a sample amplifier. |
| **+** | solves hard exploration (HopperHop 283) | planning does **directed search** in the model: it can find the hop gait by rolling out candidates, not by waiting to stumble in. |
| **+** | shrinks 5×, over-provisioned planning | for standard DMC the default capacity/planning budget is slack — most of it isn't load-bearing. |
| **−** | ~300 sps, slow | the **64 gradient updates per env-step** are a fixed, compute-bound tax (Part 20). Adding envs amortizes it but never removes it: online model-based learning ties a heavy update to *every* collected step. ~2–4× is buyable; PPO parity is architecturally off the table. |
| **−** | diverges at high-dim (Humanoid NaN) | default SimNorm latent + ensemble-Q capacity can't represent 67-dim dynamics → model error compounds inside planning → value targets explode → NaN. **Capacity must scale with dimensionality**, and default doesn't. |
| **−** | reward-hacks dense manipulation (0% real) | planning greedily optimizes the *modeled* reward over a **short horizon (H=3)**. Dense shaping has a near, reachable local optimum (hover = high `gripper_box`) while the true goal (a long contact-rich lift+place) lies past the horizon and past the contact model's accuracy. So it climbs the shaping gradient and stops. |

### Jumpy TD-MPC2 — predicts better, controls worse

| | observation | **mechanism (why)** |
|---|---|---|
| **+** | lower multi-step prediction error (OpenCabinet k=8 err 0.15 < single-step 0.25) | a macro-model trained to predict k steps *directly* avoids the compounding error of rolling a single-step model k times. Genuinely a better forecaster. |
| **−** | null-to-harmful on control (HopperHop 283→1.7, Acrobot 420→334, CartpoleSparse 736→0) | temporal abstraction **coarsens the action space**: macro-actions can't express the fine, reactive corrections that contact/balance need, and macro-commitment means no mid-macro course-correction — so the planner overshoots the thin reward region in sparse/hard tasks. |
| **−** | the earlier "+60%" was reward-hacking | it farmed a dense shaping term *faster*, not better — on clean metrics the gain vanishes. |

**The one lesson that generalizes:** *prediction accuracy and control performance are different objectives.*
A model that forecasts well is not automatically good for planning-control if its abstraction throws away
control-relevant detail. **Better world model ≠ better policy.** That single fact is why jumpy fails and why
the structural-entropy "Glass" abstractions failed before it.

## 3. How to put planning & abstraction into RL correctly and efficiently

The campaign's failures are unusually informative — each one points at a design rule.

**A. Match the algorithm to the *binding constraint*.** This is the biggest lever and the most common
mistake.
- Samples expensive (real robot, slow sim)? → model-based + planning. Pay wall-clock to buy sample-efficiency.
- Wall-clock/throughput the cost (fast sim, ample compute)? → model-free + massive parallelism. Don't pay for
  a world model you don't need.
- In our sim-on-a-3060 setting wall-clock binds → PPO kept winning. That's not PPO being "better"; it's a
  constraint match.

**B. Deploy planning *only where exploration is the bottleneck*.** Directed lookahead paid off on
hard-exploration tasks (HopperHop 8.6× PPO) and paid *nothing* where the task wasn't exploration-limited
(FingerTurnHard — a tie). Use planning where the reward landscape is deceptive/sparse; skip it where dense
reward already guides gradient ascent. Planning is an exploration tool first.

**C. Scale model capacity (and stability) with dimensionality.** One config for all dims is wrong — it
diverges at high-dim (Humanoid NaN). Grow latent/width with state-action dimension; add high-dim stabilizers
(normalization, ensemble-disagreement penalties, value clipping). Conversely, **shrink for easy tasks**:
default TD-MPC2 is 5× over-parameterized on DMC, and shrinking reclaims the wall-clock where it's weakest.

**D. Plan on the right horizon, and guard against shaping local-optima.** Short-horizon planning *exploits*
dense-reward traps (PickCube hover). Counter with: (i) a longer/receding horizon or terminal-value bootstrap
that sees past the shaping plateau; (ii) reward shaping that gates proximity terms behind real progress so no
hover trap exists; (iii) **selection on the true task metric, never the shaped return.**

**E. Make abstraction value-/control-aware, not merely predictive.** Jumpy optimized prediction and lost
control. A correct abstraction:
- is selected by *whether it improves value/control* (value-equivalence / bisimulation-style objectives),
  not by reconstruction or forecast error;
- **preserves control-relevant detail** — don't coarsen the action space where fine corrections matter;
- is **adaptive, not fixed-k** — abstract where dynamics are smooth/predictable, stay fine-grained where
  contact/balance demands reactivity (variable-rate macros);
- is validated on a clean, un-hackable metric *before* you believe it.

**F. Evaluation is part of the design.** Every false positive in this project came from trusting shaped
return. Always measure the task metric, report peak *and* final, use multiple seeds, and treat any "win" on a
dense proxy as suspect until it survives a clean metric.

**G. Don't pick a tribe — hybridize per constraint and per phase.** The frontier is a clean diagonal
(Part 20): neither family dominates both axes. The honest design uses **model-free throughput to collect
broadly + model-based planning where exploration is hard**, or **distills a planner's behavior into a fast
reactive policy** (plan to explore, policy to deploy). The win is in the combination, not the allegiance.

### The bottom line
Use **planning as an exploration amplifier** under a sample budget, on tasks where on-policy search stalls,
with capacity scaled to the problem and selection on the true metric. Use **abstraction only if it is
value-aware and control-preserving** — otherwise a better forecaster gives you a worse controller. And above
all, **match the method to whichever resource — samples or seconds — is actually scarce.** Most of the
"which RL algorithm" question dissolves once you answer that.

### Caveats
Single-seed/config (directional), one GPU class, PPO tuned vs TD-MPC2 default. The design rules are
hypotheses the data *motivates*, not ones it fully proves — the value-aware-abstraction and
adaptive-horizon claims in particular are the next things to test, not settled results. Budgets and returns
read from logs: `exp/tdmpc_glass/dmc_ppo_vs_tdmpc2/logs/`, `exp/benchmark/`. See Parts 18 (axes), 19
(shrink/jumpy/OpenCabinet), 20 (speed/frontier), 21 (home turf).
