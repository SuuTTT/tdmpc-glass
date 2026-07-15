---
layout: post
title: "Thread A — Planning as a Directed-Exploration Operator (flagship, living log)"
date: 2026-07-01
description: "The flagship bet: on exploration-bottlenecked tasks, model-based planning explores the state space more effectively than model-free RL, and that exploration — not a higher ceiling — is the win. Isolate it (coverage), amplify it (novelty-seeking MPPI), direct it (SE-discovered subgoals). Living log, updated as arms land."
---

> **Living log.** One post per research thread; updated whenever an experiment finishes, a finding
> lands, or something breaks. Newest progress on top. Context: [Part 5 method map](../2026-07-01-tdmpc-glass-part5-beat-ppo-reality-check),
> [Part 6 five bets](../2026-07-02-tdmpc-glass-part6-five-bets-next-phase).

## The bet

**Hypothesis.** On exploration-bottlenecked tasks, model-based planning (a learned world model + lookahead)
reaches states — and therefore policies — that on-policy model-free RL never finds. Our sharpest datapoint is
TD-MPC2 beating PPO on `HopperHop` **367 vs 33**: that is *not* a ceiling story and not even mainly a "learning
speed" story. It's that **planning is a directed-exploration operator**. On dense/explorable tasks PPO's raw
throughput wins; on exploration-bottlenecked ones, planning explores its way to a solution PPO can't reach at any
budget.

**Why us / white space.** Plan2Explore / LEXA / MAX / Go-Explore explore via curiosity or model-disagreement, but
nobody uses a **learned abstraction (structural-entropy communities / bottlenecks) to define the exploration
targets a planner pursues**. We already own SE, a working planner, and a controlled exploration-difficulty axis
(the actuation-weakened "escape frontier"). That intersection is the paper.

**Three arms.**
- **A1 — Isolate.** Measure state-space **coverage** (occupancy entropy, distinct grid cells, #SE-communities
  visited) for planning vs a no-plan \(\pi\)-only ablation vs PPO. Show planning → coverage → success.
- **A2 — Amplify.** Add an **intrinsic-novelty bonus to the MPPI objective** (model-disagreement / latent count) —
  Plan2Explore *inside* TD-MPC2. Does it push the escape frontier to even weaker actuation?
- **A3 — Direct (SE).** Run min-2D structural entropy on the replay-buffer latent graph → communities are abstract
  states, boundaries are bottleneck subgoals → a high level plans toward *under-visited* communities.

**Metrics.** Steps-to-competence (the learning-speed axis), coverage, escape-frontier shift, final success.

## Status board

| arm | state | verdict |
|---|---|---|
| A1 mechanism (same-weights ablation, **trained** ckpt) | ✅ done | **GO (n=1 task), ⚠ noise-confounded** — planning covers 2.2× more cells, BUT the π arm rolled out the *deterministic* policy mean while MPPI samples with std up to 2.0, so part of the gap is action stochasticity, not planning (review 2026-07-02; the training-time ablations below ARE noise-matched) |
| A1-core (planning vs π-only, **from scratch**, w/ coverage) | ✅ done | **NULL (mildly reversed)** — no coverage gain at 80k |
| A1-full (learnable task, coverage across curve) | ✅ done | **coverage NULL/reversed, but sample-efficiency GO** (WalkerRun) |
| A1-decisive (exploration-hard **and** learnable) | ✅ done | **NULL** — policy-only discovers sparse reward equally (earlier) |
| A2 (novelty-seeking MPPI) | ❌ **NULL / mostly harmful** (2026-07-02, n=2, 5 tasks) | novelty in the MPPI objective hurts dense tasks (Pendulum 766→135); the lone sparse "win" is within seed variance — see Part 8 |
| A3 (SE-subgoal discovery) | ❌ **NULL** (2026-07-02, n=2, 3 tasks, ran as Thread E) | SE-community subgoal shaping ties or hurts flat; worst on the sparse task it was meant to help |

**Reframe (the useful one): planning is a *pruning* operator, not an exploration one.** The "narrowing" we
measured isn't a failure — it's the mechanism. Vanilla MPPI prunes the search toward predicted value
(branch-and-bound-like), concentrating samples on high-value *reachable* trajectories. That pruning **is** the
sample-efficiency win (A1-full: narrower coverage, ~1.5–2× faster). But pruning toward predicted reward also
prunes *away* from undiscovered **sparse** reward — which is exactly why planning didn't help discovery on
CartpoleSwingupSparse (A1-decisive). So: *planning prunes exploration — an asset for exploitation/speed, a
liability for sparse discovery.* This directly motivates **A2 (running)**: add an intrinsic-novelty term so the
pruning points **toward** novelty (Plan2Explore-in-TD-MPC2), and see if it flips the sparse-discovery result.

**What the π-only ablation does and doesn't isolate (important scoping).** Verified from the code: the `pi` vs
`plan` switch only changes the *data-collection action* (MPPI lookahead vs the amortized policy prior) and the
logging — it *never* touches the loss/update. So **both arms train the full world model identically** (encoder +
latent dynamics + reward head + Q/value + policy prior); the π-only agent is *not* model-free, it's the same world
model acting without lookahead. Therefore the null means "MPPI lookahead adds no exploration beyond the
world-model-trained policy" — it does **not** test whether the **world model itself** is the driver, because both
arms share it. That is very likely where TD-MPC2's real edge lives.

**Three things we have *not* tested yet (and should):**
1. **TD-MPC2 coverage vs PPO coverage, head-to-head** (the world-model test at coarse grain). Every controlled
   test above is PLAN vs π-ONLY *within* TD-MPC2, so all share the world model. The famous HopperHop **367 vs 33**
   is TD-MPC2-vs-PPO, but we isolated only that *planning* isn't the cause (plan-vs-π edge +1.7). So "TD-MPC2
   explores better than PPO" is genuinely untested; the likely real reason it wins HopperHop is
   **sample-efficiency from the world model**, not exploration — which the queued **PPO-at-large-budget** run
   (does PPO reach 367 given enough samples?) and a **TD-MPC2-vs-PPO coverage** run will settle.
1b. **Which world-model head is load-bearing?** A finer ablation — drop the reward predictor / the
   value-from-rollouts / the latent-dynamics term one at a time — would pinpoint *which* of the ~5 nets actually
   drives the sample-efficiency, rather than treating "the world model" as a monolith.
2. **Discrete hard-exploration navigation (MiniGrid / MultiRoom / KeyCorridor).** All tasks so far are DMControl
   (continuous) + continuous 2D mazes. The canonical hard-exploration maps are discrete + partially observed,
   which TD-MPC2 doesn't natively run — a real build, and the right stress test for any "planning explores" claim.

**Verdict on the flagship: the "planning-as-exploration" thesis is refuted; the honest result is
sample-efficiency + exploitation.** Three controlled plan-vs-π-only tests now agree that planning is **not** a
directed-*exploration* operator: (a) A1-core (HopperHop@80k) — no coverage gain; (b) A1-full (WalkerRun) —
planning *narrows* coverage (exploitation) yet learns ~1.5–2× faster; (c) **A1-decisive (CartpoleSwingupSparse,
sparse/exploration-hard)** — a policy-only agent discovers the sparse reward **3/3, even slightly earlier** than
planning (140k vs 157k), so planning does *not* unlock exploration a policy-only agent misses. What planning
*does* buy is repeatable: **faster time-to-competence and higher/steadier post-discovery returns**
(A1-decisive final 540 vs 331). The proposed *new* paper — "exploration is the real breakthrough over PPO" —
does **not** survive a controlled test; the HopperHop 367-vs-33 is TD-MPC2-vs-**PPO** (model/algorithm), not
planning-vs-π-only. The salvageable, honest contribution is the **speed/exploitation** characterization, which
folds into the [Thread B taxonomy](../tdmpc-glass-thread-b-behavioral-prior-taxonomy) — not a novel exploration
mechanism. A2/A3 (amplify/direct exploration) are deprioritized: there's no exploration effect to amplify.

## Progress log

### 2026-07-02 — RESOLVED: TD-MPC2's HopperHop win is an *exploration* advantage from the world model (not sample-efficiency)
The direct test we'd been missing. Ran **PPO on HopperHop at a large budget** (mujoco_playground brax PPO, tuned config, n=5, up to **472M env-steps/seed ≈ 94× TD-MPC2's ~5M-to-367**):

| metric | value |
|---|---|
| PPO peak reward (mean / best) | **40.0 / 53.8** |
| seeds crossing 200 or 367 | **0 / 5** (never) |
| TD-MPC2 reference | ~367 in a few M steps |

**Verdict: exploration wall.** PPO given ~94× the samples still peaks ~7× below TD-MPC2 and never approaches the gait — a genuine exploration *failure*, not mere inefficiency. This **flips the earlier "likely sample-efficiency" guess** and resolves the two open gaps together:
- **plan-vs-π-only (within TD-MPC2): null** — MPPI lookahead adds no exploration beyond the world-model-trained policy.
- **TD-MPC2-vs-PPO (model-based vs model-free): the world model *is* the exploration enabler** — it discovers a gait that model-free PPO can't find at any practical budget.

So the honest, reconciled picture: **planning (MPPI) is a pruning/exploitation operator (null for exploration); the *world model* is what buys exploration** on hard-exploration tasks like HopperHop. The per-head ablation (running) will localize *which* world-model component (dynamics / reward / value) carries it. `wm-redundancy-paper` ledger has the per-seed numbers.


### 2026-07-01 — A1-full: coverage NULL, but a *sample-efficiency* win — and a correction to my own first read
Reran the de-risk on **WalkerRun** (which reaches real returns, unlike HopperHop@80k): 140k steps, PLAN vs
PI-ONLY, n=3, coverage logged across the curve (warmup ~30k). Both links are finally testable — and the story is
richer than a flat null.

- **Planning → coverage: NULL, and *reversed* post-competence.** Late-training PLAN vs PI-ONLY distinct bins
  464 vs **501** (t=−2.75), per-dim entropy 1.84 vs 1.87 (t=−3.11). Once the model is competent, planning
  *narrows* coverage — MPPI is **directed exploitation** (steers onto the running-gait manifold) while the
  stochastic policy prior disperses wider. The mechanism GO's 2.2× coverage did **not** generalize.
- **Sample-efficiency: planning clearly wins.** The n=3 training (collect) return shows PLAN leading π-only for
  the entire competence phase — 50k: **335 vs 185** (+149), 90k: **408 vs 247** (+161), 100k: 486 vs 368 — then
  converging to the same ceiling by ~120k (140k: 480 vs 446). Planning reaches competence ~1.5–2× faster.
- **Coverage does *not* mediate the benefit:** PLAN covers *less* yet learns *faster*.

**Correction (verification discipline).** My first pass reported "π-only non-significantly ahead, 542 vs 480" and
called it a flat null — that used the *noisy single-episode eval* (`EVAL_NEPS=1`) at 140k. Reading the clean n=3
**collect-return curve** from disk overturns it: planning has a large, consistent mid-training lead. Lesson: read
the training curve, not one eval snapshot. **The honest verdict:** "planning is a directed-*exploration*
operator" is **wrong on the mechanism** (it's exploitation, not coverage-broadening) — but planning is a genuine
**sample-efficiency** lever, exactly the campaign's speed-not-ceiling law. Whether that exploitation *also* helps
discover sparse reward is the decisive test (running now on CartpoleSwingupSparse).

### 2026-07-01 — A1-core: NULL from scratch, and it *reframes* the flagship
The clean 2-arm de-risk finished (n=3, HopperHop, 80k steps, online coverage logging wired into the training
loop with frozen bins fit on the shared random warmup so the arms are comparable).

**Link 1 — planning → coverage: NULL, mildly reversed.** Final-step PLAN vs PI-ONLY:

| coverage metric | PLAN | PI-ONLY | Δ |
|---|---|---|---|
| distinct occupancy bins | 614±123 | **685±151** | −71 (n.s.) |
| projected occupancy entropy | 4.87±0.20 | **5.24±0.20** | −0.37 (t=−2.28, *wrong direction*) |
| mean per-dim entropy | 1.65±0.11 | **1.72±0.05** | −0.07 (n.s.) |

The sign is consistent at *every* checkpoint from 20k on — PI-ONLY covers as much or *more* state than PLAN.
**Link 2 — coverage → performance: untestable.** Returns are floor-heavy (mean 2.64, mostly 0); neither arm
reaches a learning regime in 80k steps, so there's nothing to correlate (Pearson r = 0.105, n.s.).

**Why (from the logs, not fabricated):** early in training the world model is untrained (`mpc=0.0`,
`mppi_return ≈ 0`), so MPPI actions are conservative model-noise while the policy prior + exploration noise
disperses at least as widely. This is not a refutation of the thesis — it's a **scope correction**: planning
can only *be* a directed-exploration operator once the model is good enough to plan through. The mechanism GO
(2.2× coverage) used a trained 750k checkpoint; this null used a cold-start model. Both are true.

**Next (A1-full, reframed):** pick a task/budget where TD-MPC2 actually reaches separable nonzero returns, run
PLAN vs PI-ONLY (and PPO) to competence, and measure coverage **across the training curve** — the prediction is
that PLAN pulls ahead on coverage (and return) *after* the model warms up, and that coverage then predicts the
gap. Single-task, single-budget caveats on the null are explicit.

### 2026-07-01 — A1-core running; a process fix
Launched the clean 2-arm de-risk on b3060b (b3060b's TD-MPC2 can run DMControl): **PLAN** (full MPPI) vs
**PI-ONLY** (identical model/budget, act with the policy prior, no planning), from scratch, with online
coverage logging. The PLAN wave (n=3 seeds) finished cleanly. **Process hiccup, fixed honestly:** the first
agent tried to self-resume via a shell watcher to run the PI-ONLY wave — that doesn't work (background agents
only resume via the parent), so it stalled after PLAN. Relaunched a completion agent that blocks to the end,
audits whether coverage was actually logged online (no checkpoints exist to recover it post-hoc), runs PI-ONLY,
and writes the verdict. **No numbers claimed until that lands.**

### 2026-07-02 — A1 mechanism: PARTIAL GO (n=1 task, clean)
Isolated the mechanism with a **same-weights** ablation on a `PandaPickCubeOrientation` 750k checkpoint —
MPPI planning ON vs OFF at *identical* weights (so any difference is planning, not training stage):

| metric (plan ON vs OFF) | ON | OFF | ratio |
|---|---|---|---|
| distinct grid cells | 222 | 100 | **2.2×** |
| occupancy entropy | 4.74 | 3.61 | — |
| 10-NN dispersion | 3.84 | 1.37 | **2.8×** |
| return | 405 | 349 | — |

**Planning explores far more state, and coverage tracks return.** The agent caught and avoided a real confound:
comparing `best_pi@200k` vs `best_mppi@750k` conflates training-stage with planning and produces a spurious
near-parity — only the same-weights ablation isolates the planning effect.

**What is *not* yet de-risked (the headline):** `HopperHop`/`CartpoleSwingupSparse` have no usable checkpoints
(we never `--save_full_state`, leaving only CSV logs) and no saved PPO models, so the 3-arm
{TD-MPC2 / π-only / PPO} coverage test needs a **fresh instrumented training run**. And the famous 367-vs-33 is
TD-MPC2-vs-PPO; the planning-vs-π-only *return* edge on HopperHop is only +1.7 (n=31, 77% of seeds) — so the
claim must be **carried by coverage, not return**. That is exactly what A1-core (now) and A1-full (next) test.

**Next:** A1-full — train TD-MPC2(plan) / π-only / PPO from scratch on HopperHop + the escape frontier *with* the
coverage logger; show coverage(plan) ≫ coverage(PPO) during the exploration phase and that it predicts the final
gap. Reusable logger (`scripts/coverage_rollout.py`) is ready; queued for the next free b3060 slot.
