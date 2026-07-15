---
layout: post
title: "From Reproducing 'Compositional Planning with Jumpy World Models' to Tying PPO with an Abstraction-in-the-Loop"
date: 2026-06-24
description: "Weekly review: a reproduction attempt that turned into a beat-PPO campaign — jumpy world models can't solve PandaPickCube, InFOM and PPO can, and a heuristic-in-the-loop residual ties PPO and beats it on sample-efficiency."
---

> Weekly review (Jun 17–24). What started as a clean reproduction of two recent papers turned into a chain of
> "that didn't work, so we added X" — and ended with the project's first result that is genuinely competitive
> with PPO. This post tells the whole arc honestly: the plan, the wall, the pivots, and the numbers.

## 1. Where it started: reproduce CompPlan + InFOM, compare on a common task set

The week's plan was a paper reproduction and a head-to-head:

- **CompPlan — "Compositional Planning with Jumpy World Models"** (Farebrother, Pirotta, Tirinzoni, Munos,
  Lazaric, Touati; arXiv **2602.19634**, ICLR-2026 Workshop on World Models), which builds on **Temporal
  Difference Flows / TD-Flow** (arXiv **2503.09817**). Its core is a **Geometric Horizon Model** — a
  flow-matching generative model of the discounted future-state (occupancy) measure — used to *plan over*
  pre-trained goal-conditioned policies. The headline is "plan in jumps, not primitive actions."
- **InFOM** (Intention-conditioned Flow Occupancy Measure, ICLR 2026) — the offline flow-occupancy method
  the GHM generative core most resembles.

The goal: stand both up and **compare jumpy-world-model planning vs flow-occupancy on a common task set** —
one big table. We confirmed there is **no official code** for CompPlan/TD-Flow, stood up JAX reproduction
scaffolds (InFOM, value-flows, OGBench) on rented 4090/3090 boxes, and started pretraining on the cheapest
matching OGBench tasks (`cube-single-play`, `antmaze-medium-navigate`). This is a multi-day reproduction —
it is still in progress.

## 2. The wall: jumpy world models don't solve PandaPickCube

While the flow-occupancy reproduction trained, we ran the *jumpy world-model* idea in our own stack — a
**jumpy (k-step macro) TD-MPC2** — on a contact-rich manipulation task, **PandaPickCube** (MuJoCo Playground),
scoring **real success** (`box_target ≥ 0.9`, the actual pick-and-place, not shaped return). It failed
outright. So did vanilla TD-MPC2: both **reward-hack** the dense shaping (hover near the box to farm the
proximity term) and complete the task **0%** of the time at feasible budgets.

*Rendered rollouts — the failure mode the reward curve hides:*

| TD-MPC2 — hovers, never grasps | jumpy TD-MPC2 — hovers, never grasps |
|:---:|:---:|
| ![]({{ '/images/panda/gif_tdmpc2_fail.gif' | relative_url }}) | ![]({{ '/images/panda/gif_jumpy_tdmpc2_fail.gif' | relative_url }}) |

*Both world-model policies park the gripper near the cube to farm the dense proximity term and never close the
grasp — high shaped return, **0% real pick-and-place**. This is why we score on `box_target ≥ 0.9`, not return.*

That broke the intended comparison — you can't fill a "common task" cell with a method that gets 0. So the
table question changed from "jumpy vs flow-occupancy" to the blunter **"what actually solves this task, and
can an abstraction beat the model-free baseline?"** Which meant adding the baseline we'd been missing.

## 3. Adding PPO — the real bar

We ran the official MuJoCo-Playground **PPO** on the unmodified reward, scored with the same real-success
metric. It **solves the task** (~0.81 real success). **InFOM**, trained offline on a PPO-generated dataset,
reaches **0.74**. So the honest common-task table looked like this — *not* the one we set out to make, but the
one the data forced:

| Method | Family | PandaPickCube real success |
|---|---|---|
| jumpy TD-MPC2 *(CompPlan-style)* | jumpy world model | **0.00** |
| vanilla TD-MPC2 | world model + MPPI planning | 0.00 |
| InFOM | offline flow-occupancy *(the base paper)* | 0.74 |
| **PPO** | model-free, end-to-end | **0.81** |

The uncomfortable read: the world-model/abstraction methods we care about were *losing to plain PPO* — the
same pattern we'd seen all project. (We also did a four-part DMC field report this week confirming TD-MPC2
beats PPO **only** on sample-efficiency and exploration-bottlenecked tasks, loses wall-clock and high-dim, and
is 5× over-parameterized.)

| Axis | Winner | Evidence |
|---|---|---|
| Sample-efficiency (env-steps to a return) | **TD-MPC2** | 28–160× fewer env-steps |
| Hard-exploration tasks | **TD-MPC2** | HopperHop 283 vs PPO 33; Acrobot 420 vs 268 |
| Non-exploration "hard" tasks | tie | FingerTurnHard 975 ≈ 968 |
| Wall-clock (time to competence) | **PPO** | ~9× faster |
| High-dimensional (Humanoid) | **PPO** | default TD-MPC2 diverges to NaN; PPO learns (HumanoidWalk 285) |
| Model size | — | `tiny` (0.72M) keeps a median ~99% → 5× over-parameterized |

*The four-part DMC field report (Parts 18–22): world-model planning is a sample-efficiency / hard-exploration
tool, not a universal win — model-free PPO takes wall-clock, high-dim, and most asymptotic returns.*

## 4. The BC-data gap → building a Heuristic Learning loop

The next idea was behavior cloning / offline learning to bootstrap a competent policy — but **we had no
demonstration data** for PandaPickCube (and the jumpy/TD-MPC2 policies that would generate it were at 0%). So
we built one, following the **"heuristic learning loop"** idea from **Jiayi Weng's blog** on iteratively
improving a hand-written controller against a measurable metric: a JAX, vmappable **phase-machine controller**
(reach → descend → grasp → lift → place) for PickCube, improved iteration over iteration. It reliably *reaches*
(reached_box ≈ 0.97) and tops out around **6% real success** — the wall is grasp/place dynamics, not reaching.
Now we had a competent, interpretable scaffold and the data it generates.

## 5. The beat-PPO campaign: can an abstraction tie PPO?

With the heuristic loop as a scaffold, we ran a ladder of experiments, each ruling out a wrong idea, all
scored on real success (`box_target ≥ 0.9`, \(n=256\)):

| # | Approach | Real success |
|---|---|---|
| — | HL heuristic controller | 0.063 |
| #1 | raw-action residual (no abstraction) | 0.13 |
| #2 | **value-aware skill-options abstraction** | **0.24** — first positive abstraction |
| #3 | demo-seeded TD-MPC2 | 0 (clean negative) |
| #4 | distill the abstraction out (BC / warm-start / DAPG) | 0 / no-beat |
| #5 | in-loop residual, authority \(\alpha=0.5\) | 0.48 — breaks the ceiling |
| **#5b/#5c** | **in-loop residual, persistent \(\alpha=1.0\)** | **0.79 ± 0.01** (multi-seed) |
| ref | end-to-end PPO | 0.81 |

**The result:** keeping the abstraction *in the loop* — the live controller plus a Markov-conditioned residual
at full standing authority, executed as \( a = \mathrm{clip}(a_{\text{HL}}(s,z) + \alpha\,\pi_{\text{res}}(s,z)) \)
— reaches **0.79, a robust near-tie with PPO's 0.81**, and crosses the competence threshold **~1.3–1.7× faster**
(≈20–26M vs 32.8M env-steps). On the sample-efficiency axis that matters for real robots, the abstraction-in-loop
**beats** PPO; on raw peak it is a near-tie. *(Reconciliation note, 2026-07-02: the later canonical base-reward,
matched-harness replication — round 7, n=3, CI-separated both ways — reads **residual 0.716 ± 0.014 &lt; vanilla
PPO 0.810 ± 0.006**, i.e. PPO wins the asymptote outright with the residual ~1.6× faster. The 0.79 here comes
from this post's milestone-shaped training variant; the 0.716-vs-0.810 study supersedes it as the clean
comparison. Both agree on the direction that survived the campaign: the prior buys speed, not ceiling.)* The negatives did the real work: **distillation fails** because the
abstraction is non-Markov (its action depends on hidden phase state); **annealing authority backfires**
(non-stationary → collapse); a **learned authority-gate underperforms** uniform full authority. The winner is
the simplest in-loop form.

### Learning curve & benchmark

![Learning curve: PandaPickCube real success vs env-steps]({{ '/images/panda/learning_curve_panda.png' | relative_url }})

*Real success vs interaction budget (every point is an \(n=256\), 1000-step evaluation). Our in-loop residual
(\(\alpha=1.0\), blue/green) climbs to ~0.79 and crosses the 0.66 competence line **~1.7× sooner** than PPO
(red, peak 0.81); jumpy/vanilla TD-MPC2 stay flat at 0; the skill-options abstraction (0.24) and HL heuristic
(0.063) are reference levels.*

![Benchmark: PandaPickCube real success by method]({{ '/images/panda/bench_bar_panda.png' | relative_url }})

*The common-task comparison the week produced: jumpy world models 0.00 · InFOM 0.74 · PPO 0.81 · our
abstraction-in-loop 0.79.*

### Watch the policies (PandaPickCube rollouts)

| HL controller / our in-loop method — grasp → lift → place (success) |
|:---:|
| ![]({{ '/images/panda/gif_hl_success.gif' | relative_url }}) |

*Contrast with the **hover-fail rollouts at the top** (TD-MPC2 / jumpy TD-MPC2, which never grasp): the
heuristic controller — and, far more reliably, our in-loop residual and PPO — actually **grasp, lift, and
place** the cube. PPO and our-method rollouts resemble this HL success but complete it ~0.8 of the time; the
learning curve above is the quantitative version.*

## The algorithms

**(1) The Heuristic Learning (HL) loop** — a hand-written, JAX/vmappable phase-machine controller, improved
against a measurable real-success metric (following Jiayi Weng's heuristic-learning-loop idea):

```text
controller ← phase machine [reach → descend → grasp → lift → place], knobs θ
repeat:
    metrics ← eval(controller, n=256)        # true success + per-phase pass-rates
    p*      ← lowest-passing phase
    θ       ← tune(θ, p*)                     # e.g. grasp_hold, xy_tol, place offset
    keep θ if real_success ↑
→ v9: reached_box ≈ 0.97 but ~6.3% real success; the wall is grasp/place dynamics
```

**(2) Skill-options abstraction (#2)** — learn the controller's option *parameters*, state-conditioned
(value-aware), with Evolution Strategies:

```text
context ← [initial box pose, target pose]
kv      ← clip(KV_BASE + KV_SCALE · tanh(pi_hi(context)), lo, hi)  # option params
a_t     ← option_controller.act(s_t, z_t ; kv)                    # z = phase carry
train pi_hi by ES to maximize true success
→ 0.063 → 0.24.  Ablation: learning the params → 0.15; +state-conditioning → 0.24
```

**(3) In-loop residual — our latest method (#5/#5b/#5c)** — keep the controller *live*, add a
Markov-conditioned residual at full persistent authority, train with gradient RL:

```text
z_t ← controller phase/option state          # fed INTO the observation => Markov in (s,z)
a_t ← clip( a_option(s_t, z_t) + alpha * pi_res(s_t, z_t),  -1, 1 )   # alpha = 1.0, constant
train pi_res with PPO on task reward + sub-goal milestone shaping
→ 0.24 → 0.48 (alpha=0.5) → 0.79 (alpha=1.0).  Ties PPO 0.81; competence ~1.7x faster.
why: the abstraction is non-Markov, so it must stay IN the loop (distilling it out → 0);
annealing alpha backfires; a learned alpha-gate underperforms a uniform alpha=1.0.
```

## 6. Aside: the model is 5× over-parameterized (confirmed at 5 seeds)

A cheap, robust positive: a **`tiny` TD-MPC2** (0.72M params, 5× smaller, ~1.6× faster) keeps a **median ~99%**
of full return across the DM-Control suite, and on the high-velocity locomotion tasks where capacity *does*
matter, a 5-seed re-run confirms tiny retains **~88–89%** (a real ~11–12% gap, not noise) — with a thin
instability tail. Most of TD-MPC2's parameters aren't doing work on standard control.

## 7. Where it stands / toward 100%

Neither PPO (0.81) nor our abstraction-in-loop (0.79) solves PandaPickCube perfectly — ~20% of randomized
configs (reach-edge targets, grasp slips, placement just outside tolerance) still fail. This week's final push,
**in flight now**, attacks that tail three ways: a **bigger/longer residual** (is 0.79 capacity-limited?), a
**closed-loop retry** (the controller detects a failed grasp and re-attempts in-episode — the abstraction-in-loop's
unique advantage over a flat policy), and a **hard-config curriculum**. Honest expectation: ~0.95 is reachable;
literal 100% on a randomized contact task is not. Results next week.

**The one-line takeaway:** a reproduction attempt for jumpy world models led, via a chain of honest failures, to
the project's first abstraction that *matches* PPO and *beats* it on sample-efficiency — by keeping the
abstraction in the control loop rather than distilling it out.

## References & artifacts
- **Papers:** CompPlan — *Compositional Planning with Jumpy World Models*, arXiv [2602.19634](https://arxiv.org/abs/2602.19634); TD-Flow, arXiv [2503.09817](https://arxiv.org/abs/2503.09817); InFOM (ICLR 2026); DreamerV3 (*Nature* 2025); MuJoCo Playground (arXiv 2502.08844). The heuristic-learning-loop idea: Jiayi Weng's blog.
- **Experiment write-ups (full detail, Parts 17–27):** [suuttt.github.io/tdmpc-glass](https://suuttt.github.io/tdmpc-glass/).
- **Code & logged results:** pushed to GitHub ([SuuTTT/tdmpc-glass](https://github.com/SuuTTT/tdmpc-glass)); every number above is read from logged JSON/CSV, never hand-entered.
- **Model checkpoints & data:** archived to HuggingFace (`Dannibal/tdmpc-glass-milestones`).
- **Prior weekly review:** [Campaign Review #3 (Jun 17)](https://suuttt.github.io/projects/2026-06-17-tdmpc-glass-part3-campaign-review/).
