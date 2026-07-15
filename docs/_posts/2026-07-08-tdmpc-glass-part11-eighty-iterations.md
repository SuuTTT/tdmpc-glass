---
layout: post
title: "TD-MPC-Glass, Part 11: ~80 Iterations in Seven Phases — the Whole Journey (Living Index)"
date: 2026-07-08
description: "A complete retrospective of the program: roughly eighty experimental iterations across seven phases, from the 'six mirages' redundancy nulls, through the Panda manipulation solve and the beat-PPO campaign, the LeCun hierarchy/JEPA bets, the five-bet reality check, the dissection (wall + mechanism + sufficiency), and the current SOTA push. For each phase: what we tried, where the artifacts live, and the lesson that carried forward. Then a focused failure analysis of this week's two SOTA bets — value-aware and uncertainty-aware consistency — which both lose to plain uniform consistency by ~5-9%, establishing that TD-MPC2's world-model loss is near-optimal in form. The one law that survived all eighty iterations: in a value-based planner, structure and reweighting buy nothing the TD value pathway doesn't already consume."
---

> **📌 Living index — this is the running master summary of the whole program, kept pinned to the top and updated
> in place as new results land** (last updated 2026-07-08; its date is bumped on each update to stay on top).
> For the dated week-by-week narrative see Parts 1–10; for artifact-level detail see `ITERATION_LOG.md` in the
> paper repo.

> This is the whole map. Roughly **eighty experimental iterations** over the life of the program, grouped into
> seven phases, each with what it tried, where the artifacts are, and the lesson it left behind — followed by a
> focused post-mortem of this week's two SOTA bets, which failed in an informative way. If you read one thing, read
> the law at the end: *in a value-based planner, structure and reweighting buy nothing the TD value pathway doesn't
> already consume.* Everything below is the evidence for it. All numbers are read from the append-only ledger.

## How to count "an iteration"

An iteration here is one falsifiable experiment with its own GO/NO-GO: a proposed change, a matched control, a
multi-seed read, a verdict. By that definition the program has run **~80**: ~15 foundational redundancy nulls,
~60 numbered campaign iterations (`#1`–`#60`), 5 "next-phase" bets, and 3 SOTA bets. Full artifact-level detail is
in `ITERATION_LOG.md` in the paper repo; this post is the readable tour.

## Phase 0 — The six mirages (~15 iterations)

The founding question: does adding structure to TD-MPC2 — structural-entropy ("glass") latents, entity/graph world
models, calibration-shaped or jumpy models — buy anything on control? Iteration after iteration returned **null**:
structure ties vanilla at matched budget, and its extra couplings go unused by the value head. A single mechanism
check (does the value pathway consume the new coupling?) predicted each null before the GPU bill.
**Lesson:** the SimNorm latent is already value-sufficient (held-out value-decode \\(R^2 \approx 1\\)).

## Phase 1 — Can abstraction solve Panda PickCube? (`#1`–`#17`)

We inserted abstraction into a hard contact task every way we could: skill-options, demo-seeding, warm-start,
in-loop residuals at persistent authority, curricula, learned grasps. Abstraction-in-loop climbed from ~0.24 to
~0.79 real success — but **matched-budget PPO won the asymptote (~0.81–0.83)**, and the last ~17% turned out to be
a *physical* ceiling: 99.88% of the far-reach tail is kinematically infeasible to grasp upright.
**Lesson:** abstraction buys sample-efficiency and interpretability, not a higher ceiling.

## Phase 2 — The beat-PPO campaign & the benchmark (`#18`–`#42`)

We turned anecdotes into a matched benchmark and hunted for a clean beat. The one that survived: an
abstraction-as-curriculum that warm-starts then *releases* to free RL — **≥0.95 on OpenCabinet at ≤19.66M steps
vs PPO's 29.49M, 4/4 seeds.** Most other "beats" (energy-shaping, CPG, OSC class-controllers) evaporated the moment
we ran a *same-budget vanilla-PPO control* and collapsed to sample-efficiency claims.
**Lesson:** a cross-budget baseline manufactures fake wins; always match the budget.

## Phase 3 — The LeCun bets: hierarchy and JEPA (`#43`–`#60`)

We tested LeCun's arguments directly: hierarchical planning, a faithful decoder-free H-JEPA with an anti-collapse
term, structural-entropy latents. H-JEPA was a **multi-seed null on PandaPickCube** — the bottleneck was the
low-level motor primitive, not abstraction. Anti-collapse turned out **downstream-dependent**: the exact term that
*helps* a geometric goal-conditioned nav latent *hurts* value-based control.
**Lesson:** a non-generative predictor's information term is not free — its sign flips with the objective.

## Phase 4 — Five bets, four nulls (Parts 6–7)

Five falsifiable bets: planning-as-exploration, behavioral-prior taxonomy, glass-as-variance-reduction, pure-JEPA
done right, SE-structured latents. Four came back null-to-harmful — planning-as-exploration *decisively* refuted on
CartpoleSwingupSparse, novelty-MPPI worse on 4/4 matched seeds, SE and uniformity both hurting control.
**This is the pivot of the whole program:** every attempt to *add* structure to win reproduced the oldest finding,
so we stopped adding and started explaining.

## Phase 5 — The dissection (Papers 3–4)

Instead of "what can we add to beat PPO," we asked "why does the planner beat PPO, and what inside it does the
work?" Three results:

- **A categorical wall.** Tuned PPO hits **0/5 seeds ≥ 200 on HopperHop at 472M steps** and survives the entropy
  knob (×3, ×10); TD-MPC2 clears it 6/6 by ~1M. The wall needs *contact-criticality* — contact-free-but-unstable
  Acrobot has no wall.
- **A mechanism.** A five-loss ablation over four tasks: the **value and policy losses are individually fatal**
  (without the TD value loss the agent can't even learn to *stand*); the reward head only feeds the planner; and
  the **consistency loss — the "world model" — is the mildest cut of all.**
- **A sufficiency law.** Trained OFF from scratch, the consistency loss is removable only on HopperHop (n=8) and
  load-bearing on the planner-led tasks (Walker −23%, Cheetah −38%, Acrobot −44%). It is a *rollout-quality
  regularizer for the planner.*

**Lesson:** a rigorous null program's real output is a mechanism — "why X wins" is more citable than "we added Y
and tied."

## Phase 6 — The SOTA push, and this week's failure analysis

The sufficiency law hands you a constructive lever: if the consistency loss is a rollout-quality regularizer,
*make it better.* We tried two ways, both this week:

- **Value-Aware Consistency (VAC):** weight each latent dimension's prediction error by its value sensitivity
  \\(|\partial Q/\partial z|\\). Result, paired vs matched vanilla at 5M: **Walker −8.9%, Cheetah −4.4%. No-go.**
- **Uncertainty/Rollout-reliability Consistency (URC):** weight by the model's own rollout drift (open-loop vs
  teacher-forced). Result: **Walker ≈ −7%, Cheetah ≈ −9%. No-go.**

### Why both failed (the same way)

Both schemes concentrate the consistency gradient on *some* latent dimensions and de-emphasize the rest — VAC on
the value-sensitive ones, URC on the currently-unreliable ones. The failure is identical and follows from what the
planner does:

> The MPPI planner rolls the latent dynamics forward over many candidate action sequences and ranks them. **Which
> dimensions matter for that ranking is trajectory- and horizon-dependent, not fixed.** A dimension that is
> value-irrelevant or currently-reliable *now* can become decisive three steps into a rollout under a different
> action. Starving its prediction accuracy degrades the very rollouts the planner depends on.

So the planner needs **faithful dynamics on every dimension it might explore**, and *uniform* consistency is exactly
the objective that delivers it. Any reweighting is a bet that some dimensions matter less — and over a multi-step
rollout, that bet is wrong often enough to cost 5–9%. (VAC has an extra early-training problem — \\(Q\\) is immature
when its gradient sets the weights — but URC, whose weight comes from the dynamics model, fails too, which rules out
"immature Q" as the whole story.)

**What it establishes:** the reweighting family is closed. Uniform consistency isn't just load-bearing — it's
**near-optimal in form.** A clean negative that strengthens the sufficiency paper. Full write-up:
`REPORT_consistency_reweighting_failure.md`.

## What's next — the one honest abstraction swing left

If you can't improve the world model by re-emphasizing its objective, the only remaining lever is to change what
the **value pathway sees**. That is the one route the redundancy work said structure *could* matter but never did.
So the next bet (running when the boxes free) is a **value-conditioned abstraction**: a bisimulation-style latent
metric — states pulled together by reward-and-transition equivalence — which is structure the value pathway
consumes *by construction*, followed by a value-sufficient bottleneck if that nulls. Either outcome is publishable:
a beat is the abstraction-SOTA we've chased for eighty iterations; a null is the definitive close of the redundancy
question.

## Two open mechanisms worth nailing (future work)

Two "why" questions we brushed but never mechanistically closed — each has a partial answer and one decisive,
un-run experiment (full write-up: `FUTURE_WORK_open_mechanisms.md`):

- **Why is HopperHop unique?** It is simultaneously the *hardest* task for PPO (the sharpest categorical wall) and
  the *only* task where the world-model loss is removable. The hypothesis — Hop is *exploration-hard but
  execution-simple* (a low-dimensional limit-cycle gait the policy executes without accurate multi-step rollouts),
  while Walker/Cheetah/Acrobot are *execution-precision-hard* (their return level needs the planner's rollouts) —
  is stated but not tested. Decisive probe: policy-only vs MPPI at matched weights, Hop vs the dense tasks. Would
  turn Paper 4's descriptive split into a *predictive* law.
- **Why does the JEPA latent collapse in nav but not in control?** We showed closed-loop feedback *amplifies* it
  and data-width doesn't cause it, but never isolated the trigger. Leading hypothesis: control latents are anchored
  by a *dense TD value gradient* into the encoder; nav's goal-conditioning anchors weakly, so its rank collapses.
  Decisive probe: vary anchor strength (add a graded dense auxiliary target to nav H-JEPA) crossed with
  online/offline. Would yield the actionable rule "add anti-collapse only when the downstream signal is too sparse
  to anchor latent rank."

## The law that survived eighty iterations

> **In a value-based planner, structure and reweighting buy nothing the TD value pathway doesn't already consume.**

Abstraction buys sample-efficiency where its prior fits, and interpretability always. It does not raise the ceiling,
and — as this week proved — it cannot improve a world-model objective that is already value-sufficient. The durable
contribution of the program is therefore a *dissection* of why the planner wins and a *near-optimality* result about
its world model, honestly earned across eighty tries, most of which said no.
