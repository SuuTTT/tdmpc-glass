---
layout: post
title: "TD-MPC-Glass, Part 20: A Week in Review — From Instruments to a Paper, and the Question Finally Answered"
date: 2026-07-21
description: "The week of July 15–22. Part 19 left four named hypotheses and one sharp question — is the world model just a learned abstraction, and if so which knob does the work? This week turned the value-sufficiency instrument into a finished AAAI-27 diagnostic paper (9 tasks, cross-model correlation ρ=−0.90), found the diagnostic's honest boundary, caught and corrected a one-seed artifact with multi-seed replication, and — with a matched-seed rung-ladder plus the official TD-MPC2 vs SAC numbers — answered the sharp question: what beats model-free RL is the value pathway, not the world model or the planner."
---

> A review of one week — July 15 through July 22 — picking up where
> [Part 19](https://suuttt.github.io/tdmpc-glass/2026/07/15/tdmpc-glass-part19-weekly-review/) left off.
> Part 19 converted three dissections into measured instruments and framed the program around named
> hypotheses. It ended on a sharp one, **H-WM-ABSTRACT**: *is the world model itself just a learned
> abstraction, and if so which knob is doing the work?* This week did two things — it turned the
> value-sufficiency instrument into a **finished, submission-ready paper**, and it **answered
> H-WM-ABSTRACT** with a controlled decomposition. As always, every number is read from disk and the
> nulls are reported as loudly as the positives; two of this week's most useful results are a corrected
> mistake and a failed prescription.

## The one-paragraph version

The value-sufficiency bottleneck (**VBN**) — the checkpoint-time probe that measures how much of an
agent's latent the value head actually needs — became a paper. Across **nine** DeepMind Control tasks
and **two** world-model families (Dreamer, TD-MPC2), a single scalar from VBN predicts whether removing
the learned forward model helps or hurts: **Spearman ρ = −0.90 (n = 5 value-limited tasks)**. The
diagnostic has a clean, characterized boundary — two *exploration-limited* tasks it cannot see — and one
of those boundary points was a single-seed artifact we caught only by replicating. And the sharp
question from Part 19 now has an answer: on HopperHop, the task where TD-MPC2 beats both PPO and SAC,
**the win is carried by the value pathway, not by the world model or the planner.**

## Motivation, in one figure

![Same agent, opposite verdicts]({{ '/images/part20-motivation.png' | relative_url }})

Same agent (Dreamer), same intervention (disable the learned forward model). On **ball-in-cup** the
return collapses to zero — the world model is essential. On **walker-run** the stripped agent is no
worse, even slightly better — the world model is redundant. "Does a learned world model help?" has no
architecture-level answer; it is a **property of the task**. The whole paper is about a probe that tells
you which, *before* the world model is trained.

## The nine tasks, and why they land where they do

Because the whole argument rests on *which* tasks need a world model and *why*, here is the cast in full —
each is a standard DeepMind Control Suite task, and what matters is the pairing of its **reward structure**
with its **dynamics**. (All returns are on the 0–1000 DMC scale.)

| task | reward | dynamics | why the WM does / doesn't matter |
|---|---|---|---|
| **ball-in-cup** | **sparse**: 1 while the ball is inside the cup, else 0 | a ball on a string under an actuated cup; must be *swung* in | **essential** — the reward is a needle in a haystack; without imagined rollouts to practice the catch, the agent almost never generates a positive example to learn from |
| **reacher-hard** | near-sparse: 1 inside a *small* target disc, else ~0 | 2-link arm | **essential** — the small target makes reward sparse; the model helps the agent find it |
| **acrobot-swingup** | shaped by the tip's height toward the top | **underactuated** double pendulum (torque only at the middle joint), chaotic | **essential** — reaching *and stabilising* the top needs multi-step lookahead through unstable dynamics; a reactive policy has nothing to react to until it's already up |
| **pendulum-swingup** | high only near upright; ~0 elsewhere | **underactuated** single pendulum; torque too weak to lift directly | **essential-but-exploratory** — must *pump energy* over several swings; the hard part is discovering the pump, not representing the value |
| **cheetah-run** | **dense**: ramps with forward speed toward ~10 m/s | planar running, smooth | helps a little — dense feedback every step; a reactive policy already climbs, the model just sharpens it |
| **cartpole-swingup** | dense: product of upright + centred | single cart-pole, benign | marginal — dense and low-dimensional; value learning alone nearly suffices |
| **finger-spin** | dense: high while the spinner exceeds a target angular velocity | 2-DoF finger flicking a free spinner | **redundant** — continuous contact feedback; nothing to plan |
| **walker-run** | dense: upright-torso × ramp-with-speed (~8 m/s) | planar biped, smooth | **redundant** — dense reward + stable gait cycle; a reactive value/policy is enough |
| **quadruped-walk** | dense: upright × ramp-with-speed | 3-D quadruped | **redundant** — same story in 3-D |

Read top-to-bottom this *is* the gradient of Result 1. The pattern is not about dimensionality or embodiment —
quadruped is the highest-DoF task here and needs the model *least*. It is about **how far reward has to
travel**. When reward is dense and the dynamics are stable, the value function can be learned directly from
experience and a learned forward model adds nothing; when reward is sparse, delayed, or gated behind
unstable/underactuated dynamics, the model earns its keep — either by supplying a value representation that a
reactive learner can't (the *representational* channel, which VBN measures) or by generating the rewarding
experience in the first place (the *exploratory* channel, which VBN cannot). Those two channels are the whole
story of the next three results.

## Result 1 — the nine-task gradient and the cross-model law (H-COMPRESS, confirmed within scope)

We ablate the world model per task (strip its forward-dynamics learning) and measure the change in
return. On a fixed architecture, WM-dependence spans the **full** range:

| regime | tasks (WM-dependence = fraction of return lost when stripped) |
|---|---|
| **essential** | ball-in-cup (collapse), reacher-hard (−98%, n=2), acrobot (−85%, n=3) |
| helps / marginal | cheetah (−10%), cartpole (−3%) |
| **redundant** | finger (+5%), walker (+7%), quadruped (+1%) |

The essential end is sparse-reward / long-horizon / precision control; the redundant end is dense
continuous control. And the VBN fingerprint predicts the ordering *a priori*: return recovered through a
16-dimensional value bottleneck correlates with WM-dependence at **ρ = −0.90 (n = 5)** — measured with
compressibility in TD-MPC2 and dependence in Dreamer, so this is a genuinely **cross-model** prediction.

![VBN compressibility predicts WM-dependence]({{ '/images/part20-correlation.png' | relative_url }})

This is Part 19's **H-COMPRESS** graduating from prediction to result: low value-compressibility ⇒ the
world model is load-bearing; high compressibility ⇒ redundant.

## Result 2 — the honest boundary (where the probe stops working)

VBN is a *value* probe, so it is silent on tasks whose difficulty is **exploratory** rather than
representational. Two tasks sit exactly there:

- **ball-in-cup**: its value is *maximally* compressible in TD-MPC2 (a 16-dim bottleneck recovers ~100%
  of return) yet stripping the world model collapses it in Dreamer. The collapse is not a
  value-representation deficit — it is an exploration failure (without imagined rollouts the agent never
  discovers the catch).
- **pendulum**: covered below.

Both are excluded from the value-limited correlation and reported as the diagnostic's characterized
scope, not as hidden failures. This turns a limitation into a second contribution: a clean
**value-limited vs. exploration-limited** split, grounded in a decomposition of world-model benefit into
a representational channel (what VBN sees) and an exploratory channel (what it cannot).

### Ball-in-cup is not an outlier — it is the exploration axis, caught in the act

A reviewer's first objection to the correlation figure is the lone red point at top-right: **ball-in-cup**,
where value is *maximally* compressible (a 16-dim bottleneck recovers ~100% of return) yet the world model is
essential. If it were a mystery we would be nervous. It is not — we can watch the mechanism directly. Here are
the raw Dreamer learning curves, vanilla vs. world-model-stripped, on the two sparse-reward tasks:

![Bimodal collapse on sparse-reward tasks]({{ '/images/part20-bimodal.png' | relative_url }})

Look at the left panel. Vanilla Dreamer (blue) catches the ball essentially every episode (~980). The stripped
agent (red ×) is **bimodal**: it *does* occasionally catch — you can see the ~980 hits — but it can never
*consolidate*, and most episodes score exactly 0. Without imagined rollouts, the single sparse catch signal is
too rare to stabilise the policy; the agent keeps forgetting. This is precisely an **exploration/consolidation
failure, not a value-representation failure** — and VBN, honestly, reports exactly that: the value *is* easy
(maximally compressible), so the probe correctly says "no representational deficit here." The deficit is in
generating the data, which a value probe on already-collected data cannot see.

Three independent lines of evidence pin this down, which is why we now treat it as a **finding rather than an
outlier**: (i) the bimodal curve above (the WM's job here is exploration, visibly); (ii) a tuned **SAC** — a
model-free learner with no world model — *solves* ball-in-cup (984 vs 979 for TD-MPC2, below), so the task is
not intrinsically model-requiring, it is *imagination*-requiring for on-policy Dreamer; and (iii) the
correlation itself: including cup drops Spearman ρ from **−0.90 to −0.09**. A single point that collapses the
fit is not noise to be hidden — it is a point living on a *different axis*. Ball-in-cup is the cleanest
demonstration in the paper that world-model benefit has two separable channels, and that our probe measures one
of them by design.

## Result 3 — the mistake we caught by replicating (pendulum)

Pendulum-swingup was, for two days, the paper's most dramatic collapse point: a single seed showed
stripped return of exactly **0.0** vs ~806 vanilla. Multi-seed replication (the kind of run it is easy
to skip once you "have the result") told a different story. Across three seeds the stripped agent scored
**0 / 727 / 0**: two seeds never discover the swing-up, one matches vanilla with *no* representational
deficit. Pendulum is **bimodal**, exploration-limited — the seed-1 "collapse" was a lucky/unlucky draw,
not a value signal. We removed it from the clean correlation (ρ moved −0.94→−0.90, but now on a *clean*
value-limited set) and report the full seed range. The honest version is more defensible than the
dramatic one, and the episode is a small advertisement for running the replication seed.

**Why bimodal, mechanistically?** Pendulum swing-up is *underactuated*: the motor is deliberately too weak to
lift the pole straight up against gravity. The only route to the top is to **pump energy** — swing back and
forth, building amplitude over several oscillations, until one last swing carries the pole over and the
controller catches it at the unstable upright equilibrium. That makes learning a discovery problem with two
attractors, and essentially two outcomes per run:

- **Discover the pump** (via exploration noise, early) → the near-upright reward reinforces the sequence and it
  locks in → return ~800–1000.
- **Miss it** → the agent settles into a low-energy local optimum (small oscillations at the bottom, or idle
  spinning) that pays ~0 and provides *no gradient* pointing toward the pump → stuck at 0, permanently.

With the world model, imagined rollouts let the agent rehearse the pump internally, so vanilla Dreamer finds it
across seeds (high, if noisy — blue, right panel of the figure above). Strip the model and discovery becomes a
per-seed coin flip: hence **0 / 727 / 0**. The seed-1 "collapse" was simply the losing side of that flip; the
727 seed proves there is *no representational deficit* — a stripped agent that happens to discover the pump
matches vanilla. The right panel shows one of the *recovering* seeds: it does climb, but noisily, with frequent
dips back to 0 — the fingerprint of a policy sitting right on the boundary between the two basins. So pendulum
is the same **exploration axis** as ball-in-cup, expressed as bimodality *across seeds* rather than *across
episodes*. Both are exactly what a value probe is blind to, and exactly why we quarantine them from the
value-limited correlation instead of letting them quietly degrade it.

## Result 4 — an independent cross-check we didn't have to run

The exploration-limited boundary is corroborated by data we did not generate. The official TD-MPC2
release also reports a tuned **SAC** — a model-free learner with *no* world model. On the tasks we call
exploration-influenced, SAC solves them just as well:

| task | TD-MPC2 | SAC | gap |
|---|---|---|---|
| ball-in-cup | 984 | 979 | **+5** |
| reacher-hard | 982 | 944 | **+38** |
| acrobot | 663 | 72 | +591 |
| hopper-hop | 449 | 117 | +332 |
| pendulum | 838 | 575 | +263 |

Ball-in-cup and reacher-hard collapse in Dreamer-without-a-WM but are solved by SAC — confirming their
failure is exploratory, not value-limited, with a third independent method. The large gaps elsewhere set
up the week's headline result.

## Result 5 — H-WM-ABSTRACT, answered: it's the value pathway

Part 19 asked which knob does the work when a world model helps. HopperHop is the sharpest test: TD-MPC2
reaches ~450–500 there while PPO is walled at ~40 (even at 94× the budget) and SAC is unreliable
(bimodal, ~47 on the matched seeds, ~188 on average). Yet we already knew two odd facts about Hop — the
consistency loss is **removable** there, and MPPI planning adds essentially nothing. So what beats
model-free RL?

We ran a matched-seed **rung-ladder**, reading the π-only and MPPI eval columns of the same runs:

![The value-pathway ladder on HopperHop]({{ '/images/part20-ladder.png' | relative_url }})

| rung | what it isolates | HopperHop return |
|---|---|---|
| SAC | model-free, no latent model, no planner | 47 (matched) · 188 (avg) |
| **value pathway** = strip-consistency + π-only | value-equivalent latent + off-policy value, **no WM, no planner** | **502** |
| + planning | adds MPPI | 487 |
| full TD-MPC2 | + consistency + planning | 509 |

**The value pathway alone (502) ≈ full TD-MPC2 (509) and captures essentially the entire gap over SAC.**
Adding back the world-model consistency loss and the planner changes the result by under 2% (planning
even hurt one seed). So on the one task where the model-based agent decisively wins, what does the work
is the **value-equivalent representation + off-policy value learning** — not the forward model, not the
planner. This is the seed for a second paper: a lighter, task-adaptive learner that keeps the value
pathway and pays for the world model only where a probe like VBN says it earns its keep.

### "But that's just Hopper" — the objection, taken seriously

Anyone who has read Voelcker, Hussing & Eaton's *[Can we hop in general?](https://arxiv.org/abs/2410.08870)*
(2024) should be suspicious of a headline resting on the Hopper environment. Their case study shows that
Hopper-family benchmarks are **unusually sensitive to benchmark and design choices** — the same algorithm can
be judged strong or weak depending on which Hopper variant, action repeat, and protocol you pick, and those
choices are almost never justified in the literature. We take that seriously. It is exactly why our claim is
*not* a leaderboard statement of the fragile "method X beats Y on Hopper" kind. It is a **within-task
decomposition** — which *internal component* of one fixed agent carries the win — cross-checked against two
independent model-free learners (SAC, bimodal ~47/188; PPO, walled at ~40 even at 94× the budget). A
decomposition is far more robust to benchmark-design wobble than a horse-race, because every arm runs on the
identical task, protocol, and seeds.

But the decisive answer to "is it just Hopper?" is not an argument — it is to run the *identical ladder on a
task at the opposite end of the VBN gradient*, and to predict the outcome in advance.

## Result 6 — the essential-task mirror (acrobot), predicted in advance

HopperHop is a task VBN calls **WM-redundant** (compressible value, removable consistency loss). Its opposite
is **acrobot-swingup**: VBN says its value is *not* compressible, and its Dreamer WM-dependence is −85%. If the
value-pathway story were a Hopper artifact, the ladder there would look the same. The diagnostic predicts it
will look **opposite** — and it does:

![Acrobot vs HopperHop: the same ladder, mirrored]({{ '/images/part20-acrobot-ladder.png' | relative_url }})

| rung | HopperHop (VBN: redundant) | AcrobotSwingup (VBN: essential) |
|---|---|---|
| SAC | 47 · 188 | 72 |
| value pathway (strip + π) | **502** ≈ full | **335** (only ~86% of full) |
| strip + planning (mppi) | 487 | **48** — *collapse* |
| full TD-MPC2 (π / mppi) | 509 | 388 / 395 |

Two things flip. First, the value pathway alone recovers only **~86%** of full on acrobot (vs a dead heat on
Hopper) — the world model's *representation* is doing measurable work. Second, and more striking:
**planning through the stripped latent collapses to ~48** (n=4 seeds: 7, 42, 61, 55), a fifth of full
TD-MPC2's ~395. On a chaotic, underactuated double pendulum, once you remove the consistency loss the learned
latent dynamics are no longer accurate enough to plan through — so MPPI, rolling out a broken model, produces
garbage. *The consistency loss — the "world model" term — is exactly what makes the dynamics good enough for
the planner.* On Hopper you could throw the model away; on acrobot the planner is helpless without it.

This is the result that turns the value-pathway observation from an anecdote into a **law with a sign**: VBN
tells you, before training, which regime a task is in. Where value is compressible, keep the value pathway and
drop the world model and planner; where it isn't, the world model is load-bearing — and specifically, *the
planner depends on it*.

One methodological check worth stating. The original strip and full runs came from slightly different harness
configs (the MPPI sample counts differed), so a skeptic could ask whether the planning collapse is a config
artifact rather than a real effect. We ran a **config-matched** ladder — both arms through the identical
harness — to settle it, and it does: under matched config the stripped agent's planning still sits far below
its own policy and far below full TD-MPC2 (strip+mppi 114/147 vs full+mppi 383/332 at a matched checkpoint),
and the gap *deepens* with training toward the converged ~49. The collapse is a property of the ablation, not
of the sampler. Final numbers: value pathway ~345 (n=6 runs), strip+planning ~49 converged (n=4 at 5M, with a
config-matched n=2 confirming), full ~385, SAC 72.

## Does the diagnostic generalize? JEPA and DINO

A natural question — and one worth heading off — is whether any of this is specific to the two world-model
families we tested. It is not, and the reason is architectural. Our two families already span the main axis of
world-model design:

- **DreamerV3** is a *generative* world model: it reconstructs observations, and its imagined rollouts are
  the exploratory engine we watched fail on ball-in-cup.
- **TD-MPC2** is a *value-equivalent* model — and its **consistency loss is, precisely, a JEPA objective**:
  predict the *next latent embedding*, with no decoder and no pixel reconstruction. That is exactly LeCun's
  Joint-Embedding Predictive Architecture recipe. So our "strip-consistency" ablation *is* the ablation of a
  JEPA-style latent-prediction term — and Result 6 shows that on an essential task, that JEPA term is what
  keeps the planner alive.

So the diagnostic already covers the JEPA family through TD-MPC2. What about **DINO / DINO-WM**? DINO is a
vision self-supervised *encoder* (no dynamics, no reward); DINO-WM (Zhou et al., 2024) builds a latent
dynamics model on **frozen DINOv2 features** — a JEPA-style world model in pixel space. Our state-based DMC
setup does not use pixels, so a from-scratch DINO-WM run is out of scope for *this* paper, but the diagnostic
transfers unchanged: run VBN on the value head of a DINO-WM agent, strip its latent-dynamics loss, and the
prediction is identical — *compressible value ⇒ the frozen-feature dynamics is redundant; incompressible ⇒
it's load-bearing.*

And we ran the concrete third-architecture check. Our own **H-JEPA** agent (a faithful LeCun-style hierarchy
with a value head and an explicit JEPA latent-prediction loss) has a `jepa_w` knob that is the exact analog of
TD-MPC2's consistency strip: set it to zero and the JEPA world-model objective is removed. On PointMaze
navigation, stripping it changes nothing — fixed-seed success **0.531 with the predictor, 0.531 without it**
(n=3), while the JEPA prediction loss jumps from ~0.001 to ~3.8, confirming the ablation genuinely took hold.
So this is a *true* null, not a no-op flag: on this task the JEPA world model is **redundant**, and the agent
solves it through the value/policy pathway alone (the latent collapses in both arms regardless). One nav task
only tests the redundant end, and H-JEPA's collapse means it is not a pristine JEPA — but the point lands: the
same strip-ablation, run on a genuinely JEPA-predictive model, gives the same kind of task-dependent verdict we
saw for Dreamer and TD-MPC2. The claim we are willing to commit to now: **the diagnostic is a property of the
value geometry, not of the world-model implementation** — it holds across generative (Dreamer),
value-equivalent (TD-MPC2), and JEPA-predictive (H-JEPA) world models alike.

## The nulls, reported loudly

- **The gate does not help.** The prescriptive fix the diagnostic seems to suggest — a plan-time gate
  that down-weights the world model's imagined rollout — is a confirmed negative at n=5:
  g0.0 = 701.9, g0.5 = 688.4, g1.0 = 686.5, g0.25 = 682.9 (a 19-point band, ≪ the 80–100-point seed
  spread). It is structural: with horizon 3 and γ=0.99 the gated term is ~3% of the plan score, and
  g=0 is not WM-free anyway. The contribution is a *diagnostic*, not a repaired algorithm.
- **Parity, honestly.** Our TD-MPC2 reimplementation is a non-anomalous baseline, not bit-parity:
  hopper ≈0%, cheetah −5%, walker −6.8% (once data-collection mode is matched), acrobot −23% (open).
  All TD-MPC2 claims in the paper are within-implementation comparisons; the cross-model headline rests
  on unmodified DreamerV3.

## Shipped

- **AAAI-27 submission paper.** *"When Does a Learned World Model Help? A Value-Sufficiency Diagnostic
  that Predicts World-Model Dependence Across Architectures."* Seven pages in the official
  `aaai2027.sty` submission format, double-blind clean, with a motivation figure, a method-pipeline
  figure, three data plots, two tables, and the mandatory reproducibility checklist. Compiles clean.
- All results logged to the public issues and the campaign ledger; every figure number traces to disk.

## Next

- **Paper 2 (the value pathway).** The ladder now has its second, opposite-regime point (acrobot, Result 6) —
  next is a *config-matched n=3* confirmation (running) and a third task, then turning the decomposition into a
  task-adaptive recipe: keep the value pathway, add the world model only where a VBN-style probe predicts it is
  load-bearing. The JEPA generalization (H-JEPA VBN + strip on DMC) is queued behind it.
- **Broaden the diagnostic.** The correlation is n=5 value-limited tasks; extend it, and test the
  value-limited/exploration-limited taxonomy on manipulation (Meta-World) and navigation (sparse-goal),
  which is the natural home for the exploration axis we can currently only bound with two points.
- **Training-free criterion.** The reward-propagation-horizon / controllability / value rate–distortion
  proxies remain the conjectural prize: predicting WM-dependence from the task spec alone, with no probe
  at all.
