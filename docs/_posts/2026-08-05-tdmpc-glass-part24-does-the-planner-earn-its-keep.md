---
layout: post
title: "TD-MPC-Glass, Part 24: Does the Planner Earn Its Keep? A Self-Contained Account"
date: 2026-08-05
description: "A complete, self-contained write-up of the planner experiments: what a world model is, what MPPI planning is, what a gate is, and what happens when you switch planning off. Removing the planner is free on cup-catch, finger-spin and cheetah-run and fatal on hopper-hop (364 to 103 at n=5, every planning seed above every never-planning seed, p=0.004), walker-run and acrobot. DreamerV3, which has no planner at all, does not collapse the same way - but removing ITS world model does (209.0 to 64.3, p=0.029), so the unifying claim is that the model must be exploited somehow, by whatever route. A gate that switches planning on at 150k recovers most of the gap on hopper-hop and nothing on walker-run, for ~13% less wall-clock; its cost is predictable to within 0.2 points on three configurations, but its benefit is not predictable at all - this section was published wrong twice in opposite directions before n=3/n=6 settled it, which is itself the most instructive result here. Includes every table, three figures, per-seed values, and a section answering the objections we expect."
---

> **This post is meant to be read on its own.** It assumes you know what reinforcement learning is
> and nothing else about this project. Every term is defined where it first appears, every number
> has its seed count, and the last section answers the objections we expect. If you only read one
> thing about this work, read this.

---

## 0. The question in one paragraph

TD-MPC2 is a model-based RL agent. At every environment step it does two things that cost time:
it **trains a world model**, and it **plans** with that model by simulating hundreds of candidate
action sequences before choosing one. Planning is roughly a third of the wall-clock. We asked a
simple engineering question — *can we switch the planner off and get the same result faster?* — and
the answer turned out to be: **sometimes completely free, sometimes catastrophic, and which one it
is depends on the task in a way that is not obvious in advance.**

---

## 1. Vocabulary

Skip this if the terms are familiar.

**World model.** A learned simulator. Instead of predicting pixels, TD-MPC2 learns a *latent*
dynamics model: given a compressed state `z` and an action `a`, predict the next compressed state
`z'`, plus the reward and the value. "Latent" just means the model works in its own internal
coordinates rather than in raw observations.

**Policy prior (`π`).** A small neural network that maps a state directly to an action, exactly
like a model-free actor. TD-MPC2 trains one *alongside* the world model.

**MPPI planning.** At decision time the agent does *not* simply ask the policy prior for an action.
It samples many candidate action sequences, rolls each one forward through the learned world model,
scores them by predicted reward-plus-value, and takes a weighted average of the best ones. TD-MPC2's
defaults: **512 sampled sequences, horizon 3, 6 refinement iterations, 64 elites** — so roughly
18 sequential rounds of model evaluation per environment step. This is the expensive part.

**The ablation.** TD-MPC2 exposes a config flag, `mpc`. With `mpc=true` the agent plans as above.
With `mpc=false` it acts straight from the policy prior. **The world model is still trained either
way** — we are removing the *planner*, not the model.

**A "gate".** Rather than planning always or never, switch it partway through training. We use two:
- `WMP_MPC_OFF_AT=N` — plan until step N, then stop.
- `WMP_MPC_ON_AT=N` — act from the policy prior until step N, then start planning.

A gate is the practical object here. If the planner is only useful under some conditions, then
"detect the condition and switch" is the design, and the gate is the simplest possible version of
that detector — a fixed step count.

**Return.** Sum of rewards over an episode. DMControl tasks are scaled so 1000 is roughly perfect
and 0 is failure. **Seed.** One training run with a given random initialisation; RL results vary a
lot between seeds, which is most of what this post is about.

---

## 2. The premise was wrong, and checking cost nothing

The original proposal was "remove the world model when it isn't necessary, to save time". Before
spending any GPU-hours we checked the premise against throughput telemetry already on disk:

| what you remove | throughput | saving |
|---|---|---|
| the world-model **loss** (`consistency_coef` 20 → 0) | 31.45 vs 31.54 steps/s (n≈4700 snapshots each) | **none** |
| the **planner** (MPPI on → off) | 43.3 → 66.6 steps/s | **~35% of runtime** |

Turning off the world-model *loss* saves nothing measurable, because the latent rollout still
happens — reward and value prediction need it whether or not a consistency term reads it. **All of
the available saving is in the planner.** So the experiment became "switch off planning", not
"switch off the world model".

*Correction from an earlier draft: we first reported these as 25.26 vs 24.89 steps/s over ~365
snapshots. That aggregation keyed on a field name the records do not contain, so it silently
sampled about 2% of the rows. The conclusion did not change; the numbers above are the corrected
ones.*

---

## 3. The main experiment

**Setup.** Official `nicklashansen/tdmpc2` @`e9f5932` — not our own re-implementation, which is a
distinction that has bitten this project before. State observations (not pixels), `model_size=5`,
400,000 environment steps, evaluation every 25k or 50k steps over 10 episodes. Two arms:

- **planning throughout** — stock TD-MPC2, `mpc=true`
- **never planning** — `mpc=false` from step 0, everything else identical

<img src="{{ site.baseurl }}/assets/part24-per-task.svg" alt="Bar chart of final return with and without planning across six DMControl tasks, with individual seeds shown as dots" style="width:100%;height:auto;margin:1.5rem 0;">

### Table 1 — final return at 400k steps

| task | planning throughout | never planning | Δ | verdict |
|---|---|---|---|---|
| cup-catch | 982.8 (n=1) | 981.5 (n=1) | −1.3 | tie, **uninformative** |
| finger-spin | 985.0 (n=1) | 984.7 (n=1) | −0.3 | tie, **uninformative** |
| cheetah-run | 855.9 ±120.2 (n=5) | 882.0 ±49.0 (n=4) | +26.1 | not separable |
| **walker-run** | **809.9 ±11.7 (n=3)** | **690.6 ±64.2 (n=3)** | **−119.3** | planning wins |
| **acrobot-swingup** | **417.2 (n=3)** | **305.0 (n=3)** | **−112.2** | planning wins |
| **hopper-hop** | **364.2 ±76.6 (n=5)** | **103.3 ±55.7 (n=5)** | **−261.0** | planning wins, **p=0.004** |

Per-seed values, because means hide everything that matters here:

| task | planning throughout | never planning |
|---|---|---|
| cheetah-run | 640.9, 903.6, 908.4, 910.9, 915.6 | 808.7, 901.2, 907.7, 910.4 |
| walker-run | 799.1, 808.4, 822.3 | 622.5, 699.2, 750.2 |
| hopper-hop | 319.1, 319.6, 333.4, 349.6, 499.4 | 42.8, 45.6, 120.0, 153.0, 154.9 |
| acrobot-swingup | 192.1, 504.9, 554.6 | 24.3, 409.2, 481.5 |

**walker-run is the statistically cleanest cell.** Every planning seed beats every never-planning
seed. Under exchangeability that arrangement has probability 1/C(6,3) = **0.05** exactly — an exact
rank test, not a t-test, because at n=3 per arm a t-test assumes more than we know.

**hopper-hop is the largest effect, and it is not a degradation.** 103.3 against 364.2 means the
agent barely learns to hop at all. Removing the planner there does not cost some performance; it
costs most of the task. At n=5 per arm **every planning seed beats every never-planning seed**
(319.1 is the worst planning run, 154.9 the best never-planning run), which is an exact rank test
at p = 1/C(10,5) = **0.004**.

*This cell was first reported at n=3 as 394.1 vs 69.5, a 5.7× ratio. At n=5 it is 364.2 vs 103.3,
a 3.5× ratio. The direction did not move and the evidence got stronger — but the magnitude fell by
a third when two seeds per arm were added, which is the honest measure of how well an effect size
is pinned down at n=3 on this benchmark.*

**Two of the six tasks were wasted compute.** cup-catch and finger-spin both sit at ~982–985 against
a ceiling near 1000. When both arms are at the ceiling the comparison cannot distinguish "the
planner is unnecessary" from "this task is too easy to show a difference". We flagged them as
uninformative *before* they finished, which mattered, because —

**they finished first.** Easy tasks train fastest. Had we stopped when the first two results
arrived, we would have shipped "removing the planner is free" with two confirmations behind it.

---

## 4. What happens inside a run

The final numbers hide the interesting part. On hopper-hop, the two arms are **indistinguishable
for the first 150,000 steps** and then diverge sharply.

<img src="{{ site.baseurl }}/assets/part24-hopper-curves.svg" alt="Learning curves on hopper-hop for planning, no planning, and the gate arm, showing divergence after 150k steps" style="width:100%;height:auto;margin:1.5rem 0;">

### Table 2 — hopper-hop, matched-step returns

| step | never planning (s1) | never planning (s2) | planning throughout |
|---|---|---|---|
| 50,000 | 7.4 | 7.6 | 0.0 |
| 100,000 | 42.5 | 76.5 | 59.5 |
| 150,000 | 97.0 | 77.8 | 101.5 |
| 200,000 | 81.4 | 109.0 | **258.1** |
| 250,000 | 104.2 | 68.9 | **315.5** |

The reading: **a planner is worth nothing until the model is good enough to plan through.** Early
on, rolling out a bad model gives bad action sequences, so MPPI is no better than the policy prior
— and you are paying 35% more wall-clock for it. Once the model crosses some quality threshold the
planner starts extracting real value from it, and the no-planning arm flattens out and stays flat.

This also explains an earlier oddity. When we first tried the *opposite* gate — plan early, then
switch planning **off** at 200k — that arm did worse than never planning at all. It removes the
component exactly when it has started to pay.

---

## 5. The gate

If the planner is useless before ~150k and valuable after, the obvious design is to **switch it on
at 150k** rather than paying for it from the start.

### Table 3 — hopper-hop, three schedules

| arm | return at 400k | wall-clock | seeds |
|---|---|---|---|
| planning throughout | 394.1 ±91.5 | 2h47m | 333.4, 349.6, 499.4 |
| **gate: plan from 150k** | **250.4 ±109.0** | **2h26m** | 69.1, 230.0, 304.1, 310.5, 338.6 |
| never planning | 69.5 ±43.8 | 1h49m | 42.8, 45.6, 120.0 |

<img src="{{ site.baseurl }}/assets/part24-pareto.svg" alt="Scatter plot of return against wall-clock for the three hopper-hop schedules, showing the gate as an intermediate Pareto point" style="width:100%;height:auto;margin:1.5rem 0;">

The gate recovers **56%** of the gap between never-planning and planning-throughout, for **13.0%**
less wall-clock. It is a genuine Pareto point — better return than never planning, less time than
always planning — and **not** a free recovery.

**The saving is predictable, which is the part we care about.** Planner overhead is 34.7% of
runtime; the gate skips planning for 150k/400k = 37.5% of training; 34.7% × 37.5% = **13.0%**
predicted, and 13.0% observed. Because the arithmetic closes, the knob is understood: a later gate,
or one triggered by a model-quality probe instead of a fixed step, moves along a known curve rather
than being a guess.

*Correction: at n=2 we reported this gate as recovering "essentially all" of the benefit, from
seeds scoring 310.5 and 338.5. Seeds 4 and 5 came in at 230.0 and 69.1. The corrected 56% then held
when a further seed was added, which the original figure had not.*

**One seed (69.1) never learned at all**, flat near zero from step 0, before the gate could fire.
hopper-hop is bimodal in that way — a fraction of runs simply never get off the ground. It is
included in the mean because dropping it would need a rule stated in advance, and we did not have one.

**Update 2 — this section has now been wrong twice, in opposite directions. Read this part, not
the earlier versions.**

The honest history, because it is the most instructive thing here:

1. Mid-flight, one walker gate seed tracked below both extremes. We wrote that a badly timed gate
   "may be worse than not gating at all".
2. The second walker seed came in at 811.2 and we corrected the section to say the gate recovers
   **100%** of the gap on walker and 56% on hopper — and drew a conclusion from that ordering.
3. The third walker seed (687.8) and the sixth hopper seed (541.9) arrived. **The ordering
   reversed again.**

Final figures at n=3 (walker) and n=6 (hopper):

| task | never plan | plan throughout | gate at 150k | gate recovers |
|---|---|---|---|---|
| walker-run | 690.6 (n=3) | 809.9 (n=3) | **687.1** (n=3: 562.2, 687.8, 811.2) | **~0%** |
| hopper-hop | 69.5 (n=3) | 394.1 (n=3) | **299.0** (n=6: 69.1, 230.0, 304.1, 310.5, 338.6, 541.9) | **71% (mean), 86% (median)** |

On walker the gate lands **on top of the never-planning arm** (687.1 vs 690.6) — the 811.2 seed we
built the "100%" claim on is the top of a very wide distribution spanning 562–811, not a
representative value. On hopper the gate recovers most of the gap, and one of its six seeds (541.9)
exceeds every planning-throughout seed.

**What survives.** The gate is worth something on hopper-hop and nothing on walker-run. The wall-clock
saving still matches prediction in every configuration measured (13.0/13.0, 12.6/12.8, 4.2/4.2) —
that part was a timing measurement and has never moved. What is *not* established is any rule for
when a gate recovers, and both mechanisms we proposed for it are now dead: "gates work where the
un-gated arm is useless" was contradicted at n=2, and its replacement "recovery tracks pre-gate
policy quality" is contradicted at n=3/6.

**What this cost, and the actual lesson.** Nothing was spent but compute, because every version was
published with its seed count attached. But the sequence — n=1 wrong, n=2 wrong in the opposite
direction, n=3/6 different again — is the clearest evidence in this whole campaign that **these gate
cells are too noisy to compare at the sample sizes we were reporting at.** The walker gate spans
562–811 across three seeds of one configuration. Our stated rule was "n≥3 before any cross-arm
comparison", and the 100% claim broke it at n=2 within hours of the rule being written down.

The correct treatment for a cell like this is a failure-rate plus a conditional distribution, not a
mean, and not a headline. We are reporting it here rather than quietly fixing the number because
the error pattern is more useful to a reader than the result.

---

## 6. The result does not generalise beyond TD-MPC2 — and that is the interesting part

DreamerV3 is another world-model agent. It has **no planner at all**: it trains a policy inside an
imagined rollout of its world model and acts from that policy directly. If "you need a planner for
hard control" were true, Dreamer should fail on hopper-hop.

### Table 4 — hopper-hop across agents

| agent | hopper-hop | env steps |
|---|---|---|
| DreamerV3 (**no planner**) | 209.0 (n=3: 187.6, 214.6, 224.9) | ~500k |
| TD-MPC2, planning throughout | 394.1 (n=3) | 400k |
| TD-MPC2, gate at 150k | 250.4 (n=5) | 400k |
| TD-MPC2, never planning | 69.5 (n=3) | 400k |

A planner-free agent reaches ~209 where TD-MPC2-without-its-planner reaches ~103 — on *more*
environment steps, so if anything the comparison flatters TD-MPC2.

**So the correct claim is narrower than "planning is necessary":**

> In TD-MPC2, MPPI planning is load-bearing on walker-run, hopper-hop and acrobot, and free to
> remove where the policy prior already solves the task. The collapse without planning is a
> property of **this agent's policy prior**, not evidence that model-based control requires a
> planner.

That is a better research question than the one we started with: *what does DreamerV3's prior have
that TD-MPC2's lacks?* Candidates — a much higher replay ratio, a different actor objective, a
generative rather than value-equivalent model — are all testable.

**The probe came back, and it answers the question.** Removing DreamerV3's dynamics loss
(`dyn=0`, so the world model is no longer trained) collapses it:

| DreamerV3, hopper-hop | seeds | mean |
|---|---|---|
| `dyn=1.0` — model trained | 187.6, 214.6, 224.9 | **209.0** (n=3) |
| `dyn=0.0` — model not trained | 0.0, 65.5, 81.5, 110.1 | **64.3** (n=4) |

Complete separation — every model-trained seed beats every model-untrained seed — exact rank test
p = 1/C(7,3) = **0.029**.

Line the four configurations up on the same task:

| configuration | hopper-hop |
|---|---|
| TD-MPC2, planning | 364.2 (n=5) |
| DreamerV3, model trained | 209.0 (n=3) |
| TD-MPC2, **no planner** | 103.3 (n=5) |
| DreamerV3, **model not trained** | 64.3 (n=4) |

**Both agents fall into the same 60–105 band once the world model stops being exploited**, by
whichever route that agent uses — TD-MPC2 by planning at decision time, DreamerV3 by training its
policy inside imagined rollouts. Dreamer's advantage is therefore not a better actor-critic.

That points at a more unifying statement than "planning is load-bearing": **what matters is whether
the world model is exploited at all, not the particular mechanism by which it is exploited.** It is
also the only cell in this campaign where adding seeds did not move the verdict — n=2 and n=4 agree.

Limits: the two ablations are not exact analogues (`dyn=0` removes the model's *training signal*,
`mpc=false` removes its *use at decision time*), the agents differ in many other ways, and the
budgets differ (500k vs 400k). This is the strongest lead for the next paper, not a settled result.

*Correction: we earlier reported Dreamer at 125.3 and said it "lands with the never-plan arm". That
was one run read at 256k steps and compared against 400k results. Trained to ~500k it reaches 206.3
and sits with the gate arm instead. The comparison reversed.*

---

## 7. What went wrong three times in one day

This is the part we would most like feedback on.

### Table 5 — claims that decayed under replication

| claim | first reading | after replication | outcome |
|---|---|---|---|
| removing the planner is free, 1.87× faster | cheetah-run only, n=1–2 | fails on 3 of 4 informative tasks | **falsified** |
| never-planning is 33× more reliable on cheetah | sd 157.2 vs 4.7 (n=3 each) | 2.5× (n=5/4) | **withdrawn** |
| the gate recovers the full benefit | 310.5, 338.5 (n=2) | 56% of the gap (n=5) | **corrected** |
| DreamerV3 corroborates "planner-free fails" | 125.3 at 256k (n=1) | 209.0 at 500k (n=3) | **reversed** |

All four have the same shape: **a claim formed from the runs that finished first, and the runs that
finish first are not a random sample.** Three distinct mechanisms produced that, and the second is
specific to this kind of study:

1. **Easy tasks finish first.** cup-catch and finger-spin are fast *and* uninformative.
2. **The cheaper arm finishes first.** Never-planning runs 1.9× faster, so its seeds arrive earlier,
   so any early comparison is weighted toward whatever the cheap arm happens to show. This is a
   hazard in *any* speed-versus-quality ablation.
3. **Partial checkpoints exist first.** The Dreamer error was reading a 256k checkpoint against
   400k results.

A fourth, unrelated trap worth recording: **"same seed" is not "same draw".** Changing `eval_freq`
from 50,000 to 25,000 perturbs the RNG stream enough that seed 1 under one schedule diverged
completely from seed 1 under the other (1.61 vs 97.0 at 150k, nominally identical configuration).
Anything that consumes randomness — including evaluation — breaks seed pairing.

**What kept the cost at roughly six GPU-hours** rather than a retracted paper: the first write-up
was published as a *hypothesis with named falsifiers* before the data existed — "acrobot-swingup
and hopper-hop are the tasks I expect to break it". Both broke it. Nothing had to be retracted from
a paper, only from a note that had already said it might be wrong. Every correction above is in the
git history rather than silently edited.

**Rule adopted:** no cross-arm comparison is reported until every cell is at matched budget and n ≥ 3.

---

## 8. Objections we expect, and our answers

**"n=3 is too few."** Agreed, and it is why we report exact rank tests and per-seed values rather
than t-tests, and why the cheetah cell is reported as *not separable* rather than as a win for
either arm. Where n=3 is enough is when the arms do not overlap at all — walker-run, where every
planning seed beats every never-planning seed, p=0.05 exactly. Where it is not enough, we say so.

**"You only tested six tasks, and two were useless."** True. The two ceiling-limited tasks were a
planning error on our part — we should have screened for headroom first. The three informative
tasks all point the same way, and the one that does not (cheetah) is reported as null rather than
quietly dropped.

**"Isn't this just saying TD-MPC2's planner is good? That's already known."** The novel parts are
(a) that it is *worthless for the first ~150k steps* on hard tasks, which is not in the paper; (b)
that this makes a schedule a real design axis with a predictable cost curve; and (c) that a
planner-free agent does *not* collapse the same way, so the failure is about TD-MPC2's prior rather
than about planning.

**"The wall-clock numbers depend on your hardware."** Yes. All wall-clock comparisons in this post
are within a single machine, because environment stepping is CPU-bound and our boxes differ. Returns
are hardware-independent and are pooled across boxes; timings are not. We excluded one gate seed
from the timing table for exactly this reason.

**"Your cheetah control has a seed at 640.9 and another at 915.6. Is your setup broken?"** No —
that cell genuinely has that spread. Pooled with an earlier five-seed baseline, planning-throughout
on cheetah-run spans **640.9–915.6 across eight seeds**. Establishing that *before* running
single-seed comparisons on that task would have saved us the withdrawn reliability claim.

**"Why should I believe the gate result when one seed scored 69?"** You should believe it as far as
n=5 with a bimodal task allows, which is not very far. It is reported with the degenerate seed
included, the mean and spread shown, and the walker counter-example flagged as provisional and
possibly negative. It is a direction, not a result.

**"What would change your mind?"** For the main claim: a task where the policy prior clearly cannot
solve the task alone, yet never-planning matches planning at matched steps and n≥3. For the gate:
walker-run's 50k-gate arm failing as well as its 150k arm, which would mean the disruption is
intrinsic to switching mid-run rather than a matter of timing.

---

## 9. What this is for

Two threads come out of it.

**An efficiency paper.** Not "world models are slow, here is a fast one", but *measure which
component is load-bearing per task, then remove what is not.* The planner is the only component
with a real wall-clock price; its value is task-conditional and phase-conditional; and the saving
from a schedule is predictable from two measurable quantities. A gate triggered by a model-quality
probe — rather than a fixed step — is the concrete next build, and it connects directly to the
probe-then-prune idea from the earlier diagnostic work.

**A design question.** DreamerV3's prior survives without a planner where TD-MPC2's does not. If we
can identify why, that is a statement about what makes a policy prior robust in a model-based agent
— which is a better paper than anything in this post.

---

### Reproducibility

Every number here has per-seed values committed in `data/` in
`SuuTTT/world-model-paper@claude-opus-4-8/world-model-paper`, alongside the reports the tables were
built from. The planner gate is a ~20-line patch that adds `WMP_MPC_OFF_AT` / `WMP_MPC_ON_AT` to
TD-MPC2's trainer; it is additive-only (zero deleted lines against the stock file) and inert unless
the environment variable is set, so the reference path stays the arbiter. Figures are generated by
`docs/assets/make_part24_figs.py`, stdlib only.

Total compute for everything in this post: three 4×RTX-3060 boxes at roughly $0.67/hour, about $15.
