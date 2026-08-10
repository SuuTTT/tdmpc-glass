---
layout: post
title: "TD-MPC-Glass, Part 29: The Complete Annotated Index — Every Post, Its Hypothesis, Its Result, and What We Now Think It Meant"
date: 2026-08-08
description: "Every post in this campaign with the hypothesis it was testing, the experiment, the result, and the lesson — plus a retrospective review of each of the nine phases read against what we know now. The through-line that only became visible in August: the world model's value lies in training rather than decisions, and is concentrated early in training. Most of this campaign's nulls are what that account predicts, and several were unwinnable for reasons we could not see at the time."
---

> **What this is.** Ninety posts is too many to search. Each phase gets a table — hypothesis,
> experiment, result, lesson — followed by **a review from current knowledge**: what we now think
> that phase was really testing, and why it came out as it did.
>
> **If you only want the decision:** [Part 28](../2026-08-07-tdmpc-glass-part28-the-iclr-decision-document/).
> **If you want data availability:** [Part 27](../2026-08-06-tdmpc-glass-part27-the-full-inventory-and-the-proposals-that-survive/).

**The account everything below is now read against**, established in August:
*the world model's value lies in what it makes the agent collect and learn during **training**, not
in what it computes at decision time — and that value is concentrated **early** in training.*

> **Scoped 2026-08-10.** That account holds for **online MBRL where the model is trained on
> self-collected data**. It is not a claim about planning in general: a JEPA/DINO-WM agent is a
> predictor plus a planner with *no policy and no RL*, and it works. The reconciling variable is what
> the model was trained on — broad data makes a model searchable cold; a self-collected stream does
> not. Two proposals below (**S1**, **S3**) come out of that correction, and one (**S4**) out of a
> second gap: **behaviour cloning was never run here as a baseline.** See
> [Part 28 §6](../2026-08-07-tdmpc-glass-part28-the-iclr-decision-document/#added-2026-08-10--two-gaps-one-scoping-correction-and-the-proposals-they-generate).

---

## Every proposal on the table, ranked

**Type matters more than rank.** A *finding* says what is true; a *theory* says why; a *method* says
what to do differently. A paper that lands is usually **finding + theory + method** — the phenomenon,
its mechanism, and the thing a reader can use. Rows are typed accordingly.

"Odds" are rough judgements of acceptance *if the work is finished and written well* — aids for
choosing between rows, not predictions.

<div class="wide-table wide-table--proposals" markdown="1">

| # | proposal | **type** | in one sentence | evidence | ICLR odds | to finish |
|---|---|---|---|---|---|---|
| **1** | Search is a training effect | **finding** (+ small method) | You cannot bolt a planner on at the end — it has to be there while the agent learns, because what it really does is change what the agent sees | **A, now on three legs** — 364 with planner during training vs 87 when added later; DreamerV3 shows the same with no planner at all; **cheetah dissociates the channels outright: planning buys ~1.00× there, yet removing the world-model loss costs 58–127 points (p≤0.063, n=5)** | **~60–70%** | writing only |
| **S1** | World models are representation learners, not simulators | **theory + method** *(added 2026-08-10, **running**)* | The world-model loss pays by shaping the encoder, not by giving you a simulator worth rolling out. Keep the loss, block its gradient into the encoder, and see whether the benefit survives | cheetah **dissociates already**: planning ~1.00× yet the loss is worth 58–127 pts; ablation verified on a fixed batch (loss bit-identical, encoder grad → exactly 0.0) | **~55–65%** — *both outcomes publish* | ~1 day |
| **S2** | Train the model on the **query** measure | **method** *(added 2026-08-10)* | A policy is queried where it is trained; a planner is not — MPPI evaluates latents that were never visited. Train the model on the states search actually looks at | none yet; replaces a withdrawn "circular dependency" framing (MPPI has no parameters, so no loop closes through it) | ~50% | ~2 weeks |
| **S4** | The demonstration crossover | **finding + method** *(added 2026-08-10)* | Sweep the number of demos: BC vs BC+planner vs RL+planner vs RL. Somewhere there is a crossover, and *finding it* is the result — "how many demos make your planner pointless" | none — **BC was never run as a baseline here**; the one BC arm (Part 25) failed because the controller was non-Markov, which says nothing about this | ~40%; partly defensive | ~4 days |
| **S3** | Coverage decides searchability | **theory + method** *(added 2026-08-10)* | Whether you can plan through a learned model depends on the coverage of its training data, not on whether a policy exists. This is why JEPA/DINO-WM can plan with no policy and we cannot | none yet; the natural synthesis of #1 with the JEPA objection | ~35%; highest null risk | ~2 weeks |
| **12** | Buffer-composition probe | **theory** *(the mechanism for #1)* | Prove it is the data: show planner-collected experience is measurably different, then feed that data to a planner-free agent and see if the gap closes | none yet | ~50% | ~4 days |
| **10** | Model-loss schedule | **method** *(the recipe from #1)* | Turn the world-model loss off partway through training: you lose nothing and save wall-clock. Establish where the cut-off is, across tasks | walker n=5: dropping at 250k costs 7.2 return (n.s.) and saves 7.2% time; at 50k saves 16.8% | ~40% | ~3 days |
| **2** | Anneal the update ratio | **method** | TD-MPC2 spends its time on 64 gradient updates per env step, not on planning. If the model matters early and not late, that budget probably can be annealed too — a much bigger saving | none yet; rests on two Grade-A results | ~50% *if it works* | ~2 days |
| **3** | Gate planning during training | **method** | Plan only where planning pays. A perfect gate on 10% of steps captures ~half the benefit — but the nine obvious signals for *where* all fail | **B/A** — ceiling measured (n=6); nine signals, three families, all Spearman ≈ 0 | ~45% | ~1 week |
| **11** | Aleatoric vs epistemic | **theory** | Planning should help when the policy is stuck for a fixable reason (credit assignment) and not when it is stuck on noise. Same headroom, opposite predictions — the clean way to kill the "you just had more room" objection | none yet | ~40% | ~1 week |
| **4** | The 5× shrink | **finding** (practical) | TD-MPC2 runs 5× smaller and 1.6× faster while keeping 83–100% of its score across 16 tasks — the default is heavily over-provisioned | **A** — full-suite sweep | ~40% alone; strong appendix | writing only |
| **5** | Reward structure sets search's value | **finding** | Make walker's reward a strict AND and planning becomes twice as valuable; just making the task harder does nothing | **A on walker**, **D as a law** — hopper says the opposite | ~30% alone; good section | 1 more task |
| **7** | How claims decay | **methodology** | Eight patterns looked real at 3 seeds; seven vanished at 5. Here is the anatomy, with timestamps and pre-registrations | **A** (it happened) | ~50% workshop | writing only |
| **6** | Programmatic policy as an oracle | **method** (borrowed) | Before spending 33M steps, have a coding agent write a controller: it tells you whether the task is solvable and whether your reward is hackable | **A** (it happened); method is Weng's | ~25% research, high as a workshop piece | done |
| **8** | Entity-graph compositional OOD | **finding** (orphan) | Graph latents generalise to object counts they never saw — but the advantage never reaches the controller | **B** — the campaign's first GO, honestly gated | ~25% as-is | needs a control-reaching result |
| **9** | Data-centric abstraction — **downgraded** | **method** | Apply the abstraction to what the agent *collects* rather than to its latent | **the closest test already failed**: SE-community subgoals gave Acrobot 460→382, **Cartpole-Sparse 316→7** | ~15% | needs a non-community mechanism |

</div>


### The paper you want: finding + theory + method

Your preference — method backed by theory, derived from a finding — is available and mostly built:

| role | row | status |
|---|---|---|
| **finding** — the phenomenon | **#1** the planner's value is in training, not decisions | **done, Grade A** |
| **theory** — why it is true | **#12** because it changes the buffer; show the data differs and that replaying it closes the gap | **4 days** |
| **method** — what to do about it | **#10** drop the world-model loss after X% of training: free return, 7–17% less wall-clock — and **#2** the same schedule applied to the update ratio, where the real compute goes | **3 days / 2 days** |

That is one coherent paper: *the world model earns its keep by shaping what the agent collects, early;
here is the measurement, here is the mechanism, and here is the schedule you should therefore use.*
Roughly **nine days of compute**, with #1 already finished.

Everything else is supporting material: #4 as an appendix, #5 as a section, #7 to a workshop.

**Do not** revive #5 as a law, and stop spending compute on hopper — a single hopper cell spans
289–573, wider than any effect we have ever claimed on it.

**Correction to #9, made the same day it was written.** I justified it by saying our abstraction
attempts all targeted the wrong channel. They did not. Thread E's reframing — "SE belongs on the
graph of states, not inside the latent vector" — aimed squarely at data collection, was tested as
Thread A's A3, and failed harder than the regulariser versions (Cartpole-Sparse 316→7, on the
sparse-exploration task it was built for). "Apply the abstraction to collection instead of
representation" is not untried; it is refuted in its most natural form.

### Where that leaves abstraction learning and SE

Both doors are closed, and the campaign tried both properly:

| use of SE / learned abstraction | verdict | why, read from here |
|---|---|---|
| **latent regulariser** (Glass, 13–16 levers, SE-JEPA on Panda, Thread C variance) | **null / redundant** | applied uniformly through training; changes neither what is collected nor the early schedule — the two channels that carry the value |
| **structure discovery** (SE communities → bottleneck subgoals → exploration; Thread E/A3) | **null, and harmful where targeted** | it *did* aim at data collection, so the account does not excuse it. Continuous control lives on smooth manifolds; latent-graph communities are not the subgoals it needs |
| **hierarchy / options / H-JEPA** | null on tasks whose flat policy already worked | nothing to buy where the policy is not the bottleneck |
| **entity-graph latents** | first GO — generalises compositionally, **does not reach the controller** | a representation property, unconverted; the campaign's genuine orphan |
| **abstraction *in the loop*** (analytic controller + learned residual, TAMP) | **the one thing that worked** — TAMP 0.827 with zero training; but never beats a budget-matched PPO asymptotically | it changes what the agent *does first*, i.e. collection — and it works by **encoding** the solution, not learning an abstraction |

**The honest summary:** nine months of learned abstraction produced no method that beats a
budget-matched baseline. The one abstraction that ever earned its keep was **hand-written** — an
analytic controller with a residual on top — and its benefit was sample-efficiency where the prior
fit the task's control structure, never a higher ceiling.

That is a real finding and worth stating in the paper rather than omitting: *for these tasks, the
useful abstraction was cheaper to write than to learn.*

---

## Phase 1 — Building Glass, and the basin lottery (May 13 – Jun 9)

<div class="wide-table wide-table--phase" markdown="1">

| post | hypothesis / goal | experiment | result | lesson |
|---|---|---|---|---|
| Part 1 / 1b (05-13) | **structural entropy in the latent will improve TD-MPC2's world model** | JAX/Flax reimpl ~50× faster; Glass-JAX integrated; HopperHop | Phase 1/1b/2 run; cluster-basin failure analysed | a fast reimplementation enabled everything after |
| Iter 2–7 (05-20) | more phases / better clustering will beat vanilla | μ/S/c internals, 25 phases, video rollouts | 25 phases failed | K_UPDATE=64 named as the ignored root cause |
| Iter 8–9 (05-27) | **apply Glass early, then hand off** | MPPI-vs-policy diagnostics; stability regularisers | Glass off-at-1M beat our internal TD-MPC2 mean | — |

</div>

### Reviewed now

Two things were wrong here that took two months to surface. First, **the baseline was our own
reimplementation**, which the July audit ([Part 16](../2026-07-10-tdmpc-glass-part16-reimplementation-audit/))
showed deviates from Hansen's original — so "beats TD-MPC2" meant "beats our version of it".

Second, and more interesting: the **off-at-1M handoff** was the one Glass configuration that looked
like it worked. In August we found the world-model loss is *front-loaded* — it matters early and
decays to nothing by 250k. A schedule that applies a representation prior early and removes it later
is exactly the shape the front-loading result says should work. **Iteration 9 stumbled onto the right
schedule for the wrong reason, and we read it as a Glass result instead of a scheduling result.**
It was never followed up because the framing was "does Glass help", not "when does the model matter".

---

## Phase 2 — Sixteen nulls and the redundancy criterion (Jun 9 – Jun 17)

<div class="wide-table wide-table--phase" markdown="1">

| post | hypothesis / goal | experiment | result | lesson |
|---|---|---|---|---|
| Part 2 (06-09) | **some explicit abstraction beats vanilla under a fair protocol** | every abstraction idea, strict protocol | **8 apparent wins → null**; the one real win was prior art | a mechanism-check saved a campaign |
| Part 3 (06-10) | the latent may already *be* the abstraction | cheap value-decodability probe | value decodes linearly | — |
| Part 4 (06-11) | **abstraction is redundant iff the latent is value-sufficient** | convert 16 nulls into a criterion | criterion stated | a null is useful only when it predicts |
| Part 5 (06-11) | aim abstraction where sufficiency fails | plan for the planning axes | — | — |
| Part 6 (06-12) | **`disc_err_gap` predicts when temporal abstraction pays** | pre-registered 9-candidate screen | one survived | predictions published before results |
| Part 7 (06-12) | it transfers out-of-domain | CheetahRun test | weak positive, as committed | — |
| Part 8 (06-12) | calibration fine-tuning fixes composition failure | flip experiment + control | **headline died in 6h to its own control** | the control caught us |
| Part 9 (06-13) | it will survive from scratch | 5 seeds, 3 tasks, prediction committed | **null** | fine-tune flips ≠ from-scratch effects |
| Part 10 (06-14) | **entity-factored latents generalise compositionally** | graph vs monolithic, held-out object counts | **first GO** | ~18 nulls before one positive |
| Part 11 (06-14) | the GO reaches control | random-shooting MPC on the learned models | **no** | decodability is not control |
| Part 13 (06-17) | the R² story explains the nulls | reviewer-style challenge | **criterion falsified** | you cannot probe your way to "redundant" |

</div>

### Reviewed now

Every lever in this phase attacked the **state representation, in distribution, throughout
training**. Under the August account that is the one place a world model's value does *not* live:
the model earns its keep by changing **what data gets collected** and by shaping learning **early**.
A regulariser that re-organises the latent while leaving data collection untouched should do nothing —
and sixteen of them did nothing.

So the redundancy criterion was answering the right question with the wrong instrument. "Is the
latent value-sufficient?" is a statement about representation; the real question was "does this
change what the agent experiences?" Part 13 was right to falsify the criterion, but the replacement
was not available until we measured the training/decision split in August.

The **entity-graph GO** (Parts 10–11) remains the most interesting orphan in the campaign: a genuine
compositional-generalisation win that did not reach the controller. Under the current account that is
unsurprising — decodability is a representation property — but it was never retried where data
collection is the bottleneck.

---

## Phase 3 — Jumpy world models and reproduction (Jun 18 – Jun 20)

<div class="wide-table wide-table--phase" markdown="1">

| post | hypothesis / goal | experiment | result | lesson |
|---|---|---|---|---|
| Part 14 (06-18) | build the background to reproduce the jumpy line | primer: γ-models, TD-Flow, CompPlan | — | — |
| Part 15 (06-19) | **CompPlan reproduces on OGBench** | reproduce via the InFOM scaffold | **InFOM reproduces; CompPlan does not** | reproduce the scaffold before the claim |
| Part 16 (06-20) | our skip-TD-MPC2 vs CompPlan/GHM | two roads to a jumpy model | compared | — |
| **Part 17 (06-20)** | **the +60% jumpy win on PickCube is real** | **video-evaluate it** | **reward-hacking: 0% real pick success**; hover banked 89% of return | **watch the video** |

</div>

### Reviewed now

Part 17 is, in hindsight, the most valuable post of the first two months, and it is a *negative*
about our own flagship. The lesson generalised: **the metric was measuring proximity, and the agent
optimised proximity.** Everything downstream — the HL loop, the true-success protocol, the insistence
on per-phase pass rates — descends from that afternoon.

The jumpy line itself is dead and the current account says why: coarsening the action space removes
the fine, reactive correction that contact tasks need, and it does not improve data collection. Its
one apparent success was an artifact of the metric.

---

## Phase 4 — The coding-agent loop (Jun 20 – Jun 21)

<div class="wide-table wide-table--phase" markdown="1">

| post | hypothesis / goal | experiment | result | lesson |
|---|---|---|---|---|
| **HL loop (06-21)** | **a coding agent can solve an RL task without gradients** | Weng's *Learning Beyond Gradients*: iteratively refine a programmatic controller from telemetry, per-phase rates and video | PandaPickCube **0% → ~9% real success in hours**, where gradient RL scored 0 | a solvability oracle and reward-hacking detector |

</div>

### Reviewed now

This is the only thing in nine months that solved a task gradient RL could not, and the August
account explains why it worked where learning failed. Gradient RL was stuck because **the data it
collected never contained a success** — the hover optimum meant no successful trajectory ever
entered the buffer. A programmatic controller sidesteps the data-collection bottleneck entirely by
*encoding* the solution rather than discovering it. It then generated the demonstrations that
unblocked warm-starting.

Its ~9% ceiling is open-loop control, not a limit of the method. Under the current framing the right
role for it is exactly what it did: **an oracle that answers "is this task solvable and what is the
binding constraint?" before you spend 33M steps discovering that your reward is hackable.**

It was missed entirely by the Part 25 audit because it was filed under "HL loop" rather than
"programmatic policy" — the clearest single instance of an index failure costing a research line.

---

## Phase 5 — What TD-MPC2 actually is (Jun 22 – Jun 23)

<div class="wide-table wide-table--phase" markdown="1">

| post | hypothesis / goal | experiment | result | lesson |
|---|---|---|---|---|
| Part 18 (06-22) | **TD-MPC2 beats PPO — on which axis?** | vs brax PPO, 6 DMC tasks | PPO does up to 285× more steps in *less* wall-clock | sample-efficiency ≠ wall-clock |
| Part 19 (06-22) | the default config is over-provisioned | shrink probe | **5× smaller, 1.6× faster, 85–100% of performance** | — |
| Part 20 (06-22) | the bottleneck is MPPI | profile it | **it is the 64 gradient updates per step** | our own intuition was wrong |
| Part 21 (06-22) | it wins on its home turf | hard-exploration + high-dim | wins hard-exploration; default fails Humanoid | — |
| Part 23 (06-23) | the shrink holds suite-wide | 16 tasks | it holds | the most reusable practical result here |

</div>

### Reviewed now

This phase produced the campaign's most *usable* results and they are still under-exploited. Two
now look different in light of August:

**The 64-updates-per-step bottleneck** (Part 20) sits directly next to the front-loading result. If
the world-model loss matters early and not late, the natural question — never asked — is whether the
*update ratio* can be annealed on the same schedule. We measured that ablating the model loss at 50k
saves 16.8% wall-clock; the update-ratio lever is larger.

**The 5× shrink** (Parts 19, 23) is orthogonal to everything the campaign chased afterwards and is
the one result a practitioner could use tomorrow. It never became a paper because it is not a
mechanism claim.

---

## Phase 6 — The beat-PPO arc on Panda, and its retraction (Jun 23 – Jun 29)

<div class="wide-table wide-table--phase" markdown="1">

| post | hypothesis / goal | experiment | result | lesson |
|---|---|---|---|---|
| Part 24 (06-23) | **a value-aware abstraction in the loop beats a predictive one** | skill-options inside the HL loop | first positive abstraction result, 0.24 | — |
| Part 25 (06-23) | you can distil the abstraction *out* | warm-start / BC / DAPG | **all failed** — it is non-Markov | — |
| Part 26 (06-23) | keep it live, learn a residual | Markov-corrected residual | 0.24 → 0.48; annealing authority backfires | keep the scaffold live |
| Part 27 (06-23) | full persistent authority ties PPO | ladder | 0.78 vs 0.81, 1.7× faster to competence | — |
| Parts 28–30 (06-24) | **0.81 is beatable** | six experiments, long PPO, learned grasp | **0.83 is a kinematic ceiling** (far-reach ≈99.9% IK-infeasible) | name the physical wall |
| Part 31 (06-24) | it generalises to a 2nd task | same method, PickCubeOrientation | **loses**, 0.33 vs 0.82 | the prior did not transfer |
| Parts 33–34 (06-24) | parametrising the prior closes the gap | orientation-aware prior | 0.33 → 0.67, still short | — |
| Parts 39–43 (06-25/26) | a learned/TAMP abstraction beats PPO on OpenCabinet | ladder from hand-coded to learned | **TAMP 0.827 with zero training** vs PPO peak 0.832 | a scripted controller can match a trained policy |
| **Part 49 (06-29)** | **audit: was the PPO baseline fair?** | matched-budget control | **no — the arc was vs an under-budgeted PPO** | read the baseline's budget off disk |

</div>

### Reviewed now

The retraction (Part 49) is the headline, but the phase's durable content is elsewhere and it points
back to Phase 4. **TAMP at 0.827 with zero training** and **the HL controller at 9% where RL scored
0** are the same finding twice: on manipulation, encoding the solution beats discovering it, because
discovery is bottlenecked on data the agent cannot collect.

The systematic sweep that followed (locomotion CPG, reaching OSC, sparse swing-ups) found the
analytic prior **never** beats a budget-matched PPO asymptotically — its value is sample-efficiency,
and only where the prior captures the control-relevant substructure. That is consistent with the
August account: a prior changes *what you try first*, i.e. data collection, which buys speed and not
a higher ceiling.

Parts 28–30 also produced the cleanest non-ML result in the campaign: **0.83 is kinematic**, not
algorithmic. Recognising a physical ceiling stopped a month of optimisation.

---

## Phase 7 — JEPA, five bets, and the positive chase (Jun 30 – Jul 8)

<div class="wide-table wide-table--phase" markdown="1">

| post | hypothesis / goal | experiment | result | lesson |
|---|---|---|---|---|
| Parts 50–52 (06-30) | **H-JEPA can solve a robot task** | from 0%, diagnose, climb | 0 → 0.367 → **0.72** with a learned residual | the contact wall is learnable; matched PPO still wins the asymptote |
| Thread A (07-01) | **planning is a directed-exploration operator** (flagship) | coverage, novelty-seeking MPPI, controlled plan-vs-policy | **does not survive, three ways** | the flagship died to its own control |
| Thread B (07-01) | prior-fit × exploration-difficulty taxonomy | escape-difficulty frontier | a fitting prior is a speed lever | — |
| Thread C (07-01) | SE reduces seed variance | variance analysis | **null** | — |
| Thread D (07-01) | **anti-collapse is a JEPA property, test it on a JEPA** | pure self-predictive JEPA on DMC | **reversal** — it does not collapse; anti-collapse is neutral-to-harmful | test the property on the architecture that has it |
| Parts 6–9 (07-02→04) | five bets to finally beat PPO | run, then re-run harder | mostly null; every headline rewritten by its own control | — |

</div>

### Reviewed now

Thread A is the instructive failure. "Planning is a directed-exploration operator" was **the right
intuition aimed at the wrong channel.** It framed planning as a way of *acting* — explore better,
cover more states. August's answer is that planning's value is that it changes *what enters the
buffer and therefore what the model and critic learn*, which is adjacent to exploration but not the
same claim, and is measured differently. The controlled plan-vs-policy-only test that killed Thread A
is the direct ancestor of the frozen-checkpoint toggle that established the August result.

Thread D is the phase's most honest moment: we had been treating anti-collapse as a live threat, and
testing it on the architecture that actually has the property showed it was not one.

---

## Phase 8 — The hopper dissection and the audits (Jul 8 – Jul 23)

<div class="wide-table wide-table--phase" markdown="1">

| post | hypothesis / goal | experiment | result | lesson |
|---|---|---|---|---|
| Part 12 (07-08) | **hopper's win comes from the world model** | ablate the consistency loss there | **no** — it is the off-policy TD pathway; the loss is removable (n=8) | — |
| **Part 14 (07-09)** | **the PPO wall on hopper is a conjunctive-reward artifact** | env-gate the reward, re-run tuned PPO | product **0**, additive **135**, easier hop threshold **1** | grounded in Voelcker et al. 2024, *Can we hop in general?* |
| **Part 16 (07-10)** | **our reimplementation is faithful** | line-by-line audit vs Hansen's release | **it deviates** | the whole program ran on an unvalidated base |
| **Part 21 (07-22)** | the AAAI headline is sound | audit six days before the deadline | **retracted our own headline** | audit before submission |
| Part 22 (07-23) | make the vocabulary legible | explain cells, bimodality, permutation p | — | — |

</div>

### Reviewed now

Part 14 is the single most under-used post in the corpus. It **already** identified hopper's
conjunctive reward as the cause of the PPO wall, **already** built the env-gated additive-reward knob,
and **already** named the margin confound and asked for the margin-controlled variant. In August we
rebuilt all three from scratch believing them new. That is the concrete cost of an unindexed corpus,
and it is why Parts 27 and 29 exist.

Part 14's H4 — *hopper is exploration-hard but execution-simple* — also turned out to be the most
durable idea in the campaign, though not where it was aimed. Tested directly on hopper it is null
(the variance is too large to resolve anything). Tested on **walker** the predicted shape appears
cleanly: the model's contribution decays with training step, −18.4 / −12.8 / −7.2, p=0.008 / 0.016.

The two audits (Parts 16, 21) both found real defects in our own work. The pattern that repeats
through this campaign is that **our controls and audits caught almost everything; our claims usually
preceded them by a few days.**

---

## Phase 9 — Verification and the planner (Aug 3 – Aug 8)

<div class="wide-table wide-table--phase" markdown="1">

| post | hypothesis / goal | experiment | result | lesson |
|---|---|---|---|---|
| Part 23 (08-03) | the AAAI result survives on the **official** TD-MPC2 | rebuild the diagnostic | the ~9.2× cheetah gap is **1.04×** ordinary, **3.55×** matched-data | two config lines were worth 1.9× |
| Part 24 (08-05) | **does the planner earn its keep?** | planner ablation, 6 tasks | free on cup/finger/cheetah, **fatal on hopper (3.53×)** | — |
| Part 25 (08-05) | **reward conjunctivity sets the value of search** | nine-month audit + nomination | the audit missed a whole research line | an audit from notes inherits the notes' gaps |
| Part 26 (08-06) | conjunctivity survives its own control | difficulty control (bar 0.60→0.35) | **3.53× → 1.21×** — killed as a law | the headroom objection must be answered |
| [Part 27](../2026-08-06-tdmpc-glass-part27-the-full-inventory-and-the-proposals-that-survive/) | rebuild the inventory from primary sources | re-read all 87 posts | found the HL loop and the July precedent | — |
| [Part 28](../2026-08-07-tdmpc-glass-part28-the-iclr-decision-document/) | which candidate is the ICLR paper? | evidence-grade every candidate | #1 + #5 together | **read this for the meeting** |

</div>

### Reviewed now

This phase finally asked the question the previous eight had been circling: **not whether the world
model helps, but through which channel.** The frozen-checkpoint toggle — train without the planner,
then switch it on with the policy held identical — is the design that made the answer unambiguous,
and it is a direct descendant of Thread A's plan-vs-policy control.

The conjunctivity hypothesis died as a general law but left a real result on walker
(1.84×, p=0.008) and, more importantly, forced the ceiling/headroom objection into the open. Two of
this phase's corrections were mine: an in-sample oracle frontier (0.91 → 0.53 held out) and an
over-correction of it from a single checkpoint.

### The August results

| finding | evidence | grade |
|---|---|---|
| **Search is a training effect** — 364.2 trained-with vs 87.5 planner-added-later (24% recovery); net loss on walker | frozen-checkpoint toggle, policy identical across arms | **A** |
| **The model's contribution is front-loaded** — ablate at 50k/150k/250k: −18.4 (p=0.008) / −12.8 (p=0.016) / −7.2 | walker n=5, monotone, *tightened* from n=3 | **A** |
| …and **Pareto-positive**: ablating at 250k costs 7.2 return (n.s.) and saves **7.2%** wall-clock; at 50k, **16.8%** | walker n=5 | **A** |
| **DreamerV3 replicates with no search at all** — hopper 2.40× (p=0.009), walker null | n=6 / n=13 | **A** |
| **Reward structure sets search's value on walker** — conjunctive 1.84×/1.91× med (p=0.008); merely harder 1.07× | n=5 / n=4 | **A** (walker only) |
| **Adaptive planning**: top 10% of states carry 53% of capturable value (held out), but **nine signals, three families, all Spearman ≈ 0** | frozen checkpoints, n=6 | **B / A** |
| H4 on hopper | +165.9 at n=3 → **+46.5, p=0.72** at n=5 | **D** |

---

## What the whole campaign looks like from here

Nine months, roughly 90 posts, and the shape is: **we spent Phases 1–3 and 7 asking whether a better
*representation* helps, when the answer was about *data and schedule*.** The nulls were not bad luck.
A latent regulariser applied uniformly through training cannot change what the agent collects, and
the August measurements say that is where the value is.

Three things survived, and they are consistent with each other:
1. the model's value is in **training**, not decisions (frozen-checkpoint toggle);
2. that value is **front-loaded** and can be dropped late for free, saving wall-clock (ablation timing);
3. where the policy can't collect a success at all, **encoding** the solution beats discovering it
   (HL loop, TAMP).

And the methodological through-line: **almost every claim in this campaign was corrected by a control
we wrote ourselves, usually within a week of making it.** That is the process working — but it also
means the claims consistently preceded their controls, which is the habit worth changing.

---

*Index: [suuttt.github.io/tdmpc-glass](https://suuttt.github.io/tdmpc-glass/). Numbers trace to logged
JSON/CSV in `SuuTTT/world-model-paper` under `data/`, with per-seed values; retractions live in git
history rather than being edited away. Known exception: Part 14's PPO reward-gate figures survive only
as prose — the scripts exist, the results JSON does not, so they must be re-run before citation.*
