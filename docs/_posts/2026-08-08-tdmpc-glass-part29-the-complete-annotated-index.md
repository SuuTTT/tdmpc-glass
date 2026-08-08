---
layout: post
title: "TD-MPC-Glass, Part 29: The Complete Annotated Index — Every Post, Its Experiment, Its Result, and What It Taught"
date: 2026-08-08
description: "Every post in this campaign, in order, with what was run, what came out, and the lesson that survived. Nine phases from the first structural-entropy build in May to the confirmed front-loading result in August, including the reward-hacking discovery, the beat-PPO arc and its retraction, the five-bet reality check, the reimplementation audit, and the planner experiments. Written as the navigation layer for the whole corpus: read this to find which post you need."
---

> **What this is.** Ninety-odd posts is too many to search. This is the index: every post, one row,
> with its experiment, its result, and the lesson. Where a result was later retracted, the row says
> so and points at the retraction. Where the data no longer exists, the row says that too.
>
> **If you only want the decision:** read [Part 28](../2026-08-07-tdmpc-glass-part28-the-iclr-decision-document/).
> **If you want the inventory with data availability:** [Part 27](../2026-08-06-tdmpc-glass-part27-the-full-inventory-and-the-proposals-that-survive/).
> This post is the map to everything else.

---

## Phase 1 — Building Glass, and the basin lottery (May 13 – Jun 9)

| post | experiment | result | lesson |
|---|---|---|---|
| Part 1 / Phase 1b (05-13) | JAX/Flax TD-MPC2 reimplementation ~50× faster; Glass structural-entropy integrated; HopperHop | Phase 1/1b/2 run; cluster-basin failure analysed | a fast reimplementation is the enabler for everything after — and later turned out to deviate from canonical (see Part 16, Jul 10) |
| Iterations 2–7 (05-20) | what μ/S/c actually do; 25 phases; video rollouts | 25 phases failed to beat vanilla | K_UPDATE=64 flagged as the root cause ignored for two weeks |
| Iterations 8–9 (05-27) | MPPI-vs-policy diagnostics; stability regularisers; off-schedule handoff | Glass off-at-1M beat internal TD-MPC2 mean on HopperHop | the "win" was against our own baseline, not the official one — the seed of the audit two months later |

## Phase 2 — Sixteen nulls and the redundancy criterion (Jun 9 – Jun 17)

| post | experiment | result | lesson |
|---|---|---|---|
| Part 2 (06-09) | every abstraction idea under a strict fair protocol | **eight apparent wins dissolved to null**; one real win (jumpy k-step) was prior art | a mechanism-check saved a campaign |
| Part 3 (06-10) | cheap probe: is the latent already value-sufficient? | value decodes linearly from the trained latent | the latent was already the abstraction |
| Part 4 (06-11) | convert 16 nulls into a falsifiable criterion | redundancy criterion stated | a null becomes useful only when it predicts |
| Part 5 (06-11) | aim abstraction where sufficiency fails → the planning axes | plan set | — |
| Part 6 (06-12) | **pre-registered** 9-candidate screen | `disc_err_gap` survived | predictions published *before* results |
| Part 7 (06-12) | out-of-domain test on CheetahRun | weak positive, as committed | a weak prediction, correctly weakly confirmed |
| Part 8 (06-12) | calibration fine-tuning flip | headline died in six hours to its own control | **the control caught us** |
| Part 9 (06-13) | fair from-scratch calibration test, 5 seeds | closed, null | fine-tuning flips are not from-scratch effects |
| Part 10 (06-14) | entity-graph latent, compositional OOD | **first GO** — value-decodability generalises to held-out object counts | ~18 nulls before one positive |
| Part 11 (06-14) | does the graph win reach the controller? | **no** | decodability is not control |
| Part 12 (06-16) | jumpy vs vanilla, 6 tasks × 5 seeds × 500k | genuine gains on contact manipulation | later revealed as reward-hacking (Part 17) |
| Part 13 / postmortem (06-17) | reviewer-style challenge to the R² story | **criterion falsified** | you cannot probe your way to "redundant" |
| Campaign review (06-17) | consolidated | no explicit abstraction beats TD-MPC2 | — |

## Phase 3 — Jumpy world models and reproduction (Jun 18 – Jun 20)

| post | experiment | result | lesson |
|---|---|---|---|
| Part 14 (06-18) | from-scratch primer: γ-models, TD-Flow, CompPlan | — | the explainer that made the next reproduction possible |
| Part 15 (06-19) | reproduce CompPlan on OGBench via InFOM | **InFOM reproduces; CompPlan does not** | reproduce the scaffold before the claim |
| Part 16 (06-20) | our skip-TD-MPC2 vs CompPlan/GHM on cube | two roads compared | — |
| **Part 17 (06-20)** | **video-evaluate the +60% jumpy win** | **it was reward-hacking: 0% real pick success**, hover banked 89% of return | **watch the video; the metric was lying** |

## Phase 4 — The coding-agent loop (Jun 20 – Jun 21)

| post | experiment | result | lesson |
|---|---|---|---|
| **HL loop (06-21)** | Weng's *Learning Beyond Gradients*: a coding agent iteratively writing a programmatic controller from telemetry and video | PandaPickCube **0% → ~9% real success in hours**, where gradient RL scored 0 | a solvability oracle and reward-hacking detector; ceiling ~9% open-loop vs PPO's 66–83% closed-loop |

*Missed entirely by the Part 25 audit because it was filed under "HL loop". Now in
`competition-playbook/techniques/programmatic-policy-agents.md` with our numbers.*

## Phase 5 — What TD-MPC2 actually is (Jun 22 – Jun 23)

| post | experiment | result | lesson |
|---|---|---|---|
| Part 18 (06-22) | TD-MPC2 vs brax PPO on 6 DMC tasks | PPO does up to 285× more steps in *less* wall-clock | sample-efficiency ≠ wall-clock |
| Part 19 (06-22) | shrink probe | **5× smaller (3.6M→0.72M), ~1.6× faster, 85–100% of performance** | the default is over-provisioned |
| Part 20 (06-22) | profile the bottleneck | it is the **64 gradient updates per step**, not MPPI | our own intuition was wrong |
| Part 21 (06-22) | hard-exploration + high-dim home turf | wins hard-exploration; default config fails Humanoid; jumpy harmful | — |
| Part 22 (06-22) | budgets synthesis | — | report steps **and** wall-clock |
| Part 23 (06-23) | shrink across 16 tasks | holds suite-wide | the campaign's most reusable practical result |

## Phase 6 — The beat-PPO arc on Panda, and its retraction (Jun 23 – Jun 29)

| post | experiment | result | lesson |
|---|---|---|---|
| Part 24 (06-23) | value-aware skill-options in the HL loop | first positive abstraction result, 0.24 | — |
| Part 25 (06-23) | warm-start from the abstraction | **no** — the abstraction is non-Markov; BC/DAPG all failed | you cannot distil it *out* |
| Part 26 (06-23) | keep it in the loop, learn a residual | 0.24 → 0.48; annealing authority backfires | keep the scaffold live |
| Part 27 (06-23) | full persistent authority | 0.78 vs PPO 0.81, ~1.7× faster to competence | — |
| Parts 28–30 (06-24) | five levers toward 100% | **0.83 is a kinematic ceiling** — far-reach configs are ~99.9% IK-infeasible | name the physical wall, stop optimising |
| Part 31 (06-24) | second Panda task | abstraction **loses** (0.33 vs 0.82) | the prior was fixed top-down; it did not transfer |
| Part 32 (06-24) | why hopper ≫ pick | exploration in learning vs execution | the seed of H4 |
| Parts 33–34 (06-24) | parametrise the prior | 0.33 → ~0.67, still short | fuller parametrisation does not close it |
| Part 35 (06-24) | what abstraction is *for* | interpretable, reusable, stable — not a higher ceiling | — |
| Parts 36–37 (06-25) | InFOM on OGBench; constrain the workspace | reproduces on cube-single; workspace constraint lifts PickCube to 1.0 | confirms the ceiling is orientation |
| Part 38 (06-25) | **planning advantage across 10 tasks** | ranges from ~0 to +39% | the measurement the whole August campaign is built on |
| Parts 39–43 (06-25/26) | OpenCabinet ladder; abstraction-as-curriculum; TAMP+residual | TAMP controller **0.827 with zero training** vs PPO peak 0.832 | a scripted controller can match a trained policy |
| Part 44 (06-25) | fair dual-protocol table | every cell, both protocols | — |
| Part 47 (06-25) | analytic prior on AcrobotSwingup | **backfires** | a prior that does not fit is dead weight |
| Part 48 (06-26) | full DMC benchmark, 5 methods × 16 tasks | unified table | — |
| **Part 49 (06-29)** | **matched-control audit of Parts 42–48** | the "beats PPO" arc was measured against an **under-budgeted PPO** | **read your baseline's step budget off disk** |

## Phase 7 — JEPA, five bets, and the positive chase (Jun 30 – Jul 8)

| post | experiment | result | lesson |
|---|---|---|---|
| Parts 50–52 (06-30) | H-JEPA on Panda from 0% | 0 → 0.289 → 0.367 (better grasp) → 0.72 with a learned residual | the contact wall is learnable, but matched-budget PPO still wins the asymptote |
| Thread A (07-01) | planning-as-exploration, the flagship bet | **does not survive** a controlled plan-vs-policy test, three ways | the flagship died to its own control |
| Thread B (07-01) | behavioural-prior taxonomy | prior-fit × exploration-difficulty | a fitting prior is a speed lever, not a ceiling lever |
| Thread C (07-01) | abstraction as variance reduction | **null** | — |
| Thread D (07-01) | JEPA anti-collapse done right | **reversal** — a pure JEPA does *not* collapse on DMC; anti-collapse ranges neutral to harmful | test the property on the architecture that has it |
| Thread E (07-01) | SE for structure discovery, not regularisation | repositioned | — |
| Parts 6–9 (07-02→07-04) | five bets, resolved, then re-run harder | mostly null; every headline rewritten by its own control | — |
| Parts 10–11 (07-08) | week review; ~80 iterations in seven phases | — | the retrospective index this post supersedes |

## Phase 8 — The hopper dissection and the audits (Jul 8 – Jul 23)

| post | experiment | result | lesson |
|---|---|---|---|
| Part 12 (07-08) | does hopper's win come from the world model? | **no** — it is the off-policy TD pathway; the consistency loss is removable there (n=8) | — |
| **Part 14 (07-09)** | **env-gate hopper's reward, re-run tuned PPO** | product **0**, additive **135**, product with easier hop threshold **1** | **the PPO wall is a conjunctive-reward artifact** — grounded in Voelcker et al. 2024, *Can we hop in general?* |
| Part 15 (07-09) | revision plan | — | — |
| **Part 16 (07-10)** | **audit the reimplementation against Hansen's original** | it deviated | our entire program had run on an unvalidated base |
| Part 17 (07-10) | the handbook | — | the from-scratch reference for the stack |
| Part 18 (07-12) | lab notebook of every open track | verified numbers per track | — |
| Parts 19–20 (07-15, 07-21) | week reviews; four named hypotheses | AAAI-27 diagnostic paper assembled | — |
| **Part 21 (07-22)** | **audit six days before the deadline** | **retracted our own headline**; what an ablation actually ablates | audit before submission, not after |
| Part 22 (07-23) | vocabulary explained from scratch | — | cells, bimodality, permutation p, design floors |

## Phase 9 — Verification and the planner (Aug 3 – Aug 8)

| post | experiment | result | lesson |
|---|---|---|---|
| Part 23 (08-03) | rebuild the diagnostic on **official** TD-MPC2 | the AAAI ~9.2× cheetah gap is **1.04×** under ordinary training, **3.55×** under its matched-data control | two config lines were worth 1.9× |
| Part 24 (08-05) | planner ablation, self-contained account | free on cup/finger/cheetah, **fatal on hopper (3.53×)** | — |
| Part 25 (08-05) | nine-month audit | nominated reward conjunctivity as the ICLR bet | **the audit was built from notes and missed a whole research line** |
| Part 26 (08-06) | the difficulty control fires | conjunctivity killed as a law; the ceiling objection taken seriously | headroom is the objection every "improvement" claim must answer |
| [Part 27](../2026-08-06-tdmpc-glass-part27-the-full-inventory-and-the-proposals-that-survive/) (08-06) | rebuild the inventory from **all 87 posts** | found the HL loop and the July conjunctivity precedent that Part 25 missed | an inventory built from summaries inherits every gap in the summaries |
| [Part 28](../2026-08-07-tdmpc-glass-part28-the-iclr-decision-document/) (08-07) | the decision document | five candidates, evidence-graded | **read this one for the meeting** |

### The August results in one table

| finding | evidence | grade |
|---|---|---|
| **Search is a training effect** — trained-with-planner 364.2 vs planner-added-at-deployment 87.5 (24% recovery); net loss on walker | frozen-checkpoint toggle; policy identical across arms | **A** |
| **The model's contribution is front-loaded** — ablate the world-model loss at 50k/150k/250k: −18.4 (p=0.008) / −12.8 (p=0.016) / −7.2 | walker n=5, monotone, tightened from n=3 | **A** |
| …and it is **Pareto-positive**: ablating at 250k costs 7.2 return (n.s.) and saves **7.2% wall-clock**; at 50k saves 16.8% | walker n=5 | **A** |
| **DreamerV3 replicates the task pattern with no search at all** — hopper 2.40× (p=0.009), walker null | n=6 / n=13 | **A** |
| **Reward structure sets search's value on walker** — conjunctive 1.84×/1.91× median (p=0.008); merely harder 1.07× | n=5 / n=4 | **A** (walker only) |
| **Adaptive planning**: top 10% of states carry 53% of capturable value (held out, n=6), but **nine signals across three families all score Spearman ≈ 0** | frozen checkpoints | **B / A** |
| H4 on hopper | +165.9 at n=3 → **+46.5, p=0.72** at n=5 | **D** |

---

## The lessons, collected

1. **Watch the video.** The +60% jumpy win was a hover with 0% real success (Part 17).
2. **Read your baseline's budget off disk.** The whole beat-PPO arc was against an under-budgeted PPO (Part 49).
3. **Audit your own instrument.** Our reimplementation deviated from canonical and everything ran on it for two months (Part 16).
4. **The control is the experiment.** Parts 8, 26 and the H4 null all died to controls we wrote ourselves.
5. **n=3 is a pilot.** Eight n=3 patterns; seven died at n=5. The one that survived *tightened*.
6. **Choosing the biggest effect can select the worst instrument.** We made hopper primary because its planning gain is largest (3.53×); one hopper cell spans 289–573 while all 20 walker runs span 5%. Most retracted patterns are hopper results.
7. **Never select and evaluate on the same draw.** My own oracle frontier read 0.91 in-sample, 0.53 held out.
8. **An inventory built from notes inherits the notes' gaps** (Part 25 → Part 27).
9. **Verify the mechanism fires before spending the experiment on it.** A budget cap that never enforced; a patch missing `import os`; a `torch.compile` graph freezing the planner's RNG so two "independent" draws were byte-identical.

---

*Index: [suuttt.github.io/tdmpc-glass](https://suuttt.github.io/tdmpc-glass/). Numbers trace to
logged JSON/CSV in `SuuTTT/world-model-paper` under `data/`, with per-seed values; retractions are in
git history rather than edited away. Part 14's PPO reward-gate figures are the known exception —
scripts survive, results do not, so they must be re-run before citation.*
