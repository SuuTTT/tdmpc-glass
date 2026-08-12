---
layout: post
title: "TD-MPC-Glass, Part 27: The Full Inventory — Every Experiment, Whether Its Data Still Exists, and the Proposal List That Survives"
date: 2026-08-06
description: "An audit of our own audit. Part 25 claimed to review every direction the campaign has run; a re-read of all 87 posts found it had missed several, two of them serious — the coding-agent/HL-loop programmatic-policy line that solved a task gradient RL reward-hacked to zero, and a July conjunctive-reward experiment that pre-dated (and partly pre-empted) this month's flagship hypothesis. The post opens with the full inventory: every experiment, the hypothesis it was testing, and what came back. Data availability -- which results still exist on disk versus only in a blog paragraph -- follows, along with the corrected proposal list."
---

> Part 25 was billed as "a full audit of every direction this campaign has run." It wasn't. Two
> misses were caught by our supervisor, not by us, and re-reading all 87 posts found more. This post
> is the corrected inventory, and it pairs every direction with **the hypothesis it was actually
> testing** — so a reader can see what we believed going in and what came back, side by side.
>
> **The inventory itself is §1**, immediately below — every direction links to the post that ran it.
> Everything after is provenance: how the list was rebuilt, what the earlier audit lost, where the
> data still lives, and the proposal list that survives. **§7 ranks what is left by its odds of
> acceptance if we spend the remaining time on it.**

---

## 1. Full inventory: every direction, what it proposed, and what happened

This is the core of the post. Each row is a direction the campaign ran, **the hypothesis it was
actually testing**, and what came back. Data availability — which results still exist on disk versus
only in a blog paragraph — is in §5.

<div class="wide-table wide-table--inventory" markdown="1">

### Directions with positive results

| # | direction | the proposal — what we believed going in | what happened |
|---|---|---|---|
| 1 | **[HL loop / coding-agent programmatic policy](https://suuttt.github.io/tdmpc-glass/2026/06/21/heuristic-learning-loop-for-rl.html)** (Parts 17, 21, 24) | A coding agent writing an *explicit program* can solve a task gradient RL cannot, because it can state structure RL would have to discover — and it tells you whether the task is solvable at all | PickCube **0% → ~9% real success in hours**, on a task where gradient RL reward-hacked to literally 0. The only thing in nine months that solved a task RL could not |
| 2 | **[TAMP controller, PandaOpenCabinet](https://suuttt.github.io/tdmpc-glass/2026/06/26/tdmpc-glass-part43-residual-on-tamp-beats-ppo.html)** (Part 43) | With the right task-and-motion abstraction, a *planner needs no training at all* to match RL on manipulation | **0.827 success with zero training** vs PPO's 0.832 peak; reach rate 0.05 → 0.841 |
| 3 | **[Shrink-Pareto](https://suuttt.github.io/tdmpc-glass/2026/06/23/tdmpc-glass-part23-shrink-pareto-full-suite.html)** (Parts 19, 23) | TD-MPC2's default capacity is heavily over-provisioned and most of it can be removed for free | **5× smaller** (3.6M → 0.72M params), ~1.6× faster, **83–100% of return across 16 DMC tasks** |
| 4 | **[Throughput profiling](https://suuttt.github.io/tdmpc-glass/2026/06/22/tdmpc-glass-part20-how-fast-can-tdmpc2-go.html)** (Part 20) | MPPI's ~9k rollouts per step dominate wall-clock, so cheaper planning is the way to speed up | **Wrong, and usefully so:** the bottleneck is the **64 gradient updates per env step**. Our own intuition was inverted |
| 5 | **[Abstraction-as-curriculum](https://suuttt.github.io/tdmpc-glass/2026/06/25/tdmpc-glass-part41-abstraction-as-curriculum.html)** (Parts 41–42) | Bootstrap with a structured skill then *release* it: keep the sample-efficiency without inheriting the prior's ceiling | Matched PPO on OpenCabinet at ~2.5× sample-efficiency — **but see Part 49**: that PPO baseline was under-budgeted |
| 6 | **[Entity-graph compositional OOD](https://suuttt.github.io/tdmpc-glass/2026/06/14/tdmpc-glass-part10-first-go.html)** (Parts 10–11) | A graph-structured latent factorises entities, so it should generalise to object counts it never saw | **First GO of the campaign**: value-decodability does generalise to held-out object counts — but the advantage **never reaches the controller** |
| 7 | **[InFOM reproduction](https://suuttt.github.io/tdmpc-glass/2026/06/25/tdmpc-glass-part36-infom-reproduction.html)** (Parts 15, 36) | Intention-conditioned flow occupancy models reproduce as published and transfer to new domains | Reproduces on cube-single (2/3 seeds, 0.96/0.98); **CompPlan does not reproduce**; AntMaze is outside its benchmark |
| 8 | **[Search is a training effect](https://suuttt.github.io/tdmpc-glass/2026/08/06/tdmpc-glass-part26-the-hypothesis-died-and-what-replaced-it.html)** (Part 26) | Planning's value lies in what it makes the agent *collect and learn during training*, not in what it computes at decision time | Frozen-checkpoint toggle: deployment-time planning recovers only **24%** on hopper and is **net-negative** on walker (0.960× over 8 checkpoints) |
| 9 | **[Conjunctive reward ⇒ PPO wall](https://suuttt.github.io/tdmpc-glass/2026/07/09/tdmpc-glass-part14-hopper-uniqueness.html)** (Part 14) | Hopper is hard for PPO because its reward is a *product* (standing × hopping) — a gate paying nothing until both terms are satisfied | Additive **135** vs product **0**; lowering the hop threshold does not rescue it. The conjunction, not the difficulty, is the wall |
| 10 | **[Walker conjunctivity dissociation](https://suuttt.github.io/tdmpc-glass/2026/08/07/tdmpc-glass-part28-the-iclr-decision-document.html)** | *Conjunctivity*, not difficulty, sets how much planning is worth | At matched baseline: harder-without-conjunctivity **1.07×** vs conjunctive **1.76×**. Holds on walker — and hopper says the opposite |

### Directions that closed null — and are informative

| # | direction | the proposal — what we believed going in | what happened |
|---|---|---|---|
| 11 | **[Structural entropy / Glass as latent regulariser](https://suuttt.github.io/tdmpc-glass/2026/06/10/tdmpc-glass-part3-the-latent-was-already-the-abstraction.html)** (Parts 1–3, Thread E) | Imposing minimal-structural-entropy community structure on the latent yields a better world model — the founding bet of this campaign | **13–16 levers, all null.** SE is redundant wherever the latent is already value-sufficient. Repositioned toward structure *discovery*; never fully tested |
| 12 | **[Jumpy / temporal abstraction](https://suuttt.github.io/tdmpc-glass/2026/06/20/tdmpc-glass-part17-reward-hacking-pickcube.html)** (Parts 12, 16, 17) | A k-step "jumpy" world model beats a 1-step one on long-horizon manipulation | The flagship "+60% on PickCube" was **reward-hacking** — 0% real success. Actively harmful on high-DoF |
| 13 | **[Planning-as-exploration](https://suuttt.github.io/tdmpc-glass/2026/07/01/tdmpc-glass-thread-a-planning-exploration.html)** (Thread A, Parts 6–7) | Planning helps because it *explores* better than the policy alone | **Does not survive** a controlled plan-vs-policy-only test, three separate ways |
| 14 | **[Abstraction as variance reduction](https://suuttt.github.io/tdmpc-glass/2026/07/01/tdmpc-glass-thread-c-abstraction-variance-reduction.html)** (Thread C) | Abstraction earns its keep by reducing variance in returns and gradients | NULL |
| 15 | **[JEPA anti-collapse](https://suuttt.github.io/tdmpc-glass/2026/07/01/tdmpc-glass-thread-d-jepa-anticollapse-done-right.html)** (Thread D, Parts 50–52) | A pure self-predictive JEPA collapses without an explicit anti-collapse term, so that term is load-bearing | **Reversal.** A pure JEPA does *not* collapse on DMC (state or pixels); anti-collapse ranges neutral to harmful; the **BYOL predictor+EMA asymmetry** is what actually carries it |
| 16 | **[H-JEPA on Panda](https://suuttt.github.io/tdmpc-glass/2026/06/30/tdmpc-glass-part50-making-hjepa-work-on-panda.html)** (Parts 50–52) | A hierarchical JEPA planning in a learned latent can solve PandaPickCube | 0 → 0.289 → **0.367** via a better grasp; a learned residual breaks the contact wall to **0.72** — but matched-budget PPO still wins the asymptote (0.81) |
| 17 | **[Calibration-shaped world models](https://suuttt.github.io/tdmpc-glass/2026/06/13/tdmpc-glass-part9-calibration-closed.html)** (Parts 8–9) | Shaping the world model for *calibration* improves downstream control | Our own control caught the headline: the fine-tuning flip **died under a fair from-scratch test** |
| 18 | **[R² / value-sufficiency redundancy criterion](https://suuttt.github.io/tdmpc-glass/2026/06/17/tdmpc-glass-part13-r2-criterion-fails.html)** (Part 13) | You can decide whether an abstraction is redundant by probing how well the latent decodes value | **Falsified** — you cannot probe your way to "redundant" |
| 19 | **[Analytic prior + residual, systematically](https://suuttt.github.io/tdmpc-glass/2026/06/24/tdmpc-glass-part34-the-analytic-prior-ceiling.html)** | An analytic prior plus a learned residual beats PPO's asymptote | Never beats budget-matched PPO on asymptote **anywhere tested**. The value is sample-efficiency, and only where the prior fits (actuated DOF == goal DOF) |
| 20 | **[Analytic prior on AcrobotSwingup](https://suuttt.github.io/tdmpc-glass/2026/06/25/tdmpc-glass-part47-acrobot-prior-backfires.html)** (Part 47) | The analytic-prior recipe generalises to swing-up | **Backfires** |

### Integrity events (keep these visible)

| # | event | what we believed | what was actually true |
|---|---|---|---|
| 21 | **[Part 49](https://suuttt.github.io/tdmpc-glass/2026/06/29/tdmpc-glass-part49-matched-control-correction.html)** | Our abstraction pipeline beats PPO — the whole Parts 42–48 arc | The PPO baseline was **under-budgeted**. The arc was retracted |
| 22 | **[Part 21](https://suuttt.github.io/tdmpc-glass/2026/07/22/tdmpc-glass-part21-audit-and-retraction.html)** | "World models can hurt" on hopper — our headline finding | Our JAX TD-MPC2 had **deviated from the official release**. Headline retracted |
| 23 | **[Part 23](https://suuttt.github.io/tdmpc-glass/2026/08/03/tdmpc-glass-part23-verification-week.html)** | The AAAI paper's ~**9.2×** cheetah gap | **1.04×** under ordinary training; 3.55× only under its matched-data control |
| 24 | **The budget guard** | The $40 cap had been enforcing itself all campaign | It announced "STOPPING both boxes" and **stopped nothing** — cron's PATH lacked `vastai` and the errors were swallowed |

</div>

## 2. Why this post exists

Part 25's proposal list was assembled from memory notes and recent reports rather than from the blog
corpus. That is how it lost an entire research line. Two specific failures:

**Miss 1 — the coding-agent / HL-loop line.** The campaign ran Jiayi Weng's *Learning Beyond
Gradients* method on Panda in June: a coding agent iteratively writing a programmatic policy from
telemetry and video, no gradients. It produced one of the campaign's most striking results. It
appears nowhere in Part 25's list of directions. The work was filed under "HL loop", so a search for
"programmatic policy" found nothing.

**Miss 2 — the conjunctive-reward experiment already existed.** Part 25 nominated "reward
conjunctivity sets the value of search" as the ICLR bet and described it as new. **Part 14, published
2026-07-09, had already identified hopper's conjunctive reward as the cause of the PPO wall and run a
controlled experiment on it** — including building the exact env-gated additive-reward knob we
rebuilt from scratch this week, and naming the margin confound we then "discovered" independently.

Both misses share a cause: **the inventory was built from summaries instead of from primary
sources.** That is the same failure mode as reading a monitor's log line instead of checking whether
the action happened.

---

## 3. The thing Part 25 missed most: the coding agent solved a task gradient RL could not

**PandaPickCube.** Gradient RL *reward-hacked it to zero.* Video evaluation showed vanilla and jumpy
TD-MPC2 both achieving **0% real pick success** while scoring well — they hovered the gripper beside
the cube to bank a dense proximity term. `box_target_max` was exactly `0.0000`; vanilla plateaued at
~2,500 against a ~12,550 ceiling, abandoning ~10,000 of return **by never picking**. The reward was
well-*ordered* (real success 965 vs hover 315 per 150-step episode, 3.1×) but badly *shaped*, and
89% of the hover's return came from the proximity term.

A coding agent looping on a programmatic phase machine (reach → descend → grasp → lift → place),
improved from telemetry, per-phase pass rates and rendered video, took the same task from
**0% → ~6% → ~9% video-verified real success** (99% grasp, lift 0.69, 256 envs × multiple seeds) in
**hours**, against PPO's 33M-step grind. The binding constraint was **cube orientation** — `rot_err`
capping `box_target` even on good grasps — cracked with analytic level-gripper IK.

**Why this matters more than the 9%:** it answered *does anything solve this task?* when gradient
RL's answer was an opaque zero. It is a **solvability oracle and a reward-hacking detector**, and
then a scaffold — the controller generated the demonstration data that unblocked warm-starting,
which had been impossible because nothing could produce a successful trajectory.

**Its ceiling is real and large.** ~9% open-loop against a properly-budgeted PPO's ~66–83%
closed-loop. A hand-written controller cannot sense and correct the grasp dynamics it fails on.

The reusable protocol is in the 2026-06-21 post; its four memory artifacts (`KNOWLEDGE.md`,
`LOG.jsonl`, versioned snapshots + `BEST`, `regression.py`) are the part that made it compound.
This line has now been written into `competition-playbook/techniques/programmatic-policy-agents.md`
with our numbers.

---

## 4. The conjunctivity precedent we re-derived

Part 14 (2026-07-09), tuned PPO on HopperHop, seed 50, 20M steps, with env-gated reward knobs
(byte-identical when unset):

| PPO variant | reward structure | final return |
|---|---|---|
| default (control) | product `standing × hopping` | **0** — the wall |
| **additive** | `0.5·standing + 0.5·hopping` | **135** — climbs off zero |
| product, `HOP_SPEED=1.0` | product, *easier* hop threshold | **1** — wall persists |
| tdmpc2 additive (control) | additive | 467 |

Its conclusion: **the barrier is the conjunction itself**, not the hop-speed magnitude, not early
termination (there is none), not a fundamental PPO limit. And a footnote flagged that `HOP_SPEED=1.0`
also halves the tolerance margin, so *"a margin-controlled variant would be the cleaner isolation for
the paper version."*

This week we independently built that margin-controlled variant (`WMP_HOPPER_MARGIN`), an additive
variant (`WMP_HOPPER_ADDITIVE`), and a threshold control (`WMP_HOPPER_STAND`) — without knowing the
July work existed.

**The two are complementary, not redundant, and together they say more than either alone:**

| | July (Part 14) | August (Parts 24–26) |
|---|---|---|
| agent | tuned PPO | TD-MPC2, and DreamerV3 |
| measured | can it learn at all | how much *planning* adds |
| knob | hop-speed threshold (the *smooth* factor) | standing threshold (the *binary gate*) |
| result | conjunction blocks on-policy learning; easier hop threshold does **not** help | easier standing gate collapses the planning gain 3.53× → 1.21× |

So: the conjunction is what blocks a gradient learner (July), while how hard the *gate* is to clear
modulates how much search is worth (August). July's additive result (0 → 135) is also the clean
version of the additive test that came out inconclusive at n=2 this week.

---

## 5. Where the data actually lives

- **Local, safe (control box):** `tdmpc-glass/exp/` (568 MB), `aaai27-wm-diagnostic/` (377 MB),
  `world-model-paper/data/` (~90 MB, all of this month's per-seed JSON), `tdmpc-glass/hl_pickcube/`.
- **Stopped boxes (restartable, disks kept):** box1 `45525865`, lean `46737061`, mwm `41649155`.
  Checkpoints from the A3 deployment sweep live here and nowhere else.
- **Old server (unreachable without `ssh -A`):** most June-era raw run outputs.
- **Blog only:** Part 14's PPO reward-gate numbers — the scripts survive, the results JSON does not.
  **If we cite those numbers in a paper, re-run them.**

---

## 6. The corrected proposal list

**UPDATE (later the same day): the hopper leg is now RESOLVED, and the two tasks disagree.**

Hopper additive reached n=4, scored on the stock reward:

| hopper condition | plan | never | gain | perm p |
|---|---|---|---|---|
| stock (conjunctive) | 364.2 (n=5) | 103.3 (n=5) | 3.53× | **0.008** |
| additive (conjunction BROKEN) | 149.3 | 38.5 | **3.87× mean / 2.30× median** | 0.343 |
| stock, bar lowered (still conjunctive) | 522.2 (n=4) | 432.1 (n=4) | **1.21×** | 0.657 |

Removing the AND left the gain intact; making the conjunction *easier to satisfy* destroyed it. So on
**hopper the operative variable is difficulty, not conjunctivity** — while on **walker it is
conjunctivity, not difficulty** (1.14× → 1.76× when made conjunctive; 1.14× → 1.07× when merely made
harder, n=4). Both manipulations were run on both tasks, the same way.

**No single-variable account fits both columns.** That is now the honest headline: the tasks disagree,
and any claim that "conjunctivity sets the value of search" — or that difficulty does — requires
ignoring one of them. Caveats: the additive cell is p=0.343 with several collapsed seeds
(plan 1.1 → 441.7), and additive-trained agents scored on the stock reward are off-distribution, so
only the within-cell ratio is meaningful.

**Still open (~2 GPU-hours):**
1. **Walker conjunctive cell to n=5** — running now; currently n=3 at p=0.100.

**The strongest paper-shaped result we have:**
3. **Search is a training effect.** Deployment-time planning recovers 24% on hopper and is
   net-negative on walker, measured with the policy frozen — no headroom confound available to a
   reviewer. Explains why planner-free DreamerV3 shows the same task pattern. This is the finding to
   build on.

**Revived by this audit:**
4. **Adaptive planning, gated during TRAINING.** Since the value is in training, not deployment, the
   version worth building gates planning while data is being collected. First experiment costs one
   day: on a frozen checkpoint compute Δ = Q(MPPI) − Q(π) per state; if Δ is heavy-tailed, planning
   on the top-k% of states traces the achievable Pareto frontier before any gating mechanism is built.
5. **The coding-agent loop as a first-class method, not a footnote.** It is the only thing in nine
   months that solved a task gradient RL could not, and its diagnostic value (reward-hacking
   detection, solvability oracle) is independent of its score ceiling. Underexplored because it was
   filed under the wrong name.

**Honest downgrades:**
6. **Conjunctivity as the ICLR headline** — demoted. It is one contributor among at least two
   (difficulty is the other), it now has a July precedent we must cite rather than rediscover, and
   its hopper leg is untested.
7. **"Train cheap, plan at deployment" (A3)** — **dead.** Measured: 24% recovery, negative on walker.

---

## 7. Ranked by odds of acceptance, if we spend the remaining time on it

**Added 2026-08-12**, six days after this post, once the mechanism work closed. Everything above is
what we *ran*; this is what is worth *finishing*. The odds are judgements about acceptance
**conditional on the work being completed and written well** — aids for choosing between rows, not
predictions. No venue deadline has been set yet, which is the single input that would move these
most.

<div class="wide-table wide-table--inventory" markdown="1">

| rank | proposal | what is actually left | odds | why that number |
|---|---|---|---|---|
| **1** | **The world model and the planner are one mechanism** — search earns its keep in training, and the model loss exists to make search worth doing while the agent learns | **writing only** | **~65%** | Grade A on three legs: the frozen-checkpoint toggle, a cross-agent replication in DreamerV3 (which has no search at all), and a 2×2 interaction significant on two tasks (walker 90.1, p=0.044; cheetah 129.5, p=0.011). Two rival mechanisms were eliminated with gradient-level verification. No confound left for a reviewer to attach to |
| **2** | **Adaptive planning: a real ceiling that no obvious signal can reach** | complete as a negative; ~1 week to attempt a positive | **~45%** | Converts "someone should gate the planner" into a measured statement — the opportunity is worth ~53% of search's value at 10% of states, and nine signals across three families are provably blind to it. Strongest as the second half of #1 rather than alone |
| **3** | **Anti-collapse in JEPA world models: a reproduction study** ([Part 30](https://suuttt.github.io/tdmpc-glass/2026/08/12/tdmpc-glass-part30-we-could-not-reproduce-the-collapse.html)) | ~2 weeks to add scale | **~40%** | Adversarial to a live literature and cheap to state: collapse did not appear in five settings, anti-collapse is neutral-to-harmful, and effective rank moves *opposite* to control. The weakness is scale — our JEPA is small, LeWorldModel trains 15M params from pixels — so a pixel-scale arm is what buys the last 15% |
| **4** | **The 5× shrink** | **writing only** | ~35% alone | Grade A across 16 tasks and genuinely useful, but "the default is over-provisioned" reads as engineering rather than insight. Much stronger as an appendix to #1 than as a paper |
| **5** | **How claims decay** — the methodology paper | writing only | ~50% *as a workshop paper*, ~15% main track | Eight patterns looked real at 3 seeds; seven died at 5. We have the anatomy with timestamps, pre-registrations and four silent-instrument bugs. Real, and better suited to a workshop than a main track |
| **6** | **The coding-agent loop as a first-class method** | ~1 week | ~25% research, high as a workshop piece | The only thing in nine months that solved a task gradient RL could not, and its diagnostic value is independent of its ceiling. But the method is Weng's, not ours — we contribute evidence, not invention |
| **7** | **Entity-graph compositional OOD** | needs a control-reaching result | ~25% | The campaign's first honest GO, but the advantage stops at value-decodability and never reaches the controller. Without closing that gap it is a probe result |
| **8** | **Conjunctivity sets the value of search** | 1 more task | ~20% | **Demoted.** Holds on walker, and hopper says the opposite at n≥4 on every cell. Publish as a dissociation inside another paper, never as a law |

</div>

**The honest read of that table:** one paper is ready to write and everything else is either supporting
material for it or a workshop submission. The recommendation has not changed since
[Part 28](https://suuttt.github.io/tdmpc-glass/2026/08/07/tdmpc-glass-part28-the-iclr-decision-document.html) —
write #1, fold #2 in as its second half, #4 as an appendix, send #5 to a workshop.

**Two things that would move these odds more than any experiment:** a decided venue and deadline,
and a behaviour-cloning baseline. Right now "planning helps" has no demonstration-based denominator,
which is the first thing a reviewer will ask for and among the cheapest to supply.

---

## 8. The lesson about inventories

Part 25 was wrong not because any single number in it was wrong, but because it was assembled from
notes about the work instead of the work. An audit built that way inherits every gap in the notes —
and gaps are exactly what an audit is for.

The rule going forward: **an inventory is only valid if it was built from primary sources**, and it
must carry a data-availability column, because a result whose file no longer exists is a claim, not
a measurement.

