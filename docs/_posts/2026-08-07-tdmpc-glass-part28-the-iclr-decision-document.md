---
layout: post
title: "TD-MPC-Glass, Part 28: The ICLR Decision Document — Every Candidate, Its Evidence Grade, and What It Would Cost"
date: 2026-08-07
description: "One self-contained document for choosing what to write. Every proposal the campaign has generated, each with the experiments behind it, an explicit evidence grade, the cost to finish, and what would kill it. Includes the results from this week — planning is a training effect, the adaptive-planning ceiling and the nine signals that cannot find it, the walker conjunctivity dissociation, and two nulls (H4 ablation timing, and my own retracted oracle frontier). Written to be read cold by a supervisor with no prior context."
---

> **Read this one.** Part 25 was an audit assembled from notes and missed an entire research line.
> Part 27 rebuilt the inventory from all 87 posts and added a data-availability column. This post is
> the *decision* document: what to write for ICLR, ranked, with the evidence grade for each claim
> stated honestly enough that a reviewer's first objection is already answered.

---

## 0. The one-paragraph version

We set out to design a better world model. We do not have one. What we have is a causal account of
**how** a world model earns its keep — it shapes *training*, not decisions — measured in a design
with no confound left to attack, and reproduced across two agents that exploit the model in
completely different ways. That is the paper. Everything else here is either supporting evidence,
a clean negative result worth a section, or dead.

---

## 1. The evidence grades used below

| grade | meaning |
|---|---|
| **A** | n≥5, p<0.05, replicated, and the obvious confound explicitly controlled |
| **B** | n≥4 with a clear effect, but one confound unaddressed or p between 0.05 and 0.2 |
| **C** | suggestive; n≤3, or p>0.2, or an in-sample statistic |
| **D** | null or retracted |

The campaign's base rate matters when reading these: **seven times** an n=3 pattern has evaporated
at n=5. Grade C here means "do not build a paper on this yet", not "probably true".

---

## 2. The candidates, ranked

### #1 — Search is a training effect, not a decision-time effect · **Grade A**

Train TD-MPC2 with its planner and it scores **364.2** on hopper-hop (n=5). Train it *without* the
planner and switch the planner on afterwards, on the frozen weights, and it scores **87.5** — it
recovers **24%**. On walker, across 8 checkpoints, adding the planner at deployment is a **net loss**
(mean 0.960×).

**Why this survives review.** The comparison holds the policy *identical* across arms — same weights,
same state distribution, same headroom. The reviewer's standard objection to every "planning helps"
result ("your baseline just had more room to improve") has nothing to attach to.

**Why it generalises.** DreamerV3 has no search anywhere; it uses its model only to train the policy
on imagined rollouts. It reproduces the same task pattern — hopper **2.40×** (p=0.009, n=6), walker
**0.95×** null (n=13/11). A search-based explanation cannot produce that.

**The mechanism, visible in one cell.** A hopper seed whose policy scored 0.0 could not be rescued by
the planner (0.0 with search too). MPPI searches *through the learned model*; a run that never
learned a policy never collected the data to learn a usable model. Planning is not a module you
attach to a failed agent.

**What it kills:** "train cheap, plan at deployment". Measured, and the answer is no.

**Cost to finish:** the result is complete. Writing only.

---

### #2 — Adaptive planning: the ceiling is real, and the obvious signals cannot reach it · **Grade B/A**

If the value is in training, the natural follow-up is *plan only where it pays*. Two measurements,
both on frozen checkpoints, no training:

**The ceiling (Grade B).** Δ = Q(planner action) − Q(policy action) per state. Sorting by Δ and taking
the top 10% captures a **median 53%** of capturable value, held out across 6 checkpoints (range
0.23–0.88). Δ is reproducible at **r = 0.76** between two planner draws, so it is a state property,
not dice. The planner costs ~35% of wall-clock, so a perfect gate at k=10% would pay ~3.5% for ~half
the benefit.

**The negative (Grade A).** Nine candidate signals across three independent families — value
uncertainty (Q-ensemble disagreement, spread, policy std, advantage gap), action sensitivity (a
3%-cost probe plan's return spread and top-gap), and model reliability (one-step latent error, reward
error, observation jump) — **all show Spearman ≈ 0 against Δ** and gate efficiency ≤ 0. Not one of
them finds the states where planning pays.

**Why this is a good section rather than a failure.** It converts "someone should try gating the
planner" into a measured statement: the opportunity is real and worth ~half of search's value, and
the three most natural feature families are provably blind to it. That is a well-posed open problem,
not a shrug.

**Caveat carried in the open:** I first reported this ceiling as 0.83–0.91. That was **in-sample** —
selection and evaluation on the same noisy draw. Held out it is 0.53. I then over-corrected to 0.22
by generalising from one checkpoint. Both corrections are in git.

**Cost to finish:** complete as a negative result. Turning it positive requires a feature family
outside these three.

---

### #3 — Reward structure sets the value of search, on walker · **Grade A (walker) / D (as a law)**

| walker condition | plan | never | gain | p |
|---|---|---|---|---|
| stock (smooth) | 807.4 (n=4) | 705.4 (n=4) | 1.14× | 0.029 |
| harder, **conjunctivity unchanged** | 480.2 (n=4) | 447.0 (n=4) | **1.07×** | 0.029 |
| **conjunctive** (hardened gate) | 816.9 (n=5) | 443.9 (n=5) | **1.84× / 1.91× med** | **0.008** |

Making walker's reward conjunctive nearly doubles what planning buys, at matched baseline. Making it
merely *harder* does nothing. The planning arm is tight (sd 34.3) while the no-plan arm scatters
(sd 163.4) — search rescuing runs that would otherwise fail.

**But it does not generalise to hopper.** There, breaking the conjunction left the gain intact
(3.53× → 3.87×, p=0.343) while making the task easier destroyed it (3.53× → 1.21×). **The two tasks
disagree, at n≥4 on every cell.** Publish this as a dissociation on walker, not as a law.

---

### #4 — How claims decay: the methodology paper · **Grade A (it happened)**

Seven times an n=3 pattern died at n=5. The causes are documented with timestamps:
- **arrival-order bias** — the cheaper arm finishes first, so a speed-vs-quality ablation is biased
  by the very effect it studies
- **in-sample selection** — sorting by a noisy quantity and scoring the sort with the same noise
  (my own oracle frontier, 0.91 → 0.53 held out)
- **within-cell spread exceeding the effect** — hopper's cells span 258–578, wider than any hopper
  effect ever claimed here
- **silent instrument bugs** — a Q function that randomly subsamples 2 of 5 heads per call; a
  `torch.compile` graph that freezes the planner's RNG so two "independent" draws are byte-identical;
  a budget cap that announced enforcement it could never perform because `vastai` was not on cron's PATH

**Workshop paper, not main track.** Real, and we have the receipts.

---

## 3. What is dead, and should not be revived without new evidence

| direction | verdict | grade |
|---|---|---|
| **H4: the world model has a shelf life within a run** | ablating the consistency loss at 50k/150k/250k vs never: +46.5 / +38.0 at n=5, p ≥ 0.72. The n=3 "+165.9" was the control drawing three low seeds | **D** |
| Conjunctivity as a general law | tasks disagree; hopper says difficulty | **D** |
| "Train cheap, plan at deployment" | 24% recovery, negative on walker | **D** |
| Structural entropy as a latent regulariser | 13–16 levers, all null | **D** |
| Jumpy / temporal abstraction | the flagship +60% was reward-hacking (0% real success) | **D** |
| Planning-as-exploration | fails a controlled plan-vs-policy test three ways | **D** |
| Hierarchy, options, H-JEPA, SE latents | null — all tested where the flat policy already worked | **D** |

The last row has a re-reading worth one sentence in the paper: every one of those was tested on
tasks whose plain policy already succeeded, which is precisely where #1 predicts nothing to gain.

---

## 4. The one positive result that is not ours to claim, and should be

A coding agent iteratively writing a **programmatic controller** (Weng's *Learning Beyond Gradients*)
took PandaPickCube from **0% → ~9% real success in hours**, on a task where gradient RL
reward-hacked to literally zero (hovering banked 89% of return; `box_target_max` = 0.0000). It is the
only thing in nine months that solved a task gradient RL could not.

Its value was diagnostic, not competitive: it answered *is this task solvable at all?* when RL's
answer was an opaque zero, and then generated the demonstration data nothing else could produce.
Ceiling: ~9% open-loop vs a properly-budgeted PPO's ~66–83% closed-loop.

**This belongs in the related-work and discussion, not as a contribution** — but leaving it out
would misrepresent what the campaign learned.

---

## 5. Recommendation

**Write #1 as the paper.** Title it around the finding: the world model's value is in what it makes
the agent *collect and learn*, not in what it computes at decision time. It is the only Grade A
result with a mechanism, a cross-agent replication, and no confound left standing.

**Include #2 as the second half.** It is what #1 implies (gate during training, not deployment), it
has a measured ceiling, and its negative is clean enough to be a contribution rather than an
apology.

**Use #3 as a supporting section** on walker, stated as a dissociation.

**Submit #4 separately** to a workshop.

**What I would not do:** keep hunting for the single sentence that explains both hopper and walker.
Two tasks, two answers, at n≥4 on every cell — that is the finding, and forcing a resolution would
mean ignoring one of them.

---

*Every number here is read from logged JSON/CSV in `SuuTTT/world-model-paper` under
`data/`, with per-seed values. The corrections — including two of mine from this week — are in git
history rather than edited away.*
