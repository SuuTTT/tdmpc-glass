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
| **H4 on hopper** | ablating the consistency loss at 50k/150k/250k vs never: +46.5 / +38.0 at n=5, p ≥ 0.72. The n=3 "+165.9" was the control drawing three low seeds | **D** |
| Conjunctivity as a general law | tasks disagree; hopper says difficulty | **D** |
| "Train cheap, plan at deployment" | 24% recovery, negative on walker | **D** |
| Structural entropy as a latent regulariser | 13–16 levers, all null | **D** |
| Jumpy / temporal abstraction | the flagship +60% was reward-hacking (0% real success) | **D** |
| Planning-as-exploration | fails a controlled plan-vs-policy test three ways | **D** |
| Hierarchy, options, H-JEPA, SE latents | null — all tested where the flat policy already worked | **D** |

The last row has a re-reading worth one sentence in the paper: every one of those was tested on
tasks whose plain policy already succeeded, which is precisely where #1 predicts nothing to gain.

---

### #5 — The world model's contribution is front-loaded within a run — on walker · **Grade A (confirmed n=5)**

Added after this document's first version, and it reverses what §3 said about H4.

| ablation point | walker (n=5) | p | hopper (n=5) | p |
|---|---|---|---|---|
| off @ 50k | **−18.4** | **0.008** | −56.9 | 0.468 |
| off @ 150k | **−12.8** | **0.016** | +46.5 | 0.722 |
| off @ 250k | −7.2 | 0.230 | +38.0 | 0.722 |
| per-cell sd | **3.6 – 8.9** | | 96 – 131 | |

**This is the first pattern in the campaign to survive the n=3 → n=5 jump.** It went
−25.7/−17.8/−10.2 → −18.4/−12.8/−7.2: tighter, still monotone, two cells significant. The previous
seven all collapsed at n=5.

Removing the world-model loss costs 25.7 points at 50k, 17.8 at 150k, 10.2 at 250k — a **monotone
decay**, exactly the shape H4 predicted, on the task H4 did not name.

**Why hopper shows nothing is a fact about hopper.** All 12 walker runs span 39.5 points (4.9%); a
single hopper cell spans 289–573. The instrument resolves a 10-point effect on walker and cannot
resolve a 200-point one on hopper.

**This is worth a paragraph in the paper regardless of how #5 resolves.** We chose hopper as the
primary task because its planning gain is the largest in the suite (3.53×) — and it is the task where
nothing is measurable at achievable n. Most of the seven retracted n=3 patterns are hopper results.
**Selecting the biggest effect selected the worst signal-to-noise.**

**Why it strengthens #1.** Candidate #1 says the model's value is in training rather than decisions.
This says that value is concentrated **early** in training. Two independent instruments — a
frozen-checkpoint planner toggle and a training-time loss ablation — on two different tasks,
reaching the same conclusion. Promote both together.

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

# Added 2026-08-10 — two gaps, one scoping correction, and the proposals they generate

Everything above was written as though "does planning help?" were a question about planners. Two
objections raised on 2026-08-09 show it is not, and both were correct. They change what §5
recommends, so they are recorded here rather than folded silently into the text.

## 6.1 The JEPA objection — our flagship claim is narrower than §2 states

**The objection.** A JEPA/DINO-WM agent is a next-state predictor plus a planner: *no policy, no RL,
and it works.* If planning can carry a task with no policy anywhere in the system, then "planning's
value is that it shapes what the policy collects" cannot be a claim about planning in general.

**This is right, and candidate #1 must be scoped.** What reconciles it is a variable we never
isolated — **what the model was trained on**:

| | TD-MPC2 | JEPA / DINO-WM |
|---|---|---|
| model trained on | its own self-collected stream | broad, diverse, off-policy data |
| objective | value-equivalent | pure prediction |
| accurate where | the current policy already goes | over a wide region |
| can you search it cold? | **no** (24% recovery) | **yes** — that is the whole method |

So the honest form of #1 is: **in online model-based RL where the model is trained on
self-collected data**, the planner must be present throughout, because it is what keeps the model's
training distribution good enough to search. The deployment toggle did not fail because planning is
intrinsically a training-time phenomenon — it failed because that agent's model was only ever
trained on data from a bad policy.

This is a better framing than the one we had, because it makes a **prediction**: give TD-MPC2 a
model trained on broad off-policy data and the deployment toggle should start working.

## 6.2 The behaviour-cloning gap — the baseline we never ran

**The objection.** In practical game AI, supervised behaviour cloning routinely beats both RL and
planning, and we never considered it.

**Also right.** BC appears in this campaign exactly once as a real arm — Part 25, distilling the
analytic controller *out* into a policy, where "BC / warm-start / DAPG all failed". That failure is
specific: the controller is **non-Markov**, so cloning it into a Markov policy is ill-posed. It says
nothing about BC as a demonstration-based baseline, which we never ran.

We even had the demonstrations and used them for something else: the HL-loop programmatic controller
(§4) produced working PandaPickCube trajectories, and we spent them seeding TD-MPC2 rather than
establishing a baseline. Without that baseline, "planning helps" is measured against a weak
alternative, and a reviewer will say so.

## 6.3 A retracted framing: "MBRL has a circular dependency"

I first pitched §6.5 below as breaking a circular dependency — *the planner needs an accurate model,
the model is only accurate where the planner has been.* **Withdrawn on two counts.** MPPI has no
parameters, so nothing about the planner is learned and no loop closes through it. And the residual
loop — model → behaviour → data → model — is **not specific to MBRL**; model-free RL has exactly the
same thing. That framing dresses up ordinary on-policy distribution shift as a discovery.

The real asymmetry is a **train/query mismatch**, and it *is* specific to planning:

- a policy is queried where it is trained — at visited states;
- **a planner is not.** MPPI evaluates 512 action sequences over horizon 3, sampled with noise around
  the policy's actions, so by construction it queries the model at latents that were never visited.
  Asking "what if I did something other than what I do" *is* an off-distribution query.

Nothing in the training objective targets the query measure. Model-free RL has no analogue, because
the critic is queried where it is trained.

## 6.4 The candidates these generate, ranked

### S1 — World models in MBRL are representation learners, not simulators · **RUNNING**

**Claim.** The world-model loss earns its keep by *shaping the encoder*, not by producing a
simulator worth rolling out.

**Why the existing evidence points here.** On cheetah, planning is worth ~1.00× — search through the
model buys nothing — yet removing the model loss costs **58–127 points** (n=5). Something is bought
that is not simulation. DreamerV3 reproduces the task pattern with no search anywhere. And sixteen
explicit-abstraction levers nulled precisely because the model loss was *already* doing the
representation work they duplicated.

**The decisive ablation.** Stock TD-MPC2 cannot separate the two, because its consistency loss

> `L_c = Σ_t ρ^t ‖ dyn(z_t, a_t) − sg(enc(s_{t+1})) ‖²`,  `z_0 = enc(s_0)`

sends gradient to **both** the dynamics head (through `dyn`) and the encoder (through `z_0`); the
target is already under `no_grad`, so `z_0` is the encoder's only path. Run a second latent rollout
for the consistency term starting from `z_0.detach()` and the two are separated:

| arm | dynamics head | encoder shaped | cheetah return | n |
|---|---|---|---|---|
| **A** stock | consistency + value/reward | **yes** | **900.6** (sd 17.0) | 5 |
| **B** `WMP_CC_SG=1` | consistency + value/reward | **no** | **817.9** (sd 102.1) | 5 |
| **C** `consistency_coef=0` | value/reward only | no | **656.4** (sd 98.1) | 3 |
| *(off at 50k, for reference)* | | | *842.8* (sd 66.2) | *5* |

**Because `.detach()` changes no values, the consistency loss is numerically identical in A and B.**
Verified on a fixed batch before spending the compute — same loss to the last digit
(`0.016058076173067093`), same dynamics gradient (`0.043782633957885744`), encoder gradient
`0.00626 → exactly 0.0`:

```
== STOCK ==  {"consistency_loss": 0.016058076173067093,
              "encoder_grad_norm": 0.006258090169236204,
              "dynamics_grad_norm": 0.043782633957885744}
== SG ==     {"consistency_loss": 0.016058076173067093,
              "encoder_grad_norm": 0.0,
              "dynamics_grad_norm": 0.043782633957885744}
```

**Reading.** B ≈ A → the loss buys a simulator, S1 is wrong. B ≈ C → the loss is an auxiliary
representation objective and the "world model" is not modelling the world in any way the agent uses.

#### RESULT (2026-08-10): S1's strong claim is REFUTED. B sits between A and C.

| contrast | what it isolates | Δ | perm p |
|---|---|---|---|
| **A − B** | the consistency loss **shaping the encoder** | **82.8** | 0.056 |
| **B − C** | the consistency loss **training the dynamics head** | **161.4** | 0.054 |
| **A − C** | the whole consistency loss | **244.2** | **0.018** |

**Roughly one third representation shaping, two thirds dynamics-head training** — and the split runs
*against* S1. On cheetah the loss's larger job is making the dynamics head predict well, even though
search through that head buys ~1.00×. The dynamics head still feeds the reward and value losses over
horizon 3, so "useless for planning" does not mean "useless as a model".

The arms are also monotone in how much consistency loss the run received, which is the internal
check this design gets for free:

> **656.4** (never) < **842.8** (off at 50k) < **900.6** (stock)

That ordering independently supports the front-loading result (#5): a loss you drop at 50k costs 58
points, one you never have costs 244.

**What survives.** Not "world models are representation learners". Instead a **decomposition** that
nobody has published: the consistency loss has two separable jobs, they can be measured
independently, and on this task neither dominates. Encoder shaping being worth 83 points on a task
where planning is worth ~0 is still the interesting half — that value cannot be flowing through
search.

**Honest limits.** C is **n=3**; the 34/66 split rests on it and could move materially. A−B is
p=0.056 — the effect held its direction and magnitude from n=4 to n=5 (−99.1 → −82.8), which seven
of eight patterns in this campaign failed to do, but it is **not significant** and I am not calling
it so. Arm B's spread is 6× the control's (sd 102 vs 17).

**A correction, made the same day.** At n=4 I reported B as "landing at the level of deleting the
loss outright". That was wrong: the comparator I used (`ccoff@50k`, 842.8) is not deleting the loss
outright, it is deleting it at 50k. The real delete-outright arm is 656.4, and B is well above it.
The within-batch control is what caught this — which is why it was run.

**Why ICLR still takes it:** a falsifiable, counterintuitive claim about a whole class of methods,
decided by an ablation anyone can rerun — and the ablation worked. **Both outcomes publish, and this
is the second one.**

*(Tooling: `repro/cc_stopgrad.patch.py`, `repro/verify_stopgrad.py`. The verification harness itself
failed twice first — it read `.grad` after the optimizer had zeroed it, then recorded the second
`clip_grad_norm_` call, which fires after `optim.zero_grad()` in `update_pi`. Both read as a dead
patch. The rule that caught it: verify the **mechanism**, never the log line.)*

### S2 — Train the model on the query measure, not the visitation measure · **method**

From §6.3. **Measure first, the gap nobody reports:** log the model's k-step error separately on
(a) visited transitions and (b) the latents MPPI actually queried during search. If (b) ≫ (a), the
planner has been searching a model that is wrong exactly where it looks, and the size of that gap
should predict how much planning is worth across tasks.

**Then the method follows:** draw the model-loss batch from MPPI's rollout latents rather than from
the buffer alone — a change to *which states get gradient*, with an argument that is not
"circularity". **Kills it:** error on queried latents is no worse than on visited ones.

### S3 — Coverage decides whether a frozen model is searchable · **theory + method**

From §6.1. Train the same model three ways — self-collected, broad exploratory, demonstrations —
then run the identical deployment toggle on each, with off-distribution model error as the mediating
variable. Supplies the missing piece of the pretrained-world-model agenda: **the condition under
which a frozen world model can be planned through.** Highest fashion, highest null risk.

### S4 — The demonstration crossover · **finding + method**

From §6.2. Sweep N demonstrations upward, comparing BC / BC+planner / RL+planner / RL. There is a
crossover; **locating it is the contribution** — "how many demonstrations make your planner
pointless." Partly defensive: it supplies the denominator #1 currently lacks.

## 6.5 What this does to the §5 recommendation

§5 stands — #1 is still the paper — with two amendments:

1. **State the scope.** #1 is a claim about online MBRL with self-collected data, and §6.1 belongs in
   the paper as the boundary condition plus a prediction, not as a caveat buried in limitations.
2. **S1 outranks #12 as the next experiment.** Both are mechanism work for #1, but S1 costs a day
   against #12's four, its supporting evidence is already collected, and it publishes either way.

**Betting order: S1 → S2 → S4 → S3.** S1 because a cheap ablation can flip a widely held belief; S2
because it is the better method paper if the mismatch is real; S4 because the baseline has to exist
before review; S3 last because it is the most likely to end in a null.

---

*Every number here is read from logged JSON/CSV in `SuuTTT/world-model-paper` under
`data/`, with per-seed values. The corrections — including two of mine from this week, and the
retracted "circular dependency" framing in §6.3 — are in git history rather than edited away.*
