---
layout: post
title: "TD-MPC-Glass, Part 10: A Week in Review — From Five Failed Bets to a Dissection (and a First SOTA Swing)"
date: 2026-07-08
description: "The week of July 1–8, reviewed end to end. It began with a method map and five concrete bets to finally beat PPO with abstraction or planning-as-exploration (Parts 5–6). Almost all of them nulled (Part 7). That failure was the useful part: it forced a pivot away from 'add structure to win' toward rigorously dissecting WHY the planner beats PPO in the first place — the categorical exploration wall, the loss-by-loss mechanism, and a five-task sufficiency grid (Parts 8–9, three papers). We close with the first constructive SOTA attempt built on those lessons: value-aware consistency (a clean no-go) and an uncertainty-aware variant now running. The recurring law held all week: abstraction and reweighting buy nothing the value pathway doesn't already use."
---

> This is a review of one week — July 1 through July 8 — across [Part 5](https://suuttt.github.io/projects/2026-07-01-tdmpc-glass-part5-beat-ppo-reality-check/)
> through Part 9 and the work since. The week has a clean three-act shape. **Act I:** we laid out a map of four
> method families and made *five concrete bets* to beat PPO with abstraction or planning-as-exploration.
> **Act II:** the data killed almost all of them — and that failure redirected the whole program from "add
> structure to win" to "dissect *why* the planner already wins." **Act III:** that dissection produced three
> papers (a wall, a mechanism, a sufficiency law) and, finally, a first constructive attempt at a new SOTA world-model
> objective. Every number below is read from disk; the nulls are reported as loudly as the positives.

## Where the week started: the method map (Part 5)

[Part 5](https://suuttt.github.io/projects/2026-07-01-tdmpc-glass-part5-beat-ppo-reality-check/) framed the entire program not as
"beat PPO once" but as a **map**: across Panda manipulation and DMControl, *when does each of four families win?*

- **State abstraction** — structural-entropy ("glass") / relational latents.
- **Behavioral abstraction** — analytic controllers, TAMP, skill options.
- **Planning** — TD-MPC2 (a learned latent world model + MPPI).
- **Plain PPO.**

Scored LeCun-style by *learning speed*, not just final return, one law kept recurring: **abstraction and priors buy
sample-efficiency where the prior fits, not a higher ceiling.** Part 5 also set up the JEPA pivot — LeCun's argument
that a non-generative predictor needs an information (anti-collapse) term, and that *hierarchy* is where structure
should pay off — and a beat-PPO reality check that, when the returns were audited for reward-gaming, collapsed a
"TD-MPC2 beats PPO on PandaPickCubeOrientation" claim into a verified **PPO solves, TD-MPC2 fails** (real success
0.000 under return-gaming).

## Act I — five bets (Part 6)

[Part 6](https://suuttt.github.io/projects/2026-07-02-tdmpc-glass-part6-five-bets-next-phase/) turned the open
questions into five falsifiable bets:

- **A. Planning-as-exploration** — does using the MPPI planner to *seek* novel/uncertain states break tasks a
  reactive policy can't explore?
- **B. Behavioral-prior taxonomy** — where exactly does an injected controller help vs hurt?
- **C. Glass-as-variance-reduction** — even if the mean return ties, does structure reduce seed variance?
- **D. JEPA done right** — pure (decoder-free) latent prediction + a real anti-collapse term.
- **E. SE-structured latent discovery** — structural entropy as the anti-collapse signal.

## Act II — what survived contact with the data (Part 7)

[Part 7](https://suuttt.github.io/projects/2026-07-03-tdmpc-glass-part7-five-bets-resolved/) is where the week
turned. Five bets, all multi-seed and matched-budget. Verdicts first, then the experiments and the data behind each.

| Bet | What it claimed | Verdict |
|---|---|---|
| A — planning-as-exploration | the planner *seeks* novel states a policy can't | **Refuted** |
| C — glass as variance-reduction | structure lowers seed variance even at tied mean | **Null** |
| D — pure-JEPA anti-collapse | a decoder-free predictor collapses without it | **Reversed** |
| E — SE-structured latent | structural entropy is the right anti-collapse | **Hurts** |
| B — behavioral-prior taxonomy | injected controllers raise the ceiling | **Sample-eff only** |

#### Bet A — is planning a directed-exploration operator?

The claim: TD-MPC2 beats PPO because the MPPI planner actively *explores* — it seeks novel states a reactive policy
never reaches. We tested it with a same-weights plan-vs-policy ablation, escalating to a decisive sparse task. On a
learnable dense task (WalkerRun, n=3) planning leads through the competence phase then **converges to the same
ceiling** — a speed effect, and it actually *narrows* late-training state coverage (directed exploitation onto the
gait manifold, not exploration):

| step | PLAN return | PI-only return |
|---|---|---|
| 50k | 335 | 186 |
| 90k | 408 | 247 |
| 140k | 480 | 446 (converged) |

The make-or-break test was a sparse, exploration-hard, learnable task — CartpoleSwingupSparse (n=3, 300k). Does
planning discover the sparse reward where policy-only stalls? **No — policy-only discovers it *earlier*:**

| metric | PLAN | PI-only |
|---|---|---|
| reward-discovery rate | 3/3 | 3/3 |
| mean discovery step | 156,672 | **140,032** |
| final return | 454 | 239 |

Planning's real value is *post-discovery exploitation*, not exploration. The other half of Bet A — novelty bonuses
(disagreement/RND) inside the planner on HopperHop 1M, with a **matched-seed** vanilla control — was worse on 4/4:

| seed | vanilla | + novelty |
|---|---|---|
| 61 | 442.3 | 5.6 (broke) |
| 62 | 508.5 | 284.6 |
| 63 | 321.4 | 258.7 |
| 64 | 359.4 | 273.8 |

**Refuted:** planning is a sample-efficiency + exploitation operator, not a directed-exploration one.
(`b3060b:exp/proposal_A1_coverage/`.)

#### Bet C — does glass reduce seed variance even when the mean ties?

Out-of-sample across all 16 tasks: mean seed-sd **glass 123.2 vs TD-MPC2 114.5** — glass is *higher* variance;
lower-sd on only 8/16, better worst-seed 9/16 (coin flips), and glass has its own collapses TD-MPC2 avoids
(HopperHop 0 vs 179). **Null** — no variance-reduction effect. (`b3060:exp/proposal_C_variance/`.)

#### Bet D — does a "proper" JEPA collapse without an anti-collapse term?

A pure decoder-free self-predictive latent (encoder + jumpy predictor + EMA target, no reward/value/policy),
frozen-encoder ridge probes, three tasks (n=3) plus a pixel version (30 runs). The premise was **falsified** — a
pure JEPA does *not* collapse; the predictor+EMA (BYOL) asymmetry prevents it with no anti-collapse term at all, and
adding one *hurts*:

| arm (WalkerWalk) | geom R² | value R² | eff-rank |
|---|---|---|---|
| none (no anti-collapse) | **0.795** | **0.304** | 5 |
| + uniformity | 0.583 | 0.121 | 40 |

Uniformity *maximizes* eff-rank yet *destroys* the readouts — the same inverse eff-rank↔decodability pattern held
on pixels. The Part-5 "anti-collapse taxonomy" **reversed**: anti-collapse is load-bearing only in the narrow
closed-loop nav regime where the latent actually collapses; on broad DMControl data it's neutral-to-harmful.
(`exp/proposal_D2_pure_jepa/`, `exp/proposal_D3_pixel_jepa/`.)

#### Bet E — structural-entropy latent on TD-MPC2

SE anti-collapse added to the value-anchored TD-MPC2 latent (fixed-λ, return-AUC, n=3):

| task | default | SE | uniformity | SE+unif |
|---|---|---|---|---|
| CheetahRun | 58.9 | 23.4 | 36.6 | 29.0 |
| WalkerWalk | 293.8 | 110.3 | 89.9 | 62.5 |
| FingerSpin | 249.4 | 214.4 | 171.8 | 63.8 |

**default > SE on every task**; SE+uniformity is worse than either alone (no synergy). On a value-sufficient latent
the right anti-collapse is *none-extra*. **Hurts.**

#### Bet B — behavioral-prior taxonomy

A matched-budget-controlled sweep confirmed the campaign law: injected controllers buy **sample-efficiency where the
actuated DOF match the goal DOF** (ReacherHard OSC ~3× faster), and are dead weight or an anchor on unfit tasks. No
ceiling gain survives a same-budget PPO control.

**The tally: four of five bets came back null-to-harmful**, and the one preliminary positive (a learned 2-level
feudal hierarchy beating flat) was attributable to signal density, not abstraction. This is the pivot point of the
week: every constructive attempt to *add* structure to win reproduced the program's oldest finding — the TD value
pathway is the engine, and structure the value head doesn't consume is redundant.

So instead of asking "what can we add to beat PPO," we asked the sharper question: **why does the planner beat PPO at
all, and what inside it is doing the work?**

## Act III, scene 1 — the wall (Part 8–9, Paper 3 Claim 1)

The first dissection result is that PPO doesn't merely lose to the planner on some tasks — it hits a **categorical
exploration wall** on contact-critical hopper tasks.

- **HopperHop:** tuned PPO reaches **0/5 seeds ≥ 200 return at 472M steps/seed** (peak 53.8). SAC crosses 200 in
  **6/12 seeds by 5M and ~6/9 by 8M**; TD-MPC2 does it **6/6 by ~1M**.
- The wall **survives the standard exploration knob**: entropy-cost ×3 (peaks 4.2 / 73.9) and ×10 (48.1 / 64.1) at
  150M leave PPO in the same ≤74 class — under-exploration hyperparameters are *not* the explanation.
- **HopperStand** is a graded, near-categorical barrier: PPO escapes **2/16** (all four 285M runs walled at
  105–195); entropy ×3 put 1/4 over (a rate consistent with the baseline lottery, not a repair), ×10 0/4. Final
  three-method gradient: **TD-MPC2 5/5 (≤0.9M) ≫ SAC 7/10 (5M) ≫ PPO 2/16 (120–285M).**

And the wall is **causally scoped**: on **AcrobotSwingup** — unstable but *contact-free* — PPO learns fine (267–344,
n=4), no wall. The wall needs *contact-criticality*, not mere instability.

## Act III, scene 2 — the mechanism (Paper 3 Claim 3)

If the planner is the thing that clears the wall, *which of its losses* is the engine? The five-loss ablation
(mask one loss at a time), **4 tasks × n≥4 per arm**, MPPI-best per seed:

| Arm removed | CheetahRun | HopperHop | WalkerRun | HopperStand |
|---|---|---|---|---|
| none (full) | 721–795 | 287–570 | 680–731 | 911–946 |
| **value** | 16–58 dead | ~0 dead | 28–56 dead | 6–13 dead |
| **policy** | 123–192 dead | ~0 dead | 53–83 dead | 9–34 dead |
| reward | 5–31 (π 681–805) | ~0 (π full) | 44 (π full) | 265–542 (π 943–944) |
| consistency | 367–558 mild | 185–245 mild | 483–674 mild | 816–898 near-full |

The reading, confirmed on a fourth task the full model solves in 0.3M steps: **without the TD value loss the agent
cannot even learn to stand.** Value and policy are individually fatal everywhere; the reward head only feeds the
planner (the policy still reaches full strength without it); and the **consistency loss — the actual "world model"
objective — is the mildest cut on all four tasks.** This contradicts TD-MPC2's own ablation story and is the most
checkable claim in the program. Rounding out the matrix: **humanoid (21-DoF) is where only SAC survives** (Walk
625–909 in 4/5, Stand 918/922) — PPO nan under every config, and TD-MPC2 diverges to loss=nan in 6/6 default runs
(the one stable knob variant never learns, 21.8). The efficiency anchor at publication n: **TD-MPC2 HopperHop 5M =
420 ± 113 over 12 seeds.**

## Act III, scene 3 — the sufficiency law (Paper 4, the 2×5 grid)

The ablation proves *necessity* one loss at a time. The flip side is *sufficiency*: train the stripped agent
(consistency loss OFF from scratch) at full 5M budget — does it match the full model? Five tasks in:

| Task | stripped (consistency-OFF) | full baseline | gap | verdict |
|---|---|---|---|---|
| **HopperHop** | 306/449/443/477 + 165/475/481/511 (n=8) | 420 ± 113 (n=12) | ≈0 | **removable** (7/8 in band) |
| WalkerRun | 537/574/554/594 (565) | 709/705/753/782 (737) | **−23%** | load-bearing |
| CheetahRun | 527/528/516/524 (524) | 903/904/782/806 (849) | **−38%** | load-bearing |
| AcrobotSwingup | 297/233/352/256 (284) | 533/511/513/488 (511) | **−44%** | load-bearing |
| CartpoleSwingupSparse | 0/0/0/0 | 0/0/1.3/0 | n/a | both-fail (uninformative) |

I got the first cut of this wrong and corrected it in the open: with only Hop and Walker in, "removable when
exploration-bound, load-bearing when dense" looked clean — until **Acrobot** (exploration-*flavored*) showed the
*largest* gap. The reading that survives all four informative tasks is sharper: **the consistency loss underwrites
the planner's MPPI rollout quality wherever the planner carries learning** (Walker, Cheetah, Acrobot are all
planner-led). **HopperHop is the sole removable case** — there the policy head learns the behavior directly and the
planner only amplifies it, so the self-predictive loss is redundant (confirmed at n=8, 7/8 stripped seeds inside the
full band). Cartpole-sparse is a degenerate both-fail cell (the full model can't solve it either), so the grid's
evidentiary core is the four solved tasks.

## Act III, scene 4 — the first SOTA swing

Those findings hand us a constructive lever: if the consistency loss is a *rollout-quality regularizer for the
planner*, can we make it better and beat vanilla TD-MPC2 on exactly the planner-led tasks?

- **Bet 1 — Value-Aware Consistency (VAC):** weight each latent dimension's consistency error by its value
  sensitivity \\(|\partial Q/\partial z|\\), spending model capacity where the planner's value-ranking looks.
  Result, paired same-seed vs matched vanilla: **no-go.** Walker −8.9%, Cheetah −4.4% at 5M (worse at every
  checkpoint). The interpretation *reinforces* the sufficiency thesis: the planner's rollouts need **faithful
  dynamics on every dimension they might explore**, so concentrating capacity on currently-high-value dims
  *starves* the rest. Uniform consistency isn't just load-bearing — it's near-optimal in form.
- **Bet 2 — Uncertainty/Rollout-reliability-weighted Consistency (URC):** the better-motivated idea — weight
  consistency by the model's *own* rollout drift (open-loop vs teacher-forced one-step prediction), i.e. fix the
  dynamics exactly where multi-step rollouts compound error. This is running now (paired vs matched vanilla on
  Walker and Cheetah); early signal is mildly positive but far too young to trust. Verdict pending at 5M.

## The through-line

Read top to bottom, the week is one argument delivered four ways. Planning-as-exploration, JEPA anti-collapse,
SE-structured latents, glass variance-reduction, and value-aware consistency **all failed to beat a plain baseline**,
while the value/policy pathway proved individually fatal to remove and the uniform consistency loss proved both
load-bearing (on planner-led tasks) and near-optimal in form. The consistent moral: **in a value-based planner,
structure and reweighting buy nothing the TD value pathway doesn't already consume.** That is a negative result with
teeth — it's precisely why the three papers this week are a dissection, not a new method.

## The JEPA line, reopened — a plan

A fair objection: did we drop JEPA too early? The uniformity result looks like a flip — it *helped* the nav H-JEPA
early, then *hurt* everywhere later. It isn't a flip; it's **regime-dependence**. Anti-collapse helps if and only
if the latent actually collapses, which happened in exactly one place — closed-loop online goal-conditioned nav
(eff-rank → ~0; anti-collapse restored it, point-maze 0.53 → 0.95). On broad DMControl a pure JEPA doesn't collapse
(the predictor+EMA/BYOL asymmetry prevents it), so anti-collapse is redundant-to-harmful; on value-anchored TD-MPC2
the TD loss already keeps the latent value-sufficient. So we closed **anti-collapse-as-a-penalty** — but we never
tested **SE as *structure***: an SE-community partition that *defines* an abstraction, rather than an SE gradient
that competes with the predictor.

That is the open line, and it has a home — not dense value-based control (structure is provably redundant there),
but the three niches where structure showed a pulse or where there's no value signal to hide behind:

- **J1 — SE-community anti-collapse vs uniformity/VICReg, *in the collapse regime*** (cheap; finishes an open cell).
  Does a compact SE community structure preserve goal-decodability better than uniformity where the latent really
  collapses?
- **J2 — SE *as* the H-JEPA abstraction** (the real novelty). Use min-2D-SE community detection to partition the
  latent trajectory into **temporal subgoals**; the high-level planner plans over community transitions. This is
  "SE as structure," which we never built — targeted at long-horizon tasks where flat TD-MPC2 is weak.
- **J3 — SE-structured JEPA for offline/transfer** (paper-friendly). Frozen-encoder multi-task probes, no dense
  value; does SE's community geometry transfer better than plain/VICReg JEPA?

Decision rule: run J1 first (one box-day), gate the J2 build on its result, run J3 in parallel. No "JEPA+SE SOTA"
claim without a matched multi-seed win in a niche where structure isn't already redundant. Full design in
`PLAN_jepa_se_sota.md`.

## Discussion & Q&A (2026-07-08)

A working session of nine questions that sharpened the above.

**Primer — TD-MPC2's five losses, and why "consistency" *is* the world model.** Everything below hinges on knowing
what each loss does. TD-MPC2 encodes an observation to a (SimNorm-normalized) latent \\(z_t = \text{enc}(o_t)\\) and
trains four heads jointly:

| Loss | What it trains | What it computes | Removable? (our ablation + sufficiency) |
|---|---|---|---|
| **consistency** | encoder + **latent dynamics** | roll the dynamics open-loop \\(\hat z_{t+1}=\text{dyn}(z_t,a_t)\\) and match the *encoded* next obs \\(\bar z_{t+1}=\text{enc}(o_{t+1})\\) (stop-grad): \\(\lVert \hat z_{t+1}-\bar z_{t+1}\rVert^2\\) | **removable on HopperHop (n=8); load-bearing on planner-led tasks −23/−38/−44%** |
| **value (Q)** | the critic | TD target \\(r+\gamma V(z_{t+1})\\), regressed by a Q-ensemble | **individually fatal, every task** (without it the agent can't even stand) |
| **policy** | the actor / planner seed | policy prior trained to maximize Q | **individually fatal, every task** |
| **reward** | reward head (planner scoring) | predict the (two-hot) reward \\(r(z_t,a_t)\\) | partial — the planner loses reward-scoring but the *policy still reaches full strength* |

The **consistency loss is the one that makes the latent dynamics *predictive*** — it is the JEPA-style
self-prediction objective (predict your own future latent), i.e. the part that turns the network into a *world
model* you can roll forward and plan over. The value/policy/reward heads are the RL + planning machinery, not a
predictive model of the world. That is precisely why we call consistency "the world-model loss," and why its
removability is the load-bearing question of Paper 4: it isolates *the world model itself* from the RL engine. The
answer — value/policy fatal, consistency the mildest cut — is exactly what says **the engine is the TD value
pathway, and the world model is a rollout-quality regularizer on top of it.** When you strip consistency, the
encoder+dynamics still receive gradients *through* the value/reward/policy losses, so the latent stays
value-relevant; the dynamics just stop being explicitly self-supervised to predict — which costs accuracy only on
tasks whose return depends on precise multi-step rollouts (the dense/planner-led ones), not on HopperHop.

**Q1 — For Bet A, the two tables just show planning gets *higher return*. How did we conclude that's exploitation,
not exploration?** Because "higher return" and "more exploration" are different axes, and we measured the
exploration axis directly. Exploration = discovering new states / finding the reward (measured by state coverage
and reward-discovery step); exploitation = executing the found behavior for higher return. Two measurements
separate them: **(a) coverage.** On WalkerRun, planning gets higher return yet *covers less* state late in training
(distinct bins 464 vs 501, entropy 4.81 vs 4.86) — it concentrates onto the good manifold, the signature of
exploitation, not exploration. **(b) discovery step.** On the decisive sparse task (CartpoleSwingupSparse), the
exploration claim predicts planning finds the reward *first*; instead both find it 3/3 and **policy-only finds it
earlier** (140k vs 157k). So the higher return (454 vs 239) is entirely *post-discovery* — planning exploits the
found reward and stays stable while policy-only collapses late. Return↑ with coverage↓ and no discovery advantage =
exploitation. The tables show planning wins; the *coverage + discovery-step* numbers show it wins by exploiting.

**Q2 — What is "collapse" in JEPA, and how is it measured — is it R²?** Collapse is the trivial solution a
self-predictive encoder can fall into: map every input to the *same* (or a very low-dimensional) latent, so
"predict your own future latent" becomes trivially perfect (predict a constant) while the representation carries no
information. We measure it two ways, and they can disagree: **(1) effective rank** of the latent covariance — how
many dimensions the representation actually uses (healthy = high; collapsed → ~0–1; our nav latent hit ~1e-7); and
**(2) frozen-encoder readout R²** — ridge-probe how well you can decode geometry (qpos) or value (return-to-go)
from the frozen latent (collapsed = low R²). R² is *one* of the two. Crucially they diverge: uniformity *maximizes*
eff-rank but *lowers* readout-R² — so eff-rank alone is a misleading collapse metric, and readout-R² (usable
information) is the one that matters.

**Q3 — What tasks do the JEPA variants actually benchmark on?** They are overwhelmingly *perception* benchmarks,
not RL control:

| Model | What it is | Benchmark tasks |
|---|---|---|
| I-JEPA | image SSL | ImageNet-1k linear-probe / low-shot, CIFAR, Places, iNaturalist |
| V-JEPA | video SSL | Kinetics-400 (82.1%), Something-Something-v2 (71.2%), AVA, ImageNet |
| V-JEPA 2 / 2-AC | video world model + action-conditioned | video understanding + **robot manipulation via visual MPC** from goal images (reach 100%, manipulation 60–80%; 62h robot data) |
| DINO-WM | frozen DINOv2 features + latent dynamics | **zero-shot planning** via MPC/CEM: PointMaze (0.98), Push-T, Two-Room, Wall (0.96), Rope/Granular manipulation |

The pattern is decisive for our scoping: JEPA's home is perception + *goal-conditioned* planning on manipulation/
nav — **never dense-reward DMControl locomotion** (Cheetah/Walker/Hopper), which is exactly TD-MPC2's turf. That is
*why* SE-on-TD-MPC2-control nulled, and why our JEPA+SE plan targets the goal-conditioned/transfer niche.

**Q4 — What "task" is JEPA doing, and do recent works drive it with a classical planner?** JEPA is a
*representation-learning objective* (predict a masked/future latent), not an RL algorithm — it yields an encoder.
To act, you add something on top, and **yes: the recent control-JEPA works use a classical planner, not RL.**
V-JEPA 2-AC and DINO-WM both learn a (frozen-encoder) latent dynamics model and plan with **MPC / CEM** over it;
LeCun's H-JEPA vision is explicitly *planning* (energy-based inference), not reward-maximizing RL. They largely
*avoid* the learned value/policy that TD-MPC2 relies on — which, given our finding that the value pathway is the
engine, is precisely why they target goal-conditioned tasks (a goal-image distance substitutes for value) rather
than dense-reward control. **How *we* used JEPA:** our H-JEPA arm (#51–#57) built a faithful decoder-free
hierarchical JEPA — encoder + latent predictor + EMA target + VICReg anti-collapse, a 2-level model with a
high-level latent-subgoal — and drove it two ways: a reactive high-level selector, and a **high-level latent-MPPI
planner** (the same "JEPA + planning" recipe the field uses). We evaluated it on PandaPickCube (multi-seed **null** —
the bottleneck was the low-level *motor primitive* that never learned to reach, not the abstraction) and on
closed-loop nav (the collapse studies). Separately we used JEPA as a pure *representation probe* — frozen-encoder
geometry/value readouts (D2/D3) — and added SE/uniformity as anti-collapse *terms* on both pure-JEPA and TD-MPC2's
latent (D1). So across the program JEPA served as (i) a hierarchical controller with latent planning, and (ii) a
representation-quality probe — never as a stand-alone RL learner.

**Q5 — What RL does TD-MPC2 use — SAC, PPO, on- or off-policy?** **Off-policy**, replay-buffer, and *not* PPO or
SAC — it's its own model-based method whose model-free core is a **TD actor-critic** (a TD-learned Q/value ensemble
+ a policy prior), closest in DNA to an off-policy deterministic-policy actor-critic (SAC-family), plus a learned
world model and **MPPI planning** at deploy. So the contrast in our benchmark is three-way: PPO (on-policy) hits the
wall, SAC (off-policy) partly clears it, TD-MPC2 (off-policy TD actor-critic + planning) clears it best — and our
ablation shows the TD value + policy losses are the engine, consistent with "off-policy TD actor-critic." *Briefly,
what "TD actor-critic" means and how the three differ:* a **critic** learns a value/Q by **temporal-difference
bootstrapping** (\\(Q(s,a)\leftarrow r+\gamma Q(s',a')\\) — it updates its estimate toward its *own* next-step
estimate), and an **actor** (policy) is trained to maximize that critic. **PPO** is an actor-critic too but
**on-policy** and **policy-gradient**: no replay buffer, it learns from fresh rollouts with a clipped policy update
and Monte-Carlo/GAE returns (no TD-bootstrapped Q) — stable but sample-hungry. **SAC** is an **off-policy TD**
actor-critic: replay buffer, a bootstrapped soft-Q, and an entropy-regularized stochastic policy — sample-efficient.
**TD-MPC2** shares SAC's DNA (off-policy, replay buffer, TD-learned Q-ensemble + policy prior) but adds a learned
world model and swaps stochastic exploration for **MPPI planning** at deploy. So the axis that matters for the wall
is on-policy (PPO, throws data away) vs off-policy TD-bootstrapping (SAC/TD-MPC2, reuses data and can plan) — which
is why the two off-policy methods clear the contact-critical wall and PPO does not.

**Q6 — Why didn't abstraction help complex / long-horizon tasks as expected, only HopperHop / nav?** A subtlety
worth stating plainly: our "complex" tasks (locomotion) are *dense-reward*, so the TD value pathway already does the
credit assignment and extracts a value-sufficient latent → structure is redundant. The regime where abstraction /
hierarchy *theory* predicts a win is **long-horizon AND sparse-reward AND no competent primitive** — where value
credit alone can't bridge the horizon. We never cleanly tested that: our sparse tasks were short-horizon
(Cartpole) or primitive-bottlenecked (PandaPickCube), and our long-horizon tasks were dense. So it isn't that
abstraction *failed* on the regime it should help — it's that **we haven't tested that regime yet.** That gap is
exactly what J2 (SE-hierarchy on a genuinely long-horizon sparse task like AntMaze) is for; it's the honest
steel-man for continuing.

**Q7 — What is J3, and why tie it to offline / transfer learning?** In *online* value-based RL the dense TD signal
makes the representation value-sufficient, so structure is redundant — the wrong place to demonstrate that SE
helps. In *offline / transfer* you learn the representation self-supervised on a fixed dataset with **no downstream
reward pulling it**, then evaluate transfer (frozen-encoder linear probe across tasks). There, nothing preempts
structure, so if SE's compact community geometry transfers better than plain/VICReg JEPA, that's a genuine win —
*and* it matches where JEPA methods are actually evaluated (I-JEPA/V-JEPA on ImageNet/Kinetics transfer). J3
relocates the SE claim to a regime our own redundancy result doesn't already close.

**Q8 — HopperHop is simultaneously the sharpest PPO wall *and* the only removable cell. What's the hypothesis, and
what paper/proposal?** The hypothesis is a **two-axis model of task difficulty**: an *exploration* axis (how hard to
*find* the behavior) and an *execution* axis (how much precision/planning to *execute* it for high return).
HopperHop is high-exploration (contact-critical → PPO wall) but low-execution (a low-dimensional limit-cycle gait
the policy executes without accurate multi-step rollouts → world-model removable); Walker/Cheetah/Acrobot are
high-execution (their return *level* needs the planner's rollouts → world-model load-bearing). **Formal claim:** the
world-model loss is removable exactly on exploration-hard-but-execution-simple tasks, *orthogonally* to the wall,
which lives on the exploration axis. **Paper:** "Two axes of task difficulty for model-based RL: exploration vs.
execution, and what each demands of the world model" (or a sharp section in the dissection paper). **Proposal:** a
task grid crossing the two axes, with two probes — policy-only-vs-MPPI at matched weights (execution axis → predicts
removability) and PPO escape rate (exploration axis → predicts the wall) — showing the two splits are predicted by
*independent* task properties. That turns a descriptive coincidence into a predictive law.

**Q9 — The two directions, five bullets each (state → todo):**

*Direction 1 — the HopperHop / two-axis study* (upgrade Paper 4 from descriptive to predictive):
- **State (detailed):** we have a five-task sufficiency grid at n=4–12: training the consistency loss OFF from
  scratch, HopperHop matches the full model (stripped 165/475/481/511 + 306/449/443/477 = n=8, 7/8 inside the
  full band 420±113), while WalkerRun (−23%), CheetahRun (−38%) and AcrobotSwingup (−44%) all lose non-overlapping,
  and CartpoleSwingupSparse both-fails. Independently, HopperHop is the sharpest PPO wall (0/5 ≥200 at 472M) and
  the clearest env where TD-MPC2 beats *both* PPO and SAC. Those two facts sit in tension — the hardest env for
  PPO is the one where TD-MPC2's world model is dispensable — and the *cause* is only a hypothesis
  (exploration-hard-but-execution-simple), never mechanism-checked. This is exactly the tension the new Part 12
  post dissects.
- **Todo A — the decisive probe:** policy-only vs MPPI at matched weights, Hop vs the dense tasks — if Hop is π-learnable, policy-only reaches ~full return on Hop but falls short elsewhere. Reuses the FORCE_CK harness; ~1 box-day.
- **Todo B — the rollout probe:** k-step open-loop latent rollout-error, stripped vs full, per task — expect stripped-Hop to stay accurate (periodic gait) while stripped-Walker's error explodes.
- **Todo C — behavioral characterization:** measure action-sequence periodicity / control-precision of the optimal policy per task, to *predict* the split from task structure rather than from the ablation.
- **Todo D — write it up:** a task grid crossing exploration × execution axes → the "two axes of task difficulty" paper section/proposal.

*Direction 2 — JEPA + SE, correctly scoped* (structure where the value pathway isn't already doing the job):
- **State (detailed):** across the D-thread we showed a *pure* JEPA does **not** collapse on broad DMControl state
  (none-arm readouts geom 0.795 / value 0.304, above raw-obs) or on pixels (30 runs), because the predictor+EMA
  (BYOL) asymmetry already prevents it; adding uniformity/SE there *hurts* (0.795→0.583). Collapse (eff-rank → ~1e-7)
  appeared in exactly one regime — the closed-loop online goal-conditioned nav H-JEPA — where anti-collapse helped
  (point-maze 0.53→0.95). So we closed *anti-collapse-as-a-penalty* (regime-dependent) but never tested
  *SE-as-structure*, and — a real gap — **we only ever benchmarked on our own DMControl/Panda tasks, never on the
  perception/goal-conditioned tasks JEPA was actually designed and evaluated for.** The nav-collapse *trigger* is
  also unresolved.
- **Todo J-bench — evaluate on JEPA's home turf + find real collapse tasks (added):** run the SE-JEPA on the tasks
  the JEPA literature uses (ImageNet/CIFAR-scale linear-probe for I-JEPA-style; PointMaze / PushT / goal-image
  manipulation for the V-JEPA-2-AC / DINO-WM planning setup), and deliberately *hunt for more environments where the
  latent genuinely collapses* (sparse goal-conditioned, narrow-data, closed-loop) — the only regime where any
  anti-collapse or SE-structure can pay off. A fair test of SE must be on tasks JEPA is meant for, not on
  dense-value control where every structure is redundant.
- **Todo J0 — collapse trigger (do first):** anchor-strength A/B — add a graded dense auxiliary target to the nav
  H-JEPA, crossed with online/offline; tests "dense value/state anchor prevents collapse." Tells us whether
  SE-structure even has a collapse to fix.
- **Todo J1 — SE-community vs uniformity/VICReg** *in the collapse regime only:* matched head-to-head (none/unif/vicreg/SE, fixed-λ) on the tasks where the latent collapses. Finishes open cell #59; ~1 box-day.
- **Todo J2 — SE *as* the hierarchy (the real novelty):** use min-2D-SE community detection to *define* H-JEPA temporal subgoals; the HL planner plans over community transitions on a long-horizon *sparse* task (AntMaze) — the untested regime where abstraction *should* win. Needs a build (~2–3 box-days); gated on J0/J1.
- **Todo J3 — offline/transfer (parallelizable):** SE-structured JEPA vs plain/VICReg JEPA on frozen-encoder multi-task probes, where no dense value makes structure redundant — the regime JEPA is actually evaluated in.

**The unifying frame:** structure can only help where the *value pathway isn't already doing the job* — so both
directions deliberately live *outside* dense value-based control: HopperHop probes *why* the world model is
dispensable there; JEPA+SE probes the goal-conditioned / long-horizon / offline regimes where structure still has a
pulse.

Sources for the JEPA landscape: [V-JEPA 2 (Meta AI)](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/),
[DINO-WM (arXiv 2411.04983)](https://arxiv.org/html/2411.04983v2),
[V-JEPA (OpenReview)](https://openreview.net/forum?id=WFYbBOEOtv).

## This week's TODO (from the Q&A)

A consolidated, prioritized worklist distilled from everything above. Running items first, then the probes the Q&A
surfaced.

**In flight (no new work):**
1. **Bet-3a bisimulation sweep** (value-conditioned abstraction) — CheetahRun, coef {0.1, 0.5} vs vanilla 855;
   nan-smoke passed. Verdict at 5M: a beat → abstraction-SOTA forming; a null → the redundancy result extends from
   *added* structure to *value-conditioned* structure.
2. **Bet-3b value-sufficient bottleneck** — build + run *iff* bisim nulls (\\(z=[z_v,z_r]\\), Q/π read only
   \\(z_v\\)).
3. **URC-Walker clean 5M** — finishing on the slow box; harvest for the citable second-task number.

**Direction 1 — HopperHop / two-axis (cheap, sharpens Paper 4), interleave between SOTA checkpoints:**
4. **π-only vs MPPI at matched weights**, Hop vs Walker/Cheetah/Acrobot — the decisive removability probe (~1 box-day).
5. **k-step rollout-error, stripped vs full**, per task — the mechanism behind the split.
6. Draft the **"two axes of task difficulty"** section/proposal once 4–5 land.

**Direction 2 — JEPA + SE (correctly scoped), start the cheap cells:**
7. **J0 anchor-strength A/B** on nav H-JEPA (graded dense target × online/offline) — resolves the collapse trigger;
   prerequisite for J2.
8. **J1 SE-community vs uniformity/VICReg** in the collapse regime — finishes open cell #59 (~1 box-day).
9. **J2 SE-as-hierarchy** build (min-2D-SE subgoals → HL planner on AntMaze) — *gated* on J0/J1 (~2–3 box-days).
10. **J3 offline/transfer** SE-JEPA probe — parallelizable on a free box, no RL loop.

**Lower priority (confirms rather than opens):**
11. **Bet-B controlled taxonomy** — one prior family, vary DOF-overlap only, show the sample-efficiency multiplier
    is monotone in overlap (turns scattered evidence into a citable predictive law).

**Docs:** fold the five-loss primer, the two-axis hypothesis, and the reweighting-null into the papers; keep the
Part 11 living index and `FUTURE_WORK_open_mechanisms.md` current.

Sequencing rule: cheap probes (4, 5, 7, 8) interleave on whichever box frees first; builds (2, 9) are gated on the
prior cell's result; nothing claims a win without a matched multi-seed beat.

## What's next

Two tracks, run without competing for the same claim. **(1) Value pathway:** the value-conditioned abstraction bet
(bisimulation, running now) tests structure inside the value head on control tasks — the redundancy wall.
**(2) Representation pathway:** the JEPA+SE plan above tests structure in the encoder on goal-conditioned /
hierarchical / transfer tasks — where structure already showed a pulse. If both null, the program's thesis is
airtight; if the SE-hierarchy (J2) wins, it is the abstraction-SOTA, correctly located *outside* the
value-redundant regime. Either way it is an honest result, which after a week like this one is the point.
