---
layout: post
title: "TD-MPC-Glass, Part 9: The Anatomy of \\"Beating PPO\\" — What Survived Every Control"
date: 2026-07-04
description: "The positive-chase ended with every headline claim rewritten by its own control experiment — and a sharper, truer story than any of the originals. The exploration wall is real but it is an on-policy pathology specific to gait discovery; SAC escapes it; the world model's robust advantages are consistency and ~4–5× sample-efficiency, not level; the load-bearing net is the TD value signal, not the self-predictive dynamics loss; and the hierarchy win was dense shaping in disguise. A capstone with every number disk-verified."
---

> The capstone of the positive-chase ([Part 8](../2026-07-04-tdmpc-glass-part8-chasing-positives)). This week we
> ran the control experiment for every claim we wanted to make — an off-policy baseline, a shaped-flat baseline, a
> config-bug fix, a 4× budget extension, and a per-loss ablation — on byte-identical environments. Every control
> changed its claim. What's left is smaller than what we started with, and solid. All numbers trace to JSONs/logs
> in the campaign ledger; two corroborative cells are still running (marked).

## The final scorecard

| claim as originally hoped | what the control said | final form |
|---|---|---|
| "TD-MPC2 beats PPO by exploring better" | plan-vs-π: planning isn't the explorer; SAC: model-free escapes too | **categorical wall on the hop only; graded barrier on Stand — reliability×efficiency ordering TD-MPC2 ≫ SAC ≫ PPO on hopper dynamics** (revised twice on 07-03 as n grew) |
| "the world model is the exploration lever" | per-loss ablation | **the TD *value* signal is the lever; the self-predictive loss is the least critical piece** |
| "world model buys 2× higher level" | SAC at 4× budget (n=5) | **no clean level gap — consistency + ~4–5× sample-efficiency** |
| "learned hierarchy beats flat" | shaped-flat control (n=6) | **dense shaping in disguise; the hierarchy's real trick is *self-generating* that signal** |

## 1. The wall: real, on-policy, gait-specific

On the identical MJX `HopperHop` (verified byte-for-byte — same registry env, episode length, action repeat):

- **PPO (tuned config): peak 53.8, 0/5 seeds ≥200, through 472M steps/seed.** Categorically walled.
- **SAC (tuned config): 3/3 seeds cross 200 within 5M.** The wall is an **on-policy pathology** — replay +
  entropy exploration escapes it.
- **TD-MPC2: ~282 at 1M (n=4, best 367); 5M bests 295–610, mean 420 ± 113 (final, n=12, 07-05)** — the
  flagship efficiency anchor at publication n: eight fresh seeds joined the original four (best 610, worst 295;
  a wide but honest spread).

And the wall does not generalize: FingerTurnHard PPO ≈973 (3/3, no wall); Pendulum solves once an upstream
config-case bug (`PendulumSwingUp` vs `PendulumSwingup` silently skipping the tuned override) is fixed — peaks
842/831 (n=2); BallInCup is discovery-luck for *both* algorithms (PPO 1/3, TD-MPC2 1/2). *(Addendum, later that
day — a fourth control: **CheetahRun**, which our own older Pareto study had scored "PPO never reaches 500 at
30M." With the tuned config and adequate budget, PPO reaches **892–922 (3/3 seeds at 285M)**, matching SAC
(918/912 at 10M) — the old number was a budget/config artifact, corrected in the synthesis doc. CheetahRun is
slow-but-converging, not walled; TD-MPC2's edge there is purely efficiency, ~600+ within 0.55M steps.)*

**🚩 Framing revisions (2026-07-03, the wall-boundary probe — twice, because the controls kept teaching):** we
asked whether the wall is the *hop gait* or the *hopper robot*, running the same three arms on **HopperStand**
(same morphology, easier objective). At n=2 PPO looked walled there too (149/142 @285M) and we briefly framed the
wall as morphology-categorical. **Extending to n=4 corrected that:** PPO Stand = 149 / 142 / 144 / **681 — one
seed in four escapes**; and SAC Stand = 492 / 754 / 464 / **33 — one seed in four fails**. The honest final
picture, with five no-wall control tasks:

- **HopperHop: a categorical on-policy wall, budget-indexed for the rest** — PPO **0/5 ≥200 at 472M**; SAC crosses
  200 in **6/12 seeds by 5M (final, 07-04) and 6/9 by 8M** (final 07-05: the 20M cohort crossed 5/5 at
  4.1–7.7M, but four fresh direct-8M seeds went 1/4 — 374 vs 117–140 — SAC's crossing by 8M is a ~2/3 rate,
  not a certainty); TD-MPC2 **6/6 by ~1M**. *(07-05: the wall also survives the standard exploration knob — entropy_cost ×3 and ×10 leave
  PPO at peaks 4–74, n=2 each at 150M, no better than baseline; under-exploration hyperparameters are not the
  explanation. Instructive contrast on the **graded** Stand barrier below (n=4/arm final): entropy ×3 put 1/4 seeds
  over — 627 vs 118–150 — a rate consistent with the baseline's 2/16 escape lottery, not a repair; ×10 walled
  all four (116–185). The knob neither repairs the graded barrier nor dents the categorical wall.)*
- **HopperStand: a graded but near-categorical barrier** — PPO escapes **2/16** (final, 07-04: eight more
  seeds at the full 285M budget ALL walled at 105–195; both escapes — 681/749 — came from earlier 120M runs);
  SAC
  **0/3 at 1M and 7/10 by 5M** (final, 07-04: solves 464–932; fails 33/100/300); TD-MPC2 **3/3 by ≤0.9M**
  (962 / 948 / 943 — the third seed by just 0.3M). The final three-method gradient on Stand — **TD-MPC2 3/3 ≫
  SAC 7/10 ≫ PPO 2/16** — and the matched-1M column (TD-MPC2 solves both hopper tasks at a budget where neither
  model-free method solves either) are the paper-grade statements.

So the durable, defensible claim is a **reliability × efficiency ordering on hopper dynamics — TD-MPC2 ≫ SAC ≫
PPO, with orders-of-magnitude budget gaps** — hardest exactly where the dynamics are unstable and
contact-timing-critical, categorical only for the hop itself. Both of today's framings ("gait-specific", then
"morphology-categorical") died under one more seed each; this one is stated at the n we actually have.
*(TD-MPC2 Stand later firmed to **5/5**: 962/948/943/959/959, all within 1M.)*

**Discriminator (07-04 evening, AcrobotSwingup — unstable but *contact-free*; all cells n=4 by 07-05):** PPO
learns fine (267–344 at 285M) — **no wall**, so the on-policy wall requires contact-criticality, not
instability alone. And SAC is slow *and* inconsistent here even at 20M (42/66/123/207; 54–89 at 5M) — the
off-policy advantage is also task-class-specific, not a generic superiority. **TD-MPC2 leads on both task
classes (422–454 within 1M, n=4)** — of the three methods, the value-pathway advantage is the most portable.
*(Completing the humanoid matrix: SAC also solves HumanoidStand, 918/922 — off-policy SAC solves both humanoid
tasks while PPO can't run and TD-MPC2 diverges on both.)*

## 2. Level vs efficiency: the 4× budget test

If the world model bought a higher *level*, SAC shouldn't reach TD-MPC2's plateau at any reasonable budget. At
20M steps (4× TD-MPC2's), SAC's finals are **246 / 277 / 285 / 414 / 572** (n=5; peaks to 574). The mean (~359)
sits below TD-MPC2's ~477, but the best seeds reach or exceed it — the distributions overlap. So:

> **The world model's robust advantages are *consistency* (TD-MPC2: 4/4 seeds ≥282 at 1M; SAC: wild seed spread
> even at 20M) and ~4–5× sample-efficiency. Not a level ceiling.**

This is the campaign's own law — "abstraction buys speed, not ceiling" — arriving from a fourth independent
direction, and it agrees with where the field has landed (BRO, SimBa, TD7, CrossQ: scaled/regularized model-free
methods keep erasing model-based "level" gaps; the efficiency gap is the durable one).

## 3. Which part of the "world model" does the work? The value signal.

Per-loss ablation (zero ONE loss term, from scratch, mask verified live in logs; MPPI + π-only eval returns):

**CheetahRun (decisive arms at n=4 as of 07-04, 1M):** full = 738/782/721/795 (MPPI bests). Ablate **value →
16/37/58 — dead 4/4** (π ≤12). Ablate **reward → MPPI dead at 5/31/26 but π at FULL strength 761/796/805 — n=4** (planning-only, partly by
construction — MPPI scores rollouts with it). Ablate **policy prior → MPPI limps at 123/192/141 while π is dead
at 2.5/9/11 — n=4.** Ablate **consistency — the self-predictive latent-dynamics loss itself → 367/516/558 MPPI
(π 541/575/623), the smallest drop of the four — n=4.** *(07-04: with these, every arm of the five-loss ablation
is at n≥4 on all three tasks.)*

**HopperHop — the exploration task (all five arms at n=4; the headline value cell firmed to n=6 on 07-03):**
full finds the gait on 6/6 seeds (MPPI best 287–570). **Value-ablated: 0.0 / 0.0 / 3.2 / 0.0 / 0.0 / 0.1 (n=6) —
the gait is *never* found. Ablating the TD value loss reproduces the exploration wall, six seeds out of six.** Consistency-ablated still finds it
at roughly half strength on 4/4 seeds (MPPI 185–245). Reward-ablated matches CheetahRun's pattern: the planner is
dead by construction (MPPI ≈0 — it scores rollouts with that head) but **the policy still learns the gait**
(π 519 / 241 / 226 / 189, n=4). Policy-ablated is ≈0 on *both* readouts (n=4) — and here HopperHop is *harsher*
than CheetahRun (where the planner alone still limped to 123): on the exploration task the planner cannot
compensate for a missing prior at all. So the precise statement: **the value-learning pathway — the TD value loss
and the policy trained from it — is individually necessary (each ablation reproduces total failure); the reward
head matters only for planning; and the self-predictive consistency loss is the only component whose removal
merely degrades.**

**Replicated on a third task (WalkerRun, decisive arms at n=4, 1M):** full 731/680/699/723 (MPPI best);
**value-ablated 56/28/39/38 — dead 4/4; policy-ablated 76/64/83/53 — dead 4/4** (seeds 3/4 dead through ≥900k);
reward-ablated MPPI dead (44/44) but π at full strength 711/728/681/684 (n=4); consistency-ablated 547/522 &
533/483 (π to 667) — the mildest cut yet again, now n=4. The mechanism table now spans three tasks — **CheetahRun
n=4, HopperHop value-cell n=6, WalkerRun n=4 on every arm** — with the same shape everywhere.

**Sufficiency grid (revised 07-07): removability is the exception, not a task-class rule.**
Training consistency-OFF from scratch at the full 5M budget, four tasks in:
- *HopperHop (exploration-bound)*: stripped **165/475/481/511** vs full 420 ± 113 (n=12) — three seeds at the
  top of the full band, reaching 440+ by ~2M. Removable for typical seeds; residual role = worst-seed insurance.
- *WalkerRun (dense)*: stripped **537/574/554/594 (mean 565)** vs full **709/705/753/782 (mean 737)** — −23%,
  non-overlapping.
- *CheetahRun (dense)*: stripped **527/528/516/524 (mean 524)** vs full **903/904/782/806 (mean 849)** — −38%,
  non-overlapping.
- *AcrobotSwingup (exploration-flavored, 07-07 morning)*: stripped **297/233/352/256 (mean 284)** vs full
  **533/511/513/488 (mean 511)** — **−44%, non-overlapping**.
Honest revision: our 07-06 "exploration vs dense" split — written when only Hop and Walker were in — **breaks
on Acrobot**, which is exploration-flavored yet shows the *largest* gap. The reading that survives all four
tasks: the consistency loss supports MPPI rollout quality wherever *the planner carries learning* (Acrobot,
Walker, Cheetah are all planner-led here); HopperHop — where the policy head learns the behavior directly and
the planner mostly amplifies it — is the removable case. CartpoleSwingupSparse was queued as the fifth test,
but the full-model baseline itself stalls on this sparse task (bests 0.0/0.0/1.3/0.0 at 5M, n=4 — a known
TD-MPC2 failure mode with default exploration), so that cell is a degenerate both-fail rather than an
informative sufficiency probe; the grid's evidentiary core stays the four tasks the full model solves. The HopperHop removable cell now
holds at **n=8**: four fresh stripped seeds land at 306/449/443/477 (mean 419), so 7 of 8 stripped seeds sit
inside the full-model band 420±113 — removability confirmed, with the lone 165 floor-dip the only exception.

**Replicated on a FOURTH task (HopperStand, n=4/arm, 07-05) — a task the full model solves in 0.3M steps:**
none 911–946 (all solve); **value-ablated 6–13 — dead; policy-ablated 9–34 (π ≤5) — dead**; reward-ablated MPPI
degraded to 265–542 but **π at 944/515/943/944**; consistency-ablated 816–898 (π to 930; one seed at eval 13/20
when written) — here the mildest cut is nearly no cut at all. Same shape, fourth task: **without the TD value
loss, TD-MPC2 cannot even learn to stand.**

> **"The world model explores" decomposes into: the TD value signal trained through the latent is what discovers
> and ranks behavior; the self-predictive dynamics loss is a helpful regularizer, not the key.** This rhymes with
> the entire campaign — from the redundancy nulls to the anti-collapse reversal, the value anchor keeps being the
> thing that matters.

## 4. Hierarchy: the win was the signal, not the structure

The fourroom result (feudal 4/6, sparse-flat 0/6) survived n=6 — and then the right control killed its
interpretation: **flat TD3 + dense potential shaping toward the goal solves 3/6** — statistically the same as
feudal. What carried the win was *density of learning signal*, not hierarchy. The surviving claim is narrower and
still interesting: **the feudal agent manufactured that dense signal itself** (its low level trains on distance
to *self-proposed* subgoals; no privileged goal information), whereas the control had to be handed the true goal.
This independently replicates Nachum et al. (2019): most of HRL's measurable benefit reproduces on flat agents
given better exploration/shaping.

## 4b. Addendum (07-04): does the ordering generalize? The humanoid says no — it *inverts*

The hopper result predicts "TD-MPC2 ≫ SAC ≫ PPO wherever dynamics are unstable and contact-critical." We tested
that prediction on a second unstable morphology, **HumanoidWalk** (21-DoF), same three-arm design. The prediction
failed — informatively. Every method broke, each along a **different axis**:

- **PPO: numerically unusable — reward=nan under all 3 configs tried** (brax defaults; a tuned-WalkerRun
  transplant with 512³ networks + reward_scaling 0.1; same + reward_scaling 1.0 + lr 1e-4). Notably there is *no
  official tuned humanoid config* in mujoco_playground's DMC params — the fragility is upstream reality, not our
  bug.
- **TD-MPC2 (final, 9 runs): under the default config, training diverges to loss=nan on 6/6 walk seeds (onsets
  0.53–2.51M) and on the easier HumanoidStand (0.28M) — 7/7, effectively deterministic with variable onset; best
  return before any nan was 30.4.** Lowering the update-to-data ratio shifts the onset stochastically rather than
  fixing it (k_update=32 diverged at 3.13M; k_update=64 was the *only* run of nine to finish 4M) — **and the
  nan-free survivor never learned anyway (best 21.8, flat ~20 for 4M steps).** So the humanoid failure is not
  merely numerical: under every setting tried, TD-MPC2 does not crack the 21-DoF task at these budgets. The same
  architecture solved hopper Stand at 0.3M.
- **SAC: the only robust method — 4/5 seeds solve (625–909 at 5M), 1/5 hit a nan of its own.**

So the reliability ordering is **task-class-dependent, and on the high-DoF humanoid it inverts**: plain
off-policy SAC beats both the on-policy and the model-based method — not on sample-efficiency but on *working at
all*. Config/numerical fragility is a distinct failure axis from exploration, and any "X beats PPO" claim needs
to say which axis it lives on. This sharpens rather than weakens the paper: the hopper ordering is real, matched,
and quantified — and it is a statement about hopper-class dynamics, not a universal law.

## 5. What this week actually taught

1. **Run the control before the claim.** Four headline claims went into this week; four came out rewritten by
   their own controls — every time into something smaller and truer. (The full adversarial review that mandated
   these controls, and the retraction bookkeeping, is in the campaign ledger and
   [the R²/Pareto explainer](../explainer-r2-pareto-analysis).)
2. **The durable positives of the whole program**: (a) explicit abstraction is redundant on a strong
   value-anchored latent (Paper A, submission-ready); (b) structured priors and world models buy
   sample-efficiency and consistency — never a ceiling — across priors, hierarchy, planning, and model-based RL
   (the speed-of-learning paper); (c) on-policy PPO has a real, reproducible, task-specific exploration failure
   mode that off-policy and model-based methods don't share — and its mechanism, in TD-MPC2's case, is value
   learning, not dynamics modeling.
3. **Complete.** All 20 HopperHop ablation runs finished (the mid-campaign host outage turned out to be
   network-only; the jobs never stopped). Nothing in this post is pending.

**What's next (07-05):** the 5M efficiency anchor is being extended to n=12 overnight (last wide-CI number);
then this post's content gets assembled into a conference paper (matched-env wall + budget-indexed rates +
4-task value-pathway mechanism + Acrobot/humanoid boundary results), with the redundancy-criterion paper
(Paper A) already at submission-readiness in parallel.

*Every number: campaign ledger `bet2_null_results.md` (disk paths per entry). Same-env verification, config
audits, and the two bugs found along the way (Pendulum case mismatch; an A1 noise confound) are documented in the
2026-07-02 full-arc review.*
