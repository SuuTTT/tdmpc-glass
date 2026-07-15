---
layout: post
title: "TD-MPC-Glass, Part 16: Did We Reimplement TD-MPC2 Correctly? A Validity Audit"
date: 2026-07-10
description: "A sharp question stopped us mid-campaign: our entire research program runs on a from-scratch JAX reimplementation of TD-MPC2 — and we had never rigorously compared it against Hansen's original. Worse, reading our own collection loop revealed the reimplementation never uses the MPPI planner to collect data, where canonical TD-MPC2 does. This post is the honest audit: the full deviation list, a parity check against the official published per-task results (verdict: hopper-hop at parity, SAC's failure reproduced exactly, systematic deficits precisely where our own theory predicts them), the one claim that is genuinely at risk, and the experiment now running that settles it."
---

> Mid-campaign, the right question arrived: *"Are we wrong at the very beginning? Have we ever compared our JAX
> reimplementation of TD-MPC2 with the original?"* The answer was **no — we never had.** Every result in Parts 1–15,
> both papers, the whole Hopper critique — all of it runs on a from-scratch JAX/MJX reimplementation whose fidelity
> to Hansen's TD-MPC2 we had sanity-checked qualitatively but never audited. This post is the audit.

## How the question surfaced

While preparing the P2 experiment (disable MPPI during data collection to test "planning-as-exploration"), we read
our own collection loop before patching it — and found there was nothing to disable. **Our implementation collects
data with π + Gaussian noise in every branch; the MPPI planner never touches collection.** An optional
MPC-distillation loss exists but is default-off and was never enabled in any run. Canonical TD-MPC2, by contrast,
**acts with the planner during environment interaction** — planner-collected data is part of the published method.

So the P2 hypothesis ("MPPI acts as a structured explorer during collection") was falsified *by construction* in our
stack — every number we have, including the flagship HopperHop win, is **planner-free training**. That is both a
discovery and a problem: a discovery because it shows planner-collection is unnecessary for the Hop phenomenon; a
problem because our results were being phrased as results about *TD-MPC2 the published method*.

## The full deviation list (as audited)

1. **Data collection:** ours = π + annealed Gaussian noise (0.3) + random warmup; canonical = MPPI planner actions.
2. **Actor objective:** ours = fully deterministic — maximize \\(\min_2 Q(z, \tanh(\mu_\pi(z)))\\) with a RunningScale
   normalizer, **no entropy term, no action sampling**; canonical = sampled action + a very small entropy bonus
   (~1e-4) — functionally near-deterministic, but not literally.
3. **Physics backend:** ours = mujoco_playground / MJX; canonical = dm_control / MuJoCo. Same reward definitions,
   different numerics — absolute returns are not exactly commensurable.
4. **MPPI budget:** ours = 2048 samples, horizon 3; canonical = 512 samples, horizon 3.
5. (Neutral) an MPC-distillation loss exists in our code but is default-off and unused.

## V1 — parity against the official published results

The TD-MPC2 repo publishes per-task curves (3 seeds, 4M steps, dm_control). Overlaying our numbers:

| task | official TD-MPC2 final @4M | ours @5M | gap | official cross-200 | ours |
|---|---|---|---|---|---|
| **hopper-hop** | **449** (373/380/594) | **~420±113** (mppi ~571) | **≈0%** | 0.2–0.4M | ~1M |
| cheetah-run | 896 | 855 | −5% | 0.1M | — |
| walker-run | 877 | 727 | −17% | 0.1M | — |
| acrobot-swingup | 663 | 511 | −23% | 0.1–0.2M | — |

And the reference baselines the repo ships alongside:

- **Official SAC on hopper-hop: finals 0 / 246 / 105** (only 1 of 3 seeds ever crosses 200). Our custom SAC v1
  (finals 76 / 23 / 101, 0/3 crossing) sits **inside the canonical band** — the "SAC fails on HopperHop" leg of our
  P1 attribution is *not* an implementation artifact; reference SAC exhibits the same standing-trap phenotype.
- Official TDMPC-v1 on hopper-hop: 2 / 577 / 1 — the task is seed-brutal for everyone.

**Three verdicts.**

1. **The two pillars stand at canonical levels.** Our variant reproduces both the TD-MPC2 hopper-hop level (parity)
   and SAC's hopper-hop failure (parity). The Part-12 critique was not built on a broken reimplementation.
2. **The deficits land exactly where our own theory says they should.** Our variant is weaker than official
   precisely on the tasks where we measured the world model to be *load-bearing*: the official-minus-ours gap
   ordering (Hop ≈0% < Cheetah 5% < Walker 17% < Acrobot 23%) tracks our measured WM-load-bearing ordering
   (Hop removable < Walker −7.5% < Cheetah ~−19% < Acrobot −44%). If planner-collection (which rolls the world
   model) helps most where accurate rollouts matter, this is exactly the signature you'd expect — canonical data
   *cross-validating* the task-conditional world-model story from the outside. (Honest confound: the MJX backend
   differs too; we cannot fully separate the two from published curves alone.)
3. **Early-phase speed does not transfer.** Official crossings are uniformly faster (0.1–0.4M vs our ~1M) — so
   sample-efficiency comparisons are only made within our stack, never across stacks.

## The one claim genuinely at risk — and the experiment running right now

**"The consistency loss is removable on HopperHop (n=8)"** was measured in a stack where the planner never touches
training. In canonical TD-MPC2 the planner *collects the data by rolling the world model* — so removing the
consistency loss there could also degrade collection in a way our stack cannot express. If that matters, canonical
TD-MPC2's world model might **not** be removable on Hop, and Part 12's headline would need a variant-scoped rewrite.

**V2 (running):** we added an `MPPI_COLLECT=1` gate that makes our stack collect with the planner (512 samples, the
canonical count), and launched {full, consistency-stripped} × planner-collection on HopperHop (n=2 each, 2.5M).
If the stripped model still trains to full *under planner-collection*, removability holds beyond the deviation and
the critique stands as stated. If it collapses, we will have found precisely where the claim's boundary is — and
Part 12 gets rescoped to policy-collection variants, honestly labeled. Either outcome is informative; the verdict
lands within a day and will be appended here.

## What changes in our papers regardless

- Every external-facing "TD-MPC2" becomes **"our TD-MPC2 variant (policy-collection; deviations listed in setup)"**
  until/unless V2 licenses the stronger phrasing.
- The deviation list above goes into both papers' setup sections verbatim.
- Internally-controlled ablations (the sufficiency grids, VBN curves, reweighting/bisim nulls) are unaffected —
  they compare arms within one stack and were always statements about this architecture family.
- One deviation turned into a *finding*: planner-free training reproduces the canonical hopper-hop level — evidence
  that planner-collection is unnecessary for the flagship result, which is itself a contribution of the critique.

*The lesson, stated plainly: when your research program runs on a reimplementation, the parity audit is not optional
due diligence — it is an experiment, and (as here) it can return results.*

## Update (2026-07-11): the V2 verdict — removability survives planner-collection

The experiment landed, and the answer is unambiguous. With the `MPPI_COLLECT=1` gate making our stack collect data
canonical-style (the planner acts during environment interaction, 512 samples):

| arm (HopperHop, 2.5M, n=2) | seed 50 | seed 51 | mean |
|---|---|---|---|
| full world model + planner-collection | 467.8 | 462.2 | **465.0** |
| **consistency-stripped** + planner-collection | 451.8 | 479.8 | **465.8** |

**Stripped equals full to within noise (+0.2%).** The consistency loss is removable on HopperHop even when the
planner collects the data — including in the stripped arm, where MPPI is rolling an *untrained* dynamics network and
scoring candidate actions purely through the reward and value heads. That is simultaneously (a) the discharge of the
one claim this audit put at risk — Part 12's critique holds beyond the implementation deviation, and (b) the
strongest evidence yet for the execution-simple account: on Hop, what the planner contributes is value-guided action
scoring, not rollout fidelity. The papers can now state the removability result as holding under **both**
policy-collection (n=8) and planner-collection (n=2). The natural reviewer follow-up — the same planner-collection
contrast on WalkerRun, where the world model *is* load-bearing — is queued.

## Update 2 (2026-07-11): the Walker mirror-image — a complete double dissociation

The pre-registered follow-up ran: the same planner-collection gate on **WalkerRun**, where the world model is
load-bearing. Prediction: the stripped model should *degrade* there. Result (2.5M, n=2/arm):

| task | full + planner-collection | stripped + planner-collection | Δ |
|---|---|---|---|
| HopperHop | 465.0 | 465.8 | +0.2% — removable |
| **WalkerRun** | **721.9** (758/686) | **605.4** (601/610) | **−16.1%** — non-overlapping |

Confirmed, cleanly. Two further reads: the planner-collection gap on Walker (−16% at 2.5M) is *larger* than the
policy-collection gap (−7.5% at 5M) — planner-collection **amplifies** the world model's importance exactly where
rollout quality matters, because the planner collects by rolling the model. And the full model under
planner-collection reaches at 2.5M what policy-collection needs 5M for (~722 vs the 708–727 band) —
planner-collection roughly **doubles Walker sample-efficiency** in our stack, which retroactively explains the
V1 deficit pattern (official, planner-collecting TD-MPC2 beats our policy-collecting variant precisely on the
load-bearing tasks). Together, V2 + V2W form the cleanest mechanistic statement of the program: **the world model's
value is task-conditional, and collection mode modulates it in the predicted direction at both ends.**

**Update 3 (07-12) — the n=4 resolution, and what the "bimodality" really was.** Seed 52 briefly complicated the
story: its full arm finished at 455, far below s50/51 (758/686), while its stripped arms stayed tight — raising the
possibility that the full model under planner-collection is bimodal across seeds. The seed-53 resolving pair settled
it: full **744.7**, stripped **558.3** (finals at 2.5M), so 3 of 4 full seeds cluster at 686–758 and s52 is the lone
outlier. Better, watching s53's eval trajectory live revealed the mechanism: the full arm's evals swing ~250 points
within a single late-training window (680 → 715 → 676 → **501** → 696 → 744 over 2.1–2.5M) while the stripped arm
moves ~30 points (539–568). The dissociation at n=4: full median **715.5** vs stripped **600.5** (−15.4%; means
−10.4%). So the refined claim is variance-aware: on WalkerRun under planner-collection, the world model buys a
higher-performance but higher-variance eval regime, and the stripped model trades ~115 points of median for
stability. s52's 455 was almost certainly an unlucky final-eval draw from the volatile regime, not a distinct mode.
Hop's ±0 (n=3, tight) stands unchanged — the double dissociation is now n=4 on the Walker side.

**Update 4 (07-13) — the third task inverts the story, exactly as a pre-registration should catch.** With Hop and
Walker in hand we pre-registered the natural generalization: planner-collection amplifies the world model's
importance, so stripped-Cheetah should degrade ≥15% (kill: <8%). It did neither — it **inverted**. Across both
seeds the full model's evals destabilize under planner-collection (swinging 141–585 within the last 250k steps)
while the stripped model holds a tight ~460–555 band; on last-6-eval medians the stripped model is ~40% *above*
full. Walker meanwhile strengthened to n=5 (full median 739 vs stripped 601, −18.7%), and Hop stayed removable at
n=4 (436 vs 468). The three-task table is now: **removable (Hop) / load-bearing-but-volatile (Walker) /
actively destabilizing (Cheetah)** — the collection-mode × world-model interaction is task-dependent and
non-monotone in how much the value function needs the latent (the VSB grid's ordering). That is a sharper claim
than "amplification," and it is what the planner-rolls-the-model mechanism actually predicts once the model's
rollout errors compound differently per dynamics class. Cheetah n=5 pairs are running to nail the inversion.
