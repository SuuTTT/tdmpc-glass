---
layout: post
title: "TD-MPC-Glass, Part 15: Revision Plan & Live Log — Integrating the New Findings into Papers A & 3"
date: 2026-07-09
description: "A working plan (and running log) for the next two days: fold the value-sufficiency curve, the bisim/reweighting/bottleneck nulls, the Hopper conjunctive-reward result, and the planning-vs-world-model dissection into our two existing papers — the redundancy-criterion paper (A) and the beating-PPO anatomy paper (3) — rather than spinning a fourth. It records the narrative upgrade for each paper, resolves the sharpest open question (if the world model is removable, what makes TD-MPC2 near-SOTA on HopperHop?), commits to a positive artifact per paper so these are not pure-null results, and lays out the exact experiments, GPU schedule, and time estimate. Updated in place as runs land."
---

> **📋 Living plan + log** (started 2026-07-09). We already have three drafted papers; the new findings from Parts
> 9–14 mostly *strengthen* two of them rather than justify a fourth. This post records the integration plan, the
> narrative upgrade, the one open mechanism question we still owe, the positive artifacts that keep these from being
> pure-null papers, and the experiment schedule. The **Live log** at the bottom is updated in place as runs complete.

## The decision: integrate, don't spawn

The findings we've accumulated — the value-sufficiency curve, the bisimulation / reweighting / bottleneck nulls, the
Hopper conjunctive-reward result, the planning-vs-world-model dissection — overlap almost entirely with two papers we
have already drafted:

- **Paper A — *"When Is Explicit Abstraction Redundant for a World Model? A Negative Criterion."*** The redundancy /
  value-sufficiency story.
- **Paper 3 — *"The Anatomy of 'Beating PPO': Exploration Walls, the Value Pathway, and the World Model."*** The
  dissection / attribution story.

(A third, *"Abstraction is a Speed-of-Learning Lever, not a Capacity Lever,"* is untouched by the new work.)

So the plan is to **upgrade A and 3** — and, importantly, to make each *more convincing and less purely-negative*.

## Narrative upgrade

### Paper A — from "our criterion failed" to "we built the right instrument"

Paper A already contains a strong, honest negative: the tempting way to certify value-sufficiency — a linear
decode-\\(R^2\\) — is **invalid** (it saturates at ≈0.98 for a strong *and* a collapsed policy; the return-to-go
\\(R^2\\) even *anti*-correlates with performance). But the arc ends deflating: "a valid probe is still open."

The **value-sufficient bottleneck (VSB)** closes exactly that gap. Instead of decoding value off the latent, force
the value and policy heads to read only the first \\(D\\) of the 512 latent dimensions and measure return\\((D)\\).
That is a *behavioral* measurement of how much of the latent the value pathway actually needs. New three-act arc:
1. **The trap** — why decode-\\(R^2\\) can't certify sufficiency (existing result).
2. **The instrument** — VSB: return\\((D)\\) is smooth, monotone, and distributed; *no small sufficient subspace*,
   and *no imposed structure beats the vanilla distributed latent*.
3. **The verdict on three legs** — redundancy holds because the latent is *behaviorally* value-sufficient: **theory**
   (the basis-change/data-processing proposition) + **instrument** (VSB) + **breadth** (the null campaign, now four
   taxa: added SE/graph, reweighted VAC/URC, metric bisimulation, architectural bottleneck).

### Paper 3 — from "a wall exists" to "here's why, and what the world model actually does"

Paper 3 characterizes the HopperHop PPO wall but does not explain it, and leaves the world model as merely "the
mildest cut." Two upgrades:
- **Explain the wall.** Reading the environment, HopperHop has **no early termination** and a **conjunctive,
  multiplicative reward** (`standing × hopping`). Env-gating the reward shows PPO **escapes under an additive reward**
  but **stays walled under the product even with an easier hop threshold** — so the wall is the *conjunction*, a
  benchmark-design property, not a capability limit (the Voelcker caveat, confronted head-on).
- **Resolve the world model's role.** Not "mildest cut" but a **task-conditional rollout regularizer** — *removable
  on the very task it is celebrated for* (HopperHop, n=8) and load-bearing only where precise multi-step rollouts set
  the return level (Walker/Cheetah/Acrobot).

## The sharpest open question (Point 1): if the world model is removable, why is TD-MPC2 near-SOTA on HopperHop?

Removing the *consistency loss* removes the objective that makes the latent an accurate **predictor** — it does not
remove the **planner**. TD-MPC2 differs from SAC in two things: MPPI planning, and the value architecture. Our
working hypothesis:

> TD-MPC2's HopperHop edge is **off-policy TD value learning + MPPI acting as a *structured explorer* during data
> collection** — not the predictive accuracy of the world model. On a conjunctive-reward wall the hard part is
> *finding* the joint standing+hopping behavior; MPPI's population-based, value-greedy action search finds it faster
> than SAC's local Gaussian noise (TD-MPC2 ~1M ≫ SAC ~8M ≫ PPO walls). The consistency objective is redundant because
> MPPI only needs the model to *rank actions by value*, not to roll out accurately.

This is **not yet proven** — it is the paper's key remaining experiment (**P2** below): TD-MPC2 with **policy-only
data collection** (MPPI disabled during rollouts). If it collapses toward SAC's slow curve, MPPI-as-explorer is the
edge; if it still beats SAC, the edge is the value architecture. Either way, this converts Paper 3's thesis from a
negative ("no level advantage") into a positive mechanism.

## Positive artifacts (so these are not pure-null papers)

- **Paper A → the VSB diagnostic.** A valid, checkpoint-cheap probe for value-sufficiency that predicts whether an
  abstraction objective can help on a given task/model — the correct replacement for the invalid decode-\\(R^2\\).
- **Paper 3 → the reward-conjunctivity benchmark knob + exploration-wall law.** The env-gated reward-structure
  variants (conjunctive ↔ additive) packaged as a controlled axis for probing exploration under conjunctive
  sparsity, plus the predictive statement *reward-conjunctivity → on-policy wall depth*.
- **Stretch (either) → a world-model-necessity predictor:** a checkpoint signal (k-step rollout-error / removability)
  that flags where the consistency loss is redundant — a compute-saving gate.

## Experiments, GPU schedule, estimate

A 5M TD-MPC2 run ≈ 5–6 h on a 3060 (4 concurrent/box); PPO (brax) is minutes. **b3060b cannot build HopperHop**
(mjx-warp dependency); **b3060 is the Hopper box**.

| # | Experiment | Paper | Box | Wall est. |
|---|---|---|---|---|
| A1 | VBN value-sufficiency curve → **n=5** (Cheetah/Walker/Acrobot) | A | b3060b | ~2 d |
| P4 | Sufficiency grid → **n=5** (pin removable/load-bearing %) | 3(+A) | both | ~1.5 d |
| P2 | **MPPI-as-explorer ablation** (policy-only collection) — *build + runs* | 3 | b3060 | ~1.5 d |
| P1 | **SAC-core isolation** (SAC actor-critic + WM/MPPI) — *build + runs* | 3 | b3060 | ~2 d |
| P3 | H3 seeds + 2nd conjunctive-reward task (n=5) | 3 | b3060 | ~1.5 d |
| A2 | *(stretch)* OOD/compositional value-decodability probe | A | b3060b | ~1 d |

**Two-day scheduling target:** Days 1–2 — builds (SAC-core, MPPI-disable flag) + launch A1 (b3060b) and P4 (b3060);
begin drafting both reframed abstracts + section skeletons. Then P2/P1/P3 roll behind on b3060, A2 behind A1 on
b3060b. **Honest total to n≥5 with the positive artifacts: ~1.5–2 weeks** including builds, reruns, and slack for
box flakiness. Critical path is **P2** (it answers Point 1).

## Live log

- **2026-07-09 15:00** — Plan posted. Launched **A1** (Cheetah VBN width sweep, seed 51 → toward n=5) on b3060b and
  **P4** (WalkerRun full-vs-stripped, seeds 70/71 → pins the world-model removable/load-bearing margin) on b3060.
  Both 4 arms up, 0 nan at launch. Next: build the **P2** MPPI-disable flag; extend A1 to further seeds/tasks;
  begin the Paper A VSB section and the Paper 3 wall-mechanism section. *(This log is updated in place as runs land.)*
- **2026-07-09 15:30** — GPU packing: b3060b doubled to **8 jobs** (added Walker VBN s51 alongside Cheetah s51,
  2/GPU, mem-fraction 0.35), 100% util both boxes.
- **2026-07-09 16:20 — CODE/EXP AUDIT (post-weekly review).** Four faults found and fixed:
  **(F1, data-integrity)** the `run_arm.sh` jsonl tag lacked the *task* name (`wmabl_<ablate>_s<seed>`), so the
  Walker two-axis (s62/63) appended into the *Hopper* s62/63 jsonl files — the ledgered Hop numbers were harvested
  *before* contamination (benchmark CSVs, which are task-named, remain the durable source), but this explains the
  garbage Walker π-vs-MPPI read, and a planned Cheetah batch on s70/71 would have corrupted the live Walker P4 data.
  **Fixed:** tag now `wmabl_<TASK>_<ablate>_s<seed>` (backup `.bak_tagfix`).
  **(F2, ops)** heredoc-through-compound-ssh silently failed twice — the Cheetah sufficiency batch *never launched*,
  and the earlier status report mis-attributed the 4-proc count to "compute saturation" instead of a failed launch.
  **Fixed:** printf-style drivers; Cheetah full-vs-stripped s70/71 now actually running (b3060 packed to 8 jobs,
  task-qualified tags).
  **(F3, claim precision)** Part 14's "PPO escapes the wall" overstated: the additive-reward curve climbs steadily
  to 135 at 20M (not plateaued) but is well below even the standing component's ~500 ceiling — the finding is
  *learnable signal vs flat zero*, not task-solved. Part 14 edited; `HOP_SPEED=1.0` margin-coupling footnoted
  (margin = speed/2, so that variant is threshold+margin, not threshold-only).
  **(F4, paper caveat)** VBN arms compare against van2 baselines run under earlier patch stacks; all env-gates
  (VBN/VAC/URC) verified present and default-off (byte-identical when unset) — to be stated in the paper's setup.
  Current fleet: b3060b 8 jobs (Cheetah+Walker VBN s51), b3060 8 jobs (Walker s70/71 + Cheetah s70/71
  full-vs-stripped), 0 nan.
- **2026-07-10 04:15 — A1 Cheetah s51 done (marker-gated harvest).** True 5M finals **535/582/617/723** (the earlier
  pre-completion read 594/572/636/742 is superseded — harvest-at-markers lesson re-learned). Corrected n=2 curve
  **516/572/628/738 vs vanilla 855**: *strictly monotone*, the earlier "16/32 near-tie" was a mid-fluctuation
  artifact. Cheetah VBN **s52 launched** on the freed GPUs (→ n=3).
- **2026-07-10 08:50 — P4 harvest (+ a b3060 incident).** All 8 b3060 jobs were killed ~08:20 by a host-level event
  (no reboot; ssh flapping). Walker died at 4.90M/5M, Cheetah at ~4.55M — harvested at last eval, truncation stated.
  **Walker load-bearing margin, n=4: −7.5%** (full 708.2 vs stripped 654.9; ranges non-overlapping) — the historical
  −23% is confirmed a seed/version outlier. **Cheetah (truncated, n=2): −19.2%** with wide seed spread (−28%/−9%);
  needs full-5M seeds for a paper number.
- **2026-07-10 08:55 — P2 pre-build discovery (changes the Point-1 experiment).** Reading the collection loop before
  patching it: **our TD-MPC2 implementation already collects data with π+noise — MPPI never touches collection**;
  the planner appears only in eval and in an optional MPC-distillation loss. So "MPPI as structured explorer" is
  falsified *by construction* in our stack — every TD-MPC2 result we've reported (including the Hopper 6/6-by-1M win)
  used policy-only collection. P2 is redefined: if the MPC-distill loss is default-on, ablate *it* (planning shaping
  the policy via imitation); if default-off, the attribution falls to the off-policy TD core itself and **P1
  (SAC-core isolation) becomes the decisive experiment**. Also a framing caveat for Part 12 / Paper 3: canonical
  TD-MPC2 *does* collect with the planner; our deviation is itself evidence (planner-collection unnecessary on Hop)
  but must be stated.
- **2026-07-10 10:45 — fleet incident, P1 leg 1, and the SAC-rescue arm.** (i) *Incident:* all jobs on both boxes
  were killed ~08:00-08:30 (no reboot, no OOM record; external cause unknown; the box-local keep-busy cron that had
  double-booked GPUs overnight is now disabled). Everything harvested-honest and relaunched: Walker+Cheetah VBN s52
  on b3060b, Cheetah sufficiency s72/73 full-5M on b3060. (ii) **P1 leg 1 (SAC baseline, n=3 @5M): SAC-default never
  reaches 200 on HopperHop** (76/23/101) — its auto-tuned entropy coefficient collapses to ~0.003 and the agent
  stands forever; stripped-TD-MPC2 on the same stack is 8/8 ≥200. With the planner-free-training discovery, the
  attribution sharpens to: **the TD value/actor core itself is the edge.** (iii) *Honesty check in flight:* our SAC
  is a custom v1 with target-entropy −0.5·|A| (half of canonical) — a **rescued-SAC arm** (α-floor 0.05,
  canonical −1.0·|A| target) is now running (n=3); if rescue works, the story is "entropy-collapse under conjunctive
  reward," a knob-level failure — not "SAC cannot hop." Either way it pins the mechanism.
- **2026-07-10 14:20 — P1 COMPLETE: the entropy needle (Point 1 answered).** Three SAC arms (n=3 each, 5M, same
  stack): auto-α *collapses* to ~0.003 and lands in the **stand-trap** (76/23/101 — SAC's best config, standing but
  never hopping); a fixed α-floor of 0.05 (canonical target-entropy) lands in the **noise-trap** (≈0 — persistent
  objective-stochasticity swamps the narrow contact-critical stability basin); a fixed 0.01 floor splits between the
  two traps (1/0/51). **No entropy configuration threads HopperHop's conjunctive reward at 5M, while the planner-free
  TD-MPC2 core crosses 200 by 1–2M (8/8).** The discriminating axis: SAC bakes stochasticity into the *objective*;
  TD-MPC2 optimizes a *deterministic* actor objective against a Q-ensemble and injects exploration only into the
  *data*. Scope stated honestly: this is a ≥4–8× sample-efficiency gap, not a capability wall (external SAC hops by
  ~8M); custom SAC v1. Next: the **Lean+ decomposition** asks which TD-MPC2-side ingredient (UTD, Q-ensemble,
  SimNorm latent, noise anneal) carries the speed — and the **H3 margin-controlled PPO variant** (building now)
  closes the last reward-design caveat.
- **2026-07-10 16:45 — reimplementation-validity audit (user-raised): V1 parity + V2 planner-collection test.**
  The user asked the right question: our TD-MPC2 is a from-scratch JAX reimplementation that collects data with
  π+noise, where canonical TD-MPC2 collects with the planner — *have we ever checked it against the original?*
  **V1 (done):** we pulled the official per-task results from the TD-MPC2 repo. On hopper-hop our variant matches the
  canonical level (official 449 mean, seeds 373–594 @4M vs ours ~420±113, mppi ~571) — and the official **SAC**
  reference (0/246/105 @4M, 1/3 seeds ever ≥200) sits right on top of our custom SAC v1 (76/23/101), so the P1
  "SAC fails on Hop" result is not an implementation artifact. Where our variant *is* weaker (Walker −17%,
  Acrobot −23%, Cheetah −5%, Hop ≈0%), the deficit ordering tracks our measured WM-load-bearing ordering — exactly
  what you'd predict if the missing planner-collection matters most where accurate rollouts matter. **V2 (running):**
  an `MPPI_COLLECT=1` gate now runs {full, stripped} × planner-collection on HopperHop (n=2 each, 2.5M,
  512 MPPI samples = the canonical count) — if the stripped model still trains to full *under planner-collection*,
  the Part-12 removability claim holds beyond the implementation deviation; if it collapses, we rescope Part 12 to
  policy-collection variants. Wording in Papers A/3 and Part 12 will say "our TD-MPC2 variant (policy-collection)"
  until V2 lands.
- **2026-07-11 06:30 — overnight batch: the campaign's crux results all landed.**
  (i) **P1 closed** — full SAC entropy grid (auto-α / 0.01 / 0.05, n=3 each) fails on Hop at 5M while the planner-free
  TD core is 8/8; validated externally (official SAC reference matches ours). (ii) **H3 margin-controlled variant**:
  PPO still walls (2.8/3.6 @20M) with the shaping margin held — the conjunction is the wall, de-confounded. (iii)
  **Reimplementation audit published** (Parts 16–17: validity audit + the Handbook). (iv) **V2 — the scoping verdict:
  Hop removability SURVIVES canonical-style planner-collection** (full 465.0 vs stripped 465.8 @2.5M, n=2) — Part-12's
  critique holds beyond the deviation; the stripped arm planning over an untrained dynamics net is the strongest
  execution-simple evidence yet. (v) **Sufficiency + VSB tables reach paper grade**: Cheetah suff n=4 −23.8%; task
  ordering Hop 0 (n=8) < Walker −7.5% (n=4) < Cheetah −23.8% (n=4) < Acrobot −44%; Cheetah VBN n=3 517/576/624/726
  vs 855 (strictly monotone per seed); Walker VBN n=3 622/647/669/701 vs 727 (most-compressible signature). In
  flight: **V2W** (Walker planner-collection contrast — pre-registered: stripped should degrade where the WM is
  load-bearing; ~noon) and **Acrobot VBN s52** (→ n=3 on the least-compressible task).
- **2026-07-11 18:15 — day-2 batch: the dissociation confirmed, the grid completed, the next bet designed.**
  (i) **V2W confirmed the pre-registered prediction**: under planner-collection the stripped model degrades on
  WalkerRun (−16.1%, 722 vs 605, non-overlapping) — the mirror image of Hop's ±0%. The collection-mode ×
  world-model **double dissociation is complete**, and planner-collection turns out to roughly *double* Walker
  sample-efficiency (722 @2.5M ≈ policy-collection @5M), retroactively explaining the V1 official-vs-ours deficits.
  (ii) **V2X** (seed-52 extension of the dissociation table to n=3) runs overnight. (iii) **The 3-task
  value-sufficiency grid is n=3 complete**: Cheetah 517/576/624/726 vs 855 (smooth-monotone), Walker
  622/647/669/701 vs 727 (flat-high — most compressible), Acrobot 258/267/271/428 vs 511 (step-at-128 — least
  compressible; tight-width ordering is seed noise, s53 → n=4 tonight). Paper A's positive-instrument figure is
  data-complete. (iv) **Lean+ designed and pre-registered** (ledger 0af60bf): not another robust-loss bet — our
  value loss is already two-hot CE — but *target-EMA smoothing as a replacement for the world model's anchoring
  role*, tested on the stripped model's seed-fragile Cheetah; kill numbers stated; builds when V2X frees the box.
- **2026-07-12 04:30 — overnight batch: V2X softens the Walker claim honestly; Acrobot VBN reaches n=4.**
  (i) **V2X (dissociation at n=3)**: Hop's stripped≈full is rock-solid (+1.4%, three tight seeds), but Walker's
  s52 *full* arm collapsed to 455 (vs 758/686 on s50/51) while the stripped arms stayed tight (601/610/600) —
  the full-model arm is bimodal under planner-collection. On medians the dissociation stands (−12.5% Walker vs
  ≈0 Hop); we softened the claim to median-based and launched a seed-53 resolving pair rather than over-claiming.
  (ii) **Acrobot VBN n=4**: 261/271/282/397 vs vanilla 511 — the step-at-128 persists but narrows (84%→78%),
  and s53 is the second consecutive seed where D=128 is not dominant (304 ≈ D=64's 316). Updated claim: tight
  widths flat at 51–55%; D=128 alone recovers a large fraction, with wide seed variance on the step size.
  (iii) In flight: **Lean+** (target-EMA-as-WM-anchor on stripped Cheetah, 6 arms, verdict ~evening), the V2W
  seed-53 resolving pair (~morning), and Cheetah/Walker VBN seed-53 (→ n=4 on all three tasks).
- **2026-07-12 12:40 — midday batch: the dissociation resolves at n=4, and Lean+ returns an informative null.**
  (i) **V2W seed-53 settled the Walker question** (ledger 377e319, Part-16 update 3): full 744.7 vs stripped 558.3 —
  3 of 4 full seeds cluster 686–758, and live eval-trajectory watching revealed the real phenomenon: the full arm's
  evals swing ~250 points within a run while the stripped arm's move ~30. The dissociation is confirmed at n=4
  (medians 715.5 vs 600.5, −15.4%) with a variance-aware framing: the WM buys a higher but noisier performance
  regime on Walker. (ii) **Lean+ verdict — null, as pre-registered kill criteria demanded honesty**: target-EMA
  smoothing on the stripped model (596.9, n=3) does not beat the control (607.1) and misses the ≥620 gate; it does
  halve seed variance. Conclusion: the world model's dense-task contribution is predictive structure, not mere
  target stabilization — the lightweight-TD-MPC2 recipe stays "strip only where removable." (iii) **Refills**:
  V2W s54 (Walker n=5), V2H s53 (Hop n=4), and V2C s50/51 (Cheetah joins the dissociation table; pre-registered
  prediction: stripped degrades ≥15% under planner-collection, kill if <8%). All 8 GPUs busy.
- **2026-07-12 22:30 — THE VSB GRID IS n=4 COMPLETE. Paper A's central figure is data-final.** Walker seed-53
  landed on the curve (610.7/618.2/648.9/674.9), closing the third task. The figure at n=4: **Cheetah**
  544/588/630/727 vs 855 (64→85%, strictly monotone — a smooth information gradient where every doubling of the
  value-bottleneck width buys more); **Walker** 619/640/664/694 vs 727 (85→96%, flat-high — the most compressible
  task, D=16 already retains 85%); **Acrobot** 261/271/282/397 vs 511 (51→78%, tight widths flat, only D=128
  recovers substantially — the least compressible). Three qualitatively distinct fingerprints, ordered exactly like
  the sufficiency table (Walker −7.5% < Cheetah −23.8% < Acrobot −44%) — the instrument and the ablation agree.
  Also today: the GWM line was reviewed pre-box-retirement and stays closed (Part-18 lab notebook, with the one
  banked SOLD-Distinct follow-up). In flight overnight: n=5 seeds on all Cheetah and Walker widths; the V2W s54
  (Walker dissociation n=5), V2H s53 (Hop n=4), and pre-registered V2C Cheetah planner-collection pairs on b3060.
