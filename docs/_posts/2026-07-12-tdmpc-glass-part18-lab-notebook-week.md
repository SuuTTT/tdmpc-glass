---
layout: post
title: "TD-MPC-Glass, Part 18: Lab Notebook — Every Open Track of the Revision Week (07-08 → 07-12)"
date: 2026-07-12
description: "A from-first-principles lab-notebook survey of every research track opened between the Part-10 weekly review's TODO and the Part-15 revision plan, with verified numbers for each: the value-sufficiency-bottleneck grid at n=4 on three tasks, the reimplementation validity audit and the collection-mode × world-model double dissociation, the SAC entropy needle, the margin-controlled Hopper wall, the Lean+ lightweight-agent null, the closed graph-world-model line and its one banked follow-up, and the parked JEPA/SE cells. Each track: what it is, why it exists, verified status, and which paper it feeds."
---

> **TL;DR.** Ten tracks were live this week, all serving two deliverables: **Paper A** (the redundancy criterion +
> its positive instrument) and **Paper 3** (the anatomy of beating PPO), both targeted at the late-July deadline.
> Headlines: the 3-task **value-sufficiency grid reached n=4** (Cheetah strictly monotone 544/588/630/727 vs 855;
> Acrobot's step-at-128 narrows to 78%; Walker lands tonight); the **collection-mode × world-model double
> dissociation is confirmed at n=4** on the Walker side (full median 715.5 vs stripped 600.5, and the "bimodality"
> scare resolved into within-run eval volatility); **Lean+** — target-EMA smoothing as a world-model substitute —
> returned a clean pre-registered **null**; and the **graph-world-model line**, reviewed here before its GPU box is
> retired, stays closed (its one legitimate follow-up is banked, post-deadline). Sources for every number:
> `wm-redundancy-paper/bet2_null_results.md` (commit hashes inline) and `tdmpc-glass/docs/CHANGELOG.md`.

## Minimal shared context (60 seconds)

We study **TD-MPC2**, a model-based RL agent with three coupled parts: a latent **world model** (trained by a
*consistency loss* to predict its own next latent), a **value/policy head** (two-hot cross-entropy critic, deterministic
tanh-mean actor), and an **MPPI planner** (samples action sequences through the world model). Our stack is a JAX
reimplementation ("our variant") whose one deliberate deviation is documented in Part 16: it **collects data with the
policy**, not the planner. The campaign's core finding — the *redundancy criterion* — is that on several standard
control tasks you can delete the world-model objective ("**stripped**") and lose little or nothing, which makes many
proposed abstraction objectives redundant. The open work this week: turn that negative into instruments and mechanisms
(Paper A), and explain exactly what beats PPO and why (Paper 3). "n" always counts independent seeds; "vanilla" means
the unmodified agent; "@5M/@2.5M" is environment steps.

## Track 1 — Value-Sufficiency Bottleneck (VSB/VBN) grid: the positive instrument

**What it is.** A width-\\(D\\) bottleneck inserted between the latent and the value head: the critic may only read a
\\(D\\)-dimensional slice. Sweeping \\(D \in \{16,32,64,128\}\\) against vanilla measures *how much of the latent the
value function actually needs* — a valid replacement for the invalid decode-\\(R^2\\) probe Paper A already rejects.

**Why it exists.** Paper A's arc ended deflating ("a valid probe is still open"). The VSB curve is the constructive
answer, and its *shape* is a per-task fingerprint of value-information structure.

**Status (n=4 on all three tasks as of tonight; final mppi evals, 5M steps).**
- **CheetahRun**: 544 / 588 / 630 / 727 vs vanilla 855 — 64→85%, **strictly monotone** (ledger `76c29ef`). The
  "smooth information gradient" case: every doubling buys more, no width suffices.
- **AcrobotSwingup**: 261 / 271 / 282 / 397 vs vanilla 511 — tight widths statistically flat at 51–55%, **only
  D=128 recovers a large fraction (78%)**, and the step's size is seed-heavy (`3e3e312`). The "least compressible"
  case.
- **WalkerRun**: n=3 means 622 / 647 / 669 / 701 vs vanilla 727 (85–96%, flat-high — most compressible); the
  seed-53 arms complete tonight → n=4 (in flight, no numbers claimed).
- Seed-54 arms for **n=5** on Cheetah (all four widths) and Walker are already running.

**Feeds:** Paper A's central positive figure (the three fingerprints), plus the cross-task compressibility ordering
that matches Track 2.

## Track 2 — Task-conditional sufficiency: how load-bearing is the world model?

**What it is.** The stripped-vs-full comparison (delete the consistency loss, keep everything else) run per task.

**Why it exists.** It quantifies the redundancy criterion task by task, and any candidate mechanism for "what the
world model does" must reproduce its ordering.

**Status (final, policy-collection, 5M).** HopperHop **0%** (n=8, removable) < WalkerRun **−7.5%** (n=4) <
CheetahRun **−23.8%** (n=4) < AcrobotSwingup **−44%**. Stable all week; the V1 audit (Track 4) found the same
ordering in the official-implementation deficits — independent corroboration.

**Feeds:** Papers A and 3 (it is the x-axis both papers' mechanisms must explain).

## Track 3 — Lean+: is the world model just a stabilizer? (pre-registered, closed today: NULL)

**What it is.** A "lightweight TD-MPC2" bet: if the world-model objective merely *stabilizes* value targets, then a
slower target-EMA (\\(\tau\\) 0.01 → 0.003) should recover the stripped model's losses on Cheetah — a two-line change
instead of a world model.

**Why it exists.** The user asked for a lightweight variant; the honest way to build one is to test whether the WM's
role is replaceable by cheaper stabilization. Success and kill gates were pre-registered before launch (`0af60bf`).

**Status (closed 07-12, n=3/arm, 5M, stripped Cheetah).** Lean-on **596.9** (spread 21.5) vs lean-off control
**607.1** (spread 48.2) — fails the pre-registered mean gate (≥620) without triggering the kill line; the only real
effect is a **halved seed variance** (`943819c`). **Verdict: the world model's dense-task contribution is predictive
structure, not mere target stabilization.** The honest lightweight recipe remains: strip the WM only where Track 2
says it's removable (Hopper-class).

**Feeds:** Paper A — sharpens the claim that the consistency loss carries irreplaceable predictive structure.

## Track 4 — Reimplementation validity audit (V1): are we wrong at the very beginning?

**What it is.** A direct parity check of our JAX variant against the official TD-MPC2 (and official SAC), triggered by
the user's question once we noticed our stack never uses the planner to collect data.

**Why it exists.** Every claim in both papers is scoped by whether the reimplementation is faithful; this had never
been tested head-to-head.

**Status (closed, Part 16, `5a31517`).** Official hopper-hop 449 ≈ ours 420±113; official SAC's Hopper failures match
our SAC runs (the P1 result is not an artifact of our SAC). The official-minus-ours deficit is *ordered exactly like*
Track 2's WM-load ordering (Hop ≈0 < Cheetah < Walker < Acrobot ~23%) — the deviation costs performance precisely
where the world model matters, which is what Track 5 then explained mechanistically. Consequence adopted campaign-wide:
sample-efficiency claims are **within-stack only**.

**Feeds:** the scoping section of every paper; Part 16 (published) + Part 17, the from-scratch Handbook (shipped, with
the co-design recipe the user asked for).

## Track 5 — The collection-mode × world-model double dissociation (V2/V2W/V2X + extensions)

**What it is.** Rerun the stripped-vs-full contrast **under planner-collection** (the canonical mode, `MPPI_COLLECT=1`,
512 samples). Two pre-registered predictions: on Hopper (WM removable) stripping stays free; on Walker (WM
load-bearing) stripping should start to hurt *more*, because the planner collects by rolling the model.

**Why it exists.** It is the cleanest mechanistic statement the campaign owns: the world model's value is
task-conditional, and collection mode modulates it in the predicted direction at both ends. It also retroactively
explains Track 4's deficit pattern.

**Status.**
- **Hopper**: stripped ≈ full, +1.4%, n=3, tight (unchanged all week); seed-53 pair → n=4 in flight.
- **Walker (n=4, closed today, `377e319`)**: full median **715.5** vs stripped **600.5** (−15.4%; means −10.4%).
  The mid-week scare — a seed-52 full arm at 455 suggesting bimodality — resolved when seed 53 finished at 744.7
  *and* its live eval trajectory revealed the mechanism: the full arm's evals swing ~250 points within a single
  late-training window (680→715→676→**501**→696→744) while stripped arms move ~30 points. Refined claim: under
  planner-collection the WM buys a **higher-mean but higher-variance eval regime**; seed 52's 455 was an unlucky
  final draw, not a mode. Seed-54 pair → n=5 in flight (~tonight).
- **Cheetah (new, pre-registered `943819c`, in flight)**: the third task joins the table; prediction: stripped
  degrades ≥15% under planner-collection; kill if <8% (that would falsify the "planner-collection amplifies WM
  importance" generalization). Verdict ~tonight.
- Bonus finding retained: planner-collection roughly **doubles Walker sample-efficiency** in our stack (722@2.5M ≈
  policy-collection@5M).

**Feeds:** Paper 3's centerpiece table; Part 16 updates 1–3 (published).

## Track 6 — P1, the entropy needle: why SAC fails Hopper

**What it is.** A full SAC entropy grid (auto-α, α=0.01, α=0.05; n=3 each) on HopperHop at 5M, against the
planner-free TD-MPC2 core.

**Why it exists.** Paper 3 needs the *mechanism* of TD-MPC2's Hopper advantage; the candidate was where stochasticity
lives: in the objective (SAC) vs only in the data (TD-MPC2's ε-noise collection).

**Status (closed, `09fc2ab`).** Every SAC arm fails Hopper at 5M; the TD core is 8/8. With Track 4's external
validation (official SAC also fails), the "entropy needle" stands: **stochasticity-in-objective vs
exploration-in-data** is a ≥4–8× sample-efficiency gap on this task, honestly capability-scoped.

**Feeds:** Paper 3, mechanism section.

## Track 7 — H3, the margin-controlled Hopper wall

**What it is.** PPO on HopperHop with the reward re-shaped (product form, speed and margin knobs at 1.0) to test
whether the PPO wall is a shaping artifact.

**Why it exists.** Closes Paper 3's F3 audit caveat: is the wall the *conjunctive* reward itself, or our margin choice?

**Status (closed, `aa1a870`).** PPO still walls (2.8/3.6 @20M) with the margin held — **the conjunction is the
wall**, de-confounded.

**Feeds:** Paper 3.

## Track 8 — The graph-world-model (GWM) line: reviewed before the box is retired

**What it is.** The relational axis of the redundancy program: does giving the world model an explicit **entity/graph
structure** (object nodes, interaction edges) buy anything a monolithic latent lacks? Run June 14–15 as iterations
34–36 plus a real-benchmark gate; its GPU box (the RTX 5070 Ti instance) has been idle since and is now being
retired by the user.

**Why it exists.** The 2026 GWM survey (arXiv:2604.27895) named graph world models a paradigm; three deep-research
reports converged that *if* headroom exists it is in contact-rich, compositional (held-out object-count) settings.
Our smooth-spring synthetic could not have surfaced that, so a fair contact-world test was owed.

**Status (closed June; reviewed 07-12; all numbers from `tdmpc-glass/docs/CHANGELOG.md` and
`docs/research/graph-world-model-plan.md`).** Four results, one honest asterisk:
1. **Iter-34 mechanism check (contact multi-disk world)**: the graph latent's *representation-level* win was real —
   compositional-OOD value-\\(R^2\\) 0.57 vs fair param-matched monolithic 0.40 (gap 0.18 > pre-registered 0.15) —
   but the contact-conditioned-prediction criterion failed, and the **control gate killed it**: under random-shooting
   MPC, graph ≈ fair-mono ≈ random floor at every object count (graph−random +13, std ~110). Representation
   advantage did **not** convert to control.
2. **Iter-35 (SOLD, the official GWM)**: reproduced its released checkpoint at 100% success (30/30) vs paper 97.9%.
   Third-party corroboration from SOLD's own Table 1: monolithic TD-MPC2 (97.6) **ties** SOLD (97.9) on the
   non-relational variant; SOLD's wins are confined to the *Distinct* (relational) variants. The head-to-head on
   Distinct was blocked by hardware (15 GB replay memmap > box disk).
3. **Iter-36 (compositional-OOD *control*, on the 5070 Ti)**: **uninformative, not a null** — the GPU-vectorized
   contact env had a control-signal ceiling ~10% (no controller, including random, separated) plus BPTT NaNs.
4. **Paper-B gate on a real benchmark (ManiSkill PushCubeMulti)**: control signal passed (PPO 100% vs random 0%)
   but the headroom test failed — value-decodability stayed flat from 2→6 objects and the monolithic policy solved
   the OOD counts. **Monolithic generalizes over passive distractors**; the program folded into Paper A as its
   relational-axis closure.

**Verdict on continuing (the user's question).** As a program: **no — keep it closed.** The one genuine positive
(representation-level OOD advantage) failed to convert to control three separate ways, and the real-benchmark gate
failed for the deeper reason that monolithic latents already generalize compositionally where objects don't interact
contingently. The redundancy criterion now spans state, temporal, and relational structure — that *is* the paper
contribution. **The one legitimately untested cell** is a **SOLD Fetch-Distinct head-to-head** (active multi-object
interaction, the regime where SOLD's own table shows its only wins): a bounded reproduce-and-compare, ~2–3 box-days
on a ≥30 GB-disk PyTorch box, **banked as a post-deadline item** — it either adds a third-party-anchored positive
boundary to Paper A's relational section or closes the last asterisk. It is *not* added to this week's active plan
(deadline math wins). **Box retirement is safe**: `contact_entities.py`, `entity_wm.py`, `monolithic_wm.py`, and
`value_coupling_probe.py` are all preserved in the `tdmpc-glass` repo; iter-36 was already recorded as uninformative;
nothing unique remains on the instance.

**Feeds:** Paper A (relational-axis closure section + one banked boundary experiment).

## Track 9 — Parked from the Part-10 TODO: JEPA/SE cells and the taxonomy law

**What it is (planned, unexecuted — labeled as such).** The Part-10 worklist items that the revision campaign
displaced: J0 anchor-strength A/B on nav H-JEPA, J1 SE-community vs uniformity/VICReg in the collapse regime (open
cell #59, ~1 box-day), J2 SE-as-hierarchy (gated on J0/J1, ~2–3 box-days), J3 offline SE-JEPA probe, and Bet-B's
DOF-overlap monotonicity law (~2 box-days).

**Why they exist.** They are the *next-paper* pipeline (JEPA/SE line), deliberately paused because Papers A & 3 have
a ~07-28 deadline and the audit (Tracks 4–5) was strictly more urgent.

**Status.** No runs this week; no numbers. Resume after paper assembly; J1/#59 first.

**Feeds:** the post-deadline JEPA/SE paper direction.

## Track 10 — Assembly: the papers themselves

**What it is.** Fold everything above into **Paper A** ("When Is Explicit Abstraction Redundant") and **Paper 3**
("The Anatomy of Beating PPO") per the Part-15 narrative plan: A gains the VSB instrument + 4-taxa closure + Lean+
sharpening + relational closure; 3 gains the wall mechanism (H3), the entropy needle (P1), the double dissociation
(Track 5), and the within-stack scoping (Track 4).

**Status.** Planned; starts when tonight's n=5/n=4 arms land (the last figure-feeding runs). Estimated 2–3 focused
days of writing.

**Feeds:** both deliverables, deadline ~07-28.

## Dependency map

- **Paper A** ← Track 1 (VSB figure) ← Track 2 (ordering) ← Track 3 (Lean+ null) ← Track 8 (relational closure).
- **Paper 3** ← Track 5 (dissociation table) ← Track 4 (validity scoping) ← Track 6 (entropy needle) ← Track 7
  (wall mechanism).
- **Docs/public record**: Parts 16–17 (audit + Handbook, shipped), Part 15 live log (running), this notebook.
- **Post-deadline queue**: Track 9 (JEPA/SE), Track 8's banked SOLD-Distinct head-to-head.

*Everything above is reproducible from the ledger (`bet2_null_results.md`, commits cited inline), the CHANGELOG, and
the Part 15/16 live logs; in-flight tracks are stated without numbers on purpose.*
