---
layout: post
title: "TD-MPC-Glass, Part 5: A Method Map — State vs Behavioral vs Planning vs PPO, and Where JEPA Fits"
date: 2026-07-01
description: "The question driving the whole recent program: across all Panda + DMControl tasks, when does each of four approaches win — state abstraction (SE/glass), behavioral abstraction (analytic controllers/TAMP), planning (TD-MPC2), and PPO — measured LeCun-style by learning speed, not just final score? Then the JEPA pivot: LeCun's argument that a non-generative predictor needs an information (anti-collapse) term and that hierarchical planning is where structure could pay off — and the experiments we ran on both, plus this week's beat-PPO reality check that pins down exactly where each method lives. The recurring law: abstraction and priors buy learning speed, not a higher ceiling."
---

> The real question behind this whole program isn't "can we beat PPO once" — it's a **map**: across all Panda and
> DMControl tasks, *when does each family of methods win?* We've been comparing four — **state abstraction**
> (structural-entropy / "glass" latents), **behavioral abstraction** (analytic controllers, TAMP), **planning**
> (TD-MPC2), and plain **PPO** — and, following LeCun, scoring them by **learning speed**, not just final return.
> This post lays out that map, the **JEPA** detour it took us on, and this week's beat-PPO reality check that
> nails down the boundaries. The one law that keeps recurring: *abstraction and priors buy learning speed, not a
> higher ceiling.*

> **Update (2026-07-03) — two claims in this post were later corrected by controlled experiments; see
> [Part 7](../2026-07-03-tdmpc-glass-part7-five-bets-resolved) for the resolution.**
> (1) **§2's "anti-collapse taxonomy" reversed.** Tested on a *pure* JEPA (no value loss — the right object), a
> self-predictive latent **does not collapse** on DMControl state, pixels, or narrow/on-policy data; anti-collapse
> ranges neutral-to-harmful, and the geometric-vs-value split was nav-specific, not a law
> ([Thread D](../tdmpc-glass-thread-d-jepa-anticollapse-done-right)). (2) **§7's flagship went null.** The
> "planning-as-exploration" bet does not survive a controlled planning-vs-policy-only test: planning is a
> **sample-efficiency and exploitation** lever, not a directed-exploration operator
> ([Thread A](../tdmpc-glass-thread-a-planning-exploration)). The recurring *speed-not-ceiling* law below still
> holds — it's what survived.

## The method map (what wins, measured by learning speed)
Four ways to get a policy, and — per LeCun's point that the honest axis is **how fast you learn** — where each
one actually pays off, from all our Panda + DMControl runs:

| method | what it is | wins (fast / capable) | loses | key runs |
|---|---|---|---|---|
| **PPO** | model-free on-policy | huge throughput; **solves** sample-hungry high-DoF (Leap reorient **0.99**, PickCubeOrient **0.81**, CheetahRun 928 @285M) | sample-**inefficient**; **fails** exploration-hard (HopperHop **33**) | §3–5, benchmark |
| **Planning — TD-MPC2** | self-predictive latent world-model + MPPI | 3–4 orders more **sample-efficient**; wins exploration-hard (HopperHop **367** vs 33) | slow wall-clock ⇒ can't reach the 100M+ steps dexterous tasks need ⇒ 0 there | §3–5b |
| **Behavioral — analytic/TAMP + residual** | hand-written skill / task-and-motion prior | **fast to competence** where the prior fits (Pendulum **836** vs 46; OpenCabinet **7×**; Reacher **3×**) | dead weight (**anchor**) where it doesn't (locomotion: vanilla wins 5/5); analytic-alone caps at contact physics (**0.37**) | §1 |
| **State — SE / "glass"** | structural-entropy structure on the latent | — (this was the bet) | **redundant**: glass ≈ TD-MPC2 on 16 DMC (matched benchmark, n=3–4; the old \(R^2\approx0.999\) probe evidence was retracted); SE *hurts* continuous geometry; +1.35× wall-clock | §2, §5b |

Read the "loses" column as the thesis: no state/behavioral/planning prior *raises the ceiling* — each just shifts
**where on the speed axis** you land. PPO is the reference: slow-but-eventually-capable where it can explore,
absent where it can't.

**Two SOTA paradigms not in the map (honest status).** The obvious "planning" contenders from the recent
literature are **generative jumpy world models** — Meta/FAIR's *Compositional Planning with Jumpy World Models*
(CompPlan, built on TD-Flow) — and **flow-occupancy** methods — *InFOM* (Intention-conditioned Flow Occupancy
Measure). We engaged both but they're not scored above, for different reasons: (a) the *jumpy* idea we
**reproduced in our own stack** (jumpy/k-step TD-MPC2) and it was **redundant** — null-to-mildly-negative vs
vanilla on return, sample-efficiency, and wall-clock, so it folds into the Planning row rather than being its own
column; (b) the full **CompPlan/InFOM flow-occupancy reproduction** (OGBench cube-single + antmaze) we *stood up*
— datasets, isolated envs, InFOM pretraining launched — but it's a genuine **multi-day** effort we paused when the
JEPA/beat-PPO thread took priority, so we have **no completed head-to-head** to report. Flagging that rather than
hand-waving it is the point.

## The JEPA detour (LeCun's three bets)
Midway through, LeCun's JEPA talk reframed our own "glass" idea. His three claims map onto three experiments we ran:

1. **A non-generative predictor needs an *information* (anti-collapse) term** or it collapses. → We tested exactly
   that term (VICReg vs a one-line **uniformity** loss vs **SE**), and found the right information term is
   **downstream-dependent** — §2.
2. **Hierarchical planning is where structure pays off.** → We built a faithful **H-JEPA** (latent subgoals +
   latent planning) and tried SE *for the hierarchy*. Verdict: learned hierarchy is **not** the lever — H-JEPA
   solves Panda (0.367) only because we hand it a *competent low-level primitive*; the hierarchy itself buys only
   modest speed (2-level 0.215 vs 1-level 0.184). The lever is the primitive, §1.
3. **Measure learning speed, not asymptote.** → This is now the axis of the whole map above, and of §5's honest
   split. Every "win" here is a steps-to-competence win.

The rest is the evidence, in the order we actually walked it: finish Panda (§1) → close the representation door
(§2) → hunt a real beat and nearly fool ourselves (§3–4) → the verified map + benchmark (§5–5b).

## Bottom line (the four takeaways)
- **A *learned* residual breaks the analytic contact ceiling on Panda (0.37 → 0.72), proving the "cube tips in
  the gripper" wall is *learnable* — but a matched-budget vanilla PPO still wins the asymptote (0.81).** Across
  two Panda tasks the story is identical: a structured prior buys **sample-efficiency (~1.6–7×), not a higher
  ceiling**.
- **Anti-collapse for self-predictive latents is *downstream-dependent*.** A one-line relational (uniformity)
  loss *fixes* a collapse-prone geometric latent (nav 0.53 → 0.95) but *hurts* value-based control (worst of
  three regularizers on 3/4 DMControl tasks), and a "value-aware" variant is worse still. Structural-entropy
  structure is redundant either way. There is no universal anti-collapse term.
- **We hunted for a new env where we beat PPO and did not find one — but nearly *claimed* one.** On
  `PandaPickCubeOrientation`, TD-MPC2 hit **2842 return** vs PPO's **1419** and I wrote it up as a "strong beat
  forming." Then we scored **real success**: TD-MPC2 = **0.000**. The return was pure reward-gaming. Retracted.
- **The MuJoCo Playground PPO genuinely solves these hard tasks; our TD-MPC2 (at a practical budget) does not.**
  Verified real success: PPO **0.81** on pick-and-orient, **0.99** on dexterous in-hand reorientation; TD-MPC2
  **0.00** on both at 5M. This is the *opposite* of a beat-PPO — and it maps precisely where each method wins.

## 1. Finishing Panda: a learned residual, and the ceiling that isn't a ceiling

### 1.1 The task and the exact bar
`PandaPickCube` (MuJoCo Playground) is a 7-DoF Franka that must **reach** a small cube, **grasp** it, **lift**
it, and **place** it upright at a target. We score *real* success, not shaped return: an episode counts only if

$$
\text{box\_target} = 1 - \tanh\!\big(5\,(0.9\,\lVert p_\text{err}\rVert + 0.1\,\theta_\text{err})\big) \ge 0.9,
$$

i.e. the cube must end **both** at the target *position* and near-upright *orientation* — concretely
\(0.9\,\lVert p_\text{err}\rVert + 0.1\,\theta_\text{err} \lesssim 0.02\). That `0.1·θ` term is the whole story below.

### 1.2 Where the 0.37 comes from — the tall cube tips in the grip
Our analytic skill (a hand-written reach/grasp/lift/transport/place controller) reaches, grasps, and lifts the
cube essentially every time — grasp success **1.00**, lift **~0.85**. But it caps at **~0.37** real success,
and the reason is a *contact-physics* failure that a still image can't convey:

![The failure mechanism: the tall cube rotates inside the two-finger parallel-jaw grip during lift/transport (tilt grows from ~5° at grasp to ~20° at place), so it arrives off-upright and misses the box_target≥0.9 bar even though position is fine.]({{ '/images/cube_tipping_mechanism.gif' | relative_url }})

The cube is **tall (4×4×6 cm)** and the gripper is a **two-finger parallel jaw** — a *single* compliant line of
contact with no wrist torque about the grasp axis to resist rotation. During lift and transport the cube slowly
**rotates away from upright inside the fingers**: we measured tilt growing from **~5° at grasp to ~20° at place**.
Position is fine, but that residual tilt blows the `0.1·θ_err` budget, so `box_target` lands just under 0.9. It
isn't a grasp *detection* problem and it isn't reach precision — it's that the parallel-jaw contact *cannot hold
orientation* through the carry. Instrumented on the failures: cube-tilt-at-place ~15–20° on misses vs ~2–3° on
the ~19% that succeed. (The clip above is a real miss — seed 26, tilt **4.5° → 18.3°**, box_target 0.36. For
contrast, a success keeps the cube upright the whole carry — tilt **4.5° → 2.8°**, box_target 0.96:)

![Contrast: a successful episode keeps the cube upright through transport (tilt stays ~3–4°, box_target 0.96) — the ~19% the analytic skill gets right.]({{ '/images/cube_upright_success.gif' | relative_url }})

### 1.3 Can analytic tuning fix it? (rounds 4–5) — a little, then a wall
- **Round 4 (yaw-aware place):** null — we were correcting the wrong axis (it's mid-flight *tipping*, not grasp
  yaw). Oracle barely moved (0.305 → 0.320).
- **Round 5 (closed-loop upright servo):** recompute the gripper orientation every step to actively *right* the
  live cube during transport. This helped — final tilt **20.6° → 16.4°**, and the analytic **oracle** (brute-force
  best parameters searched directly in sim) rose **0.305 → 0.352**, with the full n=512 solve landing at **0.367**.

But that's it. No analytic gain knob crosses ~0.37, because the servo *itself* perturbs placement *position* while
righting orientation — through the single compliant contact you can't satisfy both halves of the
\(0.9\,p + 0.1\,\theta \le 0.02\) budget at once. So we had a genuine ceiling **for the analytic-skill family**.
The question that names this section: is ~0.37 a ceiling of the *task/hardware*, or just of *hand-written control*?

### 1.4 Hand the wall to a learner: a residual policy
We put a **learned residual** on top of the analytic skill and trained it by RL against the true reward,
\(a = \mathrm{clip}\big(a_\text{skill} + \alpha\,\pi_\text{res},\,-1,1\big)\), sweeping the authority
\(\alpha\), and — the load-bearing control — a **budget-matched vanilla PPO** (same env, same step budget, same
eval).

| arm | PandaPickCube (success) | PandaOpenCabinet (success) |
|---|---:|---:|
| analytic skill / oracle | ~0.37 | 0.827 |
| residual α=0.25 / 0.5 (bounded) | 0.19 / 0.40 (unstable) | — |
| **learned residual α=1 (full)** | **0.716 ± 0.014** (n=3) | **0.980** (n=7) |
| **matched vanilla PPO** | **0.810 ± 0.006** (n=3) | **0.980** (n=5) |

Two clean findings, both confidence-interval-separated:

1. **The wall is *learnable* — it was never task/hardware physics.** The full-authority residual drives
   success-case cube-tilt down to **~1.9°** (below even the analytic servo's 2.5°) and reaches **0.716**, nearly
   doubling the analytic 0.37. A learner *can* hold the cube upright through transport where hand-tuned control
   can't. So "the ceiling that isn't a ceiling": ~0.37 is a hard limit of the *analytic-skill family*, not of the
   task. (Note the authority sweep: strictly *bounded* residuals — α ≤ 0.5 — are unstable and top out ~0.40; only
   near-full authority breaks through, i.e. it's effectively a *warm-started* policy, not a small correction.)
2. **But the prior does not raise the *RL* ceiling.** Matched vanilla PPO wins the asymptote outright — **0.810
   vs 0.716** on PickCube; on OpenCabinet both hit the same **0.98** structural ceiling. What the analytic prior
   actually buys is **speed**: ~1.6× faster to competence on PickCube, ~7× on OpenCabinet.

Same one-liner as the whole campaign: **a structured prior redistributes complexity into faster exploration; it
does not remove it, and it does not lift the ceiling.** (Budget-trap guarded throughout — the earlier "PPO 0.66"
was an under-budgeted baseline; the honest matched PPO is 0.81.)

## 2. Closing the anti-collapse question: it's downstream-dependent

*Why this, now?* §1 said a **behavioral** prior buys speed, not ceiling. The last place a real win could hide is
the **representation** itself — the "glass"/structural-entropy latent this whole project is named for. If a better
latent doesn't lift control, then abstraction is redundant on *both* axes and the only honest path to beating PPO
is finding the right *task* (§3). So we close it properly, with every control.

A self-predictive (JEPA/SimNorm) latent trained only to predict its own future can **collapse** — the encoder maps
everything to (nearly) one point, prediction becomes trivially perfect, and the representation is useless. You need
an **anti-collapse term**. Our "glass" program bet on **structural entropy (SE)**; the honest answer is that SE
*per se* is the wrong bet, and the right one is *downstream-dependent*. Here are the exact terms we compared and
how each is implemented:

- **VICReg** — *per-dimension* anti-collapse: a **variance hinge** (push each latent coordinate's std ≥ 1) plus a
  **covariance penalty** (decorrelate coordinates). Fights collapse one axis at a time.
- **Uniformity** (Wang & Isola) — the **one-line relational** term
  \(\mathcal{L}_\text{unif}=\log\,\mathbb{E}_{i\ne j}\,e^{-2\lVert z_i-z_j\rVert^2}\), i.e.
  `(-2*pdist(z).pow(2)).exp().mean().log()` — pushes *all pairs* of embeddings apart on the hypersphere. No graph,
  no partition, no tree.
- **SE (structural entropy)** — build a kNN graph on the latents, compute its minimal-2D-SE **community partition**
  (an encoding tree, via `selib`), and regularize the latent toward low coding-cost of that partition. Structure,
  not pairwise.
- **knnrep** — pairwise repulsion over the kNN graph only (relational, graph but no partition).

And the four controls that make the conclusion airtight:

- **random-graph** (Panda) — run the *whole SE machinery on a shuffled/random* kNN graph. Isolates: is SE's effect
  the community *structure*, or just "some regularizer"?
- **random-partition** (nav) — SE with *randomized* community assignments. Isolates: the specific SE tree, or any
  relational grouping?
- **strong-VICReg** (nav) — VICReg with a large coefficient that *fully* restores effective rank. Isolates: does
  fixing latent *health* fix *control*?
- **value-aware** (DMControl) — uniformity re-weighted by *value* similarity (down-weight repulsion between
  value-close states). Isolates: does matching the term to value structure rescue it?

### The nav collapse testbed (n=4, deterministic n=256 eval)
This is a 2-D point-maze where SimNorm+VICReg *provably* collapses, so it cleanly separates the regularizers. It is
**not** a beat-PPO task — it's a controlled representation-learning probe. The metric is success *relative to the
raw-observation ceiling* (an identity encoder that trivially solves the maze):

| arm | success | collapsed? | reads as |
|---|---:|---|---|
| raw-obs ceiling | **0.979** | — | the achievable max |
| VICReg (default) | 0.530 | 4/4 | collapses → fails |
| VICReg (strong) | 0.442 ± 0.37 | 0/4 | **un-collapses but still fails** — health ≠ control |
| SE (se_w=5) | 0.906 | 0/4 | works |
| SE (**random partition**) | 0.823 | — | ≈ real SE (overlaps) → **partition-independent** |
| knnrep (graph, no partition) | 0.887 | 0/4 | works |
| **uniformity (1 line)** | **0.954 ± 0.05** | 0/4 | works best, simplest |

Two things fall out. **(a)** SE *fixes a collapse VICReg cannot* — but the random-partition control shows the win
is the *relational/pairwise* structure, not the SE community tree; a one-line uniformity loss matches it. **(b)** On
Panda, the same SE machinery is a **null**: a frozen-encoder value/geometry probe gives VICReg \(R^2=0.987\) for
end-effector→cube vs SE's 0.736 — and **SE-on-random-graph recovers to 0.986**, proving SE's *community bucketing*
(effective rank 31→13) is what destroys the continuous manipulation geometry.

### But it flips on value-based control
Run those same arms on collapse-prone **DMControl** (return-AUC, n=3), and the relational term that *won* on nav is
now the **worst**:

| task | default (no extra) | uniformity | value-aware |
|---|---:|---:|---:|
| CheetahRun | **58.9** | 36.6 | 20.6 |
| WalkerWalk | **293.8** | 89.9 | 49.5 |
| FingerSpin | **249.4** | 171.8 | 1.6 |

Uniformity is worst on 3/4 tasks, and the **value-aware** variant is *worse still* — matching the term to value
structure didn't rescue it, it hurt more. The mechanism is clean: a relational repulsion spreads apart states that
should be *value-close*, destroying the value-sufficiency control needs. So the taxonomy:

> **There is no universal anti-collapse term.** Relational/uniformity for goal-conditioned **geometric** latents;
> **nothing extra** for **value-based control** (SimNorm's built-in pressure + the value gradient suffice);
> **never SE community structure** for either continuous case. Match the term to what you decode.

**Important scoping caveat (raised in review — see §7.1).** The DMControl arm here runs on **TD-MPC2**, which is
*not* a pure JEPA — its value/reward loss already anchors the latent, so "adding a term hurts value-based control"
is really a **redundancy** result (a regularizer fighting an already value-sufficient latent), *not* a test of
JEPA anti-collapse. Only the **nav (H-JEPA)** result — where the latent is genuinely self-predictive and collapses
to eff_rank≈0 — is a real JEPA anti-collapse test. The clean JEPA-on-DMControl experiment (strip the value loss →
pure self-predictive latent → probe) is Proposal D in §7.

**Does the 0.954 "beat PPO"? No — and it shouldn't be read that way.** It's a *recovery* number: ~97% of the
raw-obs ceiling (0.979), un-doing the collapse that cost VICReg 0.53→0.95, on an easy nav task where PPO was never
run and would also succeed. The anti-collapse result is a **representation-learning** finding (which regularizer
fixes JEPA collapse), not a control-benchmark win. The only place this touches "beat PPO" is negatively: even the
*best* fixed latent doesn't change the §1/§3 story that a structured latent doesn't raise the RL ceiling.

## 3. The hunt for a new beat-PPO env — and the mirage

With the Panda/anti-collapse threads closed, we asked: is there a *new* environment where our strongest method
(TD-MPC2, a self-predictive-latent world model + planning) beats PPO outright? We scanned four untested harder
MuJoCo Playground envs — `PandaPickCubeOrientation`, `PandaRobotiqPushCube`, and the dexterous
`LeapCubeReorient` / `LeapCubeRotateZAxis` — TD-MPC2 vs **matched-budget** PPO.

The early reads looked *thrilling*. On `PandaPickCubeOrientation`, TD-MPC2 hit **2842 episode return** at 0.95M
steps versus PPO's fully-plateaued **1419** ceiling (reached at 75M) — an apparent **both-axes beat at ~80×
fewer samples**. I recorded it (with a caveat flag) as "a genuine ceiling-beat forming."

Then we did what this project is *built* to do — score the metric that matters:

![The return-vs-success trap: TD-MPC2 accrues far more dense reward on PandaPickCubeOrientation than PPO, yet solves the task 0% of the time while PPO solves it 81%.]({{ '/images/return_vs_success.png' | relative_url }})

**TD-MPC2's real success was 0.000.** Its 2842 return was pure **dense-reward gaming** — hovering near the cube
accumulating shaping reward, never completing the pick-and-orient (reached ~0.2, `box_target` ~0). The "beat"
evaporated. This is exactly the return-vs-success trap the whole campaign is named for, and it caught me
mid-write-up. **Retracted, in the ledger, visibly.**

## 4. So: does PPO actually *solve* the hard tasks? Yes — verifiably

That left the real question, which I'd never actually measured: my PPO runs logged episode *return*, not
success. Does the Playground PPO genuinely *solve* these, or does it also just accrue return? We loaded the
trained PPO checkpoints and scored real success over 256 deterministic rollouts:

![Real success (n=256): the MuJoCo Playground PPO solves all three hard tasks at its full budget; our TD-MPC2 at a practical 5M-step budget solves none.]({{ '/images/beatppo_success.png' | relative_url }})

| env | PPO real success | budget | TD-MPC2 @5M |
|---|---:|---:|---:|
| PandaPickCubeOrientation | **0.809** (reached 1.0) | 96.7M | 0.000 |
| LeapCubeReorient | **0.988** (≈3.8 reorients/episode) | 212M | 0.000 |
| PandaRobotiqPushCube | 0.18–0.51 | 386M | ~0 |

PPO genuinely solves them. Here is its trained policy reorienting the cube in-hand — the 0.99-success dexterous
task our method never cracks:

![PPO's trained policy solving LeapCubeReorient — dexterous in-hand cube reorientation (real success 0.99).]({{ '/images/leap_reorient_solve.gif' | relative_url }})

![PPO's trained policy solving PandaPickCubeOrientation — pick and place at target pose (real success 0.81).]({{ '/images/panda_orient_solve.gif' | relative_url }})

## 5. The honest map: where each method wins

This "failed" hunt produced the most precise statement of the beat-PPO boundary we've had:

- **We beat PPO** (on both sample-efficiency *and* practical capacity) only on **exploration-bottlenecked tasks
  that are solvable within a few million steps** — `HopperHop` (TD-MPC2 367 vs PPO 33), sparse/weak-actuation
  swing-ups. PPO's on-policy exploration stalls; model-based planning gets there.
- **PPO beats us** on **sample-hungry high-DoF tasks** — dexterous in-hand reorientation, multi-object
  manipulation — that need **100M–400M** steps. TD-MPC2 is more sample-efficient *per step*, but it's
  model-based and **slow**, so ~5M steps is the practical ceiling; these tasks need far more, and PPO's ~10–30×
  throughput (512-env brax) delivers them. At *matched* 5M, neither solves these (PPO also needs ~75M).

So TD-MPC2's per-step efficiency only converts to a win when the task fits inside its practical step budget.
That's a real, useful boundary — and arguably a better result than a fake beat would have been.

## 5b. The DMControl benchmark, consolidated (the dashboard table)

The manipulation results above sit on top of a broader **DMControl suite benchmark** we've been running on a
live dashboard (16 tasks × 3 seeds, glass / TD-MPC2 / PPO). Two headlines it settles:

**(i) Representation abstraction is redundant — the full 16-task sweep.** Here is *all* of it (final-checkpoint
return, mean ± sd, n=4–5 seeds per cell; PPO = peak return at the labelled budget, subset only). The glass column
is our structural-entropy latent; TD-MPC2 is the vanilla self-predictive baseline:

| DMControl task | TD-MPC2 | tdmpc-glass (SE) | PPO (peak@budget) |
|---|---:|---:|---:|
| AcrobotSwingup | 324 ± 101 | 297 ± 10 | 268 @285M |
| BallInCup | 731 ± 422 | 577 ± 471 | — |
| CartpoleBalance | 945 ± 51 | **969 ± 11** | — |
| CartpoleSwingup | 823 ± 36 | **828 ± 19** | — |
| CheetahRun | 500 ± 274 | **599 ± 119** | **928** @285M |
| FingerSpin | 950 ± 33 | **972 ± 13** | — |
| FingerTurnEasy | 920 ± 85 | 856 ± 94 | — |
| FingerTurnHard | 869 ± 171 | 775 ± 173 | **968** @285M |
| HopperHop | **265 ± 64** | 197 ± 145 | 33 @285M |
| HopperStand | 896 ± 36 | 417 ± 330 | — |
| PendulumSwingup | 589 ± 342 | 389 ± 390 | — |
| ReacherEasy | 980 ± 3 | 933 ± 89 | — |
| ReacherHard | 883 ± 151 | **976 ± 3** | — |
| WalkerRun | 652 ± 34 | **662 ± 8** | — |
| WalkerStand | 979 ± 17 | 969 ± 15 | — |
| WalkerWalk | 971 ± 12 | 919 ± 81 | 970 @79M |

Read the two world-model columns side by side: **no systematic difference.** The gaps that look real
(HopperStand 896 vs 417, PendulumSwingup 589 vs 389) are single-collapsed-seed artifacts — note the ±330/±390
standard deviations — and they go *both* ways — **glass (SE) is bolded where it beats TD-MPC2: 6 of 16** (CheetahRun, ReacherHard, CartpoleBalance/Swingup,
FingerSpin, WalkerRun). Averaged over the suite it's a wash; the only robust effect is glass costing **~1.35×
wall-clock**. On a value-sufficient SimNorm latent an explicit representation abstraction has nothing left to
add — a conclusion carried by the **16 direct nulls of the matched benchmark**, not by the old
\(R^2\approx0.999\) value-probe reading, which our own
[postmortem](../2026-06-17-tdmpc-glass-r2-criterion-postmortem) retracted as saturated-by-construction.
**State abstraction is redundant, full stop.**

**(ii) The PPO column is the exploration split.** Where PPO was run (the far-right column), the same law as the
manipulation scan: **HopperHop** — TD-MPC2 265 vs PPO 33 (PPO never learns to hop, exploration-hard);
**CheetahRun / FingerTurnHard** — PPO 928 / 968 wins the dense ceiling, but at **285M** steps vs TD-MPC2's ~1M;
**WalkerWalk** — both solve (~970). So on DMControl too: TD-MPC2 wins exploration-bottlenecked tasks outright and
1–2 orders of magnitude faster; PPO wins the dense ceiling given its throughput. No world-model variant
Pareto-dominates PPO — the sample-efficiency ↔ wall-clock trade is fundamental, and §3–4's dexterous-manipulation
scan is just the high-DoF end of the same axis. *(Full per-task learning curves live on the dashboard generator;
PPO ran on a subset at 50–285M steps — budgets labelled, never fabricated.)*

## 6. Process notes (the part that keeps us honest)
- **Two reward-gaming traps caught by scoring real success** — TD-MPC2's, and my own premature write-up. Return
  is not success on shaped manipulation; we now score `box_target`/native-success *first*.
- **The matched-budget control did its job again**: the apparent beats only existed against the wrong baseline
  (return, or under-budgeted PPO). Every "beats X" in the ledger is qualified by a same-budget control.
- All numbers are deterministic, disk-backed, multi-seed where stated; corrections stay visible in
  `bet2_null_results.md`.

## 7. Discussion & plan (after the supervisor review)

A review reframed the whole thing. Two corrections and five proposals.

### 7.1 Two corrections
- **The real breakthrough is *exploration*, not learning speed.** TD-MPC2's HopperHop win (367 vs 33) isn't a
  ceiling or even mainly a "speed" story — it's that **planning is a directed-exploration operator**: the model +
  lookahead reach gaits PPO's on-policy sampling never finds. "Speed-not-ceiling" is the weak framing; "planning
  buys exploration, and exploration unlocks final performance on exploration-hard tasks" is the strong one.
- **Anti-collapse is a JEPA idea, and we ran it on the wrong object.** A *pure JEPA* has no reward/value loss, so
  anti-collapse is load-bearing. **TD-MPC2 is not a pure JEPA** — its value loss already anchors the latent
  (value-sufficient), so our DMControl anti-collapse study actually measured *redundancy* ("a relational
  regularizer on a value-sufficient latent hurts"), **not** JEPA anti-collapse. Only the **nav H-JEPA** result is
  a genuine JEPA anti-collapse test. The clean JEPA-on-DMControl test strips the value loss (Proposal D).

### 7.2 Five proposals
- **A — Planning as a directed-exploration operator (★ flagship).** *Isolate* it (coverage of TD-MPC2 planning vs
  PPO on HopperHop/the escape frontier), *amplify* it (intrinsic-novelty MPPI, Plan2Explore-in-TD-MPC2), and
  *direct* it with **SE-discovered subgoals** (run min-2D-SE on the replay-graph → communities/bottlenecks as
  exploration targets). White space: nobody uses learned abstraction to *define the exploration targets a planner
  pursues*, and we own SE + a planner + the exploration-difficulty frontier.
- **B — "When does a behavioral prior help RL?" — finalize.** The 2-axis taxonomy (prior-fit × exploration-
  difficulty) + escape-difficulty frontier is drafted and honest; finish figures and submit.
- **C — Abstraction as variance-reduction.** glass beats TD-MPC2 on 6/16 tasks *on the mean by noise* but with
  lower seed-variance on several — possibly a real "SE reduces collapse-seed variance / improves worst-case"
  section. Cheap per-seed check first; report honestly either way.
- **D — JEPA anti-collapse done *right*.** A **pure self-predictive JEPA** (no value loss) on DMControl, probed on
  a geometric *and* a value readout — does the downstream-dependent taxonomy hold on a real JEPA? Then **pixels**,
  where the information term is actually load-bearing (JEPA vs Dreamer).
- **E — SE for structure *discovery*, not regularization.** We used SE to regularize the latent (redundant); its
  real strength is community/hierarchy detection → subgoal/option discovery (merges into A). That's where "glass"
  belongs after SE-as-representation proved a dead end.

### 7.3 This week on the boxes
- **b3060 (DMControl):** SE-arm finishing (D1, relabelled as a redundancy datapoint) → then **A1** exploration-
  coverage (TD-MPC2 vs PPO vs pi-only) → **A2** intrinsic-novelty MPPI → **A3** SE-subgoal smoke.
- **b3060b (Panda/pixels):** beat-PPO tail → **D3** pixel env + Dreamer baseline → **D2** pure-JEPA-on-DMControl setup.
- **No-GPU / writing:** **C** per-seed robustness analysis; **B** finalize the taxonomy paper; draft the
  planning-exploration proposal after a deep-research pass.
- Priority if time-boxed: **A1 (de-risk the flagship) > C > B > D2/D3 > A2/A3.**

Full brainstorm + arm-level plan: `PROPOSALS_and_weekly_plan.md` in the results repo.
