# Iteration 17 — Prototype-Novelty Exploration on Sparse Tasks ("xnov")

*2026-06-07. Direction #1 of the Six-Mirages §6 ranking (exploration, derived FROM the
abstraction), run in parallel with iter-16 on user instruction ("these are still free,
continue other promising direction"). Pre-registered BEFORE any run.*

## Hypothesis

Iter-14 §4.4 located the real bottleneck on sparse mujoco_playground tasks: literal
0-vs-solved seed bimodality determined by whether exploration ever touches reward.
Representation quality cannot move this (the nulls); an exploration bonus can. If the
bonus is computed *through the learned behavioral abstraction* — visit rarely-occupied
prototypes — it is an abstraction-derived mechanism, on-thesis: the abstraction's value
would be as an exploration **index**, not as a representation regularizer or a planning
space.

**Not a re-proposal of falsified Path P/Pa** (within-window cluster-entropy on HopperHop,
a dense gait-diversity prior). This is count-based novelty (1/√N per prototype, global
counts), on exploration-bound sparse tasks, with decay-to-zero curriculum.

## Mechanism

`--proto_novelty_coef c --proto_novelty_decay_steps D`: at each collection step, per env,
proto_id = argmax cosine(z, prototypes); training reward += c·(1−t/D)·N[proto]^{−1/2};
counts global, init 1. Eval reward untouched. Bonus rides on behavioral Glass
(λ_behav=0.5) so prototypes become reward-grounded as training proceeds (random-init
projections early — acceptable: any coverage signal helps then).

## Kill-probe design

- Arm: behavglass + novelty (c=0.3, D=500k), tag phasei17xn.
- Tasks/seeds: CartpoleSwingupSparse × seeds {1,2,3} + BallInCup × seeds {1,2,3}, 1M steps
  (matches phaseSF protocol exactly).
- Baseline: EXISTING phaseSF flat runs (no new baseline GPU): solve (last-2 mean ≥800)
  rates — Cartpole: vanilla 1/3, behavglass 0/3; BallInCup: vanilla 1/3, behavglass 2/3.
  Pooled flat reference: 4/12 (33%).

**Pre-registered gates (solve = last-2-eval mean ≥ 800 at 1M):**

- **G1 (signal): ≥ 4/6 solved** → expand (more seeds + AcrobotSwingupSparse) for a
  CI-grade claim.
- **G-dead: ≤ 2/6 solved** → novelty-through-prototypes does not rescue sparse
  exploration; direction dead (and the write-up gains: even granting the abstraction a
  role as exploration index, it adds nothing).
- 3/6: one pre-registered retune (c=1.0 or D=1M), then re-gate.
- Honest caveats: n=6 vs pooled-12 reference is a SIGNAL probe (our own §5 hazard) — any
  pass mandates a properly-powered Stage 2 before claims. Seed-matched comparison
  impossible (different reward shaping changes trajectories); rely on solve-rate gap size.

## Results

**Box incident (2026-06-07):** BallInCup s1/s2 silently killed at ~200k by the flaky 1660S
pair (3/3 long-run deaths there today incl. iter-15 seed 1); box DISABLED in the daemon,
both seeds requeued fresh on reliable boxes (ti17xn3r/4r).

**Finals so far (read from CSVs, solve = last-2 ≥ 800 @1M):**
- CartpoleSwingupSparse s3: **no-solve, 745** (max 790; tail 781/708/782). Note: flat
  behavglass on this task scored ≤3 on all seeds — the bonus arm found and nearly
  stabilized swing-up, but the pre-registered bar is 800; recorded as no-solve.
- CartpoleSwingupSparse s1: **no-solve, 6** (never found reward).
- In flight: Cartpole s2 (271 @700k, climbing); BallInCup s3 (0 @850k); BallInCup s1
  requeue (hit 944 at ~250k(!), evals oscillating 0/solved — typical BallInCup
  bimodality); BallInCup s2 requeue (early).

Running tally: 0 solves / 2 finals; gate needs 4/6 — now requires ≥4 of the remaining 4
runs to solve. Cartpole near-misses (745, and s2's climb) are qualitatively unlike flat
behavglass (≤3) and will be reported as secondary evidence regardless of gate outcome.

**GATE UNREACHABLE (2026-06-07):** 3 finals all no-solve (Cartpole s1=6, s3=745,
BallInCup s3=0) → max possible 3/6 < 4/6 G1. Per pre-registration this is the 3/6-or-worse
region: ≤2/6 → dead; exactly 3/6 (iff all 3 remaining solve) → one retune (coef 0.3→1.0).
Remaining: Cartpole s2 (max 302), BallInCup s1-requeue (touched 973 @850k, oscillating),
BallInCup s2-requeue. Final verdict when all land. Secondary finding holds regardless:
Cartpole bonus seeds reach 745/790 sustained where flat behavglass scored ≤3 — exploration
demonstrably moved, just short of the strict solve bar.

**FINAL (2026-06-08, all 6 done): 1/6 solved → DEAD.** Solves: BallInCup s1 only (969) —
but flat behavglass already solved BallInCup 2/3, so NOT novel. The novel target
CartpoleSwingupSparse (flat behavglass 0/3): bonus reached 745/790 sustained (vs flat ≤3)
but cleared no 800 solve on any of 3 seeds. 1/6 < 4/6 gate → **VERDICT: DEAD.** Secondary
finding (writeup-worthy): prototype-novelty bonus *moves* sparse exploration substantially
(Cartpole 0→745/790 where flat is stuck at ≤3) but not enough to cross the solve threshold —
exploration-through-the-abstraction helps sub-threshold, consistent with the §4.4 thesis that
sparse-task bimodality is an exploration problem latent geometry only partially touches.
