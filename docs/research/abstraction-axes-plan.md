# Iteration 30 plan — Using abstraction *right*: aim it where sufficiency fails

*2026-06-11. The post-criterion research program. Source data: RESEARCH_LEDGER.md,
exp/tdmpc_glass/mechcheck/*.json, docs/writeup/redundancy_principle_paper_draft.md.*

## The lesson, sharpened into a positive principle

The redundancy criterion says explicit abstraction is redundant **where the latent is already
sufficient** (value linearly decodable + task-aligned interaction structure). All 16 state-abstraction
levers died exactly there: in-distribution, on the state representation, sufficiency holds.

But the campaign's own data contains the converse signal — the places sufficiency does **not**
automatically hold:

1. **The one real win was temporal.** Jumpy (k-step macro-dynamics) is itself an abstraction — a
   *temporal* one — and it beat vanilla where nothing else did (Ori +90% CI-separated
   [anchor_jumpy_vs_vanilla.json]). A value-sufficient latent does not shorten the *planning horizon*;
   that bottleneck lives outside the representation.
2. **Planning cost is a search problem, not a representation problem.** MPPI searches a k·d-dimensional
   action sequence regardless of how good the latent is. Sufficiency of the state never made the
   *action space* smaller.
3. **And the win is task-dependent** — jumpy tied on Cabinet (1050 vs 1053), won on Ori. Something
   mechanistic separates those tasks, and we don't know what it is. That is an open question with
   our name on it.

**The reframe: "using abstraction right" = aiming it at planning cost (temporal/action axes), not at
representation (state axis).** The state axis is closed by the criterion; the planning axes are open
and have our only positive evidence.

## The three probes (priority order, all single-variable on the existing stack)

### P1 — When does temporal abstraction pay? (completing the criterion's positive side)
The criterion predicts where abstraction is *redundant*; the jumpy Ori-vs-Cabinet split begs for the
mirror tool: a **cheap pre-test that predicts where temporal abstraction helps**. Candidates, all
computable from existing ckpts/dumps before any training: (a) k-step-error decay slope (jumpy-err /
iterated-1-step-err as k grows — the mechanism signal that pre-confirmed Pick), (b) reward smoothness /
contact density over rollouts, (c) MPPI plan-horizon utilization. Protocol: compute candidates on
Pick/Ori/Cabinet/Cartesian ckpts → rank-correlate against the measured jumpy gains (+90/+32/0%) →
the best signal becomes a *pre-registered prediction* for 2 unseen tasks, verified with a k∈{2,4,8}
sweep. Output either way completes the paper's predictive story.
**Cost:** analysis (dev box, ~0 GPU-h) + verification sweep ≈ 2 tasks × 3 k × 5 seeds = 30 runs.

### P2 — Hermite-spline action bottleneck (ledger priority #1, untested)
Macro-action = cubic-Hermite knots (target q,v); PD tracker executes. Shrinks MPPI's search from
k·d to 2d **with no learned codec** → no representation shift, untouched by the redundancy result.
**Mechanism-check first** (dev box, hours): re-fit splines to winning Panda replay trajectories;
GO iff return-preservation ≥95% under spline-restricted actions. Only then the gate: spline-MPPI vs
jumpy, 3 tasks × 5 seeds.
**Cost:** mechanism-check ~0; gate = 30 runs.

### P3 — Value-equivalent macro head (ledger #2 — the VE variant our null does NOT cover)
Our VE null tested *latent* value-equivalence on the 1-step model. This is different: train the
k-step head d_k to preserve **macro-return** (same macro-Q) instead of state-faithfulness — return-
equivalence riding the jumpy win, where capacity actually binds. Single loss change.
**Cost:** mechanism-check (probe d_k's state-error vs Q-error decomposition on existing jumpy ckpts,
~0) + gate = 30 runs.

### Parked — compositional-OOD state abstraction
The synthetic gate showed flat-concat value-decodability does NOT collapse at held-out object counts
(R² 0.96→0.92/0.94 [gate0.json]); without a measured C1-collapse there is no headroom signal. Revisit
only if a real multi-object env shows R² collapse (the criterion's falsification path, Prediction 1).

## Sequencing & gates
Week 1: P1 analysis + P2 mechanism-check (both on dev box, ~free) → kill or promote each.
Week 2+: whichever of P1-verify/P2-gate/P3-gate survives its mechanism-check, one at a time,
under the standing protocol (single-variable vs jumpy baseline, compute-matched, 5 seeds,
peak+final, pre-registered CI gates, read-from-JSON).

## Fleet for fast iteration (right-sized 2026-06-11)
**Keep (6):** dev 5070 Ti (24 cpu, $0.169) for mechanism-checks/dev; 4× A4000 workers — ssh4_a4000
(25 cpu, $0.089, best worker), ssh2_a4000 / ssh4_a4000b / ssh9_a4000 ($0.086, 6 cpu); ssh6_3060
($0.057, cheapest, fine for 500k Panda runs). ≈ **$0.57/hr ≈ $14/day.**
**Destroyed today (6):** orphaned ssh3 A4000 (idle for days, unnoticed), priciest A4000 ($0.103),
2-cpu A4000 (compile-starved), and 3 dead boxes (2 exited, 1 offline).
**Destroy when dgen finishes:** 2080 Ti ($0.089 — old arch, frequent disk-full) and Titan V
($0.102 — worst $/perf in the fleet).
Throughput at this size: ~16-20 Panda runs/day → one full 30-run gate every ~2 days, mechanism-checks
same-day. That cadence matches the protocol (mechanism-check gates prevent more than ~1 gate being
worth running at a time), so more GPUs would not iterate faster — the bottleneck is decision quality,
not compute.
