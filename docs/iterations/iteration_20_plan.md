# Iteration 20 — Jumpy Multi-Step Latent Model ("jumpy")

*2026-06-08. Pivot from iter-19 community-skills (Stage-2 failed: communities are motion
phases, not reachable subgoals). The iter-18 gate's real opportunity, instantiated directly.*

## The opportunity (from iter-18, locked)
Longer planning horizon helps where reward is beyond H=3's reach (CartpoleSparse H9=712 vs
H3=0) but naive H9 carries a **compounding 1-step-model-error tax** (PandaPickCube H9 collapsed
1678→419). The architectural win = long *effective* horizon WITHOUT rolling the 1-step model
many times.

## Mechanism
Add a **k-step (jumpy) dynamics head** d_k(z, a) that predicts z_{t+k} in a single call,
trained on (z_t, a_t, z_{t+k}) pairs from the buffer (k≈4). MPPI plans over the jumpy model:
H_macro jumpy steps = effective horizon k·H_macro with only H_macro model applies → far less
error compounding than rolling 1-step k·H_macro times. Single-variable vs vanilla TD-MPC2 (add
the head + a jumpy-planning eval; keep everything else). Controllability law: the jumpy step is
action-conditional by construction (d_k(z, a)) — passes iter-15.

## Pre-registered gates (staged, smoke before fanout)
- Stage-A (cheap): does the k-step head TRAIN to low prediction error? (jumpy consistency loss
  converges). Smoke 60k.
- Stage-B (the BEAT probe): jumpy-MPPI vs vanilla H3 and H9 on CartpoleSparse (sparse, the
  horizon-limited win) and PandaPickCube (where H9 collapsed):
  - WIN-1 (CartpoleSparse): jumpy finds reward (last-2 ≥ 400, ≫ H3=0), ≥3/4 seeds.
  - WIN-2 (PandaPickCube): jumpy ≥ H3 (~1490) and > H9 (419) — i.e. long horizon WITHOUT the
    collapse. This is the genuine architectural claim.
  - Compute-matched, ≥5 seeds, IQM+CI, fixed cutoff.
- If jumpy only matches H9 (no Panda improvement) → it's a hyperparameter, not a win → honest
  null; pivot to multi-task transfer (direction #2).

## Honest prior
~30%. Jumpy/temporal-difference models are known but integrating one that beats both H3 and H9
on a strong baseline is unproven. The compounding-error argument is sound; whether a learned
k-step model is accurate enough to realize it is the empirical question.

## Status
- iter-19 community-skills: Stage-2 FAILED (communities = phases not subgoals); recorded.
- Baselines ready: CartpoleSparse H3=[0,0,0,0,0,18,97] (solid 0-floor), H9 maturing; Panda H3/H9 building.

**REVISED Stage-A — the MINIMAL lever first (cheaper than a jumpy head):** vanilla H9 ALREADY
trains the dynamics over 9-step rollouts (seq_len=H+1=10), yet collapses on Panda — because
rho=0.5 down-weights long horizons (rho^9≈0.002), so the model is never trained to be ACCURATE
at depth 9. So **rho (consistency-horizon decay) is the one-parameter lever for stable deep
planning.** Added `--rho` flag (run_benchmark + run_dmc_baseline RHO env). If raising rho fixes
the H9 collapse, we get the architectural win with ~zero new code; if not, build the jumpy head.
- Stage-A smoke QUEUED (ti20smoke_rho): rho=0.9 H9 CartpoleSparse 40k — verify the high-rho
  long-rollout gradient trains (no NaN) before fanout.
- Stage-B (after smoke): rho∈{0.5(=collapse control, have it), 0.9} × H9 on PandaPickCube
  (does rho0.9 prevent the 1678→419 collapse?) + CartpoleSparse (still finds reward?), ≥5
  seeds IQM+CI. WIN = rho0.9+H9 ≥ H3 on Panda (no collapse) AND finds CartpoleSparse reward.
- Fallback if rho doesn't fix it: build the k-step jumpy dynamics head (iter-20b).

## Results

**Stage-A smoke PASS (2026-06-08, ssh8 direct):** rho=0.9 H9 CartpoleSparse 60k — 'rho
override rho=0.9' active, loss finite throughout (no NaN), 50k MPPI eval ran clean
(0.0 — expected, sparse reward not yet found). The high-rho long-rollout gradient is stable.
→ Stage-B cleared.

**Stage-B LAUNCHED (ti20rho*, 6 runs):** rho=0.9 H9 on PandaPickCube {0,1,2} + CartpoleSparse
{0,1,2}, 500k. Controls (rho=0.5): Panda H9=419 (collapse) / H3≈1490; CartpoleSparse H9≈680 /
H3=0.
GATE (when mature): WIN if rho0.9+H9 Panda ≥ ~1490 (no collapse, vs rho0.5 H9=419) AND
CartpoleSparse finds reward ≥400. Honest caveat: H9+rho0.9 vs H3 is a horizon+schedule
combo, i.e. a TUNING/algorithmic-schedule win, not a new architecture — but a single config
strictly better than vanilla's default (beats H3 on sparse, ties on manipulation) is still a
real improvement over TD-MPC2 as shipped, and if rho is the universal lever for stable deep
planning that's a genuine algorithmic insight. If Panda still collapses → rho isn't the lever
→ build k-step jumpy head (iter-20b).

**NOTE (ops):** external queue-adds race with the daemon's queue writes — adds aren't seen
until a daemon restart. Workaround: restart daemon after queueing (done here), or run
controlled smokes directly on a verified-free box (done for Stage-A).

**Stage-B FIRST READ (2026-06-08, ~450k) — MIXED, task-dependent:**
- **PandaPickCube rho0.9+H9 = 1976, 2157 (n=2 mature, mean 2067)** — NOT collapsed; *above*
  H3 baseline (~1490) and ≫ rho0.5+H9 collapse (419). **High rho cures the deep-planning
  collapse AND boosts Panda +38% over the H3 default.** Strong, needs n=5 + CI + attribution.
- **CartpoleSwingupSparse rho0.9+H9 = 0** (s0=15, s1/s2=0 @250-350k) — vanilla H9 (rho0.5)
  found reward (~680) by 200k. **rho=0.9 SUPPRESSED sparse exploration** (regression).
- **Verdict: NOT a strictly-better config** (fails the universal gate — Cartpole regressed).
  It's a TRADE: high rho helps manipulation deep-planning, hurts sparse exploration.

**Follow-ups launched (disentangle + recover):**
- Panda rho0.5+H3 baseline seeds 4,5 (n-boost the comparison baseline, currently n=2).
- Panda rho0.9+H3 seeds 0,1 (ATTRIBUTION: is the Panda win from deep planning H9, or just
  high rho? If rho0.9+H3 ≈ 1490 and rho0.9+H9 = 2067, the win is DEEP PLANNING enabled by rho).
- CartpoleSparse rho0.7+H9 seeds 0,1 (MIDDLE-GROUND: does rho0.7 keep sparse reward like
  rho0.5 AND avoid the Panda collapse? a rho that wins both would be the clean result).
- rho0.9+H9 Panda/Cartpole continue to n=5.

Honest framing stands: any rho+H finding is an algorithmic/tuning result, not a new
architecture — but "TD-MPC2 with H9+high-rho beats default H3 by 38% on manipulation" is a
real, useful improvement if it holds with CI.

*(gate verdict from CSVs only — verification discipline.)*

## FOOTNOTE CLOSED (2026-06-09, n=3): rho is a task-dependent TUNING lever, not architecture.
Panda rho0.9+H9 = [1951,1509,1863] mean 1775 (>H3~1490, NO collapse vs rho0.5-H9=419) — rho
robustly fixes the deep-planning collapse + ~19% over H3 on manipulation. BUT CartpoleSparse
rho0.9+H9 = [0,0,642] (2/3 suppressed) — high rho costs sparse exploration. TASK-DEPENDENT
trade, hyperparameter not architecture. Recorded as footnote; not pursued further.
