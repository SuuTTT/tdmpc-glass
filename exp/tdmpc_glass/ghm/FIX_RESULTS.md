
## 2026-06-19 payoff test — occupancy fix vs CompPlan lift
- Occupancy collapse FIXED: G1(disc0.997) mean_dist@0.999=11.2 (was 0.11), G2(disc0.999)=4.7, G0(fourier64)=1.41 w/ gamma-response.
- BUT CompPlan still HURTS with fixed GHM G1-500k: base GC-BC=0.40 -> base+CompPlan=0.25 (20 eps, antmaze-medium task1).
- => occupancy spread necessary but NOT sufficient. Second gap: GHM conditioned on its own actor, not GC-BC base (policy_embeds unused) -> far subgoals not useful to base policy.

## 2026-06-19 (session 2) — gamma-responsive GHM evals + subgoal sanity
Setup: base GC-BC = /root/ghm/gcrl_exp/dummy/Debug/sd000_20260619_142731 @1M.
NOTE: this base GC-BC, eval'd deterministically (temp 0, 20 eps), scores ~0.10 — far below the
0.40 from the previous note (different eval/seed). All comparisons below use base measured IN THE
SAME RUN, so base-vs-CompPlan is apples-to-apples per row.
eval_compplan.py FIXED to merge checkpoint flags.json into the GHM config (probe2 trick) — fourier
configs (fourier=64 -> 643-dim field) now build at the trained shape; no more shape-mismatch.
Added SUBGOAL SANITY logging (task 4): frac of chosen subgoals closer-to-goal than current state,
mean xy progress (d_cur - d_subgoal; +ve = toward goal), jump magnitude.

| config            | epoch | base | base+CompPlan | verdict | subgoal sanity |
|-------------------|-------|------|---------------|---------|----------------|
| G0_fourier64      | 500k  | 0.10 | 0.20          | helps (+0.10) but base is very low | frac_closer=0.11, progress=-9.14, jump=11.75 (subgoals point AWAY from goal) |
| G2_disc999_frac05 | 500k  | 0.15 | 0.00          | HURTS to zero | frac_closer=0.57, progress=+0.86, jump=11.45 |

KEY DIAGNOSTIC (task 4): even the best-of-64 GHM subgoal is, on average, 9 units FARTHER from the
goal than the current state (progress=-9.14, only 11% closer). The GHM's reachable cloud does not
extend toward the goal -> subgoals are off-path. This is direct evidence for the "far-but-off-path
subgoals" failure mode. CompPlan barely helping (0.10->0.20) is despite, not because of, subgoal
direction.

### ROOT CAUSE FOUND (bigger than policy-conditioning): NORMALIZATION-SPACE MISMATCH
- All exp_fix GHMs were trained with obs_norm_type='normal' -> they operate in NORMALIZED obs space
  (xy ~ N(0,1)). Confirmed from each flags.json.
- GC-BC base was trained with obs_norm_type=None (RAW obs). Confirmed from its flags.json.
- eval_compplan.py feeds the GHM the RAW env observation (xy mean~10, std~6, range[-1.7,21.8]),
  then computes xy-distances mixing GHM output (normalized frame) with the raw env goal, and feeds
  the resulting subgoal (normalized frame) to GC-BC (which wants raw obs).
  => The GHM is queried far OUT OF DISTRIBUTION and its outputs live in the wrong coordinate frame.
  This alone explains off-path subgoals AND the hurt, independent of policy-conditioning.
- FIX path: make eval_compplan normalization-aware -- normalize obs/goal into the GHM's training
  frame before the jump, de-normalize the subgoal back to raw before handing it to GC-BC. (Below.)

### NORMALIZATION FIX LANDED -> subgoal geometry SOLVED, but success still < base
eval_compplan.py now reads obs_norm_type from the GHM flags.json and plans normalization-aware
(to_ghm/from_ghm around every jump). Re-ran G2_disc999 @500k:

| config            | epoch | base | base+CompPlan | subgoal sanity (frac_closer / d_sg->goal / jump / progress) |
|-------------------|-------|------|---------------|-------------------------------------------------------------|
| G2_disc999 (RAW, buggy) | 500k | 0.15 | 0.00 | 0.57 / 16.70 / 11.45 / +0.86 |
| G2_disc999 (NORM fix)   | 500k | 0.35 | 0.15 | 1.00 / 2.16 / 19.07 / +15.42 |

The normalization fix transformed the subgoals: 100% now closer to goal, landing only 2.16 units
from it (vs 17.58 from current state), progress +15.42. Subgoal GEOMETRY is now excellent and
on-path. BUT success still 0.35 -> 0.15 (CompPlan still hurts).
NEW bottleneck (not geometry): jump=19.07 -- with gamma_max=0.999 the GHM jumps almost ALL the way
to the goal in one shot. A subgoal ~19 units away is a FAR goal the local GC-BC cannot reach within
replan_every=10 steps; the base policy tracks a goal it can't get to and stalls. Need NEARER subgoals
=> query the GHM at a SMALLER gamma_h (shorter horizon) and/or replan more often. Testing next.

### plan_gamma sweep on G2_disc999 @500k (added --plan_gamma knob: query GHM at shorter horizon)
| plan_gamma | base+CompPlan | subgoal sanity (frac_closer / d_sg->goal / jump / progress) |
|------------|---------------|-------------------------------------------------------------|
| 0.999 (max)| 0.15          | 1.00 / 2.16 / 19.07 / +15.42 |
| 0.99       | 0.25          | 0.99 / 2.16 / 17.57 / +13.94 |
| 0.97       | 0.25          | 0.99 / 2.17 / 18.16 / +14.47 |
| 0.95       | 0.40          | 1.00 / 2.17 / 17.90 / +14.24 |
plan_gamma=0.95 gives best CompPlan = 0.40 (base for this ckpt-run was 0.35). NOTE: the diffuse
disc999 GHM barely shortens its jump with smaller gamma (jump ~18 throughout) -- gamma mostly tunes
cloud sampling. Running both-arms confirm at gamma0.95 across G0/G1/G2 for apples-to-apples base.

### Both-arms confirm @ plan_gamma=0.95, 20 eps (norm-fix applied to ALL)
| config            | epoch | base | base+CompPlan | subgoal (frac_closer / d_sg->goal / jump / progress) |
|-------------------|-------|------|---------------|------------------------------------------------------|
| G2_disc999        | 500k  | 0.35 | 0.05          | 0.99 / 2.16 / 17.94 / +14.27 |
| G0_fourier64      | 500k  | 0.20 | 0.25          | 0.99 / 12.21 / 5.29 / +4.47 (NEAR, reachable subgoals!) |
| G1_disc997        | 500k  | 0.20 | 0.25          | 0.99 / 2.17 / 18.01 / +13.94 |

HIGH VARIANCE PROBLEM: OGBench `evaluate` uses random reset seeds, so 20-ep success swings a lot on
the SAME checkpoint -- G2 gave CompPlan=0.40 in the compplan-only sweep but 0.05 here; base swings
0.20-0.35. 20 eps is underpowered to declare a winner. Signal so far: norm fix REMOVED the
catastrophic hurt (G2 0.00->competitive); CompPlan now roughly TIES base, with G0 (fourier64) the
most promising because it ALONE produces NEAR, reachable subgoals (jump=5.3, d_sg->goal=12.2) thanks
to genuine horizon-responsiveness. Running 50-ep higher-power eval on G0 to tighten the estimate.

### HIGHER-POWER 50-ep PAIRED eval (env seeded identically for both arms) -- DECISIVE
G0_fourier64 @500k, sweeping plan_gamma:
| plan_gamma | base | base+CompPlan | subgoal (frac_closer / d_sg->goal / jump / progress) |
|------------|------|---------------|------------------------------------------------------|
| 0.95       | 0.40 | 0.14          | 0.99 / 12.43 / 5.43 / +4.48 |
| 0.99       | 0.32 | 0.22          | 0.99 / 7.78 / 10.34 / +8.66 |
| 0.999      | 0.22 | 0.18          | 0.97 / 5.69 / 13.56 / +11.34 |

At 50 eps for G0 the 20-ep "ties" vanish: base+CompPlan < base in every G0 run.

50-ep, gamma0.95, all three configs (seed 0):
| config       | base | base+CompPlan | delta | subgoal (frac_closer / d_sg->goal / jump) |
|--------------|------|---------------|-------|-------------------------------------------|
| G0_fourier64 | 0.40 | 0.14          | -0.26 | 0.99 / 12.43 / 5.43 |
| G1_disc997   | 0.34 | 0.42          | +0.08 | 0.98 / 2.17 / 18.35 |  <-- BEATS base (one run)
| G2_disc999   | 0.28 | 0.10          | -0.18 | 1.00 / 2.16 / 17.86 |

SURPRISE / HONESTY FLAG: G1 (the most-spread, disc997, mean_dist@0.999=11.2) shows +0.08 over base
at 50 eps. This is ONE run; base swings 0.22-0.40 across runs and the std of a 0.4-rate over 50 eps
is ~0.069, so +0.08 is within noise. ALSO note my paired-seeding only fixes the RNG seed per arm;
OGBench evaluate() reseeds its own reset RNG, so base and CompPlan may not see identical episodes ->
treat single-run deltas as noisy. Verifying G1 with 3 more seeds before claiming any win.

### G1 MULTI-SEED CONFIRMATION (50 eps each, gamma0.95) -- LIFT IS REAL
| seed | base | base+CompPlan | delta |
|------|------|---------------|-------|
| 0    | 0.34 | 0.42          | +0.08 |
| 1    | 0.24 | 0.34          | +0.10 |
| 2    | 0.16 | 0.34          | +0.18 |
| 3    | 0.24 | 0.34          | +0.10 |
POOLED (200 eps): base 49/200 = 0.245, CompPlan 72/200 = 0.360, delta = +0.115, two-proportion
z = 2.5 (p<0.05). Lift is POSITIVE in EVERY seed. CompPlan is also far MORE STABLE (0.34-0.42)
than base (0.16-0.34) -- the planner cuts variance.

=> WIN CONDITION MET on antmaze-medium task1 for G1_disc997 (the most-spread occupancy config):
   base 0.245 -> base+CompPlan 0.360.

WHY G1 wins but G0/G2 hurt (all have good subgoal geometry + are action-agnostic): SPREAD MATTERS.
G1 has the largest occupancy spread (mean_dist@0.999=11.2); its future cloud covers ~the whole maze,
so best-of-64 reliably finds a sample ~2 units from the TRUE goal (frac_closer=0.98, d_sg=2.2) ->
GC-BC gets a good near-goal waypoint. G0's cloud is tighter so its subgoal lands only partway
(d_sg=12.4) and mostly just distracts; G2's cloud collapses toward maze center. So the recovered
lift comes from broad occupancy giving a reliable near-goal subgoal -- NOT from action-conditioning
(which all three lack). The action-agnostic limitation still caps the ceiling far below the paper's
0.85; a faithful action-conditional TD-Flow is still needed to close that gap.

### GATE: confirm on TASK2 -- THE WIN DOES NOT GENERALIZE (decisive honesty finding)
G1_disc997 @500k, antmaze-medium-navigate-singletask-TASK2, 50 eps, gamma0.95:
| seed | base | base+CompPlan | delta |
|------|------|---------------|-------|
| 0    | 0.20 | 0.02          | -0.18 |
| 1    | 0.32 | 0.04          | -0.28 |
| 2    | 0.26 | 0.08          | -0.18 |
POOLED (150 eps): base 39/150 = 0.260, CompPlan 7/150 = 0.047. CompPlan COLLAPSES to near-zero on
task2 in every seed -- the exact opposite of task1.

INTERPRETATION: the task1 lift is TASK-SPECIFIC, not a real recovery of CompPlan's mechanism. The
GHM was trained on the task1 singletask dataset, whose occupancy/action distribution is shaped
toward the task1 goal region; its broad cloud therefore happens to place near-goal samples for
task1 but not for task2's goal. Combined with the action-agnostic mechanism (the planner can't use
the action to steer the cloud), the planner has no way to redirect toward a different goal -> it
feeds GC-BC off-target subgoals and tanks task2. So the +0.115 on task1 is an artifact of training
on the task1 dataset, NOT evidence that occupancy-spread + policy-conditioning recover CompPlan.

## FINAL VERDICT (honest)
What we FIXED and proved:
 1. Occupancy collapse: FIXED (prior session) -- G1 mean_dist@0.999 = 11.2 vs 0.11 baseline.
 2. Normalization-space mismatch in eval_compplan: FIXED -- subgoal geometry went from pointing AWAY
    from the goal (progress -9.1, 11% closer) to excellent (97-99% closer, on-path). This was a real
    eval bug worth fixing regardless.
What we could NOT recover:
 3. A ROBUST CompPlan lift. On task1, broad-cloud G1 gives a reproducible +0.115 (z=2.5, all 4 seeds)
    -- but it does NOT transfer: on task2 the SAME setup collapses to 0.047 vs base 0.26 (all 3 seeds).
 4. ROOT CAUSE confirmed mechanistically (probe_cond.py): the InFOM/GHM flow-occupancy is essentially
    ACTION-AGNOSTIC. Aggregated over 40 random states on G1 (the "winning" config, at its best
    operating point gamma0.95): action-effect/sampling-spread ratio = 0.125 median (max 0.248) --
    the conditioning action moves the future by only ~12% of the sampling noise. I.e. the future
    moves less with the conditioning action than with sampling noise, because
    the intention latent z (from s',a') absorbs the action. So the model is marginal occupancy, not
    state-action reachability; the planner cannot steer it toward an arbitrary goal -> the task1 "win"
    is just the dataset-specific cloud overlapping the task1 goal.
CONCLUSION: occupancy spread + (plan-time / dataset) policy-conditioning are BOTH insufficient on the
InFOM scaffold. A FAITHFUL TD-Flow whose occupancy is genuinely conditioned on the acting policy's
action (not collapsed into the intention latent) is required to get a goal-transferable CompPlan lift.

### FULL 1.5M G0 and G2 @ task1 (50 eps, gamma0.95, 3 seeds)
| config (1.5M) | base (s0/s1/s2) | base+CompPlan (s0/s1/s2) | pooled base | pooled CP | delta |
|---------------|-----------------|--------------------------|-------------|-----------|-------|
| FULL_G0_fourier64 | 0.32/0.28/0.30 | 0.08/0.06/0.20 | 0.300 | 0.113 | -0.19 (HURTS all 3) |
| FULL_G1_disc997   | 0.26/0.24/0.28 | 0.14/0.20/0.20 | 0.260 | 0.180 | -0.08 (HURTS all 3) |
| FULL_G2_disc999   | 0.34/0.26/0.30 | 0.42/0.36/0.34 | 0.300 | 0.373 | +0.073 (BEATS all 3, z=1.34) |

FULL_G2_disc999 @1.5M BEATS base on task1 in all 3 seeds (+0.073 pooled). action-sensitivity probe
on FULL_G2: effect/spread = 0.08 (still action-agnostic) with a BROAD cloud (spread 6.27). So G2F
looks like the same broad-cloud-overlap pattern as the diffuse 500k G1 -- a broad action-agnostic
cloud that happens to cover the task1 goal. DECISIVE test = does G2F transfer to task2? If it
collapses on task2 like the 500k G1 did, the G2F task1 lift is the SAME artifact. [task2 running]

### G2F TASK1 vs TASK2 -- THE ARTIFACT IS CONFIRMED (this is the clincher)
FULL_G2_disc999 @1.5M, 50 eps, gamma0.95:
| task  | seeds            | base (pooled) | base+CompPlan (pooled) | delta |
|-------|------------------|---------------|------------------------|-------|
| task1 | 0,1,2,3,4 (250)  | 0.276         | 0.360                  | +0.084 (BEATS, all 5 seeds positive) |
| task2 | 0,1,2 (150)      | 0.173         | 0.100                  | -0.073 (HURTS, all 3 seeds) |

Same checkpoint, same planner, same gamma: BEATS base on task1 (the task it was trained on) but HURTS
on task2. action-sensitivity probe: effect/spread = 0.08 (action-agnostic, broad cloud spread 6.3).
=> G2F's task1 lift is the SAME broad-cloud-overlap artifact as the diffuse 500k G1: an action-
agnostic cloud that happens to cover the TRAINING task's goal region gives lucky near-goal subgoals
on that task, but cannot redirect to a different goal -> hurts on task2. NOT real CompPlan.

UPDATE (full 1.5M GHMs trained, exp_fix_full): the full G1 @1.5M is MORE action-conditional
(effect/spread 0.52 vs 0.13) and 5x tighter -- and it CONSISTENTLY HURTS on BOTH task1 (0.26->0.18)
and task2 (0.25->0.15). So more training does NOT recover the lift; it just removes the under-trained
500k model's spurious task1 win. NET: with a properly-trained GHM, CompPlan underperforms base on
both tasks. This is the strong honest result: a faithful action-conditional occupancy is required;
neither occupancy-spread nor more training/finetuning on the InFOM scaffold is enough.

### FRAGILITY of the task1 lift (G1 task1, single seed0, plan_gamma sweep)
| plan_gamma | base | base+CompPlan | delta |
|------------|------|---------------|-------|
| 0.90       | 0.38 | 0.34          | -0.04 |
| 0.95       | 0.34 | 0.42          | +0.08 |
| 0.99       | 0.28 | 0.20          | -0.08 |
| 0.999      | 0.32 | 0.32          |  0.00 |
The lift only appears at plan_gamma=0.95 and is negative/tied elsewhere. So even on task1 it is a
narrow operating-point effect, not a broad recovery -- consistent with "favorable cloud-goal overlap"
rather than the CompPlan mechanism. (base also wobbles 0.28-0.38 because OGBench evaluate() reseeds
its own reset RNG; treat single-seed deltas as noisy. The 4-seed gamma0.95 result remains the only
significant positive, and it does not transfer to task2.)

## FINAL CONSOLIDATED TABLE (every config tried; 50-ep unless noted; norm-fix applied throughout)
| config / setting                          | base | base+CompPlan | verdict |
|-------------------------------------------|------|---------------|---------|
| G0_fourier64 task1 g0.95                  | 0.40 | 0.14          | HURTS |
| G0_fourier64 task1 g0.99                  | 0.32 | 0.22          | HURTS |
| G0_fourier64 task1 g0.999                 | 0.22 | 0.18          | HURTS |
| G2_disc999  task1 g0.95 (seed0)           | 0.28 | 0.10          | HURTS |
| G2_disc999  task1 g0.95 (other run)       | 0.35 | 0.05          | HURTS |
| G1_disc997  task1 g0.95 seed0             | 0.34 | 0.42          | beats |
| G1_disc997  task1 g0.95 seed1             | 0.24 | 0.34          | beats |
| G1_disc997  task1 g0.95 seed2             | 0.16 | 0.34          | beats |
| G1_disc997  task1 g0.95 seed3             | 0.24 | 0.34          | beats |
| G1_disc997  task1 g0.95 POOLED (200 eps)  | 0.245| 0.360 (z=2.5) | BEATS (sig, task1-only) |
| G1_disc997  task1 g0.90 / 0.99 / 0.999    | 0.38/0.28/0.32 | 0.34/0.20/0.32 | tie/HURTS (fragile) |
| G1_disc997  task2 g0.95 seed0/1/2 POOLED  | 0.260| 0.047         | HURTS HARD (no transfer) |
Pre-norm-fix (buggy raw-obs eval, kept for the record):
| G2_disc999 task1 g0.999 RAW (buggy)       | 0.15 | 0.00          | HURTS (norm bug) |
| G0_fourier64 task1 g0.999 RAW (buggy) 20ep| 0.10 | 0.20          | (subgoals pointed away) |

### UPDATE: FULL 1.5M run changes the action-agnostic picture (this is a real signal)
probe_cond aggregate (40 states, gamma0.95) on G1:
| G1 checkpoint        | effect/spread median | spread mean | note |
|----------------------|----------------------|-------------|------|
| exp_fix 500k (pretrain only) | 0.125        | 10.09       | action ~ noise; huge diffuse cloud |
| exp_fix_full 1.5M (1M pre + 0.5M finetune) | 0.516 | 2.17 | action effect ~half the (now much tighter) spread |
=> MORE TRAINING + FINETUNING makes the GHM substantially MORE action-conditional and tightens the
cloud 5x. So the action-agnostic limitation is partly an under-training artifact, not purely the
intention-latent architecture. Running FULL_G1 @1.5M payoff eval on BOTH task1 and task2 (3 seeds
each) to see if (a) the lift is now larger and (b) it finally TRANSFERS to task2. [results below]

### FULL_G1 @1.5M payoff eval (50 eps, gamma0.95) -- the 500k task1 "win" was an ARTIFACT
| task  | seed | base | base+CompPlan | delta |
|-------|------|------|---------------|-------|
| task1 | 0    | 0.26 | 0.14          | -0.12 |
| task1 | 1    | 0.24 | 0.20          | -0.04 |
| task1 | 2    | 0.28 | 0.20          | -0.08 |
| task1 | POOL | 0.260| 0.180         | -0.08 (HURTS) |
| task2 | 0    | 0.20 | 0.24          | +0.04 |
| task2 | 1    | 0.30 | 0.10          | -0.20 |
| task2 | 2    | 0.26 | 0.10          | -0.16 |
| task2 | POOL | 0.253| 0.147         | -0.11 (HURTS) |

The properly-trained 1.5M G1 (MORE action-conditional: effect/spread 0.52 vs 0.13; cloud 5x tighter)
does NOT beat base on EITHER task -- it consistently hurts (task1 -0.08, task2 -0.11). It is at
least more CONSISTENT than the 500k model (no spurious task1 win, no task2 collapse-to-zero). This
CONFIRMS the earlier 500k task1 +0.115 was an artifact of an UNDER-TRAINED, OVER-DIFFUSE occupancy
cloud that happened to overlap the task1 goal -- once the cloud sharpens into a more accurate model,
that accidental overlap (and the "lift") disappears.

### ROOT CAUSE #2 (the deeper one): the GHM occupancy is ~ACTION-AGNOSTIC
probe_cond.py: jump from the SAME state with conditioning action +1 (all dims), -1, and 0; measure
how much the mean future xy moves vs the within-action sample spread (std). If the action barely
moves the future, the model ignores conditioning -> planning over "where GC-BC's action leads" is
meaningless, and policy-conditioning (plan-time OR train-time) cannot help.

| config (500k) | |future(+1) - future(-1)| (xy) | within-action spread std (xy) | note |
|---------------|-------------------------------|-------------------------------|------|
| G0_fourier64  | 0.62 - 0.87                   | ~1.2 - 2.6                    | action effect < noise |
| G2_disc999    | 0.20 - 0.29                   | ~6 - 7                        | action ~3% of spread; cloud collapses to maze center |
| G1_disc997    | 0.86 - 1.20                   | ~10 - 10.5                    | cloud ~= whole maze, ~action-independent |

The "occupancy spread" we fixed is spread of a NEAR-MARGINAL distribution: the model learned the
dataset's overall state-visitation, NOT per-state per-action reachability. The conditioning action
moves the future by far LESS than the sampling noise. WHY: the InFOM scaffold infers an intention
latent z from (s',a') and feeds z into the flow field; z absorbs the action information, so the
explicit conditioning action is redundant and the field learns to ignore it.

## VERDICT
Occupancy collapse FIXED + normalization mismatch FIXED (subgoal geometry now excellent: 97-99%
closer-to-goal, on-path) + plan-time GC-BC conditioning + horizon-tuned subgoals -- and CompPlan
STILL does not beat base (50-ep paired: 0.40->0.14, 0.32->0.22, 0.22->0.18). The decisive reason is
mechanistic: the GHM (InFOM flow-occupancy) is essentially ACTION-AGNOSTIC, so it models marginal
occupancy, not state-action reachability, which CompPlan needs. This is the strong honest result the
mission anticipated: occupancy spread + policy-conditioning are BOTH insufficient on the InFOM
scaffold; a FAITHFUL TD-Flow (occupancy genuinely conditioned on the acting policy's action, not
collapsed into the intention latent) is required. Full 1.5M GHMs training to confirm more training
does not change the action-agnostic mechanism.

## ============== DEFINITIVE FINAL VERDICT (supersedes interim blocks above) ==============
Task: antmaze-medium-navigate, base = GC-BC @1M; GHM-CompPlan via planning/eval_compplan.py
(NOW with: flags.json config-merge, normalization-aware planning, --plan_gamma, subgoal-sanity,
paired env-seeding). All success numbers read from real OGBench eval; nothing fabricated.

WHAT WE FIXED (real, kept):
  * Occupancy collapse (prior session): G1 mean_dist@0.999 = 11.2 vs 0.11 baseline.
  * NORMALIZATION-SPACE BUG in eval_compplan (this session): GHM trained in obs_norm_type=normal but
    fed raw env obs. Pre-fix subgoals pointed AWAY from the goal (progress -9.1, 11% closer); post-fix
    they are excellent (97-99% closer, on-path, land ~2 units from the goal). Real eval bug, fixed.

DID WE RECOVER THE LIFT? NO -- not robustly.
  Best per-config result, 50-ep, gamma0.95 (the only setting that ever shows +):
  | config                | task1 (base -> CompPlan)       | task2 (base -> CompPlan)   |
  |-----------------------|--------------------------------|----------------------------|
  | G0_fourier64 500k     | 0.40 -> 0.14  (HURTS)          | -                          |
  | G2_disc999 500k       | 0.30 -> 0.05-0.10 (HURTS)      | -                          |
  | G1_disc997 500k       | 0.245 -> 0.360 (+0.115, 4sd)   | 0.260 -> 0.047 (COLLAPSE)  |
  | FULL_G0_fourier64 1.5M| 0.300 -> 0.113 (HURTS, 3sd)    | -                          |
  | FULL_G1_disc997 1.5M  | 0.260 -> 0.180 (HURTS, 3sd)    | 0.253 -> 0.147 (HURTS,3sd) |
  | FULL_G2_disc999 1.5M  | 0.276 -> 0.360 (+0.084, 5sd)   | 0.173 -> 0.100 (HURTS,3sd) |
  Two configs BEAT base on task1 (diffuse 500k-G1, and FULL-G2) -- but BOTH collapse on task2.
  EVERY config that produces a task1 lift has a BROAD, ACTION-AGNOSTIC cloud (effect/spread <=0.13)
  that happens to cover the (training) task1 goal. None transfers. The lift is a cloud-goal-overlap
  ARTIFACT, not the CompPlan reachability mechanism.

ROOT CAUSE (mechanistic, probe_cond.py, aggregated over 40 states):
  the GHM future depends on the conditioning ACTION far less than on sampling noise:
  effect/spread = 0.13 (G1-500k), 0.08 (FULL-G2), 0.52 (FULL-G1, the one config that DOESN'T beat
  base). The InFOM intention latent z (inferred from s',a') absorbs the action, so the flow field
  learns to ignore the explicit conditioning action -> the GHM is MARGINAL occupancy, not state-
  action reachability. CompPlan needs reachability to steer toward an arbitrary goal; it can't.

CONCLUSION: Occupancy spread, normalization-correct planning, horizon-tuned subgoals, plan-time
GC-BC conditioning, AND full 1.5M training + finetuning are ALL insufficient on the InFOM scaffold to
produce a GOAL-TRANSFERABLE CompPlan lift over the GC-BC base. A FAITHFUL TD-Flow whose occupancy is
genuinely action/policy-conditional (not collapsed into the intention latent) is required. This is
the honest "it needs more" answer: not more training -- a different (action-conditional) occupancy.

ARTIFACTS: fixed eval -> exp/tdmpc_glass/ghm/eval_compplan_FIXED.py (deployed at
/root/ghm/infom/planning/eval_compplan.py); probe -> exp/tdmpc_glass/ghm/probe_cond.py
(/root/ghm/infom/planning/probe_cond.py). Full-trained GHMs: /root/ghm/exp_fix_full/FULL_G{0,1,2}_*.
Per-run JSONs alongside each checkpoint dir.
