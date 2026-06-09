# Iteration 19 — Community-Detection Skill Discovery ("glass-skills")

*2026-06-08. User-directed (the goal is to BEAT TD-MPC2 with abstraction learning/planning).
The most Glass-native idea: skills = communities in the latent transition graph, found by the
same structural-entropy machinery we built — applied TEMPORALLY instead of to state clusters.*

## Why this can work where everything else nulled

Every nulled direction failed because abstraction had no job (proprio DMC) or was action-blind
(iter-15 reward-equivalence quotient). Community-detection skills avoid both:
- A skill = **"reach community c"**, a navigational/goal-conditioned option → inherently
  action-conditional → passes the iter-15 controllability law (I(a;next|state) > 0).
- It targets a real bottleneck: TD-MPC2's myopic H=3 (gated by iter-18 H-sweep).
- Communities in the transition graph are regions separated by **bottleneck states** — the
  classic "good subgoals" of options theory (eigenoptions / Louvain-options / bottleneck
  options), here discovered by 2D-SE / Louvain on the prototype graph Glass already builds.

## Architecture (staged; each stage gated)

**Substrate (running):** geoglass (λ_behav=0, pure SE community detection on the transition
graph) on tasks with rich multi-phase structure (AcrobotSwingup, HopperHop, PandaPickCube).
Dumps prototype transition graphs P (glass_diag npz) as it trains.

**Stage 1 — communities exist? (offline, ~0 GPU).** Run `helios.algorithms.skills` community
detection on the substrate P-matrices. Gate: modularity > 0.3 AND a stable
(seed/step-consistent) partition into 3–8 communities with clear bottlenecks. Core validated
on synthetic graph (recovery acc 1.0, modularity 0.61). If real Glass graphs have no community
structure → skills have no substrate → reconsider granularity (finer latent discretization).

**Stage 2 — skills reach communities? (cheap GPU).** Goal-conditioned low-level policy
π(a | z, c_target) trained with intrinsic reward = entering c_target's community (centroid
proximity / community-membership). Gate: skill policy reaches its target community
> baseline-random from arbitrary starts. (This is the controllability test in practice.)

**Stage 3 — planning over skills beats flat TD-MPC2 (the BEAT test).** High-level MPPI/search
over skill sequences (which community next) with the low-level executing; effective horizon =
H × skill-length. Compare to vanilla TD-MPC2 at matched compute on long-horizon tasks, IQM+CI,
≥5 seeds, fixed cutoff. WIN = CI-separated above vanilla.

## SOTA baselines (Stage 3 comparison)

- **METRA** (metric-aware temporal abstraction) — current SOTA unsupervised skill discovery.
- **DIAYN** (MI-based diversity skills) — the classic.
Our claim of novelty needs glass-skills to ≥ match these while being graph/SE-principled and
planning-integrated (most skill methods are model-free; ours plans).

## Discipline (hard-won)

Pre-register n and CI per stage; ≥5 seeds for any "beat" claim; report estimate trajectories;
a transient single-snapshot separation is a mirage (we logged seven). No Stage-N fan-out
before Stage-(N-1) passes its gate.

## Status
- skills.py community-detection core: BUILT + validated on synthetic (acc 1.0). 2026-06-08.
- Substrate geoglass runs: QUEUED (ti19sub*, AcrobotSwingup×2, HopperHop, PandaPickCube).
- Next: pull substrate P-matrices → Stage-1 gate (do real Glass graphs have communities?).

## Results

**Stage-1 community gate — PASS (with scoping), 2026-06-08.** Ran skills.community_structure
on real Glass transition graphs (glass_diag npz pulled from boxes).
- Raw P is too dense (40–76% effective edge density — 1e-4 smoothing + diffuse SimNorm
  dynamics) → Louvain sees one blob, modularity ≈ 0. **Fix baked into skills.py: sparsify to
  top 20% of edges before detection** (`keep_frac=0.2` default).
- Sparsified results: **CartpoleSwingupSparse → 4 balanced communities, modularity 0.705
  (PASS, clean skill substrate)**; HopperHop → structure present (mod 0.445) but
  over-fragmented (12 comm, needs coarser resolution/keep_frac); HumanoidWalk → diffuse
  (mod 0.0, high-dim — NOT a skill substrate).
- **Scoping conclusion:** community-skills have real substrate on **structured/bottlenecked
  low-dim tasks** (sparse swing-up, and presumably manipulation), NOT smooth high-dim
  locomotion. This tells us exactly where skill-planning could beat TD-MPC2 — and it overlaps
  the iter-18 long-horizon gate tasks (Cartpole-sparse, Panda).

**Next:**
- Stage-1b (cheap): community STABILITY across training steps/seeds (a skill must target a
  persistent community, not one that reshuffles each eval). Pull multi-step npz, check
  partition agreement (e.g., ARI across steps).
- PandaPickCube substrate bumped to priority 1 (manipulation = ideal bottlenecked substrate;
  iter-18 confirms vanilla learns it, max ~2300).
- Stage-2 build (new code): skill-conditioned policy π(a|z, c_target) + intrinsic reward =
  entering c_target community; gate = reaches target > random.

**Stage-1b community STABILITY — PASS (2026-06-08).** CartpoleSparse glass_diag across late
training (700k→950k): 6–8 communities, modularity 0.54–0.68, **consecutive-step ARI
0.90–0.97**. Communities are persistent → a skill can target one and trust it. PASS.

**Stage-2 primitives BUILT + validated offline (2026-06-08).** Added to skills.py:
`assign_community(z, prototypes, labels)` (nearest-cosine-prototype → community id) and
`skill_reward(z_next, target_comm, ...)` (sparse +1 for entering target community + 0.1·cos
shaping toward its centroid). Synthetic check: assignment correct, reward 1.1 for the true
target vs ~0 for others. These drop into the collection loop as an intrinsic reward (same
host-side pattern as the existing cluster_id_batch path).

**ALL gates green → Stage-2 TRAINING build authorized.** Spec (next, requires smoke before
fanout — touches the collection loop, do NOT blind-fanout):
- Condition the agent on a target community by APPENDING its one-hot to the raw observation
  (low-risk: obs_dim grows by n_comm, encoder+heads adapt automatically — like the distractor
  wrapper, no architecture surgery). Resample target community per episode.
- Define communities from a FROZEN pretrained Glass checkpoint (encoder+prototypes+labels via
  skills.community_structure) so the skill target is stable, not circular.
- Replace collection-loop task reward with skill_reward(encode(obs_next), target_comm, frozen
  protos, labels). 
- Probe gate: from random starts, goal-conditioned policy reaches the specified target
  community > random-policy baseline (the controllability test in practice). Cheap: 1 task
  (CartpoleSparse, where Stage-1 gave mod 0.705 + ARI 0.92), 2 seeds, 300k.
- Then Stage-3: high-level MPPI over skill sequences vs vanilla (the BEAT test) + METRA/DIAYN.

**Stage-2 progress (2026-06-08): goal-MPPI built; substrate established (with a RISK).**
- `make_goal_mppi_fn` BUILT + compiled (tdmpc_glass.py): goal-conditioned MPPI to reach a
  community centroid (per-step −L2 + terminal cosine bonus, no reward/Q heads). Serves as
  Stage-2 probe AND Stage-3 low-level.
- **SUBSTRATE-QUALITY RISK (important):** community structure is highly substrate-dependent.
  geoglass CartpoleSparse seed_0 SOLVED the task (803) but its transition graph is DEGENERATE
  (2 comm [1,31]) — a converged policy collapses latent visitation. seed_1 (exploring, 0
  reward) gives mod 0.158 (fragmented). The CLEAN substrate is iter-17 behavglass seed_2:
  7 comm, mod 0.546, stable ARI 0.92. **Lesson: skill substrate must come from a
  BROADLY-EXPLORING, non-degenerate model — and even then quality varies (0.16–0.70). If
  Stage-2/3 hinge on a fragile substrate, that's a real threat to robustness.**
- Substrate locked: iter-17 behavglass CartpoleSparse seed_2 — checkpoint + community file
  (7 comm, centroids) backed up to control plane (exp/tdmpc_glass/skill_substrate/) so box
  recycling can't lose it. Re-pushable to any box for eval.

**Next:** build skill-eval (load ckpt + community file, run make_goal_mppi_fn toward each
centroid from random starts, reach-rate = frozen-encoder nearest-prototype community == target,
vs random-action baseline). SMOKE on a box. GATE: reach-rate > random clearly on ≥3 of the 7
communities → Stage-2 PASS → Stage-3 beat test.

**Stage-2 GATE: FAIL (2026-06-08) — and a clean mechanism for WHY.**
- skill_eval reach-rate: goal-MPPI 0.14 vs random 0.14, **0/7 communities** (only c4, the
  rest-state attractor, ever "reached" — by both). goal-MPPI ≈ random.
- Diagnosis (community visitation under the trained policy): 5/7 communities ARE visited
  (c0 0.31, c4 0.31, c5 0.27, c2 0.08, c3 0.03) — they're reachable. But the policy *cycles
  through* them during swing-up/balance: **the communities are MOTION PHASES (cyclic,
  transient), not spatial subgoals.** "Reach and stay in community c" is ill-posed for a
  phase — you pass through it, you don't arrive. Centroids (mean latent) are non-physical.
- **Conclusion:** community-detection-as-SUBGOALS does not fit continuous-control phase-cyclic
  tasks. Options-via-community work in NAVIGATION (mazes, spatial manipulation) where
  communities = persistent spatial regions; in cartpole/locomotion they = gait phases. This
  is the same structural reason abstraction keeps nulling here (iter-15 action-blindness,
  iter-14/16/17 nulls): the latent has no navigational/subgoal structure to exploit.

**PIVOT (within temporal abstraction): community subgoals -> JUMPY MULTI-STEP MODEL (iter-20).**
The iter-18 gate's real lesson stands: long horizon helps on sparse (H9=712 vs H3=0) but
naive H9 has a compounding-1-step-error tax (Panda collapse). The clean way to get long
*effective* horizon WITHOUT the tax is a learned k-step dynamics head d_k(z, a) predicting
z_{t+k} in one call (fewer model applies -> less compounding) — NOT community reaching. BEAT
test: jumpy-planning matches H9 on CartpoleSparse (finds reward) AND beats H9 on Panda (no
collapse) = a genuine architectural long-horizon win, not just a hyperparameter. See
iteration_20_plan.md. (Community detection retained as a finding, not a mechanism.)
