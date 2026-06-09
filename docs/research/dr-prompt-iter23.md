# Deep-research prompt — iter-23: a NOVEL abstraction lever on a jumpy world model (continuous control)

*Copy the block below into Claude / GPT / Gemini deep-research. Collect all three, we synthesize.*

---

## Context for the researcher

I work on **model-based reinforcement learning in continuous control**. My baseline is **TD-MPC2**
(latent world model + MPPI planning; encoder → SimNorm latent, learned latent dynamics, reward/Q/
policy heads, MPPI/CEM planner over short action-sequence horizons, two-hot distributional value).
Benchmarks: MuJoCo Playground / MJX tasks (DMC-style locomotion + Franka PandaPickCube manipulation).

I have just validated a **jumpy (k-step) latent world model** on top of TD-MPC2: a head
`d_k(z_t, a_{t:t+k}) → z_{t+k}` trained on (state, k-step-action, future-state) pairs with a
horizon-consistency regularizer, plus a macro-reward head, planned by a macro-MPPI that takes
n_macro jumpy steps (effective horizon = k·n_macro) with few model applications — giving a long
effective horizon WITHOUT the compounding 1-step model error. This works (modest win + better
training stability vs vanilla on manipulation), but **the jumpy model itself is a known idea** and
I treat it only as a validated SUBSTRATE.

## What I actually want

A **genuinely NOVEL abstraction mechanism BUILT ON the jumpy substrate** that beats the jumpy model
itself (not just vanilla TD-MPC2) under a **fair, compute-matched, single-variable protocol** (no
population-based training, no per-seed tuning tricks; I report both peak/best-checkpoint AND
final/last-2 with bootstrapped CIs over >=3-5 seeds). The abstraction can be over actions, state,
time, or skills — but it must add something the plain jumpy model lacks.

## Hard constraints / what counts

1. **Single-variable & falsifiable:** the candidate must be testable as one change vs the jumpy
   baseline, with ONE metric that would kill it.
2. **Precedent honesty:** for each idea, tell me the closest prior work and exactly WHY my variant
   is or is NOT novel relative to it. I would rather hear "this is just TAP/PLAS/DIAYN re-skinned"
   than discover it later. Distinguish "novel mechanism" from "novel combination/setting".
3. **Continuous-control evidence:** has anyone shown this (or close) to BEAT a strong model-based
   baseline (TD-MPC2 / Dreamer / TAP) in continuous control? Cite. I am especially interested in
   ideas that have a sound mechanism but NO continuous-control success yet (real gap) vs ideas that
   are known-finicky (e.g. bisimulation, DIAYN/empowerment have failed for me twice in this setting).
4. **Buildability:** rough implementation cost on top of an existing jumpy-TD-MPC2 codebase.

## Candidate levers I am already considering — react to and rank these, then ADD better ones

- **A. Learned macro-action space:** action bottleneck φ(a_{0:k})→m (low-dim), jumpy dynamics
  conditioned on m, decoder back to primitives; MPPI plans over the macro-action manifold.
  (Precedent I know: TAP, PLAS, latent-action RL — is my MPPI-over-jumpy-macro-manifold version
  distinct enough?)
- **B. State-dependent adaptive jump length:** model picks k locally (small near contacts, large in
  smooth regions); variable-horizon planner.
- **C. Reward-predictive (bisimulation) temporal abstraction:** compress the k-step latent to only
  return-relevant info — bisimulation at the macro timescale.
- **D. Two-timescale hierarchical jumpy planner** (high-level waypoints, low-level primitives).
- **E. Macro-scale empowerment / controllability skill discovery** (I(m; z_{t+k}|z_t)).
- **F. Epistemic-uncertainty-gated jumpy horizon** (plan as deep as the ensemble is confident).
- **G. Compositional discrete jumpy operators** (learned composable latent action algebra).
- **H. Cross-task transferable jumpy-macro abstraction** (multi-task pretrain → transfer).

## Deliverables I want from you

1. A **ranked shortlist (top 3-4)** of the most novel-yet-feasible levers — including any NEW ones
   you propose that beat my list — scored on novelty, feasibility, single-variable cleanliness, and
   size of the prior-precedent gap.
2. For each: the **precise mechanism**, the **closest prior work + honest novelty delta**, the
   **expected effect & WHY**, a **pre-registerable falsification gate** (task + metric + threshold),
   and **continuous-control evidence (or its absence)**.
3. **Failure modes / why it might be a mirage** for each (I have been burned by abstraction ideas
   that look good on one snapshot/seed and evaporate under n-scaling).
4. A one-line verdict: **of all of these, which single lever has the best (real-novelty × feasible-
   in-3-weeks × likely-to-actually-beat-jumpy) and why?**

Be skeptical and specific. Cite papers. If the honest answer is "none of these will beat a strong
jumpy baseline and here's the one structural reason," say that.
