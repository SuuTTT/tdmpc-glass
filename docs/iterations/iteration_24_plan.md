# Iteration 24 — SI2E-style SE-driven exploration (the SE lever's correct home)

*2026-06-09. SE-k/adaptive-k died (jumpy error uniform → nothing to gate). But pre-check 1 confirmed the
SE structure is REAL (53% gap). Its right job is COVERAGE/exploration, not jump-length. User pointed to
two reference impls: github.com/SuuTTT/learn-si2e (faithful + fast SI2E) and github.com/SuuTTT/selib
(fast Louvain SE). This iter ports the FAITHFUL formulation to TD-MPC2 on sparse tasks.*

## Goal / gate (pre-registered, unchanged)
SE-driven exploration must **BEAT RND** on sparse MJX tasks {CartpoleSwingupSparse, BallInCup,
AcrobotSwingupSparse} — the exact iter-21 G2 bar the *geometric* Laplacian FAILED. rliable IQM, 5 seeds,
peak+final, CI. WIN = SI2E > RND, ≥10% non-overlapping CI on ≥2/3 tasks. Arms: vanilla, RND (have it),
**VCSE** (value-conditioning only), **SI2E** (VCSE + cluster term). The VCSE arm replicates the refs'
headline ("value-conditioning is the dominant win, +55pp over plain SE") and isolates SI2E's added value.

## Faithful formulation (from learn-si2e; the v1 in code is NOT this — see below)
- **VCSE** (Value-Conditional State Entropy, the dominant win): intrinsic reward = particle entropy =
  `r0 = log( ||x_i − x_i^{kNN}|| )` in the JOINT space `x = [normalize(φ(s)); β·normalize(V(s))]`, i.e.
  the state feature φ(s) CONCATENATED with the critic value V(s). kNN over a recent buffer. Effect:
  rewards states far from same-VALUE neighbors → stops wasting bonus on dead-ends (refs H1: +55pp vs SE).
- **SI2E** = `r = H(V0) − H(V1)`: leaf VCSE (r0 above) MINUS a cluster-level VCSE (r1) computed on the
  community centroids of an encoding-tree partition. The cluster term = "group novelty" → cuts seed
  variance (refs: SI2E 100%±0 vs VCSE 97.8%±3.1 on DoorKey-8x8). r1 = VCSE(cluster_centroids).
- **Fast variant = kmeans clustering** (refs benchmark `fast-si2e-kmeans`; leiden/infomap also). Our
  kmeans path = exactly this. selib.optimal_2d (se_louvain) = the "proper-SE" variant (needs sklearn;
  skip on workers — kmeans is the validated fast variant).
- Feature φ(s): a fixed **random encoder** (A2C variant) — encoder-independent, no training. Replaces
  v1's Laplacian-trained embedding (simpler, matches the reference A2C random_encoder).
- **Value V(s)**: from our critic — V(s) ≈ two_hot_inv(min-Q(s, π(s))). MUST be threaded into the
  intrinsic branch (the one new wiring vs v1).

## Code status
- [x] Harness wiring VALIDATED by v1 smoke: `--intrinsic si2e` branch + periodic rebuild run clean on
  CartpoleSparse (50k, SPS ~200, loss finite, no NaN, ~12 rebuilds). The faithful version reuses this.
- [x] kmeans community machinery + 3-64 granularity guard (fixed the kNN over-fragmentation bug).
- [ ] **REPLACE v1 reward (count-coverage) with faithful VCSE leaf−cluster:** rewrite make_si2e to a
  random-encoder φ + buffer of (φ, V) + value-conditional kNN entropy r0 − cluster entropy r1. Add a
  `make_vcse` (r0 only) for the ablation arm. Thread V(s) (critic) into the si2e/vcse branch.
- [ ] Mechanism-check before fanout: on a sparse smoke, does VCSE/SI2E find reward FASTER than RND on
  ≥1 seed (sanity that the bonus shapes exploration), and is r1 (cluster term) non-degenerate?
- [ ] Then the 4-arm × 3-task × 5-seed gate (fan out across fleet; HOLD full campaign for user go).

## Honest prior
iter-21 Laplacian (geometric SE-exploration) LOST to RND. SI2E is value-conditional (richer) + the refs
show it beats VCSE/SE/baseline on MiniGrid + DMC — so a fair re-attempt with a real chance, but the bar
(beat RND on OUR sparse MJX tasks) is exactly where abstraction-flavored exploration failed before. Kept cautious.
