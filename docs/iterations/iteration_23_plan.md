# Iteration 23 — SE-k: structural-entropy adaptive jump-length on the jumpy substrate

*2026-06-09. The genuinely-novel lever, greenlit by the SE pre-check (tier-2 PASS: real jumpy
CheetahRun latents show 53.1% k-step transition SE gap, 47.2% kNN). Anchored to our Glass/SE paper
line (SIDM/SISA/SI2E/SIHD) — all model-free/diffusion; **nobody has used structural entropy to set
temporal abstraction for an MPPI planner over a learned latent world model.** That gap is the novelty.*

## The claim (single-variable, falsifiable)
**Community-aware adaptive jump length beats fixed-k jumpy.** The jumpy model takes a long jump (large
effective k) where the latent dynamics are coherent and a short jump (small k) where they change
sharply — and structural-entropy communities of the latent transition graph identify exactly those
regions: **long k INSIDE a community (a motion phase / coherent dynamics regime), short k at community
BOUNDARIES (contacts, turning points, phase transitions)** — the very places the compounding-error tax
is worst. This is the iter-19 community structure (which failed as *skills*) put to its correct use:
*temporal abstraction*, not goal-reaching. (See docs/research/se-precheck-note.md §3.)

## Why this is novel (precedent honesty)
- SIDM (JMLR 2025) uses directed-SE to set a *skill hierarchy* executed by SAC — model-free, no planner.
- SIHD (NeurIPS 2025) uses SE to set *diffusion* temporal scales — no learned WM, no MPPI.
- Jumpy/compositional-planning works (Farebrother 2026) vary k but NOT via structural entropy.
- → SE × learned-latent-WM × MPPI-jump-length is unoccupied. New *mechanism+setting*, not a re-skin.

## Mechanism (batch-friendly — soft masks, NOT ragged horizons)
Gemini's constraint: variable per-trajectory k breaks GPU-parallel MPPI batching. So we keep a FIXED
macro-grid of K_MAX micro-steps and use a **soft continuation mask** to realize variable effective k:

1. **Periodic graph build (every G env-steps, e.g. 50k; cheap, host-side).** Roll out the current
   policy (or sample the buffer), encode latents z, k-means to N=128 nodes, build the k-step transition
   graph P (node i→j counts). Sparsify (top-frac, keep≈0.2) and run SE-optimal partition (greedy 2-D SE
   / Louvain) → community label per node + boundary score per node (fraction of out-mass crossing
   communities, = `bottleneck_prototypes` score in skills.py). Cache: node centroids C (N,latent),
   community labels, per-node boundary score b∈[0,1].
2. **At plan time (inside macro-MPPI).** For each rolled latent z along a sampled macro-trajectory,
   assign nearest node → boundary score b(z). Define a per-micro-step **continuation gate**
   c_t = σ((τ − b(z_t))/temp) ∈ (0,1): high (≈1, keep jumping) inside a community, low (≈0, stop/short
   jump) at a boundary. The effective return uses a CUMULATIVE-PRODUCT discount of the gate so reward
   past a crossed boundary is down-weighted (the planner stops "trusting" the long jump exactly where
   the dynamics regime changes):  G = Σ_t (Π_{l≤t} c_l) · γ^t · r_t  over the fixed K_MAX grid.
   This is identical machinery to lever F's uncertainty gate — so F and SE-k SHARE the soft-mask code;
   the only difference is the gate signal (community-boundary score vs ensemble disagreement). Run both.
3. No new large network; the SE machinery is host-side numpy/networkx (skills.py + se_precheck.py).
   Single new variable vs fixed-k jumpy = the boundary-gated continuation mask.

## Pre-registered gate (mechanism-check FIRST, then fanout)
- **Mechanism-check (cheap, before any multi-seed run):** on a trained jumpy ckpt, verify (a) the
  boundary score b(z) actually spikes at contact/phase-transition states (inspect on CheetahRun: do
  high-b states cluster at foot-strike?), and (b) gating long jumps by b reduces k-step prediction
  error vs fixed-k at matched mean-k. If b is uninformative (doesn't track real dynamics change) → SE-k
  is dead, fall back to F. KILL CONDITION: b–error correlation ≈ 0.
- **Beat gate (fanout only if mechanism-check passes):** SE-k vs fixed-k jumpy, compute-matched,
  single-variable, rliable IQM over **5 seeds, peak AND final, bootstrap CI**, on: CartpoleSwingupSparse,
  BallInCup (sparse), PandaPickCube (manipulation/contact), **HumanoidRun (high-DoF — where the DR says
  headroom lives, and where the jumpy substrate is being trained this window)**.
  **WIN = SE-k ≥ +10% IQM with non-overlapping 95% CI on ≥3 of 4 tasks (peak AND final).**
  If SE-k ties fixed-k everywhere → honest null (the structure exists but doesn't help planning — the
  iter-19 lesson at a new level); record and fall back to F as the safe modest lever.

## Risks / mirage-guards
- SimNorm density → raw graph is a blob; MUST sparsify/kNN (pre-check confirmed). Use keep≈0.2, SE-optimal.
- Graph staleness: rebuild every G steps; cheap. Compute-match (count the graph-build cost against the budget).
- "structure exists ≠ helps planning": the mechanism-check (b vs error) is the explicit guard before fanout.
- Distrust 1-snapshot: read ≥400k, peak+final, 5 seeds. No procedure tricks.

## Build status (scaffold)
- [x] SE machinery present: skills.py (communities, bottleneck score), se_precheck.py (SE gap, kmeans).
- [x] SE_DUMP latent-dump path in run_benchmark (tier-2 used it).
- [ ] code skeleton: `make_se_gate(centroids, labels, boundary)` host-side; thread a `--se_gate` flag +
      a cached `se_graph.npz` into make_jumpy_mppi_fn's return computation (cumulative-gate discount).
      Share the soft-mask with lever F (`--unc_gate`).
- [x] mechanism-check BUILT: SE_DUMP mech mode (run_benchmark, env-gated) dumps per-step k-step model
      error e_t; se_precheck.py mechcheck() correlates SE boundary-score b(z_t) vs e_t (Spearman + hi/lo
      tertile ratio; PASS if spearman>0.15 AND hi>1.1x lo). Plumbing self-tested on synthetic data.
      TO RUN: needs a jrew-containing fast jumpy ckpt (phasei22fast k4) on a FREE box (k2 ckpt lacks
      jrew -> warmup crashes). Deferred to next freed box; do NOT co-run heavy jax on busy boxes.
- [ ] HOLD multi-seed fanout for explicit user go.

## Parallel: F (uncertainty-gated horizon) — the safe fallback, shares machinery
Same soft-mask continuation gate, signal = jumpy-dynamics ensemble disagreement instead of SE boundary
score. Build both behind flags; if SE-k mechanism-check fails, F is the modest-but-likely lever (3/3 DR
consensus). F's own pre-check: does ensemble U_dyn track true k-step error on SimNorm latents (guard
against epistemic saturation)?
