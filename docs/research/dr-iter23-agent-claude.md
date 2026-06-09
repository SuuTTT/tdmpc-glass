# DR iter-23 — internal agent (Claude general-purpose, WebSearch+WebFetch), 2026-06-09

*Round-1 deep-research pass on novel abstraction levers atop the jumpy TD-MPC2 substrate. Verbatim
agent deliverable. Synthesis folded into docs/iterations/iteration_23_ideation.md. Headline
prior-art paper (arXiv:2602.19634) independently VERIFIED real (Farebrother et al., FAIR/Mila, 2026-02-23).*

---

## Headline finding
The directly-overlapping 2026 paper now exists: **"Compositional Planning with Jumpy World Models"
(arXiv:2602.19634)** — but it is policy-composition (composing pre-trained policies via Temporal
Difference Flow occupancy models, zero-shot), NOT MPPI-over-a-jumpy-macro-manifold online. It ships
"a novel consistency objective that aligns predictions across timescales" → our horizon-consistency
reg is now published prior art. The MPPI-over-learned-macro-manifold direction remains open in
continuous control.

## Lever-by-lever (A–H)
- **A. Learned macro-action manifold + MPPI.** Closest: TAP (ICLR 2023, offline VQ + beam search),
  PLAS (offline), DC-MPC (2025, discrete *states* via FSQ, still MPPI over primitive actions). Delta:
  new *combination* — sampling-based MPPI over a continuous learned macro-action latent decoded
  through d_k, online, in TD-MPC2, is undone. No evidence it beats TD-MPC2 online. Failure: VQ
  collapse, decoder OOD = second compounding-error source. Build: HIGH (~2wk).
- **B. Adaptive jump length.** Closest: AdaMVE, TAWM (ICML 2025), jumpy-models already sweep H. Delta:
  weakest, ≈ re-skin. No TD-MPC2 win. Failure: collapses to single best k; fair protocol (beat best
  fixed-k) brutal. Build: high mechanically but protocol kills it.
- **C. Macro-bisimulation.** Closest: DBC, MICo, BS-MPC (ICLR 2025, 1-step π*-bisim on TD-MPC,
  beats/ties TD-MPC on DMC). Delta: new variant (k-step macro-bisim undone). Win likely robustness
  not peak return. Build: medium.
- **D. Two-timescale hierarchical planner.** Closest: Director (NeurIPS 2022), LEXA, IQL-TD-MPC
  (2023). Delta: medium-low; wins are sparse-exploration not dense DMC. Failure: two-timescale credit
  assignment = most finicky. Build: LOW (4+ wk).
- **E. Macro empowerment / MI skills.** Closest: DIAYN, VIC, DADS (MPC-over-skills already!), CIC
  (NeurIPS 2022). Delta: low-med, DADS too close. Wins are URLB/sparse, not dense return. Failure:
  MI objective fights return metric; failed twice for us. Build: medium but misaligned.
- **F. Uncertainty-gated jump length.** Closest: PETS, MOPO, AOP (2019), Plan2Explore. Delta:
  medium-high — uncertainty-gated *jump length* underexplored, principled, complementary to A.
  Failure: ensemble disagreement weak epistemic proxy under deterministic dynamics; +ensemble compute.
  Build: HIGH (~2wk).
- **G. Compositional discrete operators.** Closest: TAP, DC-MPC, "Planning in 8 Tokens" (2026),
  options. Delta: ≈ A discretized; algebra hard to make load-bearing without a compositional benchmark.
  Build: medium, claim hard.
- **H. Cross-task transfer.** Closest: TD-MPC2 multitask (already does it), TD-MPC-Opt, Mixture-of-WM.
  Delta: medium; TD-MPC2 already shows transfer. Build: LOW for 3wk (multitask infra + seeds).

## New levers (off-list)
- **I. Learned proposal-distribution macro-MPPI.** State-conditioned q(z_macro|z_t) seeds macro-MPPI
  (replaces zero-mean Gaussian + policy warmstart). Closest: "Latent Geometry Beyond Search" (2026,
  amortizes search away, goal-conditioned, no TD-MPC2 compare). Delta: NEW MECHANISM — keep MPPI
  robustness + online learning; attacks MPPI sample-inefficiency in k·a_dim. Build: VERY HIGH.
- **J. Multi-k mixture-of-experts jumpy head, return-gated in planning.** {d_{k_i}} tied by consistency
  reg, macro-MPPI picks head per step by macro-return. = B done right (in-planning, return-gated).
  Build: high; single-k-collapse risk.
- **K. Value-equivalent macro jumpy head.** Train d_k value/return-equivalent over k steps (predict
  same macro-Q) not z-reconstruction. Closest: Value-Equivalence Principle (Grimm 2020/21), MuZero.
  Delta: strong, underexplored in MPPI continuous control; the most defensible reason a macro-
  abstraction beats a state-faithful one. Build: medium-high (reuse macro-reward head + macro-TD target).

## RANKED TOP-4: **I > A > F > K**
## VERDICT: pursue **I**; fallback **K**.
I is the only candidate simultaneously (a) genuinely novel in continuous control, (b) buildable in
~3wk as a single-variable change on existing infra, (c) attacks the jumpy baseline's actual weakness
(MPPI sample-inefficiency in macro space) → attributable win. K (value-equivalent macro head) is the
highest-upside fallback: changes *what the abstraction preserves* (return not state).

## Protocol note (load-bearing)
The ONLY reliably demonstrated TD-MPC2-beating gains in continuous control come from value/
representation fixes (BS-MPC ICLR 2025; TD-M(PC)2 2025), concentrated on **high-DoF tasks
(Dog/Humanoid/61-DoF)**. → Run the fair jumpy-vs-lever comparison on **Dog/Humanoid**, where headroom
provably exists, not on saturated low-dim tasks.

## Sources (agent-cited; 2602.19634 verified real)
TAP arXiv:2208.10291 · TD-MPC2 2310.16828 · Compositional Jumpy WM 2602.19634 (VERIFIED) ·
Jumpy Models 2302.12617 · TAWM 2506.08441 · BS-MPC 2410.04553 · Director 2206.04114 · DADS
1907.01657 · CIC 2202.00161 · PETS 1805.12114 · AOP 1912.01188 · DC-MPC 2503.00653 · TD-M(PC)2
2502.03550 · IQL-TD-MPC 2306.00867. (Other post-cutoff 2026 IDs cited but NOT load-bearing; verify
only if cited in a writeup.)
