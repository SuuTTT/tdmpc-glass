# Publication plan — TD-MPC-Glass → "abstraction in world models"

*2026-06-10. North star (user): explore ABSTRACTION IN WORLD MODELS. Near-term output: an honest
understanding paper. Next big bet: structural-entropy on GRAPH world models (see §Future).*

## Paper framing (decide first, with supervisor)
This is an **understanding / methodology paper, not a "we beat TD-MPC2" paper.** Thesis:
> A strong self-predictive world model (TD-MPC2 + SimNorm) already learns a *sufficient, value-aligned*
> abstraction; therefore explicit abstraction objectives (clustering, structural entropy, value-equivalence,
> adaptive temporal abstraction) are redundant or harmful on it. A cheap mechanism-check predicts this a
> priori — validated across 13 levers.

Venue: ICLR main track (understanding) is reachable IF the generality experiments land; **TMLR is the
safest strong fit** for rigorous negative/understanding work. Pick before writing.

## The 5 things that decide publishability
1. **Commit to the understanding-paper framing** (above). Don't dress a negative result as a method paper.
2. **Generality is the #1 reviewer attack** — need a 2nd world model (DreamerV3, in flight) AND the high-DoF
   regime (Humanoid/Dog, where the literature locates abstraction headroom). Highest-upside: if high-DoF is
   NOT value-sufficient, we flip to a positive result.
3. **Mechanism-check methodology is the most novel/transferable asset** — centerpiece = a **predictive-
   validity table** (probe prediction vs actual outcome, all ~13 levers) + compute saved. Make probes
   rigorous (≥3 ckpts×seeds×tasks, not n=1).
4. **Rigor is the moat** — fair protocol, peak+final, paired bootstrap, pre-registered gates, read-from-JSON.
   Close gaps: under-powered arms → n≥5 on discriminators (Pick/Ori/Cabinet); mechanism-checks beyond n=1.
5. **Be ruthlessly honest the only win (jumpy) is prior art** (Farebrother 2026). The integrity ("our protocol
   dissolved our own wins") is what gives the negative thesis weight.

## TODO (critical path, ordered)
**Science / closing**
- [ ] Harvest the running coef sweep (5070 Ti) → confirm value-equivalence null; add ledger row.
- [ ] **Generality (top priority):** value_probe on a high-DoF ckpt (Humanoid/Dog) + a DreamerV3 latent. Decides understanding-vs-positive.
- [ ] DreamerV3 baseline runnable on the suite (standalone launcher in flight) → queue van vs jumpy vs Dreamer.
- [ ] Architecture A/B (ti27a: attn/resmlp vs mlp on PandaPickCube) — STARTED (priority 0); harvest peak+final.
- [ ] Make mechanism-checks rigorous: ≥3 ckpts×seeds×{Pick,Ori,Cabinet}+1 DMC locomotion.
- [ ] Power up jumpy-generalization anchor to n≥5 (Pick/Ori/Cabinet; Cart floored, Push dropped/OOM).

**Writing / packaging**
- [ ] Predictive-validity table (the core figure).
- [ ] SimNorm structural-entropy interpretability section (§8 result).
- [ ] Blog Part-3 (understanding) — draft in flight (docs/blog/blog_phase3_understanding_draft.md).
- [ ] Paper outline: thesis → mechanism-check method → predictive-validity table → per-lever results → jumpy
      (prior art) → interpretability → limitations (single-domain; high-DoF caveat).
- [ ] Related work: Ni et al. 2024 (sufficient self-predictive abstraction), Grimm value-equivalence,
      Farebrother 2026 (jumpy), bisimulation/DeepMDP, + graph/object-centric WMs (for future-work positioning).
- [ ] Reproducibility package (public repo + fair-protocol harness + mechanism-check scripts).

## Future directions (post-paper) — toward "abstraction in world models"
The understanding paper establishes WHEN explicit abstraction is redundant (sufficient self-predictive
*vector* latents on low-DoF tasks). That precisely motivates WHERE it may NOT be:

**★ Graph world models + structural entropy (the user's north star, and the natural sequel).**
- The redundancy that killed the Glass arm was specific: SE imposed on a MONOLITHIC SimNorm latent is
  redundant because SimNorm's softmax is *already* a soft-clustering. **A graph-structured world model
  (nodes = entities/objects, edges = relations, GNN dynamics) has NO such built-in clustering — so SE
  (the information-theoretic graph-hierarchy measure, Li–Pan) is non-redundant there by construction.**
- Thesis for the sequel: *SE-guided hierarchical abstraction in graph world models* — learn a graph latent,
  use 2D/structural-entropy minimization to discover the entity/skill hierarchy, plan at the coarsened level.
  Leverages the project's existing SE tooling (selib `se_louvain`, multilevel 2D-SE).
- Apply the SAME discipline that made this paper: a **mechanism-check first** — does a trained graph-WM's
  latent graph ALREADY carry the SE hierarchy (the way SimNorm did)? If yes → redundant again; if no → headroom.
- Domain matters: needs real graph structure (multi-object manipulation, multi-agent, molecular/traffic) —
  single-arm MuJoCo has weak graph structure. Choose the benchmark for genuine relational structure.

**Other live threads (not killed by the sufficiency finding):**
- High-DoF / long-horizon regime (more state detail → possibly not sufficient).
- Off-distribution / planning-time robustness (WM sufficient on-policy; MPPI queries off-policy states).
- The jumpy-vs-iterated-1-step disagreement signal (Spearman 0.72 w/ error) for safe/abstaining planning.
- Object-centric / factored latents for compositional manipulation (different inductive bias than monolithic).
