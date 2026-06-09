# Iteration 23 — Ideation: a genuinely NOVEL abstraction lever ON the jumpy substrate

*2026-06-09. User decision: jumpy (iter-22) is a KNOWN method with (likely) a modest win — treat it
as a validated baseline/substrate and build something genuinely new ON it, aiming again at the
original goal: beat vanilla TD-MPC2 at the architecture/algorithm level with a NEW abstraction idea.*

## Honest framing (the bar)
- The jumpy k-step latent model abstracts in **time** (long effective horizon, few model applies).
  That is the substrate, NOT the innovation — temporal/jumpy models are known (TD-models, Δ-models,
  value-equivalent models, Dreamer/MuZero variants).
- A novel lever must add an abstraction the **plain jumpy model lacks**, and the claim must be
  **single-variable falsifiable against jumpy-as-baseline** (not against vanilla — jumpy already
  beats vanilla on Panda). For each candidate: (1) what abstraction does it add, (2) why it isn't
  just X from the literature (precedent honesty), (3) the ONE number that kills it, (4) risk/novelty.
- Discipline carried from iter-22: **mechanism-check BEFORE fanout**; pre-register n + CI; report
  peak AND final; distrust 1-snapshot effects (the "Eight Mirages"). No procedure tricks (no PBT).

## Dependency
Gate-blocked on the iter-22 jumpy substrate validating at n>=3 mature (PandaPickCube, ~4-6h out,
currently 5/5 seeds trending jumpy >> vanilla). Build proceeds against that validated substrate.

---

## Idea list (brainstorm — rank/expand after deep research)

### A. Learned macro-action space (abstract ACTIONS)  ★ current top pick
- **Mechanism:** action bottleneck `φ: a_{0:k} → m` (m ~4-8 dim), jumpy dynamics conditioned on m:
  `d(z, m) → z_{t+k}`, jumpy reward `r(z, m)`, decoder `ψ: (z,m) → a_{0:k}` for execution. MPPI
  plans over the low-dim **macro-action manifold** m, not raw k×act_dim primitive sequences.
- **Abstraction added:** the action space itself, discovered from the world model (temporally-
  extended abstract actions).
- **Precedent honesty:** latent/trajectory action planners exist (TAP, PLAS, latent-action RL) —
  but not, to our knowledge, as a learned macro-action manifold planned by MPPI on top of a jumpy
  model in TD-MPC2 continuous control.
- **Kill-shot metric:** macro-MPPI must beat primitive jumpy-MPPI at MATCHED effective horizon
  (sample-efficiency to threshold, or asymptote) — if equal, the abstraction adds nothing.
- **Risk:** low-med (decoder fidelity). **Novelty:** medium. **Buildability:** high (planner already
  plans over action sequences; swap the search space). → strongest single-variable test.

### B. State-dependent ADAPTIVE jump length (adaptive temporal granularity)
- **Mechanism:** learn where a big jump is reliable (small k near contacts/bottlenecks, large k in
  smooth regions); planner uses variable k per macro-step.
- **Abstraction added:** adaptive temporal granularity (vs fixed-k jumpy).
- **Precedent honesty:** weaker precedent in continuous-control MPC — genuinely underexplored.
- **Kill-shot:** adaptive-k beats fixed-k jumpy at matched compute on mixed-dynamics tasks
  (manipulation: contact + free motion).
- **Risk:** high (variable-horizon planner is fiddly). **Novelty:** high. **Buildability:** med-low.

### C. Reward-predictive temporal abstraction (macro-scale bisimulation)
- **Mechanism:** compress the k-step latent to ONLY return-relevant information — a bisimulation
  metric at the macro timescale, not 1-step.
- **Abstraction added:** reward-aware state abstraction over the horizon (drops irrelevant dynamics).
- **Precedent honesty:** bisimulation at 1-step exists (DBC, BS-MPC; our iter-14 `bisim_coef` was
  finicky). Novel at the jumpy/macro scale.
- **Kill-shot:** wins under distractor/irrelevant-dynamics conditions where plain jumpy wastes
  capacity (we have a distractor harness from iter-14).
- **Risk:** med (bisim instability). **Novelty:** med. **Buildability:** med.

### D. Two-timescale hierarchical jumpy planner
- **Mechanism:** high-level MPPI plans over jumpy macro-steps to pick latent waypoints z*; low-level
  MPPI fills primitives to reach each waypoint. One shared jumpy model, hierarchy in the PLANNER.
- **Abstraction added:** temporal hierarchy in planning (goal-conditioned low level).
- **Precedent honesty:** HRL/feudal/goal-conditioned planning is well-trodden; novelty would be the
  shared-jumpy-model grounding + fair MPPI-vs-MPPI test. Risk of iter-19 failure mode (waypoints
  may not be reachable subgoals).
- **Kill-shot:** hierarchical beats flat jumpy-MPPI at matched compute on long-horizon tasks.
- **Risk:** med-high. **Novelty:** low-med (most literature-adjacent).

### E. Empowerment / controllability at the MACRO scale
- **Mechanism:** discover macro-actions m that maximize controllability/diversity of k-step latent
  outcomes (mutual information I(m; z_{t+k}|z_t) / empowerment in the jumpy latent). Lifts the
  project's "controllability law" to the macro timescale.
- **Abstraction added:** an unsupervised macro-skill space grounded in jumpy reachability.
- **Precedent honesty:** DIAYN/empowerment/VIC exist; novelty = empowerment computed in a learned
  jumpy latent and used to seed MPPI. iter-19/21 unsupervised-abstraction attempts FAILED — caution.
- **Kill-shot:** empowerment-seeded macro-MPPI beats random/primitive macro-MPPI on sparse tasks.
- **Risk:** high (unsupervised objectives finicky; prior nulls). **Novelty:** med-high.

### F. Uncertainty-gated jumpy horizon (epistemic confidence-adaptive)
- **Mechanism:** ensemble/disagreement on the jumpy head; planner trusts the macro-rollout only as
  deep as the model is confident (epistemic-uncertainty-gated effective horizon).
- **Abstraction added:** confidence-adaptive temporal depth (a principled version of B).
- **Precedent honesty:** model-disagreement / pessimism is known (PETS, MOPO); novelty = gating the
  JUMPY horizon specifically.
- **Kill-shot:** uncertainty-gated jumpy beats fixed-horizon jumpy where the model is locally wrong
  (avoids the compounding-error tax adaptively).
- **Risk:** med. **Novelty:** med. **Buildability:** med (ensemble cost).

### G. Compositional jumpy operators (discrete macro-action vocabulary / action algebra)
- **Mechanism:** learn a small discrete codebook of k-step latent operators that COMPOSE; plan over
  sequences of discrete operators (a learned options library / latent action algebra).
- **Abstraction added:** a discrete temporal-action vocabulary with compositional structure.
- **Precedent honesty:** options + VQ skills exist; novelty = composable latent operators verified to
  compose (operator algebra) in TD-MPC2.
- **Kill-shot:** operator-sequence planning beats continuous macro-MPPI on tasks with reusable motifs.
- **Risk:** med-high. **Novelty:** med-high.

### H. Cross-task transferable jumpy abstraction (the multi-task bet, direction #2)
- **Mechanism:** train the jumpy macro-model / macro-actions on MULTIPLE tasks, transfer to a held-out
  task. Abstraction should pay off in transfer where flat per-task models don't.
- **Abstraction added:** reusable temporal abstraction across tasks.
- **Precedent honesty:** multi-task world models exist; novelty = the jumpy-macro representation as
  the transfer unit + a clean fair transfer protocol.
- **Kill-shot:** jumpy-macro pretrain transfers (sample-efficiency on new task) > from-scratch jumpy.
- **Risk:** med. **Novelty:** med. **Buildability:** med (needs a multi-task harness).

---

## Selection rubric (apply after DR)
Score each on **novelty × feasibility × single-variable-cleanliness × prior-precedent-gap**, weighted
by our track record (unsupervised-abstraction bets E/G have failed twice → higher bar; planner/action
levers A/D are most buildable). Pick ONE primary + one cheap fallback. Pre-register gates, mechanism-
check first.

---

## Deep-research synthesis — round 1 (internal agent, 2026-06-09; full report: docs/research/dr-iter23-agent-claude.md)

### VERIFIED prior-art shock (checked on arxiv, real paper)
**"Compositional Planning with Jumpy World Models"** (Farebrother, Pirotta, Tirinzoni, Bellemare,
Lazaric, Touati — FAIR/Mila/McGill, arXiv:2602.19634, 2026-02-23) publishes jumpy multi-step models
+ **a cross-timescale consistency objective** (≈ our horizon-consistency reg). → **Our iter-22 substrate
is no longer conceptually novel.** BUT theirs = Temporal-Difference-Flow occupancy models composing
*pre-trained policies* zero-shot; NOT online TD-MPC2 with MPPI over a learned latent. **The
MPPI-over-a-learned-macro-manifold lane stays open.** (Confirms our "jumpy = substrate, not the
innovation" framing was correct.)

### The single most important strategic finding
For **every** lever, "has this mechanism beaten TD-MPC2/Dreamer/TAP in *continuous control*?" =
**no evidence found**, with two exceptions: **BS-MPC** (1-step bisimulation on MPC, ICLR 2025) and
**TD-M(PC)2** (policy-constraint, 2025). Both are **value/representation fixes**, and their gains
**concentrate on high-DoF tasks (Dog, Humanoid, 61-DoF)** — NOT temporal-abstraction planning.
→ **Protocol pivot: run the fair jumpy-vs-lever comparison on high-DoF DMC (Dog/Humanoid) where
headroom over a strong jumpy baseline provably exists**, not on saturated low-dim tasks where any
"win" is noise. (We've been testing on Panda/Cartpole — add DogRun/HumanoidRun.)

### Precedent verdicts on A–H (condensed)
- **A (macro-action manifold):** new *combination* (TAP=offline beam-search VQ; PLAS=offline; DC-MPC=
  discrete *states* not actions). MPPI-over-learned-continuous-macro-actions online in TD-MPC2 = open. High build.
- **B (adaptive k):** WEAKEST — ≈ TAWM/AdaMVE re-skin; jumpy-models already sweep H. Fair protocol
  (must beat *best* fixed-k) likely kills it. Deprioritize.
- **C (macro-bisimulation):** new *variant* (BS-MPC is 1-step). But bisim gains are robustness/distractor,
  off-task for clean SOTA-chase. Medium.
- **D (hierarchical):** Director/IQL-TD-MPC adjacent; wins are sparse-exploration, not dense DMC. Worst
  build risk (two-timescale credit assignment). Deprioritize.
- **E (macro-empowerment):** DADS already = skill-MI + MPC-over-skills. MI objective fights return metric.
  Failed twice for us (iter-19/21). Deprioritize.
- **F (uncertainty-gated jump length):** new *combination*, more principled than B (grounded signal),
  complementary to A/I, attacks jumpy's own compounding-error weakness. High build. KEEP.
- **G (compositional operators):** ≈ A discretized; "algebra" hard to make load-bearing without a
  compositional benchmark. Deprioritize.
- **H (cross-task transfer):** TD-MPC2 already does multitask transfer; too expensive for 3-week single-var. Deprioritize.

### NEW levers (off our list, agent-proposed)
- **I. Learned PROPOSAL distribution for macro-MPPI ★★ VERDICT PICK.** State-conditioned q(z_macro|z_t)
  that *seeds/warm-starts* macro-MPPI over the jumpy macro-manifold (replacing zero-mean Gaussian +
  policy-prior). New *mechanism* in continuous control (cf. "Latent Geometry Beyond Search" 2026 amortizes
  search away & is goal-conditioned; we KEEP MPPI robustness + online learning). Attacks MPPI's real
  weakness — sample inefficiency in k·a_dim macro space → win is causally attributable, not tuning.
  **Very high buildability** (a proposal head + MPPI warmstart on existing infra). Single-variable clean.
- **J. Multi-k mixture-of-experts jumpy head, return-gated inside planning.** Set {d_{k_i}} tied by the
  consistency reg; macro-MPPI picks the head per step by predicted macro-return. = lever B "done right"
  (selection in-planning & return-gated, not a separate learned gate). High build; same single-k-collapse risk.
- **K. VALUE-EQUIVALENT macro jumpy head ★ FALLBACK.** Train d_k to be value/return-equivalent over k
  steps (predict same macro-Q under policy) rather than z-reconstruction-consistent → abstraction becomes
  *task-aware*, the most defensible reason a macro-abstraction should beat a state-faithful jumpy model
  (MuZero/Value-Equivalence Principle, rare in MPPI continuous control). Medium-high build (reuse macro-
  reward head + add macro-TD target; loss-function change, single-variable clean).

### RANKED TOP-4 (after round 1): **I > A > F > K**
### VERDICT: pursue **I (learned proposal-distribution macro-MPPI)**; fallback **K (value-equivalent macro head)**.
Reasons: (1) genuinely open lane in continuous control, (2) ~3-week single-variable build on existing infra,
(3) attacks the jumpy baseline's actual weakness so a win is attributable not tuning. K is the highest-upside
fallback because it changes *what the abstraction preserves* (return, not state).

### Caveat / verification status
Headline prior-art paper (2602.19634) = VERIFIED real. Other post-cutoff 2026 arxiv IDs the agent cited
(latent-geometry 2605.08732, mixture-of-WM 2602.01270, planning-in-8-tokens 2603.05438) are NOT load-bearing
for the verdict (the "I is novel" claim is a negative/no-evidence claim) — verify only if we cite them in a writeup.
Awaiting user's external Claude/GPT/Gemini DRs for round-2 synthesis before locking the lever.

---

## Deep-research synthesis — ROUND 2 (3 external DRs: Claude/Gemini/GPT, 2026-06-09)
*Full reports: docs/research/dr-external-iter23.md. User framing: ground the search in our previous
Structural-Entropy (Glass) paper → this re-centers the campaign on its SE identity.*

### Consensus map
- **F (uncertainty-gated jumpy horizon): 3/3 SAFE PICK** (Gemini#1, GPT#1, Claude#2). Reuses TD-MPC2's
  ensemble (~80 LOC), feasible+likely-to-work, but ALL THREE concede novelty is modest. The hedge.
- **SE-lever (structural entropy on the jumpy latent transition graph): the NOVELTY PICK** (Claude#1,
  and the one the user explicitly asked for by anchoring to our SE paper). Genuinely-open mechanism gap:
  no prior SE×learned-latent-WM, no SE-sets-temporal-abstraction-for-MPPI. Highest upside, higher risk.
- **Convergent KILLS (all reports):** C (bisimulation) + E (empowerment/DIAYN) — failed twice for us,
  unanimous drop. D (hierarchical) + H (cross-task) — lanes occupied (Puppeteer/HWM, multi-task TD-MPC2/HILP).
- **Two genuinely-new architectural levers (Gemini):** Hermite-spline action bottleneck (S) and bilinear-
  spectral jumpy dynamics (Spec) — see below.

### Two SimNorm-specific gating risks (must pre-check BEFORE committing either front-runner)
1. **SE on dense SimNorm latents may give trivial partitions** → CHECK the 2-D vs 1-D structural-entropy
   gap is materially > 0 on a real jumpy latent transition graph FIRST. If ≈0, demote SE, promote F.
2. **Ensemble epistemic saturation on normalized latents** (identically-confident-but-wrong OOD) → F's
   U_dyn could stay low exactly when it should fire. Validate U_dyn correlates with true k-step error first.
**Practical constraint (Gemini):** variable-k breaks GPU-parallel MPPI batching → implement SE-k / adaptive-k
as a fixed macro-grid with per-trajectory SOFT MASKS (λ-gates), NOT ragged horizons. (This unifies SE-k and F.)

### Updated levers
- **SE (NEW PRIMARY) — structural-entropy macro-abstraction over the jumpy latent transition graph.**
  Build a directed weighted transition graph over (quantized/sampled) jumpy latents; greedy directed-SE
  minimization → encoding tree → latent communities. Payoffs: (1) per-state target k from community
  membership (long jump inside a community, truncate at boundary — implemented as soft macro-grid mask),
  (2) phase-2: inter-community macro-actions as MPPI proposals. Anchored to SIDM/SISA/SI2E/SIHD line +
  our own Glass SE heritage. Novelty HIGH (mechanism+setting). Build MED. **Pre-check 2D-SE>1D-SE first.**
- **F (SAFE FALLBACK / parallel) — uncertainty-gated elastic jumpy horizon.** Ensemble of k-step dynamics
  heads; soft gate λ_j=σ((β−U_dyn)/τ) on macro-step returns + λ-return decay. ~80 LOC, reuses ensemble.
  **Pre-check U_dyn vs true error first.** Note: SE-k and F share the soft-mask machinery → cheap to run both.
- **S (Gemini) — Hermite-spline action bottleneck.** macro-action = (q_{t+k}, v_{t+k}) ∈ R^{2d}, cubic-
  Hermite interpolation + PD tracker, condition d_k on it; MPPI search dim k·d→2d. **No learned codec → no
  online representational shift** (edge over A/TAP/PLAS). Novelty HIGH, build MED. Risk: C¹ smoothness
  deprives discontinuous exploration (friction-breaking). Good clean ALT if SE pre-check fails.
- **Spec (Gemini) — bilinear-spectral jumpy dynamics.** low-rank operator z_{t+k}=Ψ diag(Λ(a)) Φᵀ1 →
  linear matrix rollout, fast+stable. Mostly a SPEED/STABILITY lever (latency gate), not a return-beater;
  risk: low-rank smooths contact discontinuities. Hold as efficiency add-on.
- A demoted to "component of SE skill-phase or of S, not standalone." B = non-SE ablation vs SE-k. C/E/D/H/G dropped.

### LOCKED PLAN (pending user go): **SE-lever as novel primary + F as safe parallel.**
Sequence: (0) PRE-CHECK on the validated jumpy substrate — does the jumpy latent transition graph have
real community structure (2D-SE ≫ 1D-SE)? AND does ensemble U_dyn track true k-step error? Cheap, decides
everything. (1) If SE pre-check passes → build SE-k (soft-mask macro-grid) as primary, F in parallel
(shares machinery). If SE pre-check fails → F becomes primary, Hermite-spline (S) the novel alt. (2)
Pre-registered gate (Claude's): vs plain jumpy, rliable IQM, 5 seeds, peak AND final, on sparse DMC
{Cartpole/BallInCup/Acrobot} + PandaPickCube + a HIGH-DoF task (Dog/Humanoid — where headroom provably
exists per round-1); WIN = ≥10% IQM, non-overlapping CI, ≥3/4 tasks. Mechanism-check before fanout.

## Status
- [x] iter-22 jumpy substrate validated (GATE RESOLVED: jumpy beats vanilla, peak+final CI-separated)
- [x] A/B/C drafted; D-H brainstormed; round-1 internal agent (I/J/K) folded
- [x] Round-2: 3 external DRs saved + synthesized → SE-lever = novelty primary, F = safe fallback,
      Hermite/Spectral added, C/E/D/H/G dropped; SimNorm pre-checks + batching constraint identified
- [x] PRE-CHECK tier-1 (proxy: geoglass/behavglass 32-node SimNorm prototype graph, CartpoleSparse):
      **PASSES on the structure question.** With SE-OPTIMAL Louvain + sparsification: transition graph
      SE gap up to 31.2% (keep=0.3, 3 comm), kNN-from-512d-latents up to 47.5% (k=3) — both ≫15% bar.
      NUANCE: raw graph ≈0% (the "blob"); structure only appears after top-frac sparsification / kNN
      (matches iter-19 docstring). Cached (suboptimal) partitions gave only 1-8% → must use SE-optimal
      partition. CAVEAT: 32-node prototype proxy, not the real jumpy latent graph → tier-2 must confirm
      on jumpy rollout latents (more nodes, jumpy k-step transitions). And iter-19 warns "structure
      exists" ≠ "useful for planning" (its community-SKILLS failed) — but SE-k/macro-action use is new.
      VERDICT: SE-lever viable enough to pursue as novelty primary; mechanism-check on real jumpy graph
      before fanout. (Ensemble-U_dyn check for F deferred — runs with the F arm.)
- [ ] PRE-CHECK tier-2 (mechanism-check, = build step 1): dump jumpy rollout latents -> k-step transition
      graph (>=128 nodes) -> SE-optimal partition -> confirm >=15% gap on the REAL substrate before fanout
- [ ] Pre-register SE-lever (or F if pre-check fails) gate; mechanism-check; build on validated substrate
- [ ] (HOLD for explicit user go before building)
