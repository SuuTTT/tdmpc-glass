# Iteration 15 — Plan in Prototype Space ("proto-plan")

*2026-06-07. Direction #2 from the Six Mirages post-mortem (user-selected: "do 2, if not work,
do others"). Pre-registered BEFORE any run.*

## Hypothesis

In every Glass variant to date, prototypes were **trained but never used at decision time** —
the planner always searched the full 512-dim latent space. If the learned behavioral
abstraction is worth anything, MPPI rollouts in the 32-dim prototype-assignment space should
retain most control performance at a fraction of the planner cost (~100× fewer rollout FLOPs:
tiny proto-dynamics head + dot-product reward/value vs three 512×512 MLPs per step).

This is NOT another auxiliary-loss test (§5.1 of the write-up dooms those). It tests whether
the abstraction can carry **control**, with compute as the win condition.

## Mechanism (all distilled, zero representation impact)

On top of behavioral Glass (λ_behav=0.5), three planner heads inside `params["glass"]`,
trained with **stop-gradient inputs** (pure distillation — encoder/prototypes/dynamics
unchanged, so any result difference is attributable to planning alone):

- `proto_reward` (exists): r̂(c) = c·r_p, trained on raw rewards (behav loss).
- `pdyn_*` (new): tiny MLP (P+act → 128 → P), CE loss against sg(c(z_{t+1})).
- `proto_value` (new): V̂(c) = symexp(c·v_p), regressing symlog of sg(min-Q(z, π(z)))
  from target params.

Proto-MPPI: identical MPPI loop (H=3, NS=512, 64 elites, 6 iters); π-trajectory action seeds
still generated in latent space (24×H dyn applies, once per call); ALL 512 samples evaluated
in prototype space: c₀ = assign(enc(obs)), cₜ₊₁ = softmax(pdyn(cₜ, aₜ)), return = Σγᵗ c·r_p +
γᴴ·max(symexp(c·v_p), 0). Units verified: rew_scale is vestigial (never applied); rewards and
Q are raw everywhere, so r_p and v_p targets are unit-consistent.

## Kill-probe design (minimal falsification)

**Within-run paired comparison** — train behavglass exactly as in iter-14 plus the distilled
heads, and at every 50k eval run BOTH planners on the same params, writing `mppi` and
`protomppi` CSV rows. Paired design removes seed variance → 2 runs suffice.

- Task: CheetahRun (best-characterized; vanilla/behavglass ≈ 540–550 @500k).
- Runs: 2 seeds × 500k steps, tag `phasei15pp`. Cost ≈ 2 × ~3 GPU-h.
- Collection/training is UNCHANGED (latent MPPI not used for collection anyway — collection
  is π+noise); the probe adds only the distillation losses (sg'd) and a second eval pass.

**Pre-registered gates (mean of last-2 evals at 500k, both seeds):**

- **G1 (viability): protomppi ≥ 0.8 × mppi** (same run, same step). PASS → Stage 2.
- **G-dead: protomppi < 0.5 × mppi** on both seeds → direction falsified, move to
  direction #3 (replace SimNorm with codebook) or others per the post-mortem ranking.
- Between 0.5–0.8: one retuning iteration allowed (pdyn capacity, temperature, horizon),
  pre-registered as the ONLY free shot; then re-gate.
- Secondary metric (logged, not gated): wall-clock per eval episode, protomppi vs mppi.

**Stage 2 (only if G1 passes):** compute-matched training — proto-MPPI as the *collection*
planner with samples/horizon scaled to matched planner wall-clock vs vanilla's latent MPPI;
win = IQM ≥ vanilla at equal wall-clock OR equal IQM at ≥2× planner speedup, ≥5 seeds,
stratified-bootstrap CI, on {Cheetah,Walker,Finger}.

## Honest prior

LOW (~15–25% G1). §5.3's information argument cuts against us: a 32-dim soft assignment is a
brutal bottleneck for 17-dim-state control, and the H=3 horizon means terminal value
dominates — V̂ through prototypes is the likely failure point. But this is the one untested
axis where "abstraction" could pay even at degraded-but-usable fidelity (compute), and the
probe is 2 runs.

## Results

**Smoke test (2026-06-07, box 29168, CheetahRun 60k co-resident, tag phasei15smoke):**
plumbing PASS — heads train, planner runs, CSV row written (`50176,1.4,protomppi,0`).
Speed claim CONFIRMED: proto 26.6 s/ep vs latent 224.2 s/ep = **8.4× faster** (NS=2048).
Performance at 50k: protomppi 1.4 vs mppi 125.1 (ratio 0.01) — loud early warning, but
NOT gate-relevant: glass warmup is 100k, so prototypes were still random-init at this
eval; the heads distilled through an untrained assignment. Gate remains at 500k.

**Probe pair queued** (ti15pp0/1, priority 1, 2026-06-07): CheetahRun seeds 0/1, 500k,
PROTO_PLAN=1, LAMBDA_BEHAV=0.5, tag phasei15pp_protoplan, CODE_SHA=i15a.

**Mid-flight diagnosis (2026-06-07, seed 0 @350k, checkpoint introspection on-box):**
protomppi flat at ~0.4 reward through 300k (mppi up to 529). Root cause established and it
is the HONEST mechanism, not a bug:

- pdyn L1 sensitivity: **action 0.012 vs state 1.567** (135× action-blind);
- planner objective over 256 random action sequences: std 0.005 on mean 5.22 — **flat to
  0.1%** → MPPI elites are noise → mu→0 → agent stands still (reward ≈ 0.4 ✓);
- not implementation failure: proto_reward_plan spans ±5, proto_value to 7.5, S sharp
  (max 1.0), P diag mass 0.032 (assignments DO move between steps — just not as a function
  of action).

Interpretation: given c (32-dim soft assignment), the action has ~zero residual predictive
value for c' — I(a; c'|c) ≈ 0 while I(z; c'|c) is large. Prototype transitions are driven
by gait phase, not control. **The abstraction carries no controllable signal at the
planner timescale.** No pdyn capacity/temperature retune can recover information absent
from c, so the 0.5–0.8 retune band is moot; awaiting the formal 500k gate read on both
seeds. Confirmed deliverables either way: 8.4× planner speedup (mechanically real), and a
crisp negative mechanism for the write-up ("why you can't plan in a behavioral-prototype
space": the projection that groups reward-equivalent states is exactly the projection that
discards action-conditional distinctions).

**FORMAL GATE — seed 0 (2026-06-07, run complete, read from mirror CSV):**
last-2 evals at 450k/500k: protomppi mean **1.4** vs mppi mean **642.9** → ratio **0.002**.
G-dead threshold is <0.5; this is 250× below it. Latent MPPI itself reached 658 at 500k
(healthy run — the comparison is fair). Seed 1 shows the identical flatline through 300k
(ratios 0.001–0.008); formal read when it completes, but the verdict is over-determined by
the mechanism diagnosis above.

**FORMAL GATE — seed 1 (2026-06-07, read from box CSV):** run died at ~496k (no traceback;
likely OOM on the 6GB 1660S at the tail — 4k short of the 500k eval). Last-2 available
(400k/450k): protomppi **2.15** vs mppi **577.6** → ratio **0.004**. Both seeds <0.5 by
~250×; the shortfall is immaterial to the verdict.

**VERDICT: direction #2 (plan in the abstraction) is DEAD** — pre-registered G-dead, with
mechanism: behavioral-prototype spaces are action-blind at the planner timescale
(I(a;c'|c)≈0). The 8.4× planner speedup is real but planning fast over a flat objective is
worthless. Successors (user-directed, already in flight): iter-16 FSQ-codebook swap
(ti16fsq0-3 running), iter-17 prototype-novelty exploration (ti17xn0-5 queued).
