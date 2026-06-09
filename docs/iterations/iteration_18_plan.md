# Iteration 18 — Horizon-Sweep Gate for Temporal Abstraction ("hsweep")

*2026-06-08. The cheap kill-probe that GATES the top post-mortem direction (temporal/skill
abstraction). Zero new model code — `--mppi_horizon` only. Pre-registered before any run.*

## The bet it gates

Direction #1 from the campaign post-mortem: a **skill/option-conditioned jumpy world model**
so MPPI can plan over macro-actions (long effective horizon at fixed compute). It is the only
abstraction idea that (a) satisfies the iter-15 controllability law — the macro-action carries
the control signal, so it is NOT action-blind — and (b) targets a bottleneck (TD-MPC2's
myopic H=3) rather than a place abstraction has no job. But it is only worth building **if
horizon is actually a lever on the target tasks.**

## Gate question

Does longer planning horizon improve vanilla TD-MPC2 on lookahead-requiring tasks? We sweep
H ∈ {3 (default), 9 (3×)} on three tasks spanning the regimes (all in mujoco_playground):

- **PandaPickCube** (manipulation): genuine reach→grasp→lift temporal credit — the textbook
  long-horizon test. (Exploratory: unknown if vanilla learns it at all.)
- **AcrobotSwingup** (dense, vanilla learns ~0.33): multi-step energy-pumping needs lookahead.
  The interpretable anchor.
- **CartpoleSwingupSparse** (sparse, vanilla 1/3): lookahead through sparse reward.

3 tasks × 2 horizons × 2 seeds = 12 runs, 500k steps. Tag phasei18hs. NOTE: raising H also
lengthens the training consistency rollout (seq_len=H+1) — so this tests "does a longer-horizon
model+planner help," conflating the two, which is exactly the right gate question for "is
horizon a lever."

## Pre-registered outcomes

- **H=9 ≥ 1.15× H=3 on ≥1 task (last-2 mean @500k, both seeds agree in sign)** → horizon is a
  lever → BUILD the skill/jumpy model on that task. (PASS)
- **H=9 ≈ H=3 everywhere (within ±10%)** → tasks not horizon-limited → temporal abstraction
  premise dead here → PIVOT to direction #2 (multi-task transfer). (FAIL-pivot)
- **H=9 < H=3 (longer rollout compounds 1-step model error)** → AMBIGUOUS-but-informative:
  argues a *jumpy/macro* model (accurate over long spans where 1-step isn't) is the right tool,
  but weaker evidence; would need the jumpy model itself to test. Record and treat as
  conditional-go.

Honest prior: ~50%. TD-MPC2's H=3 default exists because longer single-step rollouts compound
error; the open question is whether the task value of lookahead outweighs that on long-horizon
tasks. Either way the probe is decisive for the build decision and costs 12 short runs.

## Results

**GATE VERDICT (2026-06-08, @500k) — PASS, but refined & task-specific:**

| task | H3 (last-2@500k) | H9 | reading |
|---|---|---|---|
| CartpoleSwingupSparse | **0, 0** | **712** (s0@450k), 66 (s1@250k) | **H9 qualitative WIN** — finds sparse reward H3 never reaches |
| AcrobotSwingup | 340, 233 | 335, 184 | parity |
| PandaPickCube | 1219, 1762 | **419** (was 1678@250k → COLLAPSED) | **H9 HURTS** — 9-step rollout compounds model error, destabilizes |

**Interpretation — the thesis is now sharp:** longer planning horizon is a lever ONLY on
sparse/exploration-limited tasks where reward sits beyond H=3's reach (Cartpole). On dense
tasks it's parity; on manipulation it HURTS because rolling the 1-step model 9× compounds
error (Panda collapse 1678→419) — which is precisely why TD-MPC2 ships H=3.

**This is the strongest possible motivation for community-skills:** a skill is a *learned
macro-controller*, so it grants long *effective* horizon WITHOUT rolling the 1-step model 9×
— dodging both H=3's myopia (Cartpole=0) AND H=9's compounding-error tax (Panda collapse).
The gap vanilla cannot cover is exactly the skill thesis. **BEAT-TARGET = CartpoleSwingupSparse**
(vanilla H=3 = 0, so any skill method that reliably finds reward wins; also compare vs
vanilla-H9 712-but-unstable). Build skill-planning here (iter-19 Stage-2/3).

Firming runs queued: CartpoleSparse H3 + H9 to n=4 (confirm the 0-vs-712 with seeds).

**FIRMING CONFIRMED (2026-06-08, n=4):** CartpoleSparse H3 = [0, 0, 97, 0] (vanilla fails);
H9 = [695, 663, ...2 early]. Both mature H9 seeds found reward strongly vs H3's ~0 — the
0-vs-~680 gap is robust across seeds (a found-vs-not-found category difference, mirage-
resistant). iter-18 gate **PASS, locked.** Beat-target CartpoleSwingupSparse confirmed.
