# COLLEAGUE HANDOFF — TD-MPC-Glass → Jumpy World Model (CompPlan) Reproduction

Written 2026-06-18 for a colleague picking this up while the original author is away (until Jun 20).
GPU boxes are being **destroyed to save cost** — everything is backed up (see §5). This doc is the
single entry point: read it top to bottom, then `docs/INDEX.md` for the deeper map.

---

## 0. TL;DR — where we are right now

- **Original goal:** beat TD-MPC2 at the *architecture/abstraction* level under a strict fair protocol.
  After 16+ abstraction variants, **nothing beat it** — a rigorous **negative result**.
- **The explanation we attached (the "R² redundancy criterion") is dead.** We tried to prove that the
  latent is "value-sufficient" (so abstraction is redundant) via a linear-decode R². It **cannot be
  operationalized** — both metric versions fail (one variance-confounded, one saturated by
  construction). See Part 13 blog + `docs/iterations/paperA_criterion_VERDICT_2026-06-17.md`.
- **Current focus (the pivot):** the *one* technique that genuinely won was **temporal abstraction**
  (jumpy world models). So we're **reproducing the 2025–2026 jumpy-world-model line**:
  - **GHM** (Geometric Horizon Model) — model class, Janner et al. NeurIPS 2020.
  - **TD-Flow** — how to *train* a GHM (flow matching + TD bootstrap), ICML 2025, arXiv 2503.09817.
  - **CompPlan** — how to *plan* with a GHM ("Compositional Planning with Jumpy World Models"),
    ICLR-WS 2026, arXiv 2602.19634. **This is the reproduction target.**
  - **No public code exists** for any of them → we build on **InFOM** (open-source JAX flow-occupancy
    model on OGBench, `github.com/chongyi-zheng/infom`).
- **Built so far (this week):** a horizon- and policy-conditioned **GHM agent** on the InFOM scaffold
  (`ghm_repro/infom/agents/ghm.py`), **antmaze unblocked**, and a **plan-over-policies planner**
  (smoke-passing). **Not done:** planner not wired into an OGBench eval loop, no real base-policy
  library, no real antmaze/CompPlan numbers reproduced yet. This is a multi-day reproduction.

---

## 1. Full project history (the arc, so you have the why)

1. **TD-MPC2 baseline.** TD-MPC2 (Hansen et al., ICLR 2024) is a strong latent-space model-based RL
   method (SimNorm latent, self-predictive consistency, MPPI planning, TD value). We reimplemented it
   in JAX/Flax/MJX (`src/helios/algorithms/tdmpc2.py`, driver `scripts/run_benchmark.py`) and chased
   beating it with a *novel abstraction* under a **fair protocol**: single-variable changes,
   compute-matched, pre-registered peak+final CI gates, paired bootstrap, mechanism-check before
   fan-out, read-from-JSON (never fabricate), report peak AND final.
2. **The null campaign.** Every explicit *representation* abstraction failed: state/temporal/relational/
   compositional — bisimulation, value-equivalence aux losses, self-predictive extras, entity
   factoring, graph world models, frozen dynamics, motion-phase, a value-scale "collapse-fix". All
   null or confounded. See `docs/iterations/RESEARCH_LEDGER.md`.
3. **The redundancy criterion — and its death.** We explained the nulls with "the SimNorm latent is
   already value-sufficient (value decodes linearly, R²≈0.9994), so abstraction is redundant." A
   reviewer-style challenge ("does R² *prove* redundancy?") led to the discrimination experiment
   (`scripts/paperA_loop.sh`, distractor×bisim). Verdict: **the R² criterion can't be built** — the
   return-to-go decode is variance-confounded (R² anti-tracks performance), the V(z) decode is ~0.98
   flat regardless of policy quality (saturated by the near-linear value head). The "0.9994" was an
   artifact. → Paper A becomes the **honest negative** (null campaign + "why the obvious probe fails").
   Evidence: `exp/tdmpc_glass/paperA/vzprobe/*.json`, verdict doc, Part 13 blog.
4. **The one real win → the pivot.** The jumpy k-step world model *did* beat vanilla TD-MPC2 on
   contact manipulation (Part 12 blog). Since jumpy/temporal abstraction is the live lead, we pivoted
   to reproducing the SOTA jumpy line (CompPlan). That is the current work.

---

## 2. The CompPlan reproduction (current focus) — concepts

Read the from-scratch primer first: **blog Part 14** (`docs/_posts/2026-06-18-...primer.md`) — builds
GHM / occupancy measure / flow matching / TD-Flow / CompPlan from zero, vs Dreamer & TD-MPC2.

Key objects:
- **Discounted occupancy / successor measure** `d^π_γ(s'|s)` = distribution over geometrically-
  discounted future states under π. A **GHM** samples from it in one shot ("jumpy") — no step
  composition, so no compounding error over long horizons.
- **TD-Flow** trains it via flow matching with a Bellman/TD bootstrap (target network).
- **CompPlan** = horizon-conditioned + policy-conditioned GHM, used to **plan over sequences of
  pre-trained base policies** on OGBench (antmaze navigation + cube manipulation). Headline ~+200% on
  long-horizon tasks.

Design + InFOM→CompPlan mapping: `docs/ghm_repro/EXTENSION_DESIGN.md`. Plan/status:
`docs/ghm_repro/PLAN.md`.

---

## 3. What is implemented vs. not (be precise)

**Implemented + verified** (in `ghm_repro/infom/`, all additive — base InFOM untouched):
- `agents/ghm.py` — horizon-conditioned GHM (`GHMAgent`): per-example geometric-horizon discount
  `γ_h` fed to the velocity field; horizon-consistent TD bootstrap; `discount_consistency_frac`
  subset; policy-conditioning hook (relabel conditioning action). Defaults now enable horizon
  conditioning (`gamma_min=0.95, gamma_max=0.999, discount_consistency_frac=0.125`). Smoke-clean
  CPU+GPU. Trains on cube-single (peaks 0.08–0.78, ≈ InFOM range — expected, cube is short-horizon).
- `utils/networks.py :: HorizonVectorField` — velocity field + horizon/policy channels.
- `main.py` — `--ft_dataset_fallback` flag (auto-on for antmaze): loads the non-`ft` singletask
  dataset as the finetuning dataset, since antmaze's `-ft-` dataset isn't on the OGBench server and
  the generator doesn't cover antmaze. This **unblocked antmaze** (verified eval writes
  `episode.success`; numbers were 0.0 only because it was a 15k-step smoke, not a real run).
- `planning/compplan_planner.py` — beam search over base-policy sequences using GHM jumps + goal
  scoring; `planning/smoke_planner.py` ran clean on a real trained cube checkpoint.

**NOT done (your runway):**
- Planner is **not wired into an OGBench eval loop**, and uses only the GHM's own actor as the single
  base policy. CompPlan needs a **library of pre-trained goal-conditioned base policies** (GCBC/CRL
  via OGBench `impls/`) to plan over.
- **No real antmaze training run** completed (only the 15k-step pipeline verification). Antmaze is the
  long-horizon test that actually probes the jumpy hypothesis — **this is the priority.**
- **No CompPlan Table-1 reproduction.** That's the end goal and is multi-day.

---

## 4. How to resume (concrete, in order)

1. **Rent a GPU box** (vast.ai; see `memory`/`docs/operations`). A single 4090/3090 is enough to start.
2. **Set up the env** (isolated venv — do NOT reuse the tdmpc env):
   ```
   python3 -m venv /root/ghm/.venv && . /root/ghm/.venv/bin/activate
   pip install "ogbench==1.1.0" && pip install -r infom/requirements.txt && pip install -U "jax[cuda12]"
   ```
   Copy `ghm_repro/infom/` (this repo / backup) to the box. Copy the **ft-dataset backup** (§5) to
   `/root/.ogbench/data/` (cube ft dataset is NOT re-downloadable). OGBench standard datasets
   auto-download on first `make_env_and_datasets`.
3. **Train GHM on antmaze** (the real test):
   ```
   cd infom && python main.py --agent=agents/ghm.py \
     --env_name=antmaze-medium-navigate-singletask-task1-v0 --enable_wandb=0 --save_dir=exp
   ```
   (`--ft_dataset_fallback` auto-triggers on the `antmaze` name.) Read peak `episode.success` from
   `exp/<run>/sd*/finetuning_eval.csv`. Compare to CompPlan Table 1 antmaze-medium.
4. **Build the base-policy library** (OGBench `impls/`: GCBC, CRL) and **wire the planner**
   (`planning/compplan_planner.py`) into an OGBench eval loop that plans over them.
5. **Reproduce CompPlan Table 1** (antmaze-medium + cube-single first).
6. Keep-busy orchestration if you rent several GPUs: `scripts/ghm_loop.sh` (file-driven env rotation
   `exp/tdmpc_glass/ghm/envs.txt`, disk-pruning, watchdog cron). Edit the SLOTS/box aliases for your
   new boxes.

---

## 5. Backups & where everything is

- **EC2 control box = `ubuntu@ip-172-31-35-6`** (this machine; NO GPU; never trains). Primary backup.
- **Full data backup:** `/home/ubuntu/ghm_backup_2026-06-18/` — `code/` (launch script), `data/`
  (the generated cube ft-dataset, ~136 MB, NOT re-downloadable), `results/` (all eval CSVs +
  checkpoints from both boxes + the antmaze pipeline test).
- **Code (GitHub):** `github.com/SuuTTT/tdmpc-glass` (this repo — tdmpc2 impl, scripts, docs, blog,
  and `ghm_repro/infom/` with our GHM extension). Blog also at `github.com/SuuTTT/suuttt.github.io`.
- **GHM code in-repo:** `ghm_repro/infom/` (modified InFOM). InFOM/OGBench/value-flows upstream clones
  are also under `ghm_repro/`.
- **OGBench standard datasets** (cube/antmaze pretraining): re-downloadable via OGBench — not backed up.
- **Old tdmpc campaign checkpoints were deleted** to free disk (CSVs harvested; key ones on HF). The
  campaign results live as CSVs in `exp/tdmpc_glass/` + the blog.

## 6. Metric & env quick reference

- **`episode.success` ∈ [0,1]** = fraction of ~50 eval episodes reaching the goal. **Headline metric.**
  Report **PEAK over training**, not final (online-finetune eval collapses late). `return` ≈ −(steps),
  so −200 at length 200 = timed-out failure.
- **cube-single** = arm manipulation, short-horizon (obs 28, act 5). **antmaze-medium** = ant maze
  navigation, long-horizon (obs 29, act 8) — where jumpy planning should pay off.

## 7. Working rules (please keep)

- **Read from JSON/CSV, never fabricate.** Report peak AND final. This project fabricated numbers ~7×
  early on; the discipline is non-negotomeable.
- **Mechanism-check before fan-out.** A GO licenses a test, never predicts success.
- **vast.ai: NEVER destroy instances yourself** — recommend a destroy list, the user destroys.
- Off-limits boxes (other projects): LTSF 38664456, mahjong, StructMamba 40230626, SeSE 34838954, SIDM.
- Publish review docs/milestones as blog posts (repo `docs/_posts/` Jekyll + homepage Hugo).

## 8. The narrative (blogs)

Parts 1–11 = the TD-MPC2 campaign + nulls. **Part 12** = when jumpy planning helps. **Part 13** = the
R² criterion post-mortem. **Part 14** = jumpy-world-models primer (GHM/TD-Flow/CompPlan from scratch).
Index: `https://suuttt.github.io/tdmpc-glass/`.
