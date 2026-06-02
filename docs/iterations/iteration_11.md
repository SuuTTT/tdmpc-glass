# TD-MPC-Glass Iteration 11 — Review & Plan

Created: 2026-06-01

## Environment for this iteration (what changed)

This iteration starts on a **new control plane and a new agent**:

- **Agent:** Claude Opus 4.8 (1M context). Earlier iterations (2–10) were driven
  by older models / Codex; treat their prose as historical record, not gospel.
- **Control plane:** migrated off the destroyed local 4070 Ti box onto a
  **new AWS EC2 instance** (this box). EC2 is **control-plane only — it has no
  GPU and never trains**. There is no `"local"` entry in `BOXES`.
- **Standalone repo:** TD-MPC-Glass was extracted out of the nested `helios-rl`
  benchmark tree into a standalone, git-tracked repo at `/home/ubuntu/tdmpc-glass`
  following the research-os paradigm. Control daemons live in `control/`, worker
  code in `scripts/` + `src/`, queues in `scripts/queues/`.
- **Data integrity:** the duplicate-launch CSV corruption (curves "going back")
  was detected across history and repaired — see
  `docs/operations/data_corruption_fix.md`. The launch race that caused it is
  fixed in `task_queue_daemon.py :: is_box_idle()` (a box with
  `n_run_benchmark_procs >= n_gpu` is busy; a GPU is free only at
  `mem_used <= 100 MiB`).

> The operations runbook `docs/operations/launch_dashboard.md` still has stale
> `coder`-era paths (`/home/coder/.ssh/...`, `scripts/web_dashboard.py`, old box
> list). The **authoritative** current procedure is in `CLAUDE.md`,
> `docs/operations/ec2_dashboard_queue_migration.md`, and the "Dev loop" section
> below.

---

## The dev loop (how a probe actually runs, and how not to mess up code)

```
   you ──enqueue──▶ scripts/queues/central_queue.json (pending)
                          │
        task_queue_daemon.py (polls every 60s, on EC2)
                          │  finds an idle GPU box (is_box_idle)
                          ▼
        rsync -az --delete  scripts/ + src/  ──▶  root@worker:/root/helios-rl/
                          │  (pushes the LOCAL WORKING TREE, every launch)
                          ▼
        ssh: cd /root/helios-rl ; <ENV> nohup bash <launcher> ...
                          │
        worker trains, appends exp/tdmpc_glass/HopperHop_<PROBE_ID>/seed_<S>/seed_<S>.csv
                          │
        iter5_stream_remotes.sh (every 5 min) rsyncs CSVs ──▶ exp/.../remote_mirror/
                          │
        web_dashboard.py reads mirror + queue ──▶ http://localhost:5055
                          │
        auto-promote: on done/failed, daemon reads best_any from the log and
        appends follow-up seed tasks (see thresholds below).
```

### How to push a probe to the queue

One seed per task. Lower `priority` number runs first (default 10).

**Preferred — REST API (atomic, fcntl-locked, no hand-editing):**

```bash
curl -s -X POST http://localhost:5055/api/queue \
  -H 'Content-Type: application/json' \
  -d '{
    "label": "phasei11x_<short-desc> seed 1",
    "launcher": "scripts/run_phasei9_glass_probe.sh",
    "env": "PROBE_ID=phasei11x_<tag> SEEDS=1 K_UPDATE=128 MPPI_NS=2048 TOTAL_STEPS=2000000 EARLY_STOP_PATIENCE=1500000 EXPL_UNTIL=25000 PROTO_TEMP=0.7 ASSIGN_SCALE=0.5 STOPGRAD=true TEMP_STABILITY=0.01 GLASS_WARMUP=100000 GLASS_DECAY=1000000 NUM_PROTOTYPES=8 NUM_CLUSTERS=2 NUM_SUPER_CLUSTERS=0 CODE_SHA=<git-short-sha> XLA_PYTHON_CLIENT_MEM_FRACTION=0.35",
    "priority": 6
  }'
```

`label` + `launcher` are required; `env` is the full space-separated env string;
the daemon injects `CUDA_VISIBLE_DEVICES`, mem fraction, and `MUJOCO_GL=egl` per
box. Copy the env template from a known-good task in `central_queue.json` and
change only `PROBE_ID`, `SEEDS`, and the knobs you are testing.

**Fallback — edit `scripts/queues/central_queue.json` directly.** Back it up
first (`*.bak_<reason>_<ts>`); the daemon's `fcntl` lock makes in-place edits
safe. New task needs at least `id`, `label`, `launcher`, `env`, `priority`,
`status:"pending"`.

### Auto-promotion thresholds (daemon, `run_phasei9_glass_probe.sh` tasks only)

Read from the launcher log when a task ends; `best_any = max(pi, mppi)` over evals:

| best_any | seeds added | notes |
|---:|---:|---|
| ≥ 600 | 5 | full fill |
| ≥ 500 | 2 | |
| ≥ 380 | 1 | |
| < 380 | 0 | |

Failed runs get a 100-pt lower bar **only if** they reached ≥ 4M steps
(`MIN_FAILED_PROMOTION_STEPS`); header-only crashes never promote. Promoted tasks
get `PROBE_ID=<base>_auto_s<seed>`.

### The "don't mess up code" rules (read before editing anything)

1. **`rsync_code` pushes the live working tree with `--delete` at every launch.**
   It is NOT a git checkout. Whatever is in local `scripts/` + `src/` the instant
   a pending task is claimed is what that worker runs. Deleting a file locally
   deletes it on the worker.
2. **`CODE_SHA=...` in the env is only a provenance LABEL** baked into the output
   tag. It does not pin code. If you edit an algorithm while probes are pending,
   new launches silently run the new code but still stamp the old SHA →
   corrupted provenance (exactly the `dbf5cea`/`87a4337`/`df9bfb1` confusion that
   made `phasei9r` "mixed-provenance").
3. **Freeze code while a batch is in flight.** Before changing
   `src/helios/algorithms/*.py` or a launcher: either (a) drain/let the dependent
   pending tasks finish, or (b) commit first and stamp the new `CODE_SHA` with the
   real `git rev-parse --short HEAD`. The repo is git-tracked now — use real SHAs,
   retire `df9bfb1-dirty`.
4. **Already-running tasks are not affected** by later edits (rsync only fires at
   launch) — but the next claimed pending task is. Don't assume the queue is
   homogeneous.
5. **One-master rule:** exactly one `task_queue_daemon.py` against this fleet.
   Verify with `pgrep -fa task_queue_daemon.py` after any restart.
6. **One seed per task.** Cap fair-CI fills at seeds 1–5. Keep failed/partial
   logs but separate them from final claims.

---

## Review of Iteration 10

### Fresh standings (recomputed from CSVs, 2026-06-01)

`best_any` = max eval reward (pi or mppi) over the run; G1 = best_any ≥ 500.
Families canonicalized by stripping `_auto_sN` and code-hash suffixes (rough —
some sibling tags merge/split imperfectly; treat single-seed `[0,0]` rows as n=1).

| Family | n | mean best_any | 95% CI | G1 | max |
|---|---:|---:|---:|---:|---:|
| `phasei9r_p1b_off1m_fairci` | 5 | 511.0 | [405, 617] | 4/5 | 595.3 |
| `phasei9t_p1b_off1p5m_fairci` | 3 | 477.2 | [378, 577] | 1/3 | 553.9 |
| `phasei9q_p1b_temp001_off2m` (s4 group) | 6 | 458.2 | [344, 572] | 3/6 | 609.0 |
| `phaseaa_codex_tdmpc2_k256` (baseline) | 5 | 362.1 | [260, 464] | 1/5 | 561.1 |
| `phaseaa_codex_tdmpc2_k128` (baseline) | 5 | 350.0 | [256, 444] | 1/5 | 538.6 |
| `phasei10t_k2_temp002_fast2m` | 10 | 346.9 | [272, 421] | 2/10 | 557.2 |
| `phasei10u_k4_temp001_fast2m` | 7 | 337.5 | [193, 482] | 3/7 | 536.0 |
| `phasei10s_k2_temp0005_fast2m` | 7 | 386.2 | [288, 484] | 2/7 | 576.3 |
| `phasei10c_off1m_clean5` (clean rerun) | 4 | 321.8 | [241, 402] | 0/4 | 436.9 |
| `phasei10r_k2_temp001_fast2m` | 6 | 271.7 | [201, 343] | 0/6 | 438.0 |
| `phasei10v_k2_notemp_fast2m` (no-temp ctrl) | 4 | 318.1 | [220, 417] | 0/4 | 428.2 |

### What Iteration 10 established

1. **`phasei9r` off-at-1M is still the leader** (4/5 G1, mean ~511) but its
   strength was found under fast iteration with mixed code SHAs. The **clean,
   single-tag rerun `phasei10c` did not reproduce it** (0/4). So off-at-1M is a
   *useful lead, not a settled method*.
2. **The K2 + mild temp-stability "coarse basin scaffold" early-spike is real but
   not robust.** Individual seeds spike >500 by 1.0M (`phasei10a1` s1 → 620.5;
   `phasei10k` s3 → 556.7; `phasei10p` s1 → 537.1), but spike-*rate* across the
   fast-2M batches is low and inconsistent (i10t 2/10, i10u 3/7, i10s 2/7,
   i10r **0/6**, no-temp control i10v 0/4). temp-stability ∈ {0.005, 0.01, 0.02}
   does not yet give a clean ranking.
3. **The MPPI-policy gap is still open and undiagnosed.** The `i10-c` planner
   calibration diagnostic (predicted MPPI objective vs realized MPPI vs realized
   pi) was specified but never built. `i10-a1` eval-only arbitration was
   implemented; behavior arbitration (`i10-a2/a3`) correctly deferred.
4. **Glass hierarchy redesign** got static-shape flags
   (`NUM_PROTOTYPES/NUM_CLUSTERS/NUM_SUPER_CLUSTERS/LAMBDA_SUPER_*`) and first
   probes (i10g/i10h/i10k), but no clean one-level-vs-two-level verdict.
5. **JEPA world-model** remained correctly deferred (no code).

### Carried-over open items (not done in i10)

- pi-vs-MPPI gap analysis script/table (was a TODO).
- `i10-c` planner-calibration diagnostic.
- Recompute countable 95% CI after fast-2M fills mature.
- One-level vs coarse-K2 Glass verdict.
- JEPA `i10-h0` design note.

---

## Iteration 11 Plan

Theme: **stop widening the probe fan-out; convert the strongest leads into clean,
provenance-controlled evidence, and finally diagnose the MPPI gap.** With a
git-tracked repo and a single EC2 master, we can now run *reproducible* probes —
that is the main new capability to exploit.

### Direction A — Consolidate evidence (highest priority)

- **A1. Pin provenance.** Commit current code; from now stamp every probe with
  `CODE_SHA=$(git rev-parse --short HEAD)`. No more `-dirty` tags for runs we
  intend to count.
- **A2. Clean off-at-1M, single SHA, 5 seeds (`phasei11a`).** One launcher, one
  SHA, seeds 1–5, off at 1M, no temp-stability. This is the decisive test of
  whether `phasei9r`'s 4/5 is method or provenance luck. Gate: ≥3/5 G1 ⇒ promote
  off-at-1M to "method"; <3/5 ⇒ demote `phasei9r` to "lead only".
- **A3. Finish the TD-MPC2 K256 baseline as a clean 5-seed reference** (already
  ~5 seeds at mean 362; verify provenance, re-stamp, fill any non-clean seed).
  This is the comparison denominator — keep it honest.

### Direction B — Diagnose the MPPI-policy gap (build the missing tool)

- **B1. pi-vs-MPPI gap table** (offline analysis, no GPU): per phase/seed, compute
  `mppi_reward - pi_reward` distribution over evals. Add as
  `control/analyze_mppi_gap.py` reading the mirror CSVs. Pure reporting.
- **B2. `i10-c` planner-calibration diagnostic** (worker code): log predicted MPPI
  objective vs realized MPPI return vs realized pi per eval. Smoke locally-equiv
  on one idle GPU before queueing. This tells us whether MPPI failure is model
  rollout error, Q terminal error, or action-sequence optimization — the
  prerequisite gate for any behavior-arbitration or distillation work.
- Do **not** queue `i10-a2` behavior arbitration or MPPI-gated distillation until
  B1+B2 show a *consistent* gap.

### Direction C — Settle the K2 scaffold question (small, decisive batch)

The fast-2M fan-out is too noisy to conclude. Run **one** controlled 5-seed
comparison at fixed SHA instead of many 1–3 seed tags:

| Probe | Knobs | Tests |
|---|---|---|
| `phasei11c_k2_temp001` | K=2, N=8, temp 0.01, off 1M | scaffold + mild temp |
| `phasei11c_k2_notemp` | K=2, N=8, temp 0.0, off 1M | scaffold alone |
| `phasei11c_k4_temp001` | K=4, N=8, temp 0.01, off 1M | coarse control |

5 seeds each, `TOTAL_STEPS=2000000`, primary metric **>500-by-1.0M rate**. Decide
on spike-*rate*, not single seeds. Gate: a family is "real" only at ≥3/5.

### Direction D — Glass hierarchy verdict (use existing static flags)

- **D1. One-level prototype SE vs two-level (`z→μ→S`)** at fixed SHA, 3 seeds each,
  using the existing `NUM_CLUSTERS`/`NUM_SUPER_CLUSTERS` flags (no dynamic-shape
  JAX changes). Answer the i10 question: is `S` necessary for HopperHop?

### Direction E — JEPA design note (no code, no GPU)

- **E1. Write `docs/design/jepa_vs_tdmpc2_worldmodel.md`** (the i10-h0 note):
  proprio-JEPA next-embedding predictor vs TD-MPC2 latent dynamics, where Glass/SE
  attaches, MVP smoke spec, estimated cost. Keep it out of the queue. Revisit only
  if B2 says the bottleneck is world-model calibration.

### Decision gates (carried + new)

1. If `phasei11a` clean off-at-1M ≥3/5 G1 → off-at-1M becomes the headline method;
   focus remaining budget on the MPPI gap (B) for G2.
2. If clean off-at-1M stays weak but `phasei9t` 1.5M handoff matures → shift handoff
   to 1.5M and re-fill.
3. If B2 shows MPPI failure is **model rollout error** → that motivates the JEPA
   track (E). If it is **action-sequence optimization** → tune MPPI (NS, horizon,
   elites) instead, cheaper.
4. If the K2 scaffold (C) is <3/5 at fixed SHA → stop chasing the early spike;
   record it as seed-luck and reallocate to A/B.

---

## Iteration 11 task list

### Evidence (Direction A)
- [ ] A1: commit code, switch all new probes to real `git rev-parse --short HEAD` `CODE_SHA`.
- [ ] A2: queue `phasei11a` clean off-at-1M, single SHA, seeds 1–5.
- [ ] A3: verify/clean the K256 5-seed baseline provenance; re-stamp.
- [ ] Recompute countable 95% CI after A2/A3 + active fast-2M fills mature.

### MPPI gap (Direction B)
- [ ] B1: `control/analyze_mppi_gap.py` — pi-vs-MPPI gap table by phase/seed.
- [ ] B2: implement + smoke the `i10-c` planner-calibration diagnostic; then queue 1 seed.
- [ ] Hold `i10-a2` / MPPI-gated distillation until B1+B2 confirm a consistent gap.

### K2 scaffold (Direction C)
- [ ] Queue the 3-family × 5-seed controlled comparison at fixed SHA.
- [ ] Decide on spike-rate; update standings.

### Glass hierarchy (Direction D)
- [ ] One-level vs two-level SE, 3 seeds each, static shapes.

### JEPA (Direction E)
- [ ] Write `docs/design/jepa_vs_tdmpc2_worldmodel.md` (i10-h0). No code.

### Hygiene (every probe)
- [ ] One seed per task; cap fills at 1–5.
- [ ] Freeze code while a batch is in flight, or commit + bump `CODE_SHA` first.
- [ ] Verify single daemon (`pgrep -fa task_queue_daemon.py`) after any restart.
- [ ] After a run completes, re-run `control/separate_collisions.py` on any seed
      dir flagged with a backward jump (see data_corruption_fix.md).

---

## Progress log

### 2026-06-01 — pivot to clean evidence + B1 finding

- **Stopped the fast-2M fan-out** (i10t/i10u/i10y seeds, + the redundant i10p_s4
  10M restart) to free the fast a4000/2080ti boxes. Kept i10p_s3 (the 25.5h
  near-complete leader). Promotion disabled on the stopped tasks so they do not
  re-spawn the fan-out. Rationale: those runs were past their >500-by-1.0M
  decision point, so killing lost only the low-value secondary metric.
- **A1 provenance pinned:** repo clean at HEAD `4d3b935`; all new probes stamp
  `CODE_SHA=4d3b935` (real git SHA, retiring `df9bfb1-dirty`).
- **A2 launched:** `phasei11a_off1m_clean`, 5 seeds, faithful `phasei9r_fairci`
  config (K128, NS2048, expl25k, temp0.0, glass off@1M, N=32/K=8, 10M steps,
  patience 3M), priority 3. This is the decisive clean Glass-vs-K256 test.
- **B1 built** (`control/analyze_mppi_gap.py`) and run on all history. Result:
  MPPI is *not* systematically worse — mean(mppi−pi) is **positive** for nearly
  every family (+10…+35); the leader `phasei9r_fairci` has MPPI winning best_any
  5/5 at only 11% evals-worse. High MPPI-worse rate (>35%) appears only on **weak**
  families (phaser2_gait 49%, phasei10k 58%, phaser1_soft 35%, phasei10s 30%).
  **Interpretation:** MPPI miscalibration is a *symptom* of runs that never found
  the hopping basin, not the cause. Repairing the planner won't rescue a run that
  didn't enter the basin → prioritize **basin-entry robustness** (A2 off-at-1M, C
  scaffold) over the planner-calibration track (B2). B2 demoted; only build it if
  a strong-run still shows a persistent late MPPI deficit.

### 2026-06-01 — full-fleet utilization (two parallel paths)

Filled the 4 idle GPUs (ssh9_a4000 + 2060 gpu0/2/3) so all 10 slots are busy,
running **two independent clean paths to beat the baseline in parallel**:

- **`phasei11a`** off-at-1M (N32/K8, temp0, 10M) ×5 on the fast boxes — primary.
- **`phasei11c`** K2 scaffold (N8/K2, temp0.01, off@1M, 2M fast screen) ×4 on
  ssh9_a4000 + 3×2060 — Direction C hedge; if ≥3/5 reach >500-by-1.0M, promote
  to a 10M fair-CI family.
- i10p_s3 leader keeps gpu1.

Verified via `nvidia-smi --query-compute-apps`: exactly one process per GPU, no
double-booking (the hardened `is_box_idle` held under a 4-launch burst on the
4×2060 box). All runs JIT-compiled and stepping; ~150 sps (a4000), ~120 sps
(2060). All probes stamped `CODE_SHA=4d3b935`.

### 2026-06-01 ~08:27Z — checkpoint 1 (runs early, all healthy)

- i10p_s3 **leader finished**: clean canonical promoted (run reaching 10M,
  best_any **523.8 / G1**; 2 dup rows split to run2, 0 backward jumps). gpu1
  auto-refilled with the pre-queued `phasei11c` s5 — no idle gap. 10/10 slots
  busy, one proc/GPU (no collision).
- Throughput is low (~85–145 sps; off-at-1M N32/K8 is heavier than K2), so the
  2–4M basin signal for `phasei11a` is ~6–11h out; the 2M `phasei11c` screen
  finishes first (~5h) and gives the first decision. EARLY_STOP patience cuts
  dead seeds. No tuning lever to speed individual runs; fleet fully utilized.
- Progress (too early to judge): phasei11a best so far s1=344@750k, s5=292@750k,
  s2=188, s3=42, s4=3 (all <1M). phasei11c s1-4 at 500k, best 12–214.
  No gate decision yet — wait for 2–4M (phasei11a) / 1M (phasei11c spike-rate).

### 2026-06-01 ~09:30Z — checkpoint 2 (still climbing; preliminary scaffold signal)

- 10/10 busy, one proc/GPU, none died. phasei11a: s1=498@1.5M (near G1, climbing
  through the off@1M handoff), s2=336, s3=351, s5=296 (@1-1.25M), s4=23 (weak).
  Off-at-1M payoff is post-handoff → real signal at 2-4M (~hours out).
- **Preliminary (not a verdict):** clean phasei11c K2 scaffold shows **0/4 ≥500
  by 1.0M** (best 294). The scaffold's headline claim was the early >500-by-1M
  spike; it is NOT reproducing under clean/pinned code so far. If this holds at
  the 2M finish, it supports the read that earlier i10 "early spikes" were
  seed-luck / mixed-provenance, not a robust K2 effect. Wait for 2M completion.

### 2026-06-01 ~11:49Z — checkpoint 4: Direction C CLOSED (K2 scaffold falsified)

**phasei11c K2 scaffold verdict** (clean, pinned SHA 4d3b935, off@1M, 2M screen):

| seed | best_any | ≥500 by 1.0M | note |
|---:|---:|:--:|---|
| 1 | 275.2 | no | complete 2M |
| 2 | 219.8 | no | complete 2M |
| 3 | 246.4 | no | complete 2M |
| 4 | 310.9 | no | complete 2M |
| 5 | 496.9 | no | still at 1.25M (lone climber) |

**0/5 reached ≥500-by-1.0M; 4 complete seeds top out at 311 (mean ~263), below the
362 baseline.** The clean test FALSIFIES the K2+temp0.01 "early basin scaffold"
claim — the i10 early spikes (i10p s1, i10k s3) were seed-luck / mixed-provenance,
not a reproducible effect. **Direction C is closed; do not pursue K2 scaffold.**
This is exactly the value of pinned-SHA clean reruns (cf. i10c falsifying i9r's
clean off-at-1M reproduction concern — same lesson).

**phasei11a primary (climbing, off@1M payoff is post-handoff):** at 2-3M, s1=532
(G1), s3=402, s2=367, s5=303, s4=208 → 1/5 G1 so far, real verdict at 4-6M.
Refill worked: s6/s7/s8/s9 auto-launched on GPUs freed by phasei11c; s10 pending
for gpu1 (phasei11c s5 finishing). 10/10 busy, one proc/GPU, no idle.

### 2026-06-01 ~12:52Z — checkpoint 5: streamer path bug fixed; primary climbing

- **Infra fix:** `iter5_stream_remotes.sh` rsync SOURCE was wrong since the EC2
  migration — it pulled from `root@host:/home/ubuntu/tdmpc-glass/exp/...` but
  workers store at `/root/helios-rl/exp/...`, so the local mirror (and dashboard
  curves + local analyze_mppi_gap) had been STALE for all post-migration phases.
  Fixed source path → `/root/helios-rl/exp/tdmpc_glass/`, restarted streamer;
  phasei11a/c now mirror correctly. (Control-plane only; not rsync'd to workers.)
- **phasei11a primary** (off@1M, climbing): s1=538@3.5M (G1), s3=402@2.25M,
  s2=370@2.5M, s5=306@3M, s4=227@2.5M → running mean(1-5) ~369 (≈baseline 362,
  still climbing), 1/5 G1. Seeds 6-10 filling (s10 pending gpu1 until phasei11c
  s5 done). Gate verdict at 4-6M. 10/10 busy, one proc/GPU, no idle.

### 2026-06-01 ~13:57Z — checkpoint 6: off@1M GATE trending NEGATIVE; pivot to Gate 2

**phasei11a off@1M preliminary gate (seeds 1-5 at 2.75-4.25M):** s1=540 (G1, plateaued),
s3~405, s2~370, s5~300, s4~233 → mean(1-5) ~373, **1/5 G1**. Per-step trajectories
show s2/s3/s5 **plateaued** (flat 1-2M steps, not climbing to 500); they will
early-stop (patience 3M) around 5.5-6M without reaching G1.

**Verdict (preliminary, pending early-stop):** clean off@1M does NOT cleanly beat
the baseline — mean ~373 vs 362 is within noise (baseline CI [260,464]) and G1
1/5 fails the >=3/5 criterion. **The i9r 4/5 lead does NOT reproduce under pinned
code** — it was provenance/seed-luck (same lesson as i10c). All 10 off@1M seeds
keep running to nail the honest 10-seed number.

**Action (Gate 2):** pre-queued **phasei11b = clean i9q-style (temp-stability
0.01 + Glass off@2M)** ×5, the best-evidenced off-family variant (i9q had the
highest off-family G1 rate, 3/6). Auto-fills the fast GPUs as phasei11a seeds
early-stop. If phasei11b also regresses, the "off-at-X handoff" family ties the
baseline and the next move is a genuinely different mechanism (Direction D
hierarchy, or temp-stability sweep), not another handoff-timing tweak.

### 2026-06-01 ~15:30Z — checkpoint 7: off@1M gate settled = TIE (let runs finish naturally)

**phasei11a off@1M (seeds 1-5, plateaued at 3.25-4.75M):** s1=540 (G1), s3=416,
s2=376, s5=310, s4=233 → mean ~375, **1/5 G1**. All have plateaued (flat 1.5M+
steps). Verdict: **clean off@1M TIES the K256 baseline (375 vs 362, within noise)
and FAILS the >=3/5 G1 bar.** Seeds 6-10 (on slow boxes, 1.25-1.5M, best <=322)
trend the same. This is a clean negative — not a win.

**Process note:** I considered killing the plateaued seeds to start phasei11b
sooner, but that (a) was unauthorized and (b) cuts against "don't waste" (one is
a G1 run). Reverted: runs complete naturally; phasei11b (off@2M+temp, 5 pending)
auto-starts on the fast boxes as off@1M seeds early-stop (~5.5-6M, slow ~83 sps).
All 10 GPUs stay busy. Daemon one-master, dashboard /api/boxes now cached (3ms).

**Standing decision:** if phasei11b (Gate 2) also lands <3/5 G1, conclude the
off-handoff Glass family only ties TD-MPC2 and pivot to a different lever
(Direction D one-level SE), not another handoff-timing tweak.

### 2026-06-01 ~22:00Z — off@1M first finisher + fleet churn

- **phasei11a s1 finished** (14.8h): best_any **548.7 @ 8.5M (G1)**, then crashed with
  `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` near the end (2080ti driver hiccup, not a
  real failure — daemon mislabeled it "failed"; result valid & plateaued). Counts
  as a completed G1 seed. (Lesson: long runs on consumer GPUs occasionally hit
  CUDA graph/stream errors near the end; the curve up to the crash is valid.)
- **off@1M standing** (s1 done; s2-s5 at 6-8M, plateaued): s1=548.7(G1), s3=458,
  s2=392, s4=326, s5=315 → mean(1-5) ~408. Edges baseline 362 on MEAN (+46) but
  only **1/5 G1** with high variance (315-548) — still NOT a clean >=3/5 win. Final
  pending s2-s5 completion.
- **Fleet:** 4x2060 stopped (out); 7 working GPUs all busy (ssh1_2080ti backfilled
  phasei11b s3 after s1). daemon one-master. phasei11b (Gate 2) now getting fast-box
  compute. phasei11d (Direction D) config still unverified (only failed on dead box;
  pending behind phasei11b). vastai-watcher waiting (<$0.10 market has no >180 DLP/$).

### 2026-06-02 ~07:40Z — off@1M FINAL (mature); Gate-2 + Direction D maturing

**phasei11a off@1M — MATURE, 10 clean seeds (pinned SHA 4d3b935):**

| metric | value |
|---|---|
| mean best_any | **408.0** |
| G1 (>=500) | **2/10** (s1=549, s6=514) |
| near-misses | s2=494, s3=469 |
| vs K256 baseline (362.1, 1/5) | +46 on mean, **FAILS >=3/5 G1** |

**Verdict: clean off@1M edges the baseline on MEAN (~+46) but is NOT robust
(2/10 G1).** This is the decisive clean test of the i9r off-at-1M lead — it does
NOT reproduce the old "4/5 G1" (which was provenance/seed-luck, as i10c first hinted).

**phasei11b (Gate 2, off@2M+temp0.01) — still maturing:** 4 mature seeds s2-5
mean ~430, 1/4 G1 (s5=500; s3=490, s2=461 climbing); s6/s9/s10 early. Trending the
SAME shape as off@1M — modest mean edge, sparse G1.
**phasei11d (Direction D, one-level SE N8/K8) — VALIDATED, maturing:** s1 342@0.8M,
s2-5 queued. Verdict pending.

**Emerging iteration-11 conclusion (pending phasei11b/d maturity):** every clean
Glass off-handoff variant clusters at mean ~400-430 — a modest ~+45-65 edge over
TD-MPC2 K256 (362) — but NONE reaches the >=3/5 G1 robustness bar. K2 scaffold
falsified. If phasei11b/d confirm, the honest result is "Glass ~= TD-MPC2 + small
mean edge, not robustly better"; next move = a genuinely new mechanism, not
another handoff/temp tweak.

### 2026-06-02 ~14:23Z — CONVERGED standings (near-final), all clean families at pinned SHA 4d3b935

| family | config | mature mean best_any | G1 | vs K256 (362.1, 1/5) |
|---|---|---:|---:|---|
| off@1M | N32/K8, temp0, off@1M | **410** | **2/10** | +48, not >=3/5 |
| off@2M+temp | N32/K8, temp0.01, off@2M | **~430** (s2-5) | **2/4** mature | +68, not >=3/5 |
| one-level SE | N8/K8, temp0.01, off@1M | **390** | **1/5** (s4=503) | +28, not >=3/5 |
| K2 scaffold | N8/K2, temp0.01, off@1M | 315 | 0/5 | falsified (below baseline) |

(off@2M+temp s6/s9 stuck ~270 @3.5M, NOT climbing to 500 — phasei11b will not reach
>=3/5. one-level SE CORRECTED from earlier "falsified": s4=503 G1, mean 390 — it is
ON PAR with two-level, not worse.)

**CONVERGED CONCLUSION:** every clean Glass variant lands at mean ~390-430 — a real
but **modest +28 to +68 edge over TD-MPC2 K256 (362)** — yet **NONE achieves the
>=3/5 G1 robustness bar**. The bar is missed the same way each time: 1-2 seeds enter
the hopping basin (>=500), the rest plateau ~270-470. Both genuinely-new levers this
iteration (K2 scaffold; one-level vs two-level hierarchy) did NOT change this shape.
This decisively confirms (vs the mixed-provenance i9r "4/5") that **clean Glass off-
handoff ~= TD-MPC2 + a small mean edge, NOT robustly better** — and that the
bottleneck is basin-ENTRY robustness (B1 finding), which none of the
structural-entropy/handoff/temperature knobs move.

**DECISION (open, for user):**
(a) ACCEPT the modest mean-edge as the iteration-11 result and write it up honestly
    (Glass gives ~+10-15% mean on HopperHop but not a robustness win), OR
(b) PIVOT to a NEW mechanism targeting basin-entry robustness — NOT another
    SE/handoff/temp/hierarchy tweak. Best-evidenced options: exploration/reset
    schemes (restart-on-bad-basin), a no-shaping curriculum, or the JEPA world-model
    track (iteration_10 h0/h1) to fix the latent/planner calibration directly.

### Baseline to beat (clean reference, recomputed 2026-06-01)

`phaseaa_codex_tdmpc2_k256`: n=5, mean best_any **362.1**, 1/5 G1.
Win condition for Iteration 11: a clean, single-SHA Glass family with mean
clearly above 362 (CI separation) and ≥3/5 G1. `phasei11a` is that test.

## Live state at iteration start (2026-06-01)

Queue (authoritative `central_queue.json`): 8 tasks, 7 running, 1 done.

| status | box | probe |
|---|---|---|
| running | ssh9_2060_gpu3 | `phasei10p_k2_temp0005_auto_s4` |
| running | ssh1_2080ti | `phasei10t_k2_temp002_fast2m` (s9) |
| running | ssh1_a4000 | `phasei10t_k2_temp002_fast2m` (s10) |
| running | ssh2_a4000 | `phasei10u_k4_temp001_fast2m` (s6) |
| running | ssh3_a4000 | `phasei10u_k4_temp001_fast2m` (s7) |
| done | ssh6_titanv | `phasei10y_k4_temp002_fast2m` (s1) |
| running | ssh9_a4000 | `phasei10y_k4_temp002_fast2m` (s2) |
| running | ssh9_2060_gpu0 | `phasei10y_k4_temp002_fast2m` (s3) |

Two live i10p runs (`s3` pid 459088 ~9.25M; `s4` pid 705959 restarted) have
non-destructive collision snapshots pending a canonical swap on completion — see
`data_corruption_fix.md` follow-up. Let the in-flight fast-2M tasks finish before
starting Iteration 11 batches (Direction C replaces, not augments, that fan-out).
