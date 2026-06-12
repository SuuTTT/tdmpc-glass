# CHANGELOG — dev & training log

*Append-only. Each entry: **Dev** (what was built/analyzed and where), **Train** (what was
queued/running/harvested, on which boxes), **Verdicts** (results that changed the ledger/paper).
Maintained every monitor tick. Current live state: dashboard (port 5055) + `TaskList`; campaign
verdicts: `docs/iterations/RESEARCH_LEDGER.md`.*

## 2026-06-11

### Dev (EC2 control + ssh7 5070 Ti dev box)
- **Paper**: redundancy-criterion draft completed; every number persisted to JSON (ve-probes re-run on
  ssh7; SimNorm SE gap reproduced on EC2; anchor + clustering aggregations scripted). Published as blog
  Part 4; full TMLR-style LaTeX repo **github.com/SuuTTT/wm-redundancy-paper** (compiles clean, 10pp,
  evidence/ JSONs bundled).
- **VG-SE mechanism-check infra** (B0/B1): synthetic multi-entity env (`synthetic_entities.py`),
  entity-factored transformer WM (`entity_wm.py`), gate runner + selftest, value-coupling probe
  (`value_coupling_probe.py`, cross-Hessian vs similarity vs attention, shuffle-null AP). Ran on ssh7.
- **Iter-30 plan** (`docs/research/abstraction-axes-plan.md`, blog Part 5): aim abstraction at the
  planning axes (temporal/action), not state. P1 temporal-abstraction predictor; P2 Hermite-spline
  action bottleneck; P3 value-equivalent macro head.
- **P1 dumps**: rescued Titan-V Ori ckpts (941MB) + ssh4 Cab ckpt to EC2; `ori_mech.npz` /
  `cab_mech.npz` mech dumps running on ssh7.
- **P2 script**: `spline_mechcheck.py` (Hermite + ZOH control, open-loop replay, GO ≥0.95 return
  preservation) — chained on ssh7 behind the dumps.
- **Control plane**: dashboard refactor (earlier); mirror filter fixed to sync dreamer dirs + added
  missing ssh6_3060 box; daemon/stream registries right-sized + restarted.

### Train (worker fleet)
- **Harvested**: behav-on-Panda n=5 (final 1586 ≈ van 1416, null); geoglass-on-Panda n=5 (1247, null —
  clustering matrix complete); jumpy anchor finalized at n=5/arm (persisted, claim narrowed);
  DreamerV3 generality closed (~190–340 on Panda @150k, persisted; dgen_Pick2 lost to dead 2080Ti).
- **Queued + running (wave 1, 24 runs)**: P1 k-sweep `phasei30_jumk{2,8}` × Pick/Ori/Cab × s0-2
  (eff. horizon fixed at 24, MPPI_H=2k) + Pick n-boost `ti30_{jum,van}_Pick{5,6,7}` (n=8 resolves the
  +32% CI-crosses-zero trend). Saturating ssh2_a4000, ssh9_a4000, ssh4_a4000, ssh4_a4000b, ssh6_3060.
- **Fleet**: destroyed 6 boxes (orphan ssh3 A4000, $0.103 A4000, 2-cpu A4000, 3 dead) under explicit
  authorization — **policy since: Claude only recommends, user destroys manually**. Destroy-ready:
  36994217 (2080 Ti, unreachable), 38751740 (Titan V, ckpts rescued). Keepers (~$0.57/hr): ssh7 dev
  5070 Ti, ssh4_a4000 (25 cpu), ssh2/ssh4b/ssh9 A4000s, ssh6_3060.

### 16:50 update
- **P2 spline mechanism-check: NO-GO at the pre-registered gate** (`spline_mechcheck_PandaPickCube.json`):
  mean return-preservation 0.364 at knot k=4 (gate ≥0.95); ZOH 0.305; spline adds only +0.06 over ZOH.
  Expert action L2 deviation ~0.49/step → TD-MPC2's winning Panda actions carry high-frequency content a
  2d-per-knot bottleneck cannot express open-loop. Caveat (pre-stated): open-loop replay is an upper-bound
  test; closed-loop spline-MPPI could differ — but per pre-registration, do NOT build. Missing control to
  add before final write-up: exact-action open-loop replay (isolates reconstruction-error vs chaos).
- **P1 dumps**: first launch failed on a CLI flag (`--seeds`→`--seed`); relaunched on ssh7.
- 2080 Ti + Titan V confirmed destroy-ready to user.

### 18:30 update — P1 first result
- **P1 candidate-signal screen** (`p1_temporal_signals.json`): of 9 checkpoint signals, exactly ONE
  orders the tasks like the measured jumpy gains (+90/+32/0): **disc_err_gap** = median(disagreement)/
  median(true k-step error) — Ori 1.33 > Pick 1.12 > Cab 0.95. Story: jumpy pays where the k-step model
  is accurate AND calibrated-conservative (disagreement ≥ error); fails where bad + overconfident
  (Cab err 3× Ori at same latent scale, disc/err < 1). Survivor is scale-invariant (ratios only —
  absolute latent scales differ ~30× between the Pick dump era and ori/cab dumps; flagged).
  n=3 honesty: hypothesis-generating only.
- **Pre-registered out-of-sample test chained**: PandaPickCubeCartesian phasei27 jum/van ran but was
  never harvested → computing disc_err_gap on its ckpt FIRST (dump chained on ssh7), prediction
  committed, THEN harvest its gain. Plus the running k-sweep as dose-response.
- Dev-box chain: ori+cab mech dumps done → value-probes (Ori/Cab) running → P3 ×2 → Cartesian dump.

### 19:15 update — P3 closed, C1 cross-task, P3-mechcheck falsification
- **C1 holds on all 3 anchor tasks** (`value_probe_jum_{ori,cab}_n12.json`): Ori R²=0.9983, Cab
  R²=0.9995 (Pick 0.9994 prior). Cross-task evidence for the paper's §4.1.
- **P3 mechanism-check contrast mirrors the jumpy gains** (`p3_macroq_{ori,cab}.json`): Ori GO
  (value_cost_ratio 0.352, ρ=0.57 — errors systematically cost value); Cab NO-GO (ρ=0.23, errors
  large but value-unstructured). Reinforces the P1 "accurate+calibrated vs bad+overconfident" story.
- **BUT P3 is closed by existing data**: phasei27_ve already ran the value-equivalent macro head on
  Ori — final 583 (n=3) vs jumpy 2145 (n=5), catastrophic harm on the exact task the mechcheck said
  GO. Verdict: P3 dead; AND a falsification-grade lesson — mechanism-check GO licenses a test, it
  does not predict success (NO-GO direction remains reliable). Goes into the paper's §6/§7.
- Iter-30 now rides on P1 alone: disc_err_gap + k-sweep (4 more finals landed, 94 done) +
  Cartesian pre-registration (dump chained).

## 2026-06-12

### 02:30 — P1 pre-registered predictions committed (before any harvest)
- All 8 k2 ckpt dumps processed (grinder needed two fixes: rsync nested-dir + ssh-eats-stdin-in-
  while-read). `p1_ksweep_prediction.json` committed: **disc_err_gap k2 < k4 within every task →
  predict k2 jumpy gain LOWER than k4 on Pick/Ori/Cab**; cross-task ordering Ori>Pick>Cab preserved
  at k2 (1.03/0.97/0.67). No ti30 final has been read. Grinder pass 4 on the 3 finished k8 ckpts.
- Fleet: k8 wave training (5 boxes); Pick n-boost + Cartesian queued behind.

### Verdicts (ledger/paper updated)
- **Entity-graph NO-GO** (3rd redundancy data point, cleanest): value-coupling cross-Hessian recovers
  known-by-construction pairs at chance (AP 0.50, z −0.08) with near-perfect reward fit (0.0026);
  similarity graph beats it (0.75). VG-SE bet fails its instrument gate.
- **Anchor narrowed at n=5**: Ori +90% CI-separated; Pick +32% not separated; Cabinet null (tie).
- **Clustering-on-Panda complete**: geo + behav both null on manipulation.
- **DreamerV3 generality**: far below TD-MPC2-class on Panda at matched modest budget.

### ~13:10 — Control plane: dashboard cleanup + queue archiving (user request)
- **Dev (EC2)**: dashboard restarted so the box panel follows the daemon's current BOXES (phantom
  rows for destroyed ssh1_a4000b/ssh8_a4000 gone; ssh1_2080ti + ssh6_titanv labeled destroy-pending).
  Queue panel now hides done/superseded/old-failed rows by default with an "N archived — show"
  toggle (`?all=1` honored); new top panel "Live: Experiment / Dev" renders
  `exp/tdmpc_glass/live_status.json` (new `/api/live_status`). ETA scheduler guarded against
  queue rows whose box left the fleet. Files: `control/dashboard/{__init__.py,boxprobe.py,
  queue_api.py,templates/index.html}`.
- **New script** `scripts/archive_done_queue.py` (NOT wired into daemon/dashboard; for the monitor
  loop): moves done/superseded_dup/superseded_oom rows with ended_at >48h from central_queue.json
  to queues/archive_done_failed.jsonl under the daemon's fcntl-lock + tmp/rename pattern, with
  .bak_archive_<epoch> backup. First run: 14 rows archived, 113 remain (105 done <48h kept).
- Daemon + streamer untouched (one-master rule respected); no mirror data deleted.
