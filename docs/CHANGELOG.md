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

### 13:40 — SCORING EVENT (milestone: Part 6 update published)
- Predictions finalized (k8 + cheetah committed pre-harvest), then ALL ti30 finals read for the
  first time. **Score: k2 block 4/4, k8 block 0/3** → disc_err_gap = real cross-task predictor at
  fixed k; NOT k-invariant (iteration-drift confound). **k=4 unimodal optimum on all 3 tasks.**
  **Pick anchor CI-separated at n=8 (+45%, CI [66,1153])** — jumpy: 2/3 CI-separated + 1 null.
  Ledger + Part 6 update + dashboard history updated. Cheetah OOS reports tonight (Part 7).

### 15:40 — Part 7 published (cheetah OOS scored) + iter-31 auto-k Wave A queued
- CheetahRun: jum 620 vs van 558 (+11%, CI [-120,248], not separated) = "weak-positive" exactly as
  the committed gap-1.017 prediction said. Predictor final scoreboard: 8/8 ordering facts at fixed k,
  0/3 cross-k upward. Part 7 live; p1_cheetah_oos_score.json persisted.
- NEW DIRECTION (user standing order: beat TD-MPC2 with abstraction done right): auto-k =
  calibration-selected temporal grain. Wave A (6× 100k probes, k∈{2,4,8} × Pick/Ori) queued —
  gate: short-budget disc_err_gap must reproduce the known full-budget ordering. GO -> Wave B
  positive-method gate (auto-k vs vanilla on unseen tasks). Cartesian OOS dropped (no ckpts saved).

### 17:10 — composition is calibration-gated (the iter-32 thesis emerges)
- Ori composition GO (rho 0.805, win 85%) vs **Cab composition NO-GO (rho 2.54, win 0.3%)**:
  d4∘d4 compounds accuracy on a good+calibrated base model and compounds ERROR on a bad+overconfident
  one. Pyramid viability is predicted by the calibration signal -> "compose only what is calibrated."
  M1 (calibration fine-tune of the Cab model) running on ssh7; if it raises disc/err toward >=1 AND
  the recomposed test flips toward GO, the full system claim writes itself.
- Pick composition test lost to daemon double-booking on ssh9 (GPU contention); retry queued later.
- Fleet: cheetah n-boost x3 running, Wave A k8 finishing, M1 on dev box.

### 19:30 — M1 GO + re-composition FLIP (pending control); M2 queued with committed prediction
- **M1 GO**: calib fine-tune (100k) on Cab: disc/err 0.949→1.301, err_med 0.630→0.211 (3× better).
- **Re-composition FLIPPED**: rho 2.54→0.692, win 0.3%→75.7% — "compose only what is calibrated"
  confirmed on the failure task. CONFOUND CONTROL running (calib_coef=0 fine-tune, same +100k):
  milestone blog held until it reports (~2h).
- **M2 flagship queued** (15 calib-jumpy runs, 3 tasks × 5 seeds, priority behind Ori wave) with
  pre-registered prediction committed first (m2_prediction.json): beats jumpy on Cab, ties Ori/Pick.
- Dev queue: Pick composition retry → control chain on ssh7. Workers: Ori n-boost + last cheetahs.

### 20:00 — Pick composition deferred (ckpt loss), control chain running
- All full-budget Pick k4 ckpts lost to worker disk self-heal during new waves; Pick composition
  test deferred until M2's fresh Pick ckpts land (becomes the calibrated-composition test).
  Non-gating: Cab flip + more-training control carry the thesis.
- ssh7: control fine-tune (calib_coef=0, +100k) running; then ctrl dump + ctrl composition.

### 20:50 — Part 8 published; ledger updated; Cab-600k pre-registered test queued (priority 2)

### 22:00 — P2 exact-replay control: floor is 0.943, spline NO-GO stands
- Replaying EXACT actions open-loop preserves 0.943 mean (min 0.754) — not 1.0 (contact amplifies
  tiny divergence). Spline's 0.364 vs the 0.943 floor: reconstruction error dominates; the P2
  NO-GO is confirmed with an honest denominator. Dev queue item closed.

### 23:35 — Ori n=8 harvested: +54% CI-separated [250,1083] (settled from +90% at n=5)
- Anchor now n=8/n=8 on Pick (+45% sep) and Ori (+54% sep); Cab null at 500k (600k test training).
  Ordering Ori>Pick>Cab preserved (predictor consistent). Paper v3 updates batch with Cab-600k.

## 2026-06-13
### 02:00 — Cab anchor-flip test (600k): partial — directional +12%, not CI-separated
- jum 797 vs van 711 (n=3 each, +12%, CI [-147,318]). The Cab null OPENS directionally at 600k
  (was ~0% at 500k) — weakly supports "undertraining", but not confirmed at n=3. Anchor stays
  2/3 CI-separated (Pick +45%, Ori +54%); Cab = honest weak-trend, not a flip. M2 calib runs draining.

### 06:45 — Pyramid lead: depth sweep on converged Ori = NO-GO (return flat over horizon)
- plan_depth Ori: return flat across n_macro 3/6/9/12/16 (H 12-64): 2856/2808/2827/2679/2795, best
  +1.7% (noise). Deeper compositional planning does NOT raise return on a converged model — composition
  is ACCURATE (mechanism-check) but the converged model is already uniformly-accurate enough that
  horizon depth is saturated. Pyramid-as-deeper-planning joins the null pile. Cab depth sweep running
  (the under-converged case — the only place depth might still matter).

### 07:40 — Cab depth NO-GO too; pyramid lead closed (both tasks). 5070 freed.
- Cab depth flat (1774→1701, H 12-64), best depth 3. Pyramid: composition accurate, return-saturated
  on converged models. Temporal axis now as exhausted as the state axis — both reduce to "converged
  model already good enough." M2 calib 8/15 (last running experiment). Next dev: compounding-error-
  vs-horizon curve (the paper figure explaining WHERE composition stays accurate vs return-saturates).

### 08:40 — compounding-error curve harvested (paper figure); 5070 free
- Composed-d4 error sub-linear in horizon, 1.3-1.8x below 1-step iteration on both tasks; reconciles
  pyramid-mechcheck GO vs depth-sweep NO-GO (accuracy real but unused by the short-horizon controller).
  M2 calib 9/15. 5070 idle after curve.

### 09:05 — M2 partial (calib from-scratch): NULL-to-harmful, prediction missed (honest)
- Cab calib 1053 (n=5) ≈ plain jumpy 1050 (TIE); Ori calib 1385 (n=4) < jumpy 2145 (WORSE).
  Committed prediction "calib beats jumpy on Cab" = MISS. Confirms the control: calibration loss
  adds nothing from-scratch either. iter-32 (calibration-shaped WM) = dead, both fine-tune & scratch.
  Pick calib seeds finishing -> full score + closing milestone post then.

### 11:25 — iter-32 CLOSED (Part 9): calibration null-to-harmful from scratch
- M2 full: Cab 1053=1050 (0%), Ori 1377 vs 2145 (-36%), Pick 849 vs 1969 (-57%, n2). Prediction missed.
  Temporal axis exhausted. Queue EMPTY. Paper is the deliverable; high-DoF the only open bet (user cost call).

### 12:45 — M2 finalized at n=5: calibration harmful (Cab ~0%, Ori -36%, Pick -57%). Queue empty; campaign idle.

### 12:55 — iter-33 high-DoF bet launched (user: keep GPU busy)
- The ledger's last open headroom: jumpy vs vanilla on HumanoidRun + HumanoidWalk @1.5M steps (3x the
  500k that floored), 2 seeds/arm, pre-registered (iter33_highdof_prereg.json), compute-matched. 8 runs
  saturating all 5 workers ~1.5-2 days. Gate: jumpy beats vanilla CI-separated on >=1 humanoid task ->
  high-DoF headroom real; else -> paper gets both-axes + high-DoF closure.

### ~16:35 — iter-33: vanilla HumanoidRun FLOORED at 1.5M (final ~5, peak 9, n=2)
- HumanoidRun too hard for vanilla TD-MPC2 even at 1.5M (matches ledger 'Humanoid floored'). Makes the
  jumpy HumanoidRun arm the decisive cell: rescue-off-floor = striking win; also-floor = wash. Jumpy
  arms ~400k (slow, ~1.5d to go). HumanoidWalk arms (jum+van) now running = the likely-informative task.

### 06:05 — iter-34 GWM FIRST GO survives fair control (Part 10)
- Graph OOD value-R2 0.57 vs pooled-mono 0.21 (gap .35) AND vs fair pad-mono 0.40 (gap .18>.15). GO
  survives pooling control across seeds; pooling inflated ~half. Win = compositional OOD generalization
  NOT contacts (crit B failed); representation-level only. Part 10 posted. pad s2,s3 running.
  NEXT GATE: control-benefit (return under planning at held-out N) before any ManiSkill escalation.

### iter-34 GWM CLOSED (Part 11): control-benefit NO-GO
- Random-shooting MPC: graph ~= fair-mono ~= random floor at all N (graph-rand +-13, std ~110). OOD
  value-decode advantage does NOT convert to control. GWM closes like explicit abstraction; redundancy
  criterion now spans state+temporal+relational/graph. No ManiSkill escalation. Part 11 posted. Queue empty.

### iter-33 high-DoF CLOSED: jumpy hurts on learnable cell (HumanoidStand van 167 >> jum 34)
- Run both floor; Walk overlap; Stand (learnable) vanilla 5x jumpy. Temporal abstraction doesn't
  extend to high-DoF, hurts where learnable. Both research directions now closed; paper is deliverable.

## 2026-06-14 (later) — Real GWM comparison env stood up + SOLD reproduced
- ManiSkill 3.0.1 isolated env on ssh4_a4000b (torch cu124); official PPO baseline TRAINING on
  StackCube-v1 (relational), the monolithic anchor on a real benchmark. /root/ms_ppo.log.
- **SOLD (official GWM, github.com/maltemosbach/sold) REPRODUCED** on ssh4_a4000 (isolated /root/sold_venv,
  torch 2.5.1+cu124, mujoco-py built clean): provided reach_red ckpt eval = 100% success (30/30) vs
  paper 97.9% — MATCHES. Fresh train run confirmed learning (dyn_loss 517->0.43), stopped on disk (15GB
  replay memmap > box). Benchmark = custom multi-object Fetch (Reach/Push/Pick x Specific/Distinct),
  metric = success rate.
- **KEY for our paper:** SOLD Table 1 Reach-Specific: SOLD 97.9 ~= TD-MPC2 97.6 >> DreamerV3 87.4 — i.e.
  monolithic TD-MPC2 TIES the graph WM on the non-relational variant; SOLD's wins are on Distinct/
  relational variants. This is THIRD-PARTY published evidence for the redundancy criterion (graph helps
  only where relational structure is essential). Slot-MPC code "coming soon", ObjectZero none.

### ManiSkill PPO anchor: StackCube too hard (0% success @10M) — not a useful anchor
- StackCube-v1 state-PPO floored at 0% success (return ~25) at 10M — StackCube is hard for PPO at this
  budget. A useful ManiSkill monolithic anchor would need an easier task (PickCube/PushCube) or more
  compute. Moot anyway: no published GWM targets ManiSkill; SOLD's own Fetch suite is the real
  comparison platform (and SOLD is reproduced there). Decision: cite SOLD's published table vs
  reproduce SOLD-Distinct head-to-head (needs a bigger-disk box for the 15GB replay buffer).

### 2026-06-14 — iter-33 fully closed (HumStand n=2) + 38768950 destroyed (user-instructed)
- HumanoidStand FINAL n=2: jumpy [33.5, 15.3] mean 24.4 vs vanilla [162, 171] mean 166.6 — jumpy ~7x
  WORSE on the learnable high-DoF task. iter-33 closed: temporal abstraction does not extend to high-DoF
  and hurts where learnable. Data mirrored to EC2.
- Destroyed 38768950 (ssh2_a4000) per user's explicit instruction after its run finished. tdmpc fleet
  now = ssh4_a4000 (39109169, SOLD box) only. Queue empty/drained.

## 2026-06-14 (B) — SOLD Distinct training live but throughput-bound (~3wk/run)
- SOLD installed+verified on ssh2:38955 (3090, reach_red eval ~98-100%); HF cache cleared (19G,
  re-downloadable), SeSE untouched. Distinct (Odd) SOLD training RUNNING (step 4200, ep_return
  0.17->1.29, losses healthy) but collection is single-core mujoco-py: ~2400 steps/hr -> ~500h (~3wk)
  to the paper's 1.2M-step checkpoint. Authors ship only Reach-Specific ckpt (no Distinct shortcut).
  DECISION SURFACED: full independent Distinct repro = multi-week/multi-box; A (cite reproduced SOLD +
  Table 1 as redundancy evidence) is better value. Run left accumulating curve pending user call.
