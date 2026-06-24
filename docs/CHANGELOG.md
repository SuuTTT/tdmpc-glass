# CHANGELOG — dev & training log

## 2026-06-22 — Part 18: "Does TD-MPC2 actually beat PPO?" two-axis DMC comparison
Dev: built EC2 analysis pipeline exp/tdmpc_glass/dmc_ppo_vs_tdmpc2/build_results_ec2.py
(d2 5-seed TD-MPC2 sample-eff@500k for Cheetah/Hopper; fresh 1-seed for the 3 hard tasks;
fresh same-box 1M runs for final+sps+wall-clock; PPO from brax logs). 5 tasks:
CheetahRun, HopperHop, AcrobotSwingup, CartpoleSwingupSparse, HumanoidRun (WalkerWalk dropped=saturated).
Train (b3060, 4x3060): GPU0 CheetahRun TD-MPC2 1M (same-box wall/sps); GPU1 AcrobotSwingup,
GPU2 CartpoleSwingupSparse, GPU3 HumanoidRun fresh TD-MPC2 1M. PPO driver waits for GPU0 then runs
HopperHop/Cartpole/Acrobot(100M)/Humanoid PPO serially. PPO uses per-task default num_timesteps
(Acrobot=100M, rest=60M). CheetahRun PPO + HopperHop TD-MPC2 reused from earlier same-box session.
Verdicts: pending run completion -> RESULTS.json + Part 18 post + plots.

*Append-only. Each entry: **Dev** (what was built/analyzed and where), **Train** (what was
queued/running/harvested, on which boxes), **Verdicts** (results that changed the ledger/paper).
Maintained every monitor tick. Current live state: dashboard (port 5055) + `TaskList`; campaign
verdicts: `docs/iterations/RESEARCH_LEDGER.md`.*

## 2026-06-18 — GHM/CompPlan: antmaze UNBLOCKED + plan-over-policies planner
**Dev.** (1) ANTMAZE UNBLOCK (additive, no existing files broken): InFOM's `-ft-` finetuning dataset
does not exist on the OGBench server for antmaze (the data_gen_scripts generator only covers
cube/scene/puzzle), and eval only runs during the finetuning phase (`main.py` gates eval on
`i > pretraining_steps`). Root cause: `envs/env_utils.make_env_and_datasets(reward_free=False)`
rewrites the env name to insert `-ft-`. Fix: added flag `--ft_dataset_fallback` to `main.py`
(default 0; AUTO-enabled for any env containing `antmaze`). When active, the finetuning dataset is
loaded from the SAME non-ft singletask dataset used for pretraining (which DOES auto-download), still
reward-labeled because OGBench relabels rewards for any singletask dataset unconditionally. Eval
(`episode.success`) then runs end-to-end. Existing cube runs are untouched (flag off, no antmaze).
(2) PLANNER (CompPlan gap c): new `planning/compplan_planner.py` — `CompPlanPlanner` does beam search
over sequences of base policies, using the GHM's `compute_fwd_flow_goals` to jump to M sampled future
states per (state, policy), scores by -L2-to-goal (or learned reward head), receding-horizon (returns
first action, re-plan at next state). GHM actor wrapped as base policy via `actor_base_policy`. New
`planning/smoke_planner.py` builds the agent (restores a checkpoint if found, else random-init) and
runs one plan step. Synced `main.py` + `planning/` to BOTH boxes (`/root/ghm/infom`). Appended two
antmaze envs to `exp/tdmpc_glass/ghm/envs.txt` so the keep-busy loop folds antmaze into rotation.
**Train.** VERIFIED on g3 (4090, mem-capped 0.12, shared with cube loop): antmaze GHM run
(`--agent=agents/ghm.py --env_name=antmaze-medium-navigate-singletask-task1-v0`, 10k pretrain +
5k finetune) trained at ~320 it/s and wrote `finetuning_eval.csv` with `evaluation/episode.success`
columns populated (= eval pipeline works). Numbers (tiny budget, random-init actor, NOT a paper
repro): `episode.success = 0.0` at steps 10001 and 15000 over 10 episodes, `episode.return = -1000`
(1000-step episodes). Planner smoke on g3 (mem-capped 0.10) restored a real cube checkpoint
(`ghm_a8_t4_s1/params_1500000.pkl`, obs28/act5) and ran ONE plan step clean: returned a (5,) action,
3-deep policy sequence, score -6.16. **Verdicts.** Antmaze is now runnable WITH eval; no paper
numbers claimed (full repro needs ~1M+ steps + planner-driven eval, not the train-actor eval used
here). Planner is implemented + smoke-passing but not yet wired into OGBench eval (next step).

## 2026-06-17

### Dev (EC2 control + g3 2x4090 / g3090 2x3090)
- **GHM/CompPlan repro — pipeline standup** (`docs/ghm_repro/PLAN.md`). InFOM (flow-occupancy)
  scaffold stood up on BOTH GPU boxes in an ISOLATED venv `/root/ghm/.venv` (helios `.venv` untouched).
  Recipe that worked per box: fresh `python3 -m venv`, `pip install ogbench==1.1.0`,
  `pip install -r infom/requirements.txt`, `pip install -U "jax[cuda12]"` → jax 0.10.1 sees both CUDA GPUs.
- **Gotchas hit & fixed**: (1) two concurrent pip installs into one venv corrupted jaxlib → Bus error on
  `jax.devices()`; fix = recreate venv, single install. (2) g3090 reqs install silently hit
  `No space left on device` (box at 94%) so wandb/flax/distrax were missing → jobs crashed
  `ModuleNotFoundError: wandb`; fix = `pip cache purge` + `--no-cache-dir` reinstall of the missing pkgs.
  (3) InFOM finetuning needs a `-ft-` dataset NOT on the OGBench server → generated locally with
  `data_gen_scripts/generate_ogbench_manispace.py` (~2.3s/episode); cube-single ft (500+50 ep) made on g3,
  rsynced to g3090 via EC2. antmaze unsupported by that gen script → used cube-single tasks instead.
- Smoke verified: `ogbench.make_env_and_datasets` cube-single (obs 1e6×28, act 5) + antmaze-medium
  (obs 1e6×29, act 8); InFOM `main.py` 300-step run advances at ~14-18 it/s after JIT.

### Train (g3 + g3090, 4 GPUs, InFOM pretraining)
- 4 InFOM cube-single runs launched (1.5M pretrain + 0.5M finetune each), ~400 it/s after JIT:
  g3 GPU0 task1/sd0, g3 GPU1 task2/sd0, g3090 GPU0 task1/sd1, g3090 GPU1 task3/sd1.
  Logs `/root/ghm/logs/<name>.log`. NOT reproduction — pipeline standup only.

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

## 2026-06-15 — SOLD Distinct throughput collapses in training phase (~200 steps/hr → months)
- Step 13,075->13,225 in ~45min = ~200 steps/hr once gradient training engaged (10x slower than the
  early collection-only ~2300/hr). 1.2M-step paper ckpt now ~months away on this single-core mujoco-py
  setup; 3090 stuck at 1-3% util (GPU-starved by single-env collection). B is INFEASIBLE here without
  parallelizing collection (SOLD/mujoco-py = single-env, hard). STRONG recommendation: kill + bank A
  (we already reproduced reach_red + can cite SOLD Table 1 as redundancy evidence). Holding for user.

## 2026-06-15 — B (SOLD reproduction) PAUSED: infeasible on available hardware; box relieved
- Parallelized SOLD collection (SubprocVecEnv N=4) — but only 1.5x (collection isn't the bottleneck;
  per-step SAVi GPU encode + tiny batches are, GPU starved by design). ~13 days to 1.2M still.
- Box reality: 7.68-CPU cgroup quota (not 32). N=16 wedged it; even N=4 + box services pushed load to
  10.3 -> SeSE box appeared "unreachable" on the dashboard (saturated, not down).
- KILLED the SOLD run to restore the user's SeSE box (load 10.3->6.7, GPU 17G->2G freed). B is NOT
  viable on this hardware without changing training math (bigger/more-frequent grad updates) or a
  faster sim (would abandon SOLD's Fetch benchmark = our iter-34). Recommendation: BANK A — we already
  reproduced reach_red (98-100%) and can cite SOLD Table 1 (TD-MPC2 97.6 ~= SOLD 97.9 non-relational)
  as third-party redundancy evidence. SOLD code/.bak edits preserved on box for later if desired.

## 2026-06-15 — iter-36 compositional-OOD control INCONCLUSIVE (testbed too low-SNR); program at hardware edge
- Real Dreamer-style learner, graph vs mono on GPU-vectorized contact_entities (5070 Ti, ~64M steps/s).
  No controller beats random floor (graph==mono byte-identical) -> env control-signal ceiling ~10%
  (within noise) + BPTT-NaN -> UNINFORMATIVE, not a null. Compositional-OOD control remains untested
  (needs higher-SNR real env + compute; SOLD/ManiSkill blocked by iter-35 walls). BANK A.
- User: deleted SOLD on 3090 (re-installable, repro result safe), took 3090 for LLM. 5070 Ti idle.

## 2026-06-15 — Paper-B B0 NO-GO (real ManiSkill benchmark): no compositional headroom; program closed
- ManiSkill PushCubeMulti (GPU-sim, Blackwell torch cu128): PPO 100% vs random 0% (control signal PASS)
  but value-decode flat N2->6 + policy solves OOD (headroom FAIL). Passive distractors -> monolithic
  generalizes. Paper B fails its gate on an honest benchmark; folds into A as capstone negative.
  Redundancy criterion now spans real-benchmark compositional axis too. No more GPU; 5070 Ti -> user.

## 2026-06-15 — D1 (phase-gated switching WM): mechanism-check PRE-REGISTRATION
Direction reopened after user pushback on "nothing left": redeploy the verified
"communities = motion phases" finding from subgoals (failed) to **dynamics switching**.
Hypothesis: SimNorm discrete-code phases are real temporal structure a switching WM
can exploit on contact tasks (where jumpy already wins: Pick +45%, Ori +54%; Cab null).

Data: archival se_dump k-step mech dumps (Zt latent, Ztk true latent k ahead, err =
true jumpy k-step error, disc = jumpy-vs-iterated disagreement). SimNorm = 8 groups
× 64-way softmax → phase code = 8 per-group argmaxes; boundary = code-flip burst.

PRE-REGISTERED GO/NO-GO (must distinguish D1 from the dead lever-11/F uncertainty-gate):
  L1 phase-stratified error: err variance explained by phase-id (KMeans on Zt) exceeds
     random-grouping control (η² real > shuffled CI). [headroom for phase-experts]
  L2 boundary concentration BEYOND displacement: partial Spearman ρ(err, upcoming code
     churn | window displacement ‖Ztk−Zt‖) ≥ 0.10, CI>0. [novel beyond motion-magnitude]
  L3 beyond-uncertainty: partial ρ(err, churn | disp, disc) ≥ 0.10, CI>0. [beyond the
     already-failed F/disc signal]
  DISCRIMINATIVE: all effects stronger on Pick & Ori (jumpy-win) than Cab (jumpy-null).
GO = L1 & L2 & L3 hold on BOTH Pick and Ori AND are discriminative vs Cab.
NO-GO = phase adds nothing beyond displacement/disc (→ lever-11 redux) OR non-discriminative.

## 2026-06-15 — D1 mechanism-check VERDICT: NO-GO (lever-11 redux, displacement-confounded)
Ran scripts/d1_phase_mechcheck.py on archival se_dump (Pick/Ori jumpy-win, Cab jumpy-null; 3 seeds each).
Result (exp/tdmpc_glass/mechcheck/d1_phase_mechcheck.json):
  raw ρ(err, code-churn) positive (0.12–0.41) BUT entirely displacement-driven.
  L2 partial ρ(err, churn | ‖Ztk−Zt‖): Pick −0.18/−0.089/−0.065, Ori −0.063/−0.21/−0.034 → all ≤0.
  L3 (| disp, disc): negative/near-zero everywhere.
  L1 η² "passes" vs shuffle BUT non-discriminative — Cab η²=0.97 ≥ Pick/Ori (0.19–0.47).
PRE-REGISTERED GO required L2&L3 ≥0.10 CI>0 on BOTH Pick&Ori AND discriminative vs Cab → FAILS all.
Interpretation: SimNorm motion-phases carry no WM-error structure beyond raw latent displacement
(the already-falsified lever-11/F uncertainty-gate). Phase-gated switching WM has no headroom.
Bonus negative: the jumpy win on Pick/Ori is NOT explained by phase-structured error (non-discriminative).
Cost: zero GPU (ran on archival dumps). Discipline worked.

## 2026-06-15 — Option-3 precondition PRE-REGISTRATION (error-prioritized capacity)
New axis (not abstraction): prioritize WM replay by model error so capacity targets
persistently-hard dynamics regions. NECESSARY precondition (offline, scripts/opt3_recurring_error.py):
converged-WM error must be RECURRING (train-region difficulty predicts held-out error) AND
STRUCTURED BEYOND displacement (else = lever-11/D1 redux). Train/test split by episode.
GO precondition: rho_recurring ≥ 0.2 AND rho_beyond_disp ≥ 0.10 (CI>0) on Pick & Ori.
NO-GO: error not recurring, or fully explained by displacement → no targetable headroom.
(Precondition only — return conversion still requires GPU training.)

## 2026-06-15 — Option-3 precondition VERDICT: INCONCLUSIVE (underpowered, fold into GPU campaign)
opt3_recurring_error.json: seed-unstable on Pick (0.10/-0.32/0.77) & Ori (0.52/-0.44/-0.59);
shuffle control not null on Ori (~0.41-0.55); only 6 held-out episodes → huge CIs. Pre-reg GO
fails. NOT a clean NO-GO — too few episodes. PLAN: re-test recurrence at proper power from the
D2 training-suite rollouts + add error-prioritized-replay as a single-variable arm in the campaign.

## 2026-06-15 — D2 suite LAUNCHED (box g4070 = 2x RTX 4070Ti, US, inst 41094828)
After a long env fight: mujoco_playground's modern warp backend is UNUSABLE (released warp-lang
lacks warp._src.jax_experimental.ffi.GraphMode that mujoco-mjx 3.8/3.9 needs). Solution: pin
mujoco==3.8.0/mujoco-mjx==3.8.0 + force MJPG_IMPL=jax (pure-JAX MJX) + JAX persistent compile cache
(seed-major order warms cache → ~12 unique compiles shared by 60 runs). Steady-state ~650 steps/s
(initial XLA compile ~350s/config is the only slow part; old Xeon E5-2673v4 is compile-bound).
Suite: vanilla vs jumpy(k8,plan,nmacro3) TD-MPC2 × {Pick,Ori,Cab,Cheetah,Hopper,CartpoleSwingupSparse}
× 5 seeds × 500k steps = 60 runs, 4 concurrent (MEMFRAC0.45, 2/GPU). ETA ~4-5h. Orchestrator
scripts/launch_d2.py (PID on box), HF ckpt backup scripts/hf_backup.py watch streaming to
Dannibal/tdmpc-glass-milestones. Harvest: scripts/aggregate_d2.py → mechcheck/d2_suite_results.json.

## 2026-06-15 — D2 jumpy-arm fix + relaunch
First launch: jumpy arm crashed (act_b[:,:kk].reshape — buffer seq_len=mppi_horizon+1=4 < k=8).
Fix: jumpy arm runs with --mppi_horizon 16 (anchor convention MPPI_H>=2k → seq_len 17 >= 8).
Smoke confirmed: jumpy trains clean, mechanism visible (jumpy_err<iter1_err ratio~0.78 "WIN").
Jumpy ~6x slower than vanilla (sps ~110 vs ~650; horizon-16 macro-MPPI). GPU util ~17% (host-bound)
→ relaunched at 6 concurrent (PER_GPU=3, MEMFRAC=0.30). 60 runs, ETA ~8-11h. Both arms healthy.

## 2026-06-16 — D2 partial harvest (15/60 done, 0 failed): PROMISING, clean task-type pattern
Jumpy reproduces the contact-manipulation win at fresh seeds: PandaPickCube +37% (CI[515,932] sep),
PandaPickCubeOrientation +215% final / +22% peak (CI-sep; final inflated by vanilla late-instability).
Jumpy HURTS locomotion: CheetahRun -27%, HopperHop s0 van 227 vs jum 0 (breaks high-DoF, matches iter-33).
Jumpy WINS sparse: CartpoleSwingupSparse s0 van 0 vs jum 151. PandaOpenCabinet +87% (CI crosses 0, s1 incomplete).
=> sign varies interpretably by task type (contact/sparse win, locomotion lose) → real signal for the
"predict when jumpy helps" diagnostic. 1-2 seeds only; more GPUs to firm CIs. Evidence: d2_suite_results.json.

## 2026-06-16 — D2 (25 runs): CLEAN task-type split, CI-separated across the board
Pick +48% [658,1143], Ori +169% [1423,2181], Cab +119% [709,1399] → ALL manipulation JUM_WIN CI-sep.
Cheetah -68% [-518,-68], Hopper -100% [-227,-144] → BOTH locomotion VAN_WIN CI-sep (jumpy hurts/breaks).
CartpoleSwingupSparse: van 0 -> jum 76 (sparse jumpy-win, CI[0,151] borderline). n=2-3 each.
=> jumpy/temporal-abstraction helps contact-manipulation + sparse, hurts locomotion — sign cleanly
predictable by task type. Mechanism (jumpy_err<iter1_err) holds on ALL tasks incl. where return loses.
4090 ~2.0x 4070Ti measured. 0 failures. Strong basis for the "when to go jumpy" D2 claim.

## 2026-06-16 — D2 CORRECTION (peak vs final): "manipulation wins" largely a final-metric artifact
Checking peak vs final per-seed (discipline!): vanilla manipulation policies reach high PEAK then
COLLAPSE at final (Ori van peak ~2300-2600 -> final 627-2186; Cab peak up to 3129 -> final 313-1942).
Jumpy is steadier at final. So the big FINAL "wins" (Ori +854, Cab +1053) are vanilla late-collapse,
NOT jumpy capability:
  Pick:  Δpeak +845 [536,1196] AND Δfinal +877 [579,1172]  -> GENUINE jumpy win (both metrics, all seeds)
  Ori:   Δpeak +33 [-846,719] (null) ; Δfinal +854 (vanilla-collapse artifact, jum s4 also collapsed)
  Cab:   Δpeak -377 [-885,128] (jumpy WORSE on peak) ; Δfinal +1053 (vanilla-collapse artifact)
  Cheetah Δpeak -85, Δfinal -109 ; Hopper -151/-151  -> jumpy HURTS locomotion (real, both metrics)
  Cartpole null/borderline both metrics.
HONEST D2: jumpy = genuine capability win on Pick only; late-training STABILITY benefit on Ori/Cab
(different mechanism, peak~wash); reliably HURTS locomotion. Must report BOTH metrics. Supersedes the
earlier "manipulation 3/3 CI-sep wins" (that was final-only). n=4-5.

## 2026-06-16 — D2 FINAL VERDICT (n=5, fresh reproduction, peak+final, paired bootstrap CI)
Jumpy(k8,plan) vs vanilla TD-MPC2, 6 tasks x 5 seeds x 500k (Pick n=4: 1 vanilla seed diverged to NaN).
GENUINE capability win (CI-sep on BOTH peak AND final):
  PandaPickCube            Δpeak +1017 [817,1238]  Δfinal +1114 [829,1369]   (+60%)
  PandaPickCubeOrientation Δpeak  +697 [573,794]    Δfinal +1625 [1276,1956]  (+161%)
STABILITY-ONLY (final win = vanilla late-collapse; peak no gain):
  PandaOpenCabinet         Δpeak  -372 [-729,1] null Δfinal +929 [542,1345] JUM
NEUTRAL / HARMFUL:
  CheetahRun  Δpeak -62 null, Δfinal -87 null   (jumpy ~neutral, slightly worse)
  HopperHop   Δpeak -121 [-182,-59] VAN, Δfinal -91 VAN  (jumpy HURTS, breaks high-DoF)
  CartpoleSwingupSparse  Δpeak +168 null, Δfinal +110 null  (borderline, high-var)
HEADLINE: jumpy/temporal-abstraction gives genuine capability gains on contact-manipulation (Pick,Ori),
training-stability on Cabinet, neutral-to-harmful on locomotion+sparse. "When to go jumpy" = contact
manipulation. Mechanism (jumpy_err<iter1_err) held on ALL tasks incl. where return doesn't improve →
better WM prediction != better control except on contact manip. Evidence: mechcheck/d2_suite_results.json
(HF evidence/d2_suite_FINAL_n5.json). g4070 stopped; g3 running option-3 dumps.

## 2026-06-16 — H4 ablation (jumpy-loss-only, no planning) n=3: jumpy is a PURE PLANNING intervention
vanilla vs jumpy-full(plan) vs jumpy-loss-only(jumpy_k=8, NO jumpy_plan; harvest mppi eval):
  Pick   van 2283/1854 | jum-full 3282/2979 | jum-loss-only 2301/1979 (~= vanilla)
  Hopper van 122/91    | jum-full 1/0(break)| jum-loss-only 125/91   (~= vanilla)
  Cheetah van 540/474  | jum-full 477/387   | jum-loss-only 438/348  (~= vanilla, slightly <)
=> jumpy's GAIN (Pick) and HARM (Hopper) BOTH come from macro-MPPI PLANNING; the jumpy loss/representation
alone is NEUTRAL. Refutes "decouple value-horizon from planning-horizon" (no free representational lever).
Constructive: the adaptive method must be a TASK-ADAPTIVE PLANNER (macro-plan on contact, vanilla on
locomotion), not a new loss. n=3 directional (effects large + consistent).

## 2026-06-16 — ARCHITECTURE VERDICT (clean n=5): resmlp & attn do NOT beat TD-MPC2 (iter-27 was a mirage)
vanilla+{resmlp,attn} vs vanilla+mlp (TD-MPC2), n=5, 6 tasks, peak+final paired bootstrap:
  resmlp: WORSE Pick(fin -305), Ori(pk -620), Cheetah(pk -305/fin -378); null Hopper/Cartpole;
          ONLY "win" Cab-final +1370 = vanilla-late-collapse stability artifact (Cab-peak null).
  attn:   mostly null, slightly worse Pick/Ori; one small genuine win Hopper (pk +72[6,141], fin +102[24,176]).
=> The iter-27 "resmlp +40/26% beats TD-MPC2" did NOT replicate at n=5 — small-n (2-4) mirage. A fancier
backbone does NOT beat TD-MPC2's MLP. Architecture line CLOSED (MLP dynamics already sufficient; redundancy
theme holds). NOTE: resmlp sweep on g3 still completing (Pick n=4); direction is clear and stable.
Supersedes the earlier "resmlp beat TD-MPC2" statement (that was based on the unreliable iter-27 numbers).

## 2026-06-17 — Motion-phase abstraction: COMPREHENSIVELY CLOSED (two new nulls)
(1) kNN-graph SE clustering: η²(err|kNN-SE comm)=0.439 == η²(err|displacement-bins)=0.439; plain KMeans
    geometry (0.515) BEATS it → graph/SE adds nothing over geometry, and nothing beyond motion magnitude
    (= D1/community-detection null). The graph construction does NOT rescue SE-clustering.
(2) Phase-balanced-replay mechanism-check (NEW "right way" — data, not model structure): does visited
    phase-entropy DROP as vanilla collapses? REFUTED — in 3/4 runs (Cab/Ori, peak→final collapse to
    -80%/-59%) phase-entropy INCREASED (1.63→1.84, 1.22→1.72, 1.18→1.36); corr(ret,ent) no consistent
    sign. The degraded policy FLAILS (broader phase visitation), it does not narrow → collapse is
    value-overestimation, not phase-forgetting. Phase-balanced replay won't help.
VERDICT: motion-phase abstraction tested across subgoals/jump-boundaries/switching-dyn/clustering/kNN-SE/
phase-replay — ALL null. Real phenomenon (gait/contact phases) but no control lever in any framing
(redundant with value-sufficient latent for model-structure uses; unrelated to the failure mode for data uses).
Bonus micro-finding: late-collapse = policy degradation with INCREASING state-visitation entropy.

## 2026-06-17 — Collapse diagnosis + fix probe (SCALE_MAX). PRE-REGISTRATION
Diagnosis (from existing collapse-run logs): value RunningScale PINNED at cap 4.00 the whole run
(IQR p95-p5 > 4 throughout); returns OSCILLATE wildly (not monotonic collapse); pi-eval more stable
than the mppi-eval D2 harvested (so part of the "59-80% collapse" was mppi-eval variance + final-of-noisy).
NOTE: critic already has LayerNorm (NormMLP) — the textbook fix is a no-op here.
HYPOTHESIS: saturated scale cap → advantages over-normalized (÷4) → policy gradients too large → late
oscillation. FIX PROBE: raise SCALE_MAX 4→16 so the scale tracks the true IQR.
PRE-REG GATE: SCALE_MAX=16 reduces peak-final gap / raises final on Cab+Ori (paired-bootstrap CI vs
D2 vanilla SCALE_MAX=4) AND does not hurt CheetahRun (locomotion control). n=3.

## 2026-06-17 — Collapse-fix SCALE_MAX=16: PARTIAL GO (mechanism confirmed, 1 CI-win)
Raising the RunningScale cap 4->16 (scale was pinned at 4, true IQR>=16). n=3 vs vanilla(=4):
  Cabinet:     Δfinal +640 [254,1034] WIN; peak-final gap 2159->1452 (stabilized)
  Orientation: Δfinal -51 null; gap 1367->779 (stabilized) BUT Δpeak -644 (16 over-normalizes -> lower peak)
  Cheetah:     Δfinal/Δpeak null (no locomotion harm)
=> Diagnosis CONFIRMED: under-normalized advantages (cap 4 << IQR) cause late oscillation; raising the cap
shrinks the peak-final gap on both manip tasks. Converts to a CI win on Cab; Ori needs a less-aggressive cap.
NEXT: sweep SCALE_MAX in {8, uncapped} to find stabilization-without-peak-loss. First non-abstraction
positive signal toward beating TD-MPC2.

## 2026-06-17 — Collapse-fix cap sweep: optimum is TASK-DEPENDENT → test uncapped (adaptive)
SCALE_MAX sweep (mppi-final, n=3 vs vanilla=4):
  Cabinet:     cap8 -263 null (insufficient); cap16 +640 WIN  -> Cab IQR large, needs >=16
  Orientation: cap8 +386 [-1,800] borderline; cap16 -51 null (over-normalizes peak) -> Ori prefers ~8
  Cheetah:     null both (no harm)
=> No fixed cap wins both; value-IQR varies by task. Principled fix = REMOVE the cap (scale tracks true
IQR per task, self-normalizing). Mechanism (under-normalized advantages -> late instability) confirmed;
the CI win on Cabinet (cap16 +640) is real but the constant-cap is task-dependent. Testing SCALE_MAX=uncapped.

## 2026-06-17 — Farebrother 2026 comparison (CORRECTION) + jumpy regime mechanism
Verified the paper: Farebrother et al. 2026 "Compositional Planning with Jumpy World Models" — jumpy models
of PRE-TRAINED POLICY occupancies + macro-planning over SEQUENCES OF POLICIES/OPTIONS (compositional, off-policy),
+200% on long-horizon manip/navigation. OUR jumpy = k-step LATENT dynamics from PRIMITIVE actions + macro-MPPI
over k-step primitive chunks (online TD-MPC2, no policy library). => DIFFERENT methods (both have macro-planning
but compose different units). Earlier "jumpy = Farebrother prior art" was imprecise; ours is the generic
k-step-latent variant, distinct from Farebrother.
Mechanism of OUR regime map (NOT model accuracy — jumpy_err<iter1_err uniformly, incl Hopper):
governed by (task credit-horizon) x (dynamics forgiveness). Win = long-horizon + forgiving (manipulation).
Harm = unstable fall-prone reactive-control (Hopper: long macro-plan brittle, can't react, falls -> 0).
Null = dense-reward stable locomotion (Cheetah) / exploration-limited (Cartpole). The locomotion-HARM regime
is NEW vs Farebrother (they don't test fast unstable locomotion). Enables a task-adaptive macro-planner.

## 2026-06-17 — Supervisor feedback + deep-research positioning (logged)
Supervisor: method trivial (tdmpc2+jumpy existing). Deep research confirms several threads are pre-empted:
- TD-M(PC)2 (2502.03550) owns the late-collapse fix (value overestimation from policy mismatch -> policy
  constraint). Our scale-cap is a weaker variant. DO NOT claim novel.
- Adaptive temporal resolution crowded: ICLR'24 Adaptive Temporal Abstractions from Discrete Latent; Adaptive
  Skip Intervals (2018); RLC'25 Predictable Skills. "When does TA help" partly answered (2406.00483 + known
  compounding-error/short-horizon-for-locomotion).
- Farebrother 2026 = pre-trained-policy occupancies + compositional policy planning, off-policy zero-shot,
  AntMaze; no public code; DIFFERENT setting.
Technical fact: k = TRAINING param (jumpy head input k*adim -> retrain to change); n_macro = PLANNING param
(eval-time, NO retrain). So "adaptive=finetune k" is expensive+trivial. Non-trivial = horizon-conditioned
multi-k model (free-at-deploy) + selector, must beat ICLR'24/Adaptive-Skip + be grounded in the criterion.
Diagnoses: Cabinet learns-then-collapses (Δpeak~0; candidate for skill abstraction); Hopper jumpy=wrong tool;
sparse edge borderline (testing AcrobotSparse/CartpoleBalanceSparse now).
SURVIVING THESIS: horizon-dependent redundancy — abstraction redundant short-horizon, necessary long-horizon;
build free-at-deploy adaptive planner switching on the criterion. Running: cleaneval (g3), sparse-gen (g3090).

## 2026-06-17 — Gemini DR saved + plan locked + clean-eval flips prong-3
Gemini deep-research saved (docs/research/deep_research_gemini_2026-06-17.md, HF evidence/). Corrections:
Farebrother=GHM on TD-Flow/OGBench; RunningScale-saturation likely UNPUBLISHED (TD-M(PC)2 owns policy-mismatch
only -> dual failure). CLEAN EVAL (EVAL_NEPS=10) FLIPS the scale-fix: Cabinet van final 1625 > cap16 1160 >
uncap 1103 -> scale-fix is NULL/negative on clean measurement; the +640 was n=3 MPPI-eval noise (user's
skepticism vindicated). PLAN: prong-1 value-sufficiency-sieve = LEAD (instrumenting linear value-decode R²
over training/tasks via eval_pi value_r2 logging); prong-2 adaptive deferred (crowded); prong-3 scale-fix
reported as negative. Running: cleaneval (g3), sparse-gen (g3090), value_r2 smoke.

## 2026-06-17 — Value-sufficiency sieve: CONFOUNDED metric; self-correction
Full sieve R²(linear z -> MC return-to-go) over training: Pick 0.82->0.53, Ori 0.93->0.53, Cab 0.89->0.92,
Cheetah 0.89->0.14, Hopper 0.18, Cartpole 0.997. CORRECTION: this metric decodes the NOISY single-rollout
return-to-go -> dominated by RETURN VARIANCE, not latent sufficiency (locomotion high-variance => low R²
regardless; cartpole-balance near-deterministic => 0.997). => the "declining R²" and the earlier "#3
value-sufficiency-collapses-with-instability" are return-variance ARTIFACTS. RETRACT #3. The ORIGINAL
criterion evidence stands and is NOT circular: R²=0.9994 was linear decode of the value-NET V(z) (value is
near-linear in the latent — a real geometric property). Prong-1 criterion intact; the MC-return sieve is
dropped as confounded. Net: collapse-fix NULL (n=4 clean), sparse non-generalizing (Acrobot both ~6), jumpy
pre-empted -> no method-novelty win; the honest deliverable is the redundancy criterion + null campaign
(understanding/negative paper, TMLR/RLC).

## 2026-06-24 (cont.) — Ceiling PROVEN physical + generalization staged
- **Part 30 upgraded & live (200):** the 0.83 PandaPickCube ceiling is PROVEN physical, not learnable.
  box_target metric verified from pick.py:198 = 1-tanh(5*(0.9pos+0.1rot)), target upright → rot_err 0.40
  caps box_target≤0.80 even at perfect position; need rot≤0.147 (~8°). Upright-grasp IK test
  (b3060 hl_tail/upright_ik_test.json): 99.88% of hard far-tail (n=801, reach~0.865) INFEASIBLE
  (only 0.12% admit in-limit joints holding cube within 8° upright); reach-gated 17.5%@0.80-0.82 → ~0%
  beyond 0.84m. Part 28: 100% position-reachable → position-reachable ≠ upright-graspable. Escapes:
  side-grasp+reorient (untested) or change metric. Rollout rot_err corroboration (smoke n=16) running.
- **Generalization staged (user's "try other Panda tasks"):** 2nd task = PandaPickCubeOrientation
  (sample_orientation=True, one-line flip, reuses controller+residual; stresses the orientation finding).
  Honest premise correction recorded: campaign "tie ~0.79" = α=1.0 residual PEAK (final collapses, fragile
  optimum); baselines PPO = 0.66@32M (the 0.81/0.83 is high-budget 100M+ PPO) → generalization run is
  budget-matched (~35M both arms). Staged in generalization_PandaPickCubeOrientation/ (ppo s1, residual s1/s2).
- **Reprioritized fleet:** InFOM queue-refill cron PAUSED (#PAUSED-FOR-ORIENT) on b3060 so freed GPUs go to
  the orientation run first; 4 running InFOM jobs untouched (still producing repro numbers). Heartbeat
  re-enables InFOM cron once orientation's 3 jobs start.

## 2026-06-24 (cont.2) — Disk incident fixed + Part 31 (generalization = LOSS)
- **Disk:** b3060b filled by whyhopper --save_full_state (3.96G latest_full.pkl). Deleted, killed job, installed
  disk_guard cron (*/5, always rm *full*.pkl, prune <5G). Orientation jobs had crashed → relaunched clean.
  Rule: NEVER --save_full_state. /tmp on b3060b is mahjong — untouched.
- **Part 31 (live): abstraction does NOT generalize.** On PandaPickCubeOrientation (random target yaw):
  PPO peak 0.805/0.836 vs abstraction residual 0.320/0.344 — a ~0.5 LOSS. Cause: fixed top-down/fixed-yaw
  grasp is the wrong prior for random yaw; residual can't fix a wrong primitive (reached~1.0, box_target~0.83,
  misses orient bar). Nuance: residual leads early (prior=head-start) but caps ceiling. Reusability (#39):
  abstraction reusable only within its prior's regime. Ties orientation thread (Part 30 kinematic tail vs
  Part 31 wrong-prior whole-task). Open follow-up: orientation-AWARE controller (feed target yaw).
- **Fleet:** b3060b 4 GPUs → whyhopper TD-MPC2 (PandaPickCube + HopperHop, 2 seeds, NO full_state) for #40
  k-step leg. b3060 InFOM cron RE-ENABLED (orientation done on b3060b). ssh3 still down.

## 2026-06-24 (cont.3) — InFOM cube reproduction lands; Part 31 live (200)
- **InFOM reproduction (cube-single-play-task1), original mission #34:** 3 seeds — s0 best 0.96 (latest 0.92),
  s2 best 0.94, s1 best 0.16 (unstable outlier seed). 2/3 seeds cleanly reproduce ~0.95, matching/exceeding
  the 0.86 reference; flag the 0.16 seed honestly (training instability). Antmaze cells still pre-eval; 6 queue
  cells remain. Defer the InFOM repro NOTE until antmaze lands (need the cross-task table).
- whyhopper TD-MPC2 (b3060b, 4 jobs Panda+Hopper 2 seeds) still training; k-step model-error leg next.
- Disk: b3060b 9.4G free (whyhopper ckpts), disk_guard OK; b3060 29G. All 8 GPUs busy.
