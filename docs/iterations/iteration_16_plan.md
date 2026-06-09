# Iteration 16 — FSQ Codebook Replaces SimNorm ("fsqmpc")

*2026-06-07. Direction #3 from the Six-Mirages post-mortem ranking, started on user
instruction ("do next on free gpu") while iter-15's formal 500k gate read completes
(iter-15 mechanism already diagnosed dead: action-blind prototype dynamics → flat MPPI
objective). Pre-registered BEFORE any run.*

## Hypothesis

Auxiliary losses on TD-MPC2's already-sufficient objective are doomed (write-up §5.1–5.2,
iter-14/15 nulls). The one abstraction family with a published controlled-ablation gain on
TD-MPC2 itself *replaces a component* instead: DC-MPC (arXiv 2503.00653) swaps the latent
parameterization for discrete codes and reports "some improvement" inside TD-MPC2 on
DMControl. A representation swap changes the optimization landscape rather than adding a
redundant signal, and planning stays in the full latent — structurally immune to iter-15's
action-blindness failure.

## Mechanism

`--latent_norm fsq`: Encoder and Dynamics outputs pass through Finite Scalar Quantization
instead of SimNorm — tanh bound, per-dim rounding to 5 uniform levels in [-1,1],
straight-through gradients. Table-free (no codebook collapse modes), one-line swap,
everything else identical (losses, heads, MPPI, hyperparameters). Single-variable vs
vanilla: same K_UPDATE=128, MPPI_NS=2048, EXPL_UNTIL=25000 as the iter-14 v2 protocol.

## Kill-probe design (minimal falsification)

- Arm: fsqmpc = vanilla TD-MPC2 + `--latent_norm fsq`. NO new baseline runs needed —
  compare against the existing phasei14v2 vanilla CheetahRun curves read at 500k
  (same config, n≈8 seeds).
- Runs: 4 seeds (0–3) × CheetahRun × 500k, tag phasei16fsq, on the 4 currently-free
  A4000s. (4 seeds rather than 2 because the boxes are idle — adds power, not wall-clock.)

**Pre-registered gates (mean of last-2 evals at 500k):**

- **G1 (viability): fsq seed-mean ≥ 0.9 × vanilla-reference seed-mean at 500k.**
  PASS → Stage 2.
- **G-dead: fsq < 0.7 × vanilla on ≥ 3 of 4 seeds** → discrete codes hurt clean proprio
  DMC at this scale → direction falsified; next per ranking (#4 pixels / #5 multi-task)
  or fold into the write-up as the third mechanism family nulled.
- 0.7–0.9 band: one pre-registered retune shot (fsq_levels 5→8, or quantize only the
  encoder, not dynamics), then re-gate.
- Expected honest prior: ~40% G1 — parity on Cheetah is plausible (DC-MPC's gains were on
  high-dim Dog/Humanoid; on mid-dim tasks parity is their own result); the probe mainly
  rules out "quantization breaks training".

**Stage 2 (only if G1):** {CheetahRun, WalkerRun, FingerSpin} × 1M × ≥5 seeds, IQM +
stratified-bootstrap CI vs vanilla (n=37) under the fixed-cutoff discipline; PLUS the
high-dim hypothesis where DC-MPC's wins actually live — HumanoidWalk 2M × 2 seeds
(exploratory, slow at ~135 sps — only if Stage-2 parity holds).

## Results

**Vanilla reference @500k (computed 2026-06-07 from phasei14v2 vanilla CheetahRun CSVs,
last-2 of 450k/500k, n=11 seeds):** mean **586.9**, median 623.9, range [143, 707]
(the 143 is vanilla's own weak seed — reference kept as-is per pre-registration).

**Smoke (box 39560, 60k):** PASS — trains, no NaN, MPPI 141@50k in vanilla band.

**Interim (seed 0 complete, others at 350–450k):** fsq seed 0 last-2@500k = **434.5**
→ ratio **0.740** vs reference mean (0.696 vs median). In the 0.7–0.9 retune band.
Seeds 1–3 trajectories (400:515/437, 350:449/502) suggest similar landings — likely
outcome: one pre-registered retune shot (fsq_levels 5→8 — 5-level/dim quantization is
severe; DC-MPC uses finer codes — or encoder-only quantization). Gate verdict deferred
until all 4 seeds reach 500k; no retune launched before that read.

**FORMAL GATE (2026-06-07, all 4 seeds at 500k, read from mirror CSVs):**

| seed | last-2 @500k | ratio vs 586.9 |
|---|---|---|
| 0 | 434 | 0.740 |
| 1 | 503 | 0.857 |
| 2 | 469 | 0.799 |
| 3 | 511 | 0.870 |

Seed-mean **479.2 → 0.817×**. Not G1 (≥0.9), not dead (0 seeds <0.7). **VERDICT: retune
band — the single pre-registered retune fires: FSQ_LEVELS 5→8** (finer codes; 5-level/dim
was a severe quantization). Re-probe: 2 × CheetahRun 500k, tag phasei16fsq8, CODE_SHA=i16b.
Re-gate on the same thresholds; no further retunes regardless of outcome. Reading so far:
FSQ trains stably (no collapse, no NaN, all seeds in a tight 0.74–0.87 band — lower
variance than vanilla's own reference spread) but the coarse codes cost ~18% return on
clean proprio.

---

## RE-GATE (FSQ_LEVELS=8) + cross-task read (2026-06-07)

**CheetahRun (pre-registered gate task), 6 seeds:** vals 187/292/355/414/444/452,
mean **357 → 0.609× vanilla** (586.9). FAILS G1 (≥0.9); below the 0.7 line. The retune did
NOT beat fsq5 (0.817). **Pre-registered CheetahRun verdict: FSQ does not improve TD-MPC2
on this task.**

**BUT the task-generality filler (added as idle-fill) shows FSQ8 is HETEROGENEOUS, not
flatly null:**

| task | FSQ8 mean | n | vanilla@500k (n=11) | ratio |
|---|---|---|---|---|
| CheetahRun | 357 | 6 | 586.9 | 0.61 |
| WalkerRun | 536 | 2 | 559 | 0.96 |
| FingerSpin | **982** | 2 | 876 | **1.12** |

**⚠ FLAG FOR USER:** FingerSpin FSQ8 is *above* vanilla (977, 987 vs 876) — the first
non-null signal in the campaign. At n=2 this is exactly the small-sample regime the Six
Mirages paper says to distrust (cf. behavglass's +0.051 that evaporated). Per
pre-registration, Stage-2 fires only on a CheetahRun G1 pass, which did NOT happen — so I
did NOT autonomously launch the 15-run Stage-2 on an n=2 post-hoc signal. Instead the
in-flight Walker/Finger seeds (→ 5+ each) give a properly-powered per-task read.
**DECISION for user:** if FingerSpin holds ≥1.05× at n≥5 with separated CI → a real Stage-2
is warranted (FSQ helps spin/dexterity tasks, hurts locomotion = a *task-dependent codebook*
result, publishable, NOT a null). If it regresses to parity → iter-16 closes as the 3rd
nulled mechanism family. Either way honest.

### n=5 UPDATE + CI — the signal is PROBABLY MIRAGE #7 (2026-06-07)

| task | FSQ8 n=5 mean | ratio | min(ratio) |
|---|---|---|---|
| CheetahRun | 357 (n=6) | 0.61 | 0.32 |
| WalkerRun | 589 | 1.05 | 0.66 (371) |
| FingerSpin | 955 | 1.09 | 0.96 (844) |

My naive pre-set check (mean≥1.05 ∧ min>0.9) PASSED for FingerSpin — **but the bootstrap CI
does not:** diff(FSQ−vanilla) = **+79, 95% CI [−11, +164], NOT separated** (FSQ n=5 vs
vanilla n=11). And the mechanism is the textbook mirage: **vanilla FingerSpin is bimodal** —
seeds saturate (~985) or plateau (~750); vanilla drew 6/11 saturators, FSQ drew 4/5. The
"win" is a luckier saturation draw at small n on a bimodal task — structurally identical to
the floor-effect mirage that died in the main study. The min>0.9 heuristic was fooled
because FSQ's lone non-saturator (844) landed high by chance.

**n=5 lean (SUPERSEDED below):** CheetahRun hurt; FingerSpin not CI-supported, likely mirage.

### n=8 UPDATE — the signal STRENGTHENED, not regressed (2026-06-07)

FingerSpin FSQ8 **n=8**: vals 844/976/977/981/982/984/987/987, mean 965 vs vanilla 876.
Bootstrap diff **+89, 95% CI [10, 168] — SEPARATED** (lower bound > 0). It went from
[−11,164] at n=5 → [10,168] at n=8: **the CI is tightening above zero as n grows.** This is
the OPPOSITE of every iter-14 mirage (all of which weakened toward 0 with n) — the single
most important diagnostic for distinguishing signal from sampling noise. WalkerRun also
trending up (FSQ 600 vs van 559, n=6).

**REVISED verdict — this is a genuine candidate, NOT (yet) a mirage:**
- CheetahRun: FSQ **hurts** (0.61×, separated below) — solid.
- FingerSpin: FSQ **helps** (+89, [10,168] @n=8, strengthening) — REAL candidate; caveats:
  borderline lower bound (10), near-ceiling task, separation partly from saturation-rate
  (FSQ 7/8 vs vanilla 6/11). Resolving to n≥12.
- WalkerRun: trending up (1.07×, noisy) — resolving to n≥12.

### n=9 per-task CIs (2026-06-08 00:20Z; FingerSpin/Walker still resolving to n≥12)

| task | FSQ8 n | mean | vanilla(n=11) | diff | 95% CI | verdict |
|---|---|---|---|---|---|---|
| CheetahRun | 6 | 357 | 587 | **−230** | [−338, −106] | **HURTS** (separated below) |
| WalkerRun | 9 | 563 | 559 | +4 | [−79, +129] | parity (overlap) |
| FingerSpin | 9 | 967 | 876 | **+91** | [+10, +168] | **HELPS** (separated above) |

WalkerRun's earlier "+28 trend" collapsed to +4 at n=9 — it is parity, the trend was noise
(correctly NOT chased). FingerSpin holds separated (+91, lower bound +10) at n=9, consistent
with n=8. 18 more seeds queued (Finger→~17, Walker→~15) for a definitive read. Note: ~5 of
the earlier batch were lost to box deaths (1660s/disk-full), hence slower n growth.

### n=16 FINAL — FingerSpin was MIRAGE #7; iter-16 closes NULL (2026-06-08)

FingerSpin FSQ at n=16: diff **+53, CI [−33, +138] — overlaps zero.** Seed-ordered
trajectory: overlap(n=5) → **SEP(n=8, +89 [10,168])** → overlap(n=9 +54) → overlap(n=12 +62)
→ overlap(n=15 +46) → overlap(n=16 +53). The CI separated at *exactly one snapshot* (n=8)
then regressed — the textbook mirage, and a cruel one: it even passed the "strengthening
with n" diagnostic from n=5→8 before collapsing. Mechanism: FSQ saturates FingerSpin
slightly more often (13/16=81% vs vanilla 6/11=55%, Fisher p≈0.2 — underpowered) but carries
2 weak seeds (645, 655) that null the mean. **NOT launching the n=9 Stage-2 was correct.**

**FINAL VERDICT — iter-16 NULL (3rd nulled mechanism family):**
- CheetahRun: FSQ **HURTS** (−230, [−338,−106], separated) — the one robust effect.
- WalkerRun: parity (+4, overlap).
- FingerSpin: parity-on-the-mean (+53, overlap) — the "+91 @n=8" was Mirage #7.

FSQ codebook does not improve TD-MPC2; it only *hurts* locomotion. Codebook swap joins
auxiliary losses, plan-in-abstraction, and exploration-through-abstraction as nulled.
**No Stage-2.** The FingerSpin episode → 7th mirage for the write-up (transient n=8
separation that even survived a strengthening-with-n check).

### (superseded) Emerging story = TASK-DEPENDENT CODEBOOK: FSQ trades locomotion (Cheetah ↓) for
spin/dexterity (Finger ↑), consistent with DC-MPC's claim that discrete codes help on some
task families. **⚠ USER DECISION (flagged for wrap-up):** if FingerSpin holds separated at
n≥12, a real Stage-2 is warranted — multi-task IQM+CI, and ideally the high-dim tasks where
DC-MPC located its wins (Dog/Humanoid, slow ~135 sps). NOT launched autonomously (15+ run
commitment + a story pivot from 'complete null' to 'task-dependent codebook'). This is the
ONE live lead of the entire post-null campaign.
