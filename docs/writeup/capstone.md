# Does Abstraction Help a Strong Latent World Model? A Fair-Protocol Campaign on TD-MPC2
## Eight mirages, one real (but not novel) win, and a methodology for marginal-effect claims in deep RL

*Capstone draft v0.1 — 2026-06-09. Consolidates the iter-14→iter-22 campaign. All numbers verified
from run CSVs (mirror); final tables to be regenerated at submission. Compute: ~10–12 vast.ai GPUs
(A4000/2080Ti/TitanV), mujoco_playground (MJX). **Status: DRAFT — not for submission without user
sign-off.** The abstraction-null core is written up in full in `draft.md` (the "Six Mirages" paper);
this capstone is the wider campaign frame that adds the jumpy result, the peak-vs-final methodology,
and the rho/RND asides.*

---

## 0. One-paragraph summary

We asked whether adding *abstraction* — over state, reward, time, or skills — improves TD-MPC2, a
strong latent world model, under a strict fair protocol (single-variable changes, compute-matched,
pre-registered gates, peak-AND-final reporting with bootstrap CIs). Across eight distinct abstraction
mechanisms the honest answer is **no, with one exception**. Geometric/behavioral latent clustering,
bisimulation auxiliaries, Laplacian/eigenpurpose exploration, community-detection skills, a
consistency-horizon (rho) schedule, and several apparent wins that dissolved under n-scaling all
returned to null — "eight mirages," each separately publishable at some interim snapshot. The lone
durable positive is a **jumpy (k-step) world model**, which on manipulation (PandaPickCube) beats
vanilla TD-MPC2 on BOTH peak (+966, 95% CI [714, 1248]) and final return (+1266, [877, 1642]),
CI-separated, n=5 vs n=7. We are explicit that **the jumpy model is not our invention** — it is
published prior art (Farebrother et al., 2026) — so this is a fair-protocol *reproduction-and-
evaluation* win, not an architectural innovation. The campaign's genuinely novel abstraction bets
were among the nulls.

---

## 1. The program and its discipline

TD-MPC2 [Hansen et al., 2024] reaches strong continuous-control performance with a representation
trained only by reward, value, and latent self-consistency losses. Theory suggests this objective may
already be a *sufficient self-predictive abstraction* [Ni et al., 2024], so any added abstraction must
clear a high bar. Marginal-effect claims in deep RL are also notoriously fragile to small samples and
to reporting choices [Agarwal et al., 2021; Henderson et al., 2018]. We therefore fixed three rules
for the entire campaign:

1. **Fair protocol** — the only change between arms is the abstraction term; all hyperparameters,
   network sizes, planner budgets, env steps and eval schedules identical. No restarts, no
   population-based training, no per-task tuning. (This rule itself overturned the project's original
   motivation: the iter-11/12 "basin entries" on HopperHop that started it occurred only under
   procedure interventions, never under the clean protocol — see `draft.md` §3.)
2. **Pre-registered gates** — sample-size bars and decision thresholds fixed before data; falsifications
   sized to the minimum informative experiment; *mechanism-checked before fanout*.
3. **Honest aggregation** — per-task-normalized IQM with stratified-bootstrap 95% CIs; and (added
   mid-campaign, see §4) BOTH peak/best-checkpoint AND final/last-2 reported for every arm.

## 2. The eight mirages (abstraction does not help)

| # | Mechanism | Iter | Result | Why it failed / dissolved |
|---|---|---|---|---|
| 1 | Geometric prototype clustering (structural entropy) | 14 | null (IQM 0.748 vs 0.738, overlap) | redundant with SimNorm's soft-categorical latent |
| 2 | Behavioral (reward-grounded) clustering | 14 | null at n=34 (0.767 vs 0.738, CI overlap) | gain crossed CI separation **3×**, wandered both sides of 0 |
| 3 | Bisimulation auxiliary (BS-MPC-style) | 14 | **hurts** (0.549) | brittle; untuned coef collapses training |
| 4 | Distractor robustness (64-dim OU) | 14 | falsified (1.23× < 1.5×) | both encoders crushed identically |
| 5 | Sparse-task rescue via behavioral grouping | 14 | null (0/3 vs 1/3) | bimodality is *exploration*, not geometry |
| 6 | "Floor effect" / zero-weak-seed tail | 14 | inverted | behavioral arm produced the study's single worst seed (0.127) |
| 7 | Laplacian/eigenpurpose exploration (DCEO-style) | 21 | null vs RND | abstraction-flavored novelty ≤ generic RND everywhere |
| 8 | Community-detection skills (Louvain on latent graph) | 19 | null | communities = motion phases, not reachable subgoals |

Mirages 1–6 are documented in full in `draft.md`. Two further dissolved *apparent* wins are
methodologically important and covered in §4: a thrice-CI-separated behavioral-IQM gain, and a
jumpy-on-CartpoleSparse "growing lead" that oscillated 76>45 → 309>199 → 203<294 and reversed —
caught only by reading at ≥400k rather than 250k.

**Takeaway.** This is a strong, multi-mechanism empirical confirmation that explicit abstraction does
not improve TD-MPC2 across final return, sample efficiency, distractor robustness, sparse exploration,
or seed reliability — consistent with the sufficient-abstraction theory.

## 3. The one real win: a jumpy world model on manipulation

**Mechanism.** A k-step (jumpy) latent head d_k(z_t, a_{t:t+k}) → z_{t+k} predicts k steps ahead in a
single call, plus a horizon-consistency regularizer ‖d(z,a,2k) − d(d(z,a,k),a,k)‖² and a macro-reward
head. A macro-MPPI plans n_macro jumpy steps (effective horizon k·n_macro) with few model applies —
long effective horizon WITHOUT the compounding 1-step-model error that sinks naive deep planning
(vanilla H9 collapses on Panda).

**Mechanism confirmed before fanout.** The k-step head is measurably more accurate than iterating the
1-step model k times, and the advantage *grows* with k: ratio jumpy_err/iter1_err = 0.991 (k=2), 0.910
(k=3), 0.821 (k=8). The architectural premise holds empirically.

**Result (PandaPickCube, fair protocol, mature ≥400k, paired jumpy-vs-mppi, 20k-resample bootstrap):**

| metric | jumpy (n=5 mature) | vanilla-H3 (n=7) | diff | 95% CI | verdict |
|---|---|---|---|---|---|
| **peak** (best ckpt) | 3183 | 2217 | **+966 (+44%)** | [714, 1248] | **SEPARATED** |
| **final** (last-2) | 2708 | 1442 | **+1266 (+88%)** | [877, 1642] | **SEPARATED** |

Both metrics separate. The peak gap is the clean "jumpy plans better" claim; the larger final gap is a
**stability** finding — vanilla TD-MPC2 itself collapses peak→final on Panda (−35%, 2217→1442), and the
jumpy model resists that late collapse. The result is 5/5-consistent across seeds (no Cartpole-style
oscillation) and held across snapshots (s1: 2597→2808→2846), distinguishing it from a snapshot mirage.

**The honest caveat (load-bearing).** The jumpy world model + cross-timescale consistency is **published
prior art**: Farebrother, Pirotta, Tirinzoni, Bellemare, Lazaric, Touati, *Compositional Planning with
Jumpy World Models*, arXiv:2602.19634 (2026) — verified. Their setting differs (TD-Flow occupancy models
composing pre-trained policies zero-shot, vs our online TD-MPC2 macro-MPPI), but the *concept* is not
ours. So §3 is a fair-protocol reproduction-and-evaluation result — "a known temporal-abstraction method,
evaluated honestly, beats vanilla TD-MPC2 on manipulation on both peak and final" — not the architectural
innovation the program set out to find. That remains open (§6).

## 4. Methodology: peak vs final, and the anatomy of dissolving effects

Within-run instability is real in deep RL (deadly triad, plasticity loss/primacy bias [Nikishin 2022,
Lyle 2023], policy churn, Q-divergence). It creates two failure modes for naive reporting:

- **Cross-seed small-sample mirages** (peak-insensitive): an effect that exists at n≤9 and dissolves as
  n grows. The behavioral-Glass IQM crossed "significant win" and "confirmed null" readings six times
  (0.818→0.736→0.829→…→0.767) and wandered to both sides of baseline even past n=30 (`draft.md` §5).
  These nulls stand under *any* reporting metric.
- **Within-run collapse mirages** (peak-sensitive): an effect visible only at one checkpoint. The
  vanilla-Panda "−45% collapse" inflated an apparent jumpy "+104% on final"; the fair best-checkpoint
  comparison shrank it (peak +37%). The jumpy-Cartpole "growing lead" reversed entirely by 450k.

**Field standard, applied.** rliable [Agarwal et al., 2021] prescribes fixed-budget, many-seed, IQM with
stratified-bootstrap CIs — *not* peak-picking [Henderson et al., 2018 flags peak bias]. Best-checkpoint
reporting is valid for *deployment* claims iff applied identically to all arms and disclosed. We therefore
adopted **report-both**: peak AND final for every arm, with the gate requiring CI separation. Under this
discipline the jumpy win survives on both metrics (§3) while the abstraction effects do not survive on
either.

## 5. Asides (negative/partial, recorded for completeness)

- **rho (consistency-horizon decay), iter-20.** Raising rho 0.5→0.9 cures the deep-planning collapse on
  Panda (H9: 419→1775 mean, n=3, > H3 ~1490) but *suppresses* sparse exploration (CartpoleSparse H9:
  [0,0,642], 2/3 killed). A **task-dependent tuning lever, not architecture** — a trade, not a free win.
- **RND vs Laplacian exploration, iter-21.** Generic novelty (RND) rescues some sparse TD-MPC2 (Cartpole
  max 561 vs vanilla 0), but the *abstraction-flavored* Laplacian/eigenpurpose bonus adds nothing over
  RND anywhere (gate G2 fails). Exploration helps; the abstraction flavor does not.

## 6. What's next (genuine-novelty attempt, iter-23)

Treating the validated jumpy model as a *substrate*, we are scoping a mechanism that beats the jumpy
baseline itself, single-variable, on high-DoF tasks (Dog/Humanoid — where a deep-research survey finds
the only demonstrated TD-MPC2-beating headroom lives). The current lead candidate is **a learned proposal
distribution that seeds macro-MPPI over the jumpy macro-manifold** (attacks MPPI's sample-inefficiency in
the k·a_dim macro space; no continuous-control precedent found), with a **value-equivalent macro head**
(abstraction preserves return, not state) as fallback. Pre-registration pending (see
`docs/iterations/iteration_23_ideation.md`, `docs/research/dr-iter23-agent-claude.md`).

## 7. Conclusion

Under a fair, pre-registered protocol with peak-and-final reporting, explicit abstraction does not
improve TD-MPC2 on the axes we measured — eight mechanisms, eight nulls — consistent with the
sufficient-abstraction theory. The one method that *does* beat vanilla on manipulation (a jumpy world
model, peak +37% / final +82%, CI-separated) is a known method we evaluated fairly, not an innovation.
The campaign's second contribution is methodological: an anatomy of how marginal-effect "wins" in deep
RL arise and dissolve, and a concrete report-both (peak+final, CI, fixed-budget, estimate-trajectory)
protocol that would have flagged every mirage in advance.

---

### References
TD-MPC2 2310.16828 · Ni et al. (self-predictive abstraction) 2401.08898 · rliable 2108.13264 ·
Henderson et al. (Deep RL That Matters) 1709.06560 · Nikishin et al. (primacy bias) 2022 · Lyle et al.
(plasticity) 2023 · DBC 2006.10742 · BS-MPC 2410.04553 · DC-MPC 2503.00653 · behavioral-metric study
2506.00563 · Compositional Jumpy World Models (prior art) 2602.19634 · TAP 2208.10291.

### Reproducibility
Code: helios-rl (iter-14 SHA 4d3b935 for the null core; iter-22 jumpy adds JumpyDynamics/JumpyReward +
make_jumpy_mppi_fn in `src/helios/algorithms/tdmpc2.py`, `--jumpy_k/--jumpy_plan/--jumpy_n_macro` flags).
Per-run CSVs: `exp/tdmpc_glass/remote_mirror/**/phasei22*`, `phasei18hs_H3*`, `phasei14v2*`. Analysis:
inline IQM + stratified bootstrap (20k resamples, seed 0). Iteration records: `docs/iterations/`.

---

## Addendum (2026-06-10): post-jumpy novel-lever attempts — all honest nulls
After the jumpy result we treated it as a substrate and chased a *genuinely novel* abstraction lever.
Two families, both killed cheaply by the mechanism-check-before-fanout discipline:
- **Adaptive jump-length (iter-23: SE-k structural-entropy k-selection, and F uncertainty-gated horizon).**
  SE pre-check passed (53% structural-entropy gap in the latent graph) but the mechanism-check failed: the
  jumpy model's k-step error is small and ~uniform (in-distribution AND under MPPI-perturbed actions,
  inflation 1.06x), so there is *nothing to adapt to* — which is exactly why fixed-k jumpy already works.
- **SE-driven exploration (iter-24: faithful VCSE + SI2E, plus the novel `wmsi2e` = SE over the
  world-model latent + critic value).** Pre-registered solve-rate gate on sparse DMC. NULL: no arm
  rescued CartpoleSparse/AcrobotSparse beyond vanilla, none beat RND, and the WM-latent novelty added
  nothing over a random encoder; at coef 1.0 the bonus mildly hurt. Third abstraction-flavored
  exploration null (after iter-19 community-skills, iter-21 Laplacian).
**Standing conclusion strengthens:** a strong self-predictive world model (TD-MPC2 + SimNorm) is a high
bar; explicit abstraction — over state, reward, time, skills, or exploration — is largely redundant with
what it already learns. The one durable positive remains the (known) jumpy model on manipulation. Live,
untested ideas that do NOT depend on these null mechanisms: a Hermite-spline action bottleneck and a
value-equivalent macro head (both action/value abstraction). See docs/iterations/RESEARCH_LEDGER.md.
