# Iteration 24 — SI2E-style SE-driven exploration (the SE lever's correct home)

*2026-06-09. SE-k/adaptive-k died (jumpy error uniform → nothing to gate). But pre-check 1 confirmed the
SE structure is REAL (53% gap). Its right job is COVERAGE/exploration, not jump-length. User pointed to
two reference impls: github.com/SuuTTT/learn-si2e (faithful + fast SI2E) and github.com/SuuTTT/selib
(fast Louvain SE). This iter ports the FAITHFUL formulation to TD-MPC2 on sparse tasks.*

## Goal / gate (pre-registered, unchanged)
SE-driven exploration must **BEAT RND** on sparse MJX tasks {CartpoleSwingupSparse, BallInCup,
AcrobotSwingupSparse} — the exact iter-21 G2 bar the *geometric* Laplacian FAILED. rliable IQM, 5 seeds,
peak+final, CI. WIN = SI2E > RND, ≥10% non-overlapping CI on ≥2/3 tasks. Arms: vanilla, RND (have it),
**VCSE** (value-conditioning only), **SI2E** (VCSE + cluster term). The VCSE arm replicates the refs'
headline ("value-conditioning is the dominant win, +55pp over plain SE") and isolates SI2E's added value.

## Faithful formulation (from learn-si2e; the v1 in code is NOT this — see below)
- **VCSE** (Value-Conditional State Entropy, the dominant win): intrinsic reward = particle entropy =
  `r0 = log( ||x_i − x_i^{kNN}|| )` in the JOINT space `x = [normalize(φ(s)); β·normalize(V(s))]`, i.e.
  the state feature φ(s) CONCATENATED with the critic value V(s). kNN over a recent buffer. Effect:
  rewards states far from same-VALUE neighbors → stops wasting bonus on dead-ends (refs H1: +55pp vs SE).
- **SI2E** = `r = H(V0) − H(V1)`: leaf VCSE (r0 above) MINUS a cluster-level VCSE (r1) computed on the
  community centroids of an encoding-tree partition. The cluster term = "group novelty" → cuts seed
  variance (refs: SI2E 100%±0 vs VCSE 97.8%±3.1 on DoorKey-8x8). r1 = VCSE(cluster_centroids).
- **Fast variant = kmeans clustering** (refs benchmark `fast-si2e-kmeans`; leiden/infomap also). Our
  kmeans path = exactly this. selib.optimal_2d (se_louvain) = the "proper-SE" variant (needs sklearn;
  skip on workers — kmeans is the validated fast variant).
- Feature φ(s): a fixed **random encoder** (A2C variant) — encoder-independent, no training. Replaces
  v1's Laplacian-trained embedding (simpler, matches the reference A2C random_encoder).
- **Value V(s)**: from our critic — V(s) ≈ two_hot_inv(min-Q(s, π(s))). MUST be threaded into the
  intrinsic branch (the one new wiring vs v1).

## Code status
- [x] Harness wiring VALIDATED by v1 smoke: `--intrinsic si2e` branch + periodic rebuild run clean on
  CartpoleSparse (50k, SPS ~200, loss finite, no NaN, ~12 rebuilds). The faithful version reuses this.
- [x] kmeans community machinery + 3-64 granularity guard (fixed the kNN over-fragmentation bug).
- [ ] **REPLACE v1 reward (count-coverage) with faithful VCSE leaf−cluster:** rewrite make_si2e to a
  random-encoder φ + buffer of (φ, V) + value-conditional kNN entropy r0 − cluster entropy r1. Add a
  `make_vcse` (r0 only) for the ablation arm. Thread V(s) (critic) into the si2e/vcse branch.
- [ ] Mechanism-check before fanout: on a sparse smoke, does VCSE/SI2E find reward FASTER than RND on
  ≥1 seed (sanity that the bonus shapes exploration), and is r1 (cluster term) non-degenerate?
- [ ] Then the 4-arm × 3-task × 5-seed gate (fan out across fleet; HOLD full campaign for user go).

## Honest prior
iter-21 Laplacian (geometric SE-exploration) LOST to RND. SI2E is value-conditional (richer) + the refs
show it beats VCSE/SE/baseline on MiniGrid + DMC — so a fair re-attempt with a real chance, but the bar
(beat RND on OUR sparse MJX tasks) is exactly where abstraction-flavored exploration failed before. Kept cautious.

## CAMPAIGN LAUNCHED (2026-06-09 ~20:40Z)
All 3 new arms SMOKE-VALIDATED on GPU (vcse/si2e/wmsi2e run clean past collection loop + kmeans rebuilds,
no NaN; fixed a KeyError:onorm — onorm only for rnd/laplacian). 75-run campaign live: {van,rnd,vcse,si2e,
wmsi2e} x {CartpoleSparse,BallInCup,AcrobotSparse} x 5 seeds. Fleet = 8 tdmpc boxes (ssh1_2080ti,
ssh1_a4000b, ssh4_a4000, ssh4_a4000b, ssh6_3060, ssh6_titanv, ssh8_a4000, ssh9_a4000), all saturated.
PROTECTED (user's forecasting, NOT tdmpc — do NOT clobber/re-add): ssh2_a4000(18950)=MultiRel,
ssh3b_a4000(17426)=Crossformer. 1660s disabled. Baselines (prio3) run first, new arms (prio5) after.
~9h for all 75 on 8 boxes -> by +8h expect van/rnd complete + partial vcse/si2e/wmsi2e -> first verdict.

### tick ~21:00Z: baseline only (campaign early, ~45min in)
van CartpoleSparse + BallInCup near-complete (n5 @500k); rnd/vcse/si2e/wmsi2e not yet started (priority
order). KEY: vanilla is NOT floored on these (best Cart 768, Ball 975) — bimodal/seed-dependent. So the
gate metric must be SOLVE-RATE across seeds (how many of 5 find reward), not just mean/best — the
intrinsic must rescue MORE seeds than vanilla/rnd. No comparison yet (only van has data). Health OK, 8 boxes busy.

### tick ~22:20Z: FULL vanilla baseline (n=5/task) — discriminating tasks identified
van solve-rate: Cart 1/5 (m195), Ball 4/5 (m754), Acro 0/5 (m21).
=> AcrobotSparse (0/5) = CLEANEST discriminator (vanilla fully fails); CartpoleSparse (1/5) second;
BallInCup (4/5) nearly saturated, least useful. The gate hinges on Acro+Cart: can rnd/vcse/si2e/wmsi2e
solve MORE seeds than van (0/5 Acro, 1/5 Cart)? New arms still immature (seed-0 just launched seed-major).
rnd Cart seed0 final=2 (no reward, like a typical van seed). No comparison yet. 8 boxes busy, 0 failures.

### tick ~23:10Z: new arms n=1 (uninformative yet). done: van15 rnd3 vcse2 si2e1, 0 failures.
Early (n=1, NOT a verdict): on discriminators all new-arm seed0 = 0 solve (Cart/Acro), == typical van seed.
rnd Ball 1/1(485), vcse Ball 0/1(361). Need n>=3/arm for solve-rate signal (~2-3 ticks out, seed-major).

### tick ~02:20Z: LEANING NULL (n=2-4 on discriminators). done: van15 rnd6 vcse5 si2e6 wmsi2e4, 0 fails.
SOLVE-RATE on discriminators (mature >=400k):
  van    Cart 1/5 (best768)  Acro 0/5 (best66)
  rnd    Cart 0/4 (best147)  Acro 0/3 (best0)
  vcse   Cart 0/2 (best1)    Acro 0/1
  si2e   Cart 0/2 (best2)    Acro 0/2
  wmsi2e Cart 0/2 (best0)    Acro 0/1
=> NO intrinsic (incl RND) rescues Cart/Acro; their BEST seeds are WORSE than vanilla's best (768 Cart).
Not just bad luck — the bonus isn't finding sparse reward, may mildly over-shape at coef=1.0. RND not
rescuing here either (unlike iter-21 max561 — seed variance or harder cfg). NOT final (need n=5), but
firmly null-leaning: SE-exploration (and value-conditioning/cluster/WM-latent) shows no rescue so far.
Fleet 10 boxes, ssh3b cleaned 89->69%, forecasting not resumed, ~16 pending (~1.5h to n=5).

## FINAL VERDICT (2026-06-10, n=3-4/arm, stable; last ~8 seeds pending but picture unambiguous): NULL
SOLVE-RATE @>=400k on discriminators (van baseline: Cart 1/5 best768, Acro 0/5 best66):
  rnd    Cart 0/4(best147)  Acro 0/3(best0)
  vcse   Cart 0/3(best1)    Acro 0/3(best22)
  si2e   Cart 0/3(best2)    Acro 0/3(best0)
  wmsi2e Cart 0/2(best0)    Acro 0/3(best6)
GATES ALL FAIL: (a) rescue vs vanilla = NO (all intrinsics 0/n on both); (b) vcse>rnd = NO; (c) si2e>=vcse
= NO; (d) NOVELTY wmsi2e>si2e = NO (both 0). Worse: intrinsics' BEST Cart seeds (<=147) are BELOW vanilla's
best (768) -> at coef=1.0 the bonus mildly HURTS (over-shaping), doesn't find sparse reward.
CAVEAT: even RND failed to rescue here (0/4 Cart; cf iter-21 RND max561) -> the coef=1.0 setting may be too
strong on these tasks; a coef sweep is the only thing that could revive the test, but with THREE prior
exploration nulls (iter-19 community-skills, iter-21 Laplacian, iter-24 SI2E/wmsi2e) the expected value is low.
=> SE-driven exploration (value-conditional VCSE, cluster-term SI2E, AND the novel world-model-latent
wmsi2e) does NOT rescue sparse TD-MPC2 beyond RND/vanilla. The WM-latent novelty adds nothing. Honest null.

### VERDICT CONFIRMED stable (n=4-6/arm): rnd Cart0/6 Acro0/4, vcse/si2e/wmsi2e Cart0/4 Acro0/3-4. NULL holds.
Re-queued dropped BallInCup (fleet-fill, proven, completes 5x3 dataset for writeup). No scientific change.
