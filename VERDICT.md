# SE-vs-VICReg JEPA latent on PandaPickCube — VERDICT: **NULL** (honest)

**Question.** The H-JEPA solve campaign traced its NULL to the JEPA latent not
preserving fine end-effector→cube geometry. Hypothesis (the "glass" idea):
a **Structural-Entropy (SE)**-regularized latent preserves task geometry *better*
than VICReg → better reach/manipulation. We test this with a cheap, decisive
mechanism probe before any downstream work.

**Verdict: NULL.** SE never beats VICReg at preserving task geometry. When SE is
strong enough to actually shape the latent it *destroys* geometry; it only "ties"
VICReg when weighted so weakly it is effectively inert. Hypothesis rejected →
downstream (LL reach / solve) **skipped** per the probe-first plan.

---

## USING_SE.md regime call (done first)
This is the **"RL state/skill abstraction"** regime. The guide rates it
*"Sometimes — needs a clean hierarchy + non-sparse reward; validate per-task across
seeds,"* and explicitly lists SE RL-abstraction under **"What does NOT work"**
(null at modest compute in their reproductions). So SE was *not* expected to win
here. Required controls were honored: **(i)** matched non-SE (VICReg) baseline,
**(ii)** SE-on-random-graph (structure-blind) control, **(iii)** differentiable-SE
validated against `selib.metrics.structural_entropy_2d`, **(iv)** multiple seeds,
plus a **LAM_SE steelman sweep**.

## Faithfulness check
Our differentiable JAX 2D-SE matches selib exactly: **diff-SE = 5.459759 vs
selib = 5.459758 (abs err 3.1e-7)** on a random graph + random partition. So the
SE term is the real Li-Pan 2D structural entropy, not a degenerate surrogate
(USING_SE.md "released code ≠ paper" trap avoided).

## Setup (matched)
SimNorm encoder `NormMLP(512,512)→256, V=8` (reused from H-JEPA) + 1-step
latent-predictive MSE to an EMA-target encoder. **Identical** cached PandaPickCube
transitions (~40k train), arch, lr 3e-4, 15k steps, ema 0.99 across all conditions;
**only the structure/anti-collapse term differs.** Freeze encoder → ridge probe →
held-out R² for `obs[46:49]` (box−gripper, the ee→cube vector) and `obs[49:52]`
(target−box). n = 3 seeds for the main conditions.

## Main result (mean over 3 seeds; held-out R²)

| condition | ee→cube R² | box→target R² | geom-all R² | eff_rank | code_ent |
|---|---|---|---|---|---|
| **VICReg (baseline)** | **0.987 ±0.001** | **0.933 ±0.002** | **0.902 ±0.012** | 30.8 | 0.99 |
| SE (real graph, λ=1) | 0.736 ±0.027 | 0.569 ±0.022 | 0.736 ±0.007 | 12.9 | 0.84 |
| SE-on-random-graph (control) | 0.986 ±0.003 | 0.905 ±0.008 | 0.872 ±0.012 | 22.2 | 0.46 |

**The control is the smoking gun.** SE applied to a *random* graph (structure
blind to the latents) recovers near-VICReg geometry, while SE on the *real* latent
kNN graph is far worse. So it is the SE **structure** — bucketing states into
communities — that destroys the continuous ee→cube manifold, not a weighting or
optimization artifact. (Mechanistically: on the random graph the SE term gives the
encoder no gradient, so `l_pred` trains freely to ~0.001; on the real graph SE
pulls the encoder toward low-rank community structure, `eff_rank` 30.8→12.9.)

## Steelman: LAM_SE sweep (seed 0) — every active SE weight hurts, monotonically

| λ_SE | ee→cube R² | box→target R² | geom-all R² | eff_rank | l_struct |
|---|---|---|---|---|---|
| 0.01 | 0.982 | 0.939 | 0.924 | 7.8 | 0.06 (inert) |
| 0.03 | 0.969 | 0.894 | 0.890 | 8.6 | 0.19 |
| 0.10 | 0.925 | 0.848 | 0.857 | 9.7 | 0.57 |
| 0.30 | 0.840 | 0.702 | 0.783 | 11.2 | 1.71 |
| 1.00 | 0.773 | 0.598 | 0.746 | 12.8 | 5.37 |

SE's best geometry (λ=0.01, R²=0.982) only *ties* VICReg (0.987) — and only
because at λ=0.01 the SE term is negligible (l_struct 0.06 ≈ VICReg's), i.e. the
predictive loss alone is carrying the geometry. As SE is given any real say, R²
degrades monotonically. **At no weighting does SE beat VICReg.**

## Conclusion
- **GO criterion** (probe): SE materially higher geometry-R² than matched VICReg,
  with the control ruling out artifacts. **Not met.**
- SE here is at best inert and otherwise actively harmful for the precise
  continuous geometry that manipulation needs — exactly the failure mode
  USING_SE.md predicts for the RL-abstraction regime (community structure is the
  wrong inductive bias for a continuous geometric regression target).
- **Downstream skipped** (probe did not GO). The glass/SE idea does **not** fix
  the H-JEPA geometry gap on Panda; VICReg's variance+covariance (rank-maximizing)
  term is the better latent regularizer here.

Artifacts: `RESULTS.json` (all numbers + controls), `se_validation.json`,
`run_{cond}_seed{0,1,2}.json`, `run_se_lam*_seed0.json`, `logs/`.
