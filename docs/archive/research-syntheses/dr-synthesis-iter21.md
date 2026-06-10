# Deep-research synthesis — how (and whether) to beat TD-MPC2 with abstraction/skills
*2026-06-08. 3 internal agents (Laplacian/eigenoptions, MI/metric skills, model-based+abstraction).
Awaiting user's external dr-claude/gemini/gpt pass to corroborate.*

## The convergent finding (all 3 agents agree)
**Nobody has cleanly beaten a tuned flat model-based baseline (TD-MPC2/Dreamer) with
abstraction/skills/hierarchy on DENSE continuous control. The field doesn't even make that
comparison.** Where abstraction/skills/deep-planning credibly help = **sparse-reward,
long-horizon, exploration-limited** tasks (where the flat baseline scores ~0), and transfer/
multi-task. Prior on a clean dense-return win: **~15-25%**. Our proprio-DMC null was the
EXPECTED result, not bad luck.

Per-lineage:
- **MI/metric skills (METRA, LSD/CSD, DADS, DIAYN, CIC):** METRA = SOTA but for COVERAGE /
  zero-shot goal-reaching, NOT task return. DUSDi explicitly: on standard DMC locomotion
  "performance differences are minimal." "Learn skills → beat flat baseline on return" is
  **aspirational**, not achieved. Value = exploration/zero-shot/transfer.
- **Laplacian/eigenoptions (eigenoptions, covering options, DCEO):** the principled "graph
  structure → options" line. DCEO is the mature deep version (strong on sparse navigation:
  Montezuma, MiniWorld-sparse) — but **ALL experiments are discrete-action; ZERO continuous-
  control/MuJoCo precedent, and NO world-model combination exists.** Genuine unfilled gap.
  Eigenpurpose = intrinsic reward along Laplacian eigenvectors; implementable as an add-on.
- **Skills + model-based:** SkiMo (skill-dynamics model + plan in skill space) beats flat MBRL
  on SAMPLE EFFICIENCY in maze/kitchen (not return on DMC). Hierarchical world models: NEGATIVE
  (Schiewer 2024 — don't beat flat on return; abstract-model exploitation). TD-MPC2 extensions
  (DC-MPC, BS-MPC) MATCH not beat on dense DMC.

## What this means for us (validates our campaign + reframes the goal)
1. Our dense-DMC null is the literature's expected result. The honest "beat" is NOT on dense.
2. Our own strongest signal already lives in the right regime: **iter-18 H9 took CartpoleSparse
   from 0 → ~700** (sparse, exploration-limited — exactly where abstraction/deep-planning pays).
   iter-17 prototype-novelty got 745/790 (close, sub-800-solve). We're already CLOSE on sparse.
3. **Reframe the contribution:** "TD-MPC2/flat MBRL FAILS on sparse/long-horizon exploration;
   an abstraction-grounded exploration mechanism ENABLES it (0 → solved)" — a qualitative
   (fails→solves) claim, reviewer-credible, NOT "+X% on dense Walker".

## Recommended iter-21 (pending external-dr corroboration + user OK)
**Bet: DCEO-style Laplacian eigenpurpose exploration on TD-MPC2, on sparse MJX tasks.** Why:
- It's the abstraction idea USED THE RIGHT WAY — spectral/eigen-structure of the transition
  graph as exploration *directions* (eigenpurposes-as-intrinsic-reward), NOT reach-centroid
  subgoals (which failed in iter-19 because communities are phases).
- It targets the ONE regime where abstraction wins (sparse exploration) and where we're
  already close (CartpoleSparse 0→700 with H9; 745 with novelty).
- It fills a genuine, citable GAP: no Laplacian-options work in continuous control OR with a
  world model. Even a modest positive = novel.
- Implementable as an intrinsic-reward add-on (learn Laplacian rep from replay → eigenpurpose
  bonus), no architecture surgery; DCEO's random-termination trick avoids the hard part.

**Protocol:** sparse MJX suite (CartpoleSwingupSparse, BallInCup, AcrobotSwingupSparse) where
vanilla=0/bimodal. Arms: vanilla TD-MPC2 (=0 control), + a SIMPLE intrinsic-reward baseline
(RND/disagreement — the lower-risk control the research flagged), + the Laplacian-eigenpurpose
arm (the novel bet). Gate: solve-rate (last-2 ≥ 800) — Laplacian arm solves where vanilla=0,
and ≥ the RND baseline. ≥5 seeds, fixed cutoff. WIN = qualitative rescue (0→solved) that beats
both vanilla AND simple intrinsic reward → "spectral-abstraction exploration enables sparse
control that flat MBRL can't."

**Honest prior:** modest. DCEO unproven in continuous control; Laplacian objective is finicky/
policy-dependent (a 2025 paper documents it can hurt). But it's the highest-novelty bet aligned
with where abstraction actually works, and the downside is a clean, publishable negative.

**Demote:** rho+H is a task-dependent tuning footnote (let it confirm, don't expand).

## Decision for user
Reframe accepted? Build the Laplacian-exploration arm (with RND baseline) on sparse MJX? Or
wait for the external dr pass and decide together. (No new GPU committed to iter-21 yet; rho
confirmation runs finishing as the footnote.)
