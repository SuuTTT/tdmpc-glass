# DR agent 3 — model-based + abstraction intersection (2026-06-08)

**Bottom line: NO credible/replicated result shows abstraction/hierarchy/skills beating a
TUNED flat model-based baseline on DENSE continuous control.** Abstraction reliably wins only
in a DIFFERENT regime: sparse-reward, long-horizon navigation/exploration, and high-dim
offline manipulation — and usually vs *weak* baselines, not a tuned TD-MPC2. Prior on a clean
dense win: **~15-25%**. Dominant failure mode: **abstract-level model exploitation** (high-level
finds OOD abstract actions with fake reward).

Key evidence:
- NEGATIVE: Schiewer et al. 2024 (Sci Reports, "limits of hierarchical world models") — HMBRL
  "did not outperform" flat on final return, worse on HalfCheetah, cause = abstract-model
  exploitation. Director (Hafner 2022) wins only on sparse/long-horizon/maze, NOT dense DMC.
  DC-MPC / BS-MPC: match (not beat) TD-MPC2 on dense DMC; gains only on high-dim Dog/Humanoid
  or vs weaker baselines.
- POSITIVE (different regime): IQL-TD-MPC (hierarchical, OFFLINE sparse antmaze): flat ~0 →
  ~40+. TAP (offline high-dim Adroit): beats model-based TT large margin. Jumpy models: ~3×
  on OOD long-horizon vs one-step (but vs authors' own 1-step, not SOTA flat). Director:
  sparse/maze wins.

**Implication for us — this validates our findings and reframes the goal:**
- Our proprio-DMC null = EXPECTED (dense regime, abstraction doesn't beat tuned flat).
- Our CartpoleSparse H9 win (0→~700) = EXACTLY the regime where abstraction/deep-planning
  helps (sparse, exploration-limited). The Panda manipulation +38% is the weaker/risky claim.
- HONEST REFRAME: the credible contribution is "**enables sparse/long-horizon tasks the flat
  baseline FAILS at (0→solved)**", NOT "+X% on dense Walker". Target sparse/long-horizon;
  expect qualitative (fails→solves) wins, not percentage gains. Watch for abstract-model
  exploitation if we go hierarchical.

Sources: hierarchical-WM negative arxiv 2406.00483 / nature s41598-024-76719-w · Director
2206.04114 · TD-MPC2 2310.16828 · DC-MPC 2503.00653 · BS-MPC 2410.04553 · IQL-TD-MPC
2306.00867 · TAP 2208.10291 · jumpy 2302.12617.
