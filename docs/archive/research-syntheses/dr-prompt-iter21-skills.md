# Deep-research prompt — abstraction/skills to beat TD-MPC2 in continuous control

*(Paste into Claude/Gemini/GPT deep-research. Self-contained. 2026-06-08.)*

---

I am trying to **beat vanilla TD-MPC2 at the architecture/model/algorithm level using an
abstraction or skill/temporal-abstraction idea**, on continuous-control tasks implemented in
**MuJoCo Playground (MJX/JAX)** — DM-Control suite (incl. sparse: CartpoleSwingupSparse,
BallInCup, AcrobotSwingupSparse), and manipulation (PandaPickCube). Constraints: a **fair,
compute-matched protocol** (only the mechanism changes vs vanilla TD-MPC2; identical
hyperparameters/steps), rliable IQM + bootstrap CIs over ≥5 seeds, pre-registered kill gates.
I want an **architecture/algorithm contribution, not hyperparameter tuning.**

**What I already established empirically (so don't re-suggest these):**
1. On clean proprioceptive DM-Control, TD-MPC2 is essentially **unimprovable by latent
   abstraction**: geometric prototype clustering ≈ vanilla (SimNorm already gives soft-discrete
   codes); a behavioral/bisimulation auxiliary is null-to-harmful; a discrete FSQ codebook
   swap hurts locomotion. (Consistent with Ni et al. 2024: TD-MPC2's self-predictive objective
   is already a sufficient abstraction.)
2. **Horizon probe:** longer MPPI planning horizon (H=9 vs default H=3) *helps* sparse tasks
   (CartpoleSparse 0 → ~700: finds reward H=3 never reaches) but *hurts* manipulation
   (PandaPickCube collapses, ~1700→420: compounding 1-step model error over the longer
   rollout). Raising the consistency-loss horizon-decay (rho 0.5→0.9) stabilizes/boosts
   manipulation deep planning (+38%) but suppresses sparse exploration — task-dependent
   tuning, not architecture.
3. **Community-detection skills FAILED:** I built a transition-graph over latent prototypes,
   ran Louvain → communities, defined skills as "reach a community centroid" via
   goal-conditioned MPPI. Communities exist + are stable (modularity ~0.7, ARI ~0.9), but
   goal-MPPI reaches them no better than random. Diagnosis: in continuous control the latent
   "communities" are **transient motion/gait PHASES, not persistent spatial subgoals** —
   "reach and stay in community c" is ill-posed; centroids are non-physical averages.

**Questions (please answer each with citations, methods, and concrete numbers where possible):**

1. **Which unsupervised skill / temporal-abstraction / option-discovery methods actually
   deliver in CONTINUOUS control** (not tabular/gridworld)? Compare the leading lineages —
   e.g. METRA (metric-aware/Wasserstein-MI), DADS, DIAYN/CIC (MI), LSD/CSD
   (Lipschitz/controllability), and the **graph-Laplacian/eigenoption line (eigenoptions,
   covering options, Deep Covering Eigenoptions / DCEO, successor-representation options)**.
   For each: does it scale to MJX-style locomotion/manipulation, what's its known failure
   mode, and does it improve *task return* or only *exploration/zero-shot transfer*?

2. **Has anyone combined skill/option/temporal abstraction WITH a model-based planner**
   (TD-MPC / TD-MPC2 / Dreamer / DreamerV3 / TWM) and **beaten the flat model-based baseline**
   on continuous-control returns? (e.g. options/skills planned over a learned world model;
   jumpy/temporally-abstract world models; Director; hierarchical world models.) If yes, how;
   if the evidence is weak/negative, say so plainly. I specifically need the **model-based +
   abstraction intersection**, which I suspect is underexplored.

3. **Given my findings, what is the single cleanest, lowest-risk method to try next** to beat
   TD-MPC2 on sparse/long-horizon MJX tasks, and how exactly would I instantiate it on top of
   TD-MPC2? Candidates I'm weighing — please rank and critique, or propose better:
   (a) DCEO/eigenoption-style **Laplacian options as exploration** (intrinsic eigenpurpose
       rewards — directly attacks sparse tasks where vanilla H=3 gets 0; uses graph structure
       the RIGHT way, as eigen-directions not reach-centroid);
   (b) a **jumpy/temporally-abstract world model** (predict z_{t+k} in one call) so MPPI gets
       long effective horizon WITHOUT the compounding error that sinks H=9 on manipulation;
   (c) **METRA/LSD-style skills** learned then planned over by TD-MPC2's MPPI;
   (d) something else.

4. **Honest meta-question:** given TD-MPC2 is already a strong self-predictive abstraction and
   my proprio-DMC null, **is "beat TD-MPC2 with abstraction" the right goal at all**, or should
   the contribution be reframed (e.g. exploration on sparse/long-horizon, or transfer/multi-task
   where abstraction's reuse value is measurable)? What would make a *reviewer-credible*
   architecture-level contribution here?

**Deliverable:** a ranked recommendation with (i) the method to build, (ii) why it survives my
specific failures above, (iii) the exact benchmark + metric + kill-gate to test it, (iv) key
citations, and (v) realistic expected outcome (is a clean win plausible, or is the honest
result likely another null?).
