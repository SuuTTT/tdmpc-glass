# Iteration 27 — Manipulation re-benchmark (the supervisor redirect)

*2026-06-10. DMC tasks are weak discriminators (saturate ~1000 or floor at 0). Move evaluation to graded,
contact-rich MANIPULATION (Franka/MJX), redone with all lessons. Replaces the distractor-DMC value-equiv
gate (iter-26, dropped).*

## Suite (all verified to load): PandaPickCube(66/8), PandaPickCubeCartesian(70/3),
PandaPickCubeOrientation(66/8), PandaRobotiqPushCube(48/7), PandaOpenCabinet(55/8).

## Arms (single-variable; fair protocol; peak AND final; n=3->5; >=400k; CI; no tricks; mechanism-check before expand)
- van  = vanilla TD-MPC2 (H3)            -- baseline
- jum  = jumpy WM (k4, macro-MPPI H8)     -- does the +44%/+80% PandaPickCube win GENERALIZE across Franka?
- ve   = jumpy + value-equivalent head    -- the control-useful abstraction (Q6): does VE help on manipulation?
(Glass = settled null, not re-run; §0 already shows it; could add 1 confirmation arm later.)

## Gates
- jumpy: is jum > van CI-separated on >=3/5 tasks (peak & final)? -> jumpy generalizes.
- value-equiv: is ve >= jum (esp. tasks with task-irrelevant DoF / distractor-like obs)? -> VE earns its place.
Start n=3 for a fast cross-task read; expand discriminating tasks to n=5.

## Next (item D): architecture A/B — swap MLP backbone for transformer/attention or deeper gated MLP,
single-variable vs MLP on this suite. Separate build; smoke-before-fanout.
