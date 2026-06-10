# Refactor plan — TD-MPC-Glass

*2026-06-10. Phase 1 (dead-code removal + doc consolidation) is DONE. Phase 2+ below is the
hot-path work, DEFERRED until the iter-27 campaign drains because the daemon rsyncs the live
`src/` + `scripts/` tree into every launched run (so editing it mid-campaign risks the 47 pending
runs and muddies CODE_SHA provenance).*

## ✅ Phase 1 — done (safe; no impact on running/pending runs)
- Deleted ~5.5k lines of verified-dead code (0 importers): `algorithms/versions/`,
  `tdmpc2_patched.py`, `ppo_gymnax.py`, `dynamics/jepa.py`, `dynamics/rnn_mdn.py`, `planners/cem.py`.
- Archived ~28 stale docs (HopperHop iters 1–13, old DR syntheses, superseded ops guides) →
  `docs/archive/`; added `docs/INDEX.md`; fixed `CLAUDE.md` pointers; marked `AGENT_HANDOFF_CONTEXT.md` obsolete.

## ⏸️ Phase 2 — hot-path dedup (do AFTER campaign drains; single PR, py_compile + 1 smoke per algo path)
The gate: **0 runs in flight that depend on the live tree**, OR do it on a branch the daemon
doesn't launch from and merge during a quiet window. After merge, bump a sentinel `CODE_SHA` so
post-refactor runs are distinguishable from `i27a`.

1. **De-duplicate `tdmpc2.py` ↔ `tdmpc_glass.py`** (~350 dup lines, drift risk):
   - Extract shared `Encoder`/`Dynamics`/`RewardHead`/`QEnsemble` into `src/helios/algorithms/_core_nets.py`
     (parameterized by `latent_norm` + `arch`); both modules import them. Glass keeps only its
     prototype/cluster heads.
   - Extract shared losses (td / policy / consistency) into `src/helios/algorithms/_losses.py`;
     Glass composes its cluster/temporal/behavioral terms on top.
   - **Verify**: param-tree pytrees identical pre/post on a fixed seed (init + 1 update step) for both
     vanilla and glass; identical first-eval return on a 20k smoke.

2. **Split the monolith `run_benchmark.py` (2409 lines)**:
   - Move env-gated diagnostics `SE_DUMP` (lines ~1381–1446) and `HERMITE_CHECK` (~1452–1524) — both
     `sys.exit(0)` probes — into standalone `scripts/diagnose_se.py` and `scripts/diagnose_hermite.py`.
   - Extract reward-shaping (knee/gait/soft-stand) → `scripts/_reward_shaping.py` (pure fns, testable).
   - Collapse `train_tdmpc2`'s 59 kwargs into a `TDMPCConfig` dataclass with `validate()` that rejects
     incompatible combos (glass+jumpy, glass+dyn_arch, glass+fsq) at parse time, not runtime.

3. **Tests** (`tests/`, none exist today): core TD/policy/MPPI shapes; `_arch_backbone` all 3 branches
   produce `(B, latent_dim)`; flag-combo validation; reward-shaping in isolation. Run in CI / pre-push.

## 🗂️ Phase 3 — optional
- Decide the Hydra orphan path (`main.py` + `dreamer.py`/`rssm.py`/`tdmpc.py`): either wire a real
  entrypoint test or archive it (it's never reached by `run_benchmark.py`, the only live driver).
- Consolidate remaining overlapping ops docs into `operations/launch_dashboard.md`.

## Risk notes
- `control/task_queue_daemon.py` rsyncs `-az --delete` → deleting a *used* file would break workers.
  Phase-1 deletions were all 0-importer (verified). Any Phase-2 file move must keep import paths valid.
- Provenance: each run records `launched_git_sha`. A refactor changes the SHA even when behavior is
  identical — that's why Phase 2 waits for a clean campaign boundary.
