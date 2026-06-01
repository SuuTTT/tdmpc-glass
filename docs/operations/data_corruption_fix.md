# Duplicate-launch CSV corruption — detection, repair, prevention

**Date:** 2026-06-01 · **Tool:** `control/separate_collisions.py`

## Symptom (what the user observed)

Some learning curves "go back" — the reward-vs-step plot jumps to a much lower
step partway through, then climbs again. On the raw CSV this shows up as the
`step` column **decreasing** between consecutive rows, e.g.:

```
8750080,487.4,pi,3
8750080,512.5,mppi,3
250112,27.8,pi,3      <- step jumps BACKWARD (8.75M -> 250k)
250112,49.1,mppi,3
```

## Root cause

Two training processes wrote the **same** per-seed CSV at the same time (or one
after another). Every Glass run appends to
`exp/tdmpc_glass/HopperHop_<TAG>/seed_<S>/seed_<S>.csv` (+ `_diag.csv`), keyed
only by `TDMPC_GLASS_OUTPUT_TAG` and seed. If the queue daemon launched a second
run with the **same tag+seed** on a second GPU of a multi-GPU box, both wrote the
same path and their evaluation rows interleaved.

Two distinct corruption shapes were seen:

- **Sawtooth interleave** — both processes alive concurrently, each ~1 eval
  apart, so steps oscillate `250k,500k,250k,750k,500k,…` (e.g. `phaseq_knee`
  seed_5: 36 backward jumps).
- **Sequential concat** — one run partially completed, a second run started from
  scratch and appended `250k,500k,…` after the first run's high steps (e.g.
  `phasei9r/phasei9t` seed_4: a single backward jump).

## Prevention (already in place)

`control/task_queue_daemon.py :: is_box_idle()` was hardened (prior change) so a
box reporting `n_running_benchmark_procs >= n_gpu` is treated as **busy**, and a
target GPU is only considered free when its `memory.used <= 100 MiB`. This stops
the daemon from stacking a duplicate tag+seed onto a second GPU of the same box.

## Detection

Scan every per-seed CSV for a backward `step` jump. Plot files
(`*_95ci_curve.csv`, `*_summary.csv`) start with a `series` column and legitimately
restart per series — they are **excluded**. Files named `interleaved*` / `*rerun*`
are deliberate multi-seed combos — separated by the `seed` column, not repaired.

```bash
cd /home/ubuntu/tdmpc-glass
python3 control/separate_collisions.py            # dry-run report (no writes)
```

## Repair method — de-interleave, never delete

The original is **never deleted or edited in place**. The repair splits the
tangled rows back into the distinct monotonic runs that produced them, using a
greedy "closest-behind stream" assignment (robust to non-uniform step
increments): walk `(pi, mppi)` eval pairs; append each to the output stream whose
last step is the largest value still `< s`; open a new stream when none qualifies.

### Completed/historical phases — canonical rewrite

```bash
python3 control/separate_collisions.py --apply
```

For each corrupted `seed_N.csv`:

| file | meaning |
|------|---------|
| `seed_N.csv.collided` | the **original**, renamed (preserved, never deleted) |
| `seed_N.run1.csv`, `seed_N.run2.csv`, … | the de-interleaved runs (run1 = most complete) |
| `seed_N.csv` | rewritten = the most-complete run (clean curve for plots) |

### Live phases — non-destructive snapshot (canonical swap deferred)

When a run is **still writing** the file, renaming it would detach the live
writer to a stale inode and freeze the dashboard. For those, snapshot only — the
original is left fully untouched:

```bash
python3 control/separate_collisions.py --apply --ts <stamp> \
  --snapshot-live exp/.../seed_N.csv exp/.../seed_N_diag.csv
```

Produces `seed_N.csv.collided_snapshot_<stamp>` (backup copy) and
`seed_N.snapshot_<stamp>.runK.csv` (de-interleaved views). The canonical
`seed_N.csv` is **not** swapped — re-run `--apply` after the run completes, when
the now-finished live run is correctly the most-complete one.

## What was fixed on 2026-06-01

Scanned 898 CSVs; 19 had backward jumps. Of those, 8 were real per-seed
collisions, 3 were deliberate interleaved/rerun combos, 8 were legit multi-series
plot CSVs (left alone).

**Historical — canonical rewritten (local mirror + live remote where applicable):**

| phase / seed | shape | run1 reaches | run2 reaches | repaired on |
|---|---|---|---|---|
| `phaseq_knee` seed_5 (+diag) | sawtooth, 36 jumps | 10.00M | 10.00M | local mirror (`ssh6_4060`, stale box) |
| `phasei9r_p1b_off1m_s4` seed_4 (+diag) | concat | 10.00M | 7.25M | local mirror **+ remote `ssh9` :17647** |
| `phasei9t_p1b_off1p5m_s4` seed_4 (+diag) | concat | 8.25M | 8.00M | local mirror **+ remote `ssh9` :17647** |
| `phaseg2_tempstab_0.05` seed_1 (+diag) | concat | 8.50M | 0.50M (dup stub) | local (not mirrored) |

**Deliberate interleaved combos — split by seed, original kept as-is:**

- `_final_snapshot/.../hopper-hop-tdmpc2-rerun.csv` → `.seed1.csv` / `.seed2.csv`
- `remote_mirror/ssh17637_2x3060/.../interleaved_s1_s2_latest.csv` → `.seed1/.seed2`
- `remote_mirror/ssh17637_2x3060/.../interleaved_s1_s2_partial.csv` → `.seed1/.seed2`

**Live i10p runs on `ssh9` :17647 — non-destructive snapshots, canonical swap deferred:**

| phase / seed | live writer | snapshot run1 | snapshot run2 |
|---|---|---|---|
| `phasei10p_k2_temp0005_auto_s3` seed_3 (+diag) | pid 459088 (healthy, ~9.25M) | real run 9.25M, growing | dup stub (2 rows) |
| `phasei10p_k2_temp0005_auto_s4` seed_4 (+diag) | pid 705959 (fresh, restarted) | old dead run 7.00M | live run, growing |

> **Follow-up:** when the two i10p live runs finish, re-run
> `separate_collisions.py --apply` on those two seed dirs (on the remote, then let
> the streamer mirror) to promote the now-complete live run to canonical. For
> `s4` the live (restarted) run — not the older 7M run — is the correct keeper.

## Verification

After repair, every canonical `seed_N.csv` has **zero** backward jumps; the only
remaining backward-jump files are the deliberate `interleaved*`/`rerun` inputs
(kept intact alongside their `.seedN` splits) and the `series`-column plot CSVs.
Remote and local canonical files match in size, so `rsync -av` (no `--delete`)
will not re-clobber the fix.
